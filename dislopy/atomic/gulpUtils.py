#!/usr/bin/env python
'''This program parses an input gulp file and extracts all information
that is required to run a simulation, including cell parameters, atomic
coordinates, species, potentials, etc.
'''
from __future__ import division, print_function, absolute_import

import sys

import numpy as np
import re
import subprocess, os

from numpy.linalg import norm
from shutil import copyfile

from dislopy.atomic import crystal as cry
from dislopy.utilities import atomistic_utils as util
from dislopy.atomic import transmutation as mutate
from dislopy.atomic import rodSetup as rs
#from dislopy.atomic.multisite import sites_to_replace, sites_to_replace_neb 

from dislopy.atomic.rodSetup import __dict__ as rod_classes

# Functions to write different GULP input files

class GulpAtom(cry.Atom):
    '''Same as <cry.Atom>, but contains an additional variable to record
    whether it has a shell or breathing shell.
    '''

    def __init__(self, atomicSymbol, coordinates=np.zeros(3)):
        '''Creates an atom object with additional properties relating to the
        realization of atoms in GULP.
        '''

        #cry.Atom.__init__(atomicSymbol, coordinates)
        super(GulpAtom, self).__init__(atomicSymbol, coordinates)
        self.__hasShell = False
        # records whether the atom has a breathing shell
        self.__isBSM = False

        # records whether we are using cluster-order indices (z, x, y)
        self._cluster_ordered = False

    def addShell(self, shelCoords, shellType='shel', frac=False, pfrac=False):
        '''Gives the atom a polarizable shell, with coordinates <shelCoords>.
        May add position checking later.
        '''

        self.__hasShell = True
        self.__shelCoords = self.getShellDistance(shelCoords, frac=frac, pfrac=pfrac)

        if shellType in 'bshe':
            # using a breathing shell model
            self.__isBSM = True
            
        # default to free relaxation of shell coordinates
        self._fix_shell = False

        return

    def copyShell(self, shelDisplacement, shellType='shel', fixshell=False):
        '''When copying atoms, shell is already expressed as a displacement
        from the core, which means that we do not need to invoke
        <GulpAtom.getShellDistance()>.
        '''

        self.__shelCoords = shelDisplacement.copy()
        self.__hasShell = True
        if shellType in 'bshe':
            self.__isBSM = True
            
        self._fix_shell = fixshell

        return

    def hasShell(self):
        '''Tells user if atom has a shell.
        '''

        return self.__hasShell

    def isBreathingShell(self):
        '''Tells user if the shell has a breathing radius.
        '''

        return self.__isBSM

    def getShell(self):
        '''Returns the shell coordinates.
        '''

        try:
            return self.__shelCoords.copy()
        except AttributeError:
            print('Error: atom does not have a shell.')
            return None

    def getShellDistance(self, shelCoords, frac=False, pfrac=False):
        '''Returns the distance between the centre of the shell and the
        (undisplaced) coordinates of the ionic core. If <frac> is True, coordinates
        are given in fractional units.
        '''

        if frac:
            return np.copy(shelCoords % 1 - self.getCoordinates() % 1)
        elif pfrac:
            # get shell coordinates. Note that z corresponds to index 0
            s1 = shelCoords[0] % 1
            s2 = shelCoords[1]
            s3 = shelCoords[2]
            # get core coordinates
            c1 = self.getCoordinates()[0] % 1
            c2 = self.getCoordinates()[1]
            c3 = self.getCoordinates()[2]
            
            # calculate shortest distance between the core and any periodic image
            # of the shell along the dislocation line
            d1 = s1-c1
            if abs(s1+1-c1) < abs(d1):
                d1 = s1+1-c1
            if abs(s1-1-c1) < abs(d1):
                d1 = s1-1-c1
            
            # return final core-shell distance
            return np.copy([d1, s2-c2, s3-c3])
        else:
            return np.copy(shelCoords - self.getCoordinates())

    def setShell(self, newShell):
        '''Changes coordinates of the polarizable shell to <newShell>.
        '''

        self.__shelCoords = np.array(newShell).copy()
        return

    def normaliseShelCoordinates(self, cellA=1, cellB=1, cellC=1):
        '''Normalise shell coordinates by dividing by given factors. Useful for
        setting up supercell calculations.'''

        normalisationFactor = np.array([float(cellA), float(cellB),
                                        float(cellC)])

        self.__shelCoords = self.__shelCoords/normalisationFactor
        return

    def normaliseCoordinates(self, cellA=1, cellB=1, cellC=1):
        '''Normalise coordinates of atom (including shell).
        '''

        # normalise coordinates of the atomic core
        super(GulpAtom, self).normaliseCoordinates(cellA, cellB, cellC)

        # test to see if atom has a shell, if True, normalise its coordinates
        if self.hasShell():
            self.normaliseShelCoordinates(cellA, cellB, cellC)

        return
        
    def to_cart(self, lattice):
        '''Expressions the atomic position in cartesian coordinates.
        '''
        
        # get coordinates in perfect and deformed material (in cartesian coordinates)
        newcoords = cry.fracToCart(self.getCoordinates(), lattice.getLattice())
        newdisp = cry.fracToCart(self.getDisplacedCoordinates(), lattice.getLattice())
        
        self.setCoordinates(newcoords)
        self.setDisplacedCoordinates(newdisp)
        
        if self.hasShell():
            newshell = cry.fracToCart(self.getShell(), lattice.getLattice())
            self.setShell(newshell)
        
        return

    def clusterOrder(self):
        '''Permutes indices so that the position vector reads (z, x, y), the
        order used in a GULP polymer cell calculation.
        '''
        
        if self._cluster_ordered:
            return
        else:
            # coords in undislocated cell
            atom_coords = self.getCoordinates()
            atom_coords = np.array([atom_coords[2], atom_coords[0], atom_coords[1]])
            self.setCoordinates(atom_coords)

            # coords in dislocated cell
            u_coords = self.getDisplacedCoordinates()
            u_coords = np.array([u_coords[2], u_coords[0], u_coords[1]])
            self.setDisplacedCoordinates(u_coords)
            
            # constraints
            cons = self.get_constraints()
            self.set_constraints(np.array([cons[-1], cons[0], cons[1]]))

            # shell coordinates
            if self.hasShell():
                shel_coords = self.getShell()
                shel_coords = np.array([shel_coords[2], shel_coords[0], shel_coords[1]])
                self.setShell(shel_coords)

            # recording that we are using cluster-ordered coordinate indices
            self._cluster_ordered = True
        
    def from_cluster(self, height=1):
        '''Permutes indices so that the position vector reads (x, y, z), given a
        vector with the order (z, x, y) (ie. in polymer coordinates).
        '''

        # coords in undislocated cell
        atom_coords = self.getCoordinates()
        atom_coords = np.array([atom_coords[1], atom_coords[2], atom_coords[0]])
        self.setCoordinates(atom_coords)
            
        # coords in dislocated cell
        u_coords = self.getDisplacedCoordinates()
        u_coords = np.array([u_coords[1], u_coords[2], u_coords[0]])
        self.setDisplacedCoordinates(u_coords)

        # shell coordinates
        if self.hasShell():
            shel_coords = self.getShell()
            
            # reorder
            shel_coords = np.array([shel_coords[1], shel_coords[2], shel_coords[0]])
            self.setShell(shel_coords)

        # recording that we are using cluster-ordered coordinate indices
        self._cluster_ordered = False
        
    def shell_fix(self):
        '''Sets the shell coordinates to be fixed.
        '''
        
        self._fix_shell = True
       
    def shell_free(self):
        '''Sets the shell coordinates to be free.
        '''
        
        self._fix_shell = False

    def write(self, outstream, lattice=cry.Lattice(), defected=True, to_cart=True,
                                         add_constraints=False):
        '''Writes an atom (including Dick-Overhauser or BSM polarizable shell,
        if present) to <outstream>. If <add_constraints> is true, include
        geometric constraints in output.

        If <cluster_order> is True, then the ordering of the elements of the
        atoms' locations is (z, x, y). Critically, this means that the
        '''

        atom_format = '{} {} {:.6f} {:.6f} {:.6f}'

        # test to see if atom should be output
        if not self.writeToOutput():
            return # otherwise, write atom to <outstream>

        # get coordinates of atomic core. Note that these will already be in
        # the correct units thanks to the routines in <crystal> and <rodSetup>.
        if defected:
            coords = self.getDisplacedCoordinates()
        else:
            coords = self.getCoordinates()

        outstream.write(atom_format.format(self.getSpecies(), 'core', coords[0],
                                                 coords[1], coords[2]))

        if add_constraints:
            # add constraints if non-trivial
            outstream.write(' 1.0 {:d} {:d} {:d}\n'.format(int(self.get_constraints()[0]),
                          int(self.get_constraints()[1]), int(self.get_constraints()[2])))
        else:
            outstream.write('\n')

        # test to see if the atom has a shell
        if self.hasShell():
            # calculate coordinates of shell in deformed material
            if to_cart:
                # convert to cartesian coordinates
                if self._cluster_ordered:
                    shel_cor_dist = cluster_cartesian(self.getShell(), lattice)
                else:
                    shel_cor_dist = cry.fracToCart(self.getShell(), lattice)
            else:
                shel_cor_dist = self.getShell()

            new_shel_coords = coords + shel_cor_dist
            # determine type of shell (ie. Dick-Overhauser shell model or BSM
            shell_type = ('bshe' if self.isBreathingShell() else 'shel')
            outstream.write(atom_format.format(self.getSpecies(), shell_type,
                                  new_shel_coords[0], new_shel_coords[1],
                                                      new_shel_coords[2]))

            if add_constraints and self._fix_shell:
                outstream.write(' 1.0 0 0 0 \n')
            elif add_constraints and not self._fix_shell:
                outstream.write(' 1.0 1 1 1\n')
            else:
                outstream.write('\n')

        return

    def copy(self):
        '''Creates a copy of the atom.
        '''

        # copy over everything but the displacement field
        new_atom = GulpAtom(self.getSpecies(), self.getCoordinates())

        # ...then copy the displaced coordinates and the shell coordinates (if
        # the atom has a shell)
        new_atom.setDisplacedCoordinates(self.getDisplacedCoordinates())
        
        # copy constraints
        new_atom.set_constraints(self.get_constraints())
        
        # add shell
        if self.hasShell():
            new_atom.copyShell(self.getShell(), 'bshe' if self.__isBSM else 'shel',
                                                             self._fix_shell)
        else:
            # core only
            pass

        # check to see if atom is writable to output
        if self.writeToOutput():
            pass
        else:
            new_atom.switchOutputMode()

        return new_atom

def writeSuper(cell, sys_info, outFile, relax='conv', coordinateType='frac',
                                                      add_constraints=False):
    '''Writes file to <outFile> for supercell (3D periodic) calculation. If
    the total number of atoms is <= 1000, optimization algorithm defaults to
    BFGS, otherwise, begin with conjugate gradients and then, when the gnorm
    is sufficiently small, switch to BFGS.
    '''

    # files to hold the perfect and dislocated crystals
    perOutStream = open('ndf.{}.gin'.format(outFile), 'w')
    disOutStream = open('dis.{}.gin'.format(outFile), 'w')

    for outstream in [perOutStream, disOutStream]:
        if outstream == perOutStream:
            # do not use displaced coordinates
            defected=False
            basename = 'dis.{}'.format(outFile)
        elif outstream == disOutStream:
            # use displaced coordinates
            defected=True
            basename = 'ndf.{}'.format(outFile)

        write_gulp(outstream, cell, sys_info, defected=defected, to_cart=False,
                                add_constraints=add_constraints, relax=defected)

    return

def write1DCluster(cluster, sys_info, outname, add_constraints=True, maxiter=100):
    '''Writes 1D-periodic simulation cell to file. Always write an accompanying
    undislocated cluster, so that dislocation energy can be calculated. <maxiter>
    gives the maximum number of iterations allowed. GULP defaults to 100, but
    this is somewhat too high in most circumstances.
    '''

    # open files for the perfect and dislocated clusters
    perOutStream = open('ndf.{}.gin'.format(outname), 'w')
    disOutStream = open('dis.{}.gin'.format(outname), 'w')

    for outstream in [perOutStream, disOutStream]:
        # fourth variable tells <writeRegion> which coordinates to use
        if outstream == perOutStream:
            disloc = False
            relax = ''
            do_relax = False
            basename = 'ndf.{}'.format(outname)
        else:
            disloc = True
            if add_constraints:
                relax = ''
            else:
                relax = 'conv'
            do_relax = True
            prefix = 'dis.{}'.format(outname)

        write_gulp(outstream, cluster, sys_info, defected=disloc, do_relax=do_relax, 
                 relax_type=relax, maxiter=maxiter, add_constraints=add_constraints)

    return

# Master level functions for opening and parsing GULP input

atomLine = re.compile(r'([A-Z][a-z]?\d*)\s+(core|c|shel|s|bshe|bcor)' + \
                                      '((\s+-?\d+\.\d+|\s+-?\d+/\d+){3})')
#speciesLine = re.compile('^([A-Z][a-z]?\d*?)\s+(core|c|shel|s|bshe|bcor)' + \
#                                                          '\s+-?\d+\.\d+\s*$')

def preamble(outstream, maxiter=500, do_relax=True, relax_type='', polymer=False,
                  molq=False, prop=True, add_constraints=False, transition=False):
    '''Writes overall simulation parameters (relaxation algorithm, maximum 
    number of iterations, etc.) to <outstream>.
    
    #!!! Need to consider whether or not it is necessary to parse the preamble
    #!!! of the original input file for additional keywords that may be required
    #!!! to completely describe the interatomic potentials (eg. molq).
    '''

    # construct the control line  
    if do_relax:
        if relax_type is None:
            outstream.write('opti qok bfgs')
        else:
            outstream.write('opti qok bfgs {}'.format(relax_type))
    elif transition:
        outstream.write('transition qok')
    else:
        outstream.write('qok ')
    
    if molq:
        outstream.write(' molq')
        
    if polymer and not add_constraints:
        outstream.write(' eregion\n') # DO NOT TRY TO CALCULATE PROPERTIES
    elif prop and not polymer:
        outstream.write(' prop\n')
    else:
        outstream.write('\n')

    # maximum allowable number of relaxation steps. Our default is somewhat 
    # smaller than the default in the GULP source code
    outstream.write('maxcyc {}\n'.format(maxiter))
    return
    
def write_gulp(outstream, struc, sys_info, defected=True, do_relax=True, to_cart=True,
                 add_constraints=False, relax_type='conv', impurities=None, prop=True,
               pfractional=False, maxiter=500, rI_centre=np.zeros(2), transition=False):
    '''Writes the crystal <gulp_struc> to <outstream>. If <defected> is True,
    then it uses the displaced coordinates, otherwise it uses the regular atomic
    coordinates (ie. their locations in a perfect crystal with the provided
    dimensions. If <supercell> is True, then the simulation cell has 3D periodic
    boundary conditions. If not, it is assumed to be a 1D-periodic simulation
    cell, characterised by the cell height.
    
    Note that if <relax> is False, then the variable <relax_type> will be ignored.
    '''

    # determine if boundary conditions are 1D-periodic (polymer) or 3D-periodic
    # (supercell)
    get_class = re.compile(r"<class\s+'(?:\w[\w\d]*\.)*(?P<class>\w[\w\d]*)'>")
    try:
        struc_type = get_class.search(str(struc.__class__)).group('class')
    except IndexError:
        print("Class of <struc> not found.")
        sys.exit(1)
        
    # insert impurities, if supplied. If <defected> is False, we are using the
    # perfect (ie. dislocation-free) cell, and so we DO NOT make use of the 
    # displaced coordinates when setting the coordinates of the impurity atoms
    if not (impurities is None):
        if mutate.is_single(impurities):
            mutate.cell_defect(struc, impurities, use_displaced=defected)
        elif mutate.is_coupled(impurities):
            mutate.cell_defect_cluster(struc, impurities, use_displaced=defected)
        else:
            raise TypeError("Supplied defect not of type <Impurity>/<CoupledImpurity>")   

    # write simulation cell geometry to file
    if struc_type in rod_classes:
        struc.specifyRegions(rI_centre=rI_centre)
            
        # polymer cell -> write cell height
        preamble(outstream, do_relax=do_relax, polymer=True, relax_type=relax_type,
            maxiter=maxiter, add_constraints=add_constraints, transition=transition)
        height = struc.getHeight()
        outstream.write('pcell\n')
        outstream.write('{:.6f} 0\n'.format(height))
        cell_lattice = struc.getBaseCell().getLattice()

        # write atoms to output
        if pfractional:                
            writeRegion(struc.getRegionIAtoms(), cell_lattice, outstream, 1, 
                                          defected, coordType='pfractional', 
                                             add_constraints=add_constraints)
            writeRegion(struc.getRegionIIAtoms(), cell_lattice, outstream, 2, 
                                            defected, coordType='pfractional', 
                                             add_constraints=add_constraints)
        else: # cartesian                
            writeRegion(struc.getRegionIAtoms(), cell_lattice, outstream, 1, 
                                      defected, add_constraints=add_constraints)
            writeRegion(struc.getRegionIIAtoms(), cell_lattice, outstream, 2,
                                      defected, add_constraints=add_constraints)
    else:
        # write lattice vectors
        preamble(outstream, do_relax=do_relax, relax_type=relax_type, prop=prop,
                                          maxiter=maxiter, transition=transition) 
        writeVectors(struc.getLattice(), outstream)
        
        if relax_type is None or relax_type == '':
            # give strain optimization flags
            outstream.write('0 0 0 0 0 0\n')
            # GULP requires that optimization flags be set for all atoms
            if do_relax:
                add_constraints = True

        # write atoms to output.
        outstream.write('frac\n')
        for atom in struc:
            atom.write(outstream, lattice=struc.getLattice(), defected=defected,
                          to_cart=to_cart, add_constraints=add_constraints)

    # write system specific simulation parameters (interatomic potentials, etc.)
    for line in sys_info:
        if 'output' in line:
            continue
        elif 'dump' in line:
            continue
        else:
            outstream.write('{}\n'.format(line))
        
    # add restart lines and close the output file
    restart(outstream)
    outstream.close()
    
    # undo defect insertion
    if not (impurities is None):
        mutate.undo_defect(struc, impurities)
        if struc_type in rod_classes:
            struc.specifyRegions()

    return

def restart(outstream, every=10):
    '''Adds restart (.grs) and .xyz lines to a gulp input file <outstream>. Use 
    a larger value of <every> if calculations are expensive and you need to back
    up more frequently.
    '''
    
    # find basename
    if 'gin' in outstream.name:
        name_form = re.compile('(?P<base>.+)\.gin')
        basename = name_form.match(outstream.name).group('base')
    else:
        basename = outstream.name
    
    # write dump lines
    if not every:
        outstream.write('dump every {} {}.grs\n'.format(every, basename))
    else:
        # dump restart file only at the end of the calculation
        outstream.write('dump {}.grs\n'.format(basename))
    outstream.write('output xyz {}'.format(basename))
    return

def parse_gulp(filename, crystalStruc, path='./'):
    '''Parses the file <line>  to extract structural information, atomic
    potentials, and control parameters. <crystalStruc> must be initialised (with
    dummy cell parameters) prior to use.
    '''

    gulp_lines = util.read_file(filename, path)

    # <systemInformation> stores species, potentials, etc.
    sysInfo = []
    atomsNotFound = True
    allAtoms = dict()

    i = 0

    for i, line in enumerate(gulp_lines):
        if line.startswith('#'):
            # this line is a comment, skip it
            continue
        elif line.strip() in 'vectors':
            #can read in vectors directly
            for j in range(3):
                temp = gulp_lines[i+1+j].split()
                cellVector = np.array([float(temp[k]) for k in range(3)])
                crystalStruc.setVector(cellVector, j)
        elif line.strip() in 'cell':
            #different stuff
            cellParameters = gulp_lines[i+1].split()[:6]
            cellParameters = [float(a) for a in cellParameters]
            # reformat cell vectors
            cellVectors = cry.cellToCart(cellParameters)
            for j in range(3):
                crystalStruc.setVector(cellVectors.getVector(j), j)
        elif line.strip() in 'pcell':
            # reading in a 1D-periodic cluster -> record the cell height
            cell_height = float(gulp_lines[i+1].strip().split()[0])
            crystalStruc.setC(np.array([0., 0., cell_height]))
        else:
            foundAtoms = atomLine.match(line)
            if foundAtoms:
                if atomsNotFound:
                    # record that we are now reading atom info
                    atomsNotFound = False
                    
                    # check to see if the atomic positions are specified in 
                    # fractional coordinates
                    if gulp_lines[i-1].rstrip().startswith('frac'):
                        frac = True
                        pfrac = False
                    elif gulp_lines[i-1].rstrip().startswith('pfrac'):
                        frac = False
                        pfrac = True
                    else:
                        pfrac = False
                        frac = False

                # extract atom info to <allAtoms>
                extractAtom(foundAtoms, allAtoms, frac=frac, pfrac=pfrac)

            elif (not atomsNotFound) and (not foundAtoms):
                # Locates the end of the atomic line section
                if ('dump' not in line) and ('switch' not in line):
                    if not re.match('\s*\w+\s*region\s*\d+', line):
                        sysInfo.append(line)

    for element in allAtoms:
        for atom in allAtoms[element]['atoms']:
            crystalStruc.addAtom(atom)

    # delete list of atoms
    del allAtoms

    return sysInfo

# Utility functions used to parse specific GULP input

def extractAtom(atomRegex, atomsDict, frac=False, pfrac=False):
    '''Extracts atom info found in <atomRegex> to existing dictionary of atoms
    <atomsDict>.
    '''

    atomicSymbol = atomRegex.group(1)

    if atomicSymbol in atomsDict:
        # extract atomic symbol and coordinates
        typeOfAtom = atomRegex.group(2)
        tempCoords = atomRegex.group(3)
        tempCoords = tempCoords.split()
        atomCoords = np.array([float(eval(x)) for x in tempCoords])
        
        # test to see if the "atom" is in fact a polarizable shell
        if typeOfAtom in 'shell':
            index = atomsDict[atomicSymbol]['shells']
            atomsDict[atomicSymbol]['atoms'][index].addShell(atomCoords, frac=frac,
                                                                       pfrac=pfrac)
            atomsDict[atomicSymbol]['shells'] += 1
        elif typeOfAtom in 'bshe':
            index = atomsDict[atomicSymbol]['shells']
            atomsDict[atomicSymbol]['atoms'][index].addShell(atomCoords,
                                                            shellType='bshe')
            atomsDict[atomicSymbol]['shells'] += 1
        else:
            newAtom = GulpAtom(atomicSymbol, atomCoords)
            atomsDict[atomicSymbol]['atoms'].append(newAtom)
    else:
        atomsDict[atomicSymbol] = dict()
        atomsDict[atomicSymbol]['shells'] = 0
        atomsDict[atomicSymbol]['atoms'] = []
        if atomRegex.group(2) in 'shell' or atomRegex.group(2) in 'bshe':
            print('Error')
        else:
            tempCoords = atomRegex.group(3)
            tempCoords = tempCoords.split()
            atomCoords = np.array([float(x) for x in tempCoords])
                
            # create atom and add it to the appropriate collection
            newAtom = GulpAtom(atomicSymbol, atomCoords)
            atomsDict[atomicSymbol]['atoms'].append(newAtom)

    return

def writeRegion(region_basis, lattice, outstream, regionNumber, disloc,
                   use_cart=True, coordType='cartesian', add_constraints=False):
    '''Outputs all atoms in <regionBasis> to <outstream>, preceded by
    "cart region 1 if <regionNumber> == 1 and with all refinement flags set
    to 1, or cart region 2 rigid and all refinement flags == 0 if
    <regionNumber> == 2. <disloc> tells us whether to use displaced coords.
    '''

    # output atoms to file, remembering to put them in cluster ordering, ie.
    # (z,x,y)
    if regionNumber == 1:
        outstream.write("{} region 1\n".format(coordType))
            
        for atom in region_basis:
            atom.clusterOrder()
            atom.write(outstream, lattice, defected=disloc, to_cart=use_cart,
                                                add_constraints=add_constraints)
    elif regionNumber == 2:
        outstream.write("{} region 2 rigid\n".format(coordType))
            
        for atom in region_basis:
            atom.clusterOrder()
            atom.write(outstream, lattice, defected=disloc, to_cart=use_cart,
                                                add_constraints=add_constraints)
    else:
        raise ValueError('{} is not a valid region.'.format(regionNumber))

    return

def writeVectors(cellVectors, outstream):
    '''Writes <cellVectors to output file <outstream>.
    '''

    outstream.write('vectors\n')
    for i in range(3):
        outstream.write('{:.5f} {:.5f} {:.5f}\n'.format(cellVectors[i][0],
                             cellVectors[i][1], cellVectors[i][2]))
    return

def extractRegions(cluster_file, rIBasis, rIIBasis, nocomment=True):
    '''Given a gulp output (for a cluster calculation), extracts a list of
    atoms in region I and region II.
    Precondition: <rIBasis> and <rIIBasis> have been initialised and contain
    no atoms.
    '''

    cluster = util.read_file(cluster_file, nocomment=nocomment)

    dictRI = dict()
    dictRII = dict()

    inRI = False
    inRII = False
    for line in cluster:
        if 'region' in line:
            # checks to see if we are at the start of a region list
            if '1' in line:
                inRI = True
            elif '2' in line:
                inRI = False
                inRII = True
        # test to see if we have finished scanning the atomic coordinates block
        if inRII and ('species' in line):
            # end of atomic coordinates block
            inRII = False
        else:
            # determine if line is an atom line
            foundAtom = re.search(atomLine, line)
            if inRI and foundAtom:
                extractAtom(foundAtom, dictRI, pfrac=True)
            elif inRII and foundAtom:
                extractAtom(foundAtom, dictRII, pfrac=True)

    # copy atoms from the temporary, element-sorted dictionaries into regions
    # and II. Since we are extracting from the dump file of a polymer calculation,
    # we need to reorder the coordinates, which are given as (z, x, y)
    for element in dictRI:
        for atom in dictRI[element]['atoms']:
            atom.from_cluster()
            rIBasis.addAtom(atom)

    for element in dictRII:
        for atom in dictRII[element]['atoms']:
            atom.from_cluster()
            rIIBasis.addAtom(atom)

    del dictRI, dictRII

    return

def cluster_cartesian(atom_coords, lattice):
    '''Converts the atomic coordinates from crystal basis to cartesian if the
    they are given in cluster ordering (ie z, x, y).
    '''

    atom_x = atom_coords[1]
    atom_y = atom_coords[2]
    atom_z = atom_coords[0]

    # convert to cartesian
    new_coords = atom_x*np.copy(lattice[0]) + atom_y*np.copy(lattice[1]) + \
                                                atom_z*np.copy(lattice[2])

    # return atomic coordinates in cluster order
    new_coords = np.array([new_coords[2], new_coords[0], new_coords[1]])
    return new_coords

# additional utilities that can be helpful for working with GULP

def run_gulp(gulp_exec, basename, in_suffix='gin', out_suffix='gout'):
    '''Runs a gulp calculation (using the gulp executable <gulp_exec>) taking
    <basename>.gin -> <basename>.gout.
    '''

    gin = open('{}.{}'.format(basename, in_suffix), 'r')
    gout = open('{}.{}'.format(basename, out_suffix), 'w')
    subprocess.call(gulp_exec, stdin=gin, stdout=gout)

    gin.close()
    gout.close()
    return
    
def gulp_process(prefix, gulpexec, message=None, in_suffix='gin', out_suffix='gout'):
    '''An individual GULP process to be called when running in parallel.
    '''
    
    if message is not None:
        print(message)
    
    # create the directory from which to run the GULP simulation
    if os.path.exists(prefix):
        if not os.path.isdir(prefix):
            # the name <prefix> is taken, and NOT by a directory
            raise Exception("Name {} taken by non-directory.".format(prefix))
        else:
            # assume that using directory <prefix> is fine
            pass
    else:        
        os.mkdir(prefix)
    
    # copy .gin file into child directory, and then descend into subdirectory
    # to run simulation
    copyfile('{}.{}'.format(prefix, in_suffix), '{}/{}.{}'.format(prefix, prefix, in_suffix))    
    os.chdir(prefix)
   
    # run simulation and return to the primary impurity directory    
    run_gulp(gulpexec, prefix, in_suffix=in_suffix, out_suffix=out_suffix)
    os.chdir('../')
    
    # copy output file to main directory 
    copyfile('{}/{}.{}'.format(prefix, prefix, out_suffix), '{}.{}'.format(prefix, out_suffix))
    return 0

def cluster_from_grs(filename, rI, rII, new_rI=None, r=None):
    '''Reads in the .grs file output after a successful cluster-based dislocation
    simulation and use it to construct a <TwoRegionCluster>. <new_rI> allows 
    us to change the size of region I, whether by enlarging or reducing it.
    '''
    
    # read in the .grs file and convert it to cartesian coordinates
    grs_struc = cry.Crystal()
    sysinfo = parse_gulp(filename, grs_struc)
    
    for atom in grs_struc:
        # arrange coords in order x, y, z
        atom.from_cluster()
        # convert z from pcell units to length units
        atom.to_cart(grs_struc)
        
    # determine the region I radius to use for the new cluster
    if new_rI is None:
        new_rI = rI
    else:
        # check that <new_rI> is not ridiculous
        if new_rI >= rII:
            raise ValueError("New region I radius must be less than total radius.")
    
    # create the cluster    
    height = norm(grs_struc.getC())        
    new_cluster = rs.TwoRegionCluster(R=rI, regionI=new_rI, regionII=rII, height=height,
                                            periodic_atoms=grs_struc)                            
    
    return new_cluster, sysinfo
