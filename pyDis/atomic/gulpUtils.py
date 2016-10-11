#!/usr/bin/env python
'''This program parses an input gulp file and extracts all information
that is required to run a simulation, including cell parameters, atomic
coordinates, species, potentials, etc.
'''
from __future__ import division, print_function

import sys
import os
sys.path.append(os.environ['PYDISPATH'])

import numpy as np
import re
import subprocess

from numpy.linalg import norm

from pyDis.atomic import crystal as cry
from pyDis.atomic import atomistic_utils as util
from pyDis.atomic import transmutation as mutate
from pyDis.atomic import rodSetup as rs

from pyDis.atomic.rodSetup import __dict__ as rod_classes

# Functions to write different GULP input files

class GulpAtom(cry.Atom):
    '''Same as <cry.Atom>, but contains an additional variable to record
    whether it has a shell or breathing shell.

    ### NOT FINISHED ###
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

    def addShell(self, shelCoords, shellType='shel'):
        '''Gives the atom a polarizable shell, with coordinates <shelCoords>.
        May add position checking later.
        '''

        self.__hasShell = True
        self.__shelCoords = self.getShellDistance(shelCoords)

        if shellType in 'bshe':
            # using a breathing shell model
            self.__isBSM = True

        return

    def copyShell(self, shelDisplacement, shellType='shel'):
        '''When copying atoms, shell is already expressed as a displacement
        from the core, which means that we do not need to invoke
        <GulpAtom.getShellDistance()>.
        '''

        self.__shelCoords = shelDisplacement.copy()
        self.__hasShell = True
        if shellType in 'bshe':
            self.__isBSM = True

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

    def getShellDistance(self, shelCoords):
        '''Returns the distance between the centre of the shell and the
        (undisplaced) coordinates of the ionic core.
        '''

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

            # shell coordinates
            if self.hasShell():
                shel_coords = self.getShell()
                shel_coords = np.array([shel_coords[2], shel_coords[0], shel_coords[1]])
                self.setShell(shel_coords)

            # recording that we are using cluster-ordered coordinate indices
            self._cluster_ordered = True
        
    def from_cluster(self):
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
            shel_coords = np.array([shel_coords[1], shel_coords[2], shel_coords[0]])
            self.setShell(shel_coords)

        # recording that we are using cluster-ordered coordinate indices
        self._cluster_ordered = False

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
            outstream.write(' 1.0 {} {} {}\n'.format(self.get_constraints()[0],
                          self.get_constraints()[1], self.get_constraints()[2]))
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

            if add_constraints:
                # always relax shells
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
        if self.hasShell():
            new_atom.copyShell(self.getShell(), 'bshe' if self.__isBSM
                                                             else 'shel')
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

def writeSlab():
    '''Write a slab simulation cell to file.
    '''

    pass

def write1DCluster(cluster, sys_info, outname, maxiter=1000):
    '''Writes 1D-periodic simulation cell to file. Always write an accompanying
    undislocated cluster, so that dislocation energy can be calculated. <maxiter>
    gives the maximum number of iterations allowed. GULP defaults to 1000, but
    his is somewhat too high in most circumstances.
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
            relax = 'conv'
            do_relax = True
            prefix = 'dis.{}'.format(outname)

        write_gulp(outstream, cluster, sys_info, defected=disloc, do_relax=do_relax, 
                                                                    relax_type=relax)

    return

def writeDefectCluster():
    '''Sets up cluster calculation (eg. Mott-Littleton).
    '''
    pass

# Master level functions for opening and parsing GULP input

atomLine = re.compile(r'([A-Z][a-z]?\d*)\s+(core|c|shel|s|bshe|bcor)' + \
                                      '((\s+-?\d+\.\d+|\s+-?\d+/\d+){3})')
#speciesLine = re.compile('^([A-Z][a-z]?\d*?)\s+(core|c|shel|s|bshe|bcor)' + \
#                                                          '\s+-?\d+\.\d+\s*$')

def preamble(outstream, maxiter=500, do_relax=True, relax_type='conv',
                                   polymer=False, molq=False, prop=True):
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
    else:
        outstream.write('qok ')
    
    if molq:
        outstream.write(' molq')
        
    if polymer:
        outstream.write(' eregion\n') # DO NOT TRY TO CALCULATE PROPERTIES
    elif prop:
        outstream.write(' prop\n')
    else:
        outstream.write('\n')

    # maximum allowable number of relaxation steps. Our default is somewhat 
    # smaller than the default in the GULP source code
    outstream.write('maxcyc {}\n'.format(maxiter))
    return
    
def write_gulp(outstream, struc, sys_info, defected=True, do_relax=True, to_cart=True,
                    add_constraints=False, relax_type='conv', impurities=None, prop=True,
                                                pfractional=False):
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
        if not (impurities is None):
            struc.specifyRegions()
            
        # polymer cell -> write cell height
        preamble(outstream, do_relax=do_relax, polymer=True, relax_type=relax_type)
        height = struc.getHeight()
        outstream.write('pcell\n')
        outstream.write('{:.6f} 0\n'.format(height))
        cell_lattice = struc.getBaseCell().getLattice()

        # write atoms to output
        if pfractional:
            writeRegion(struc.getRegionIAtoms(), cell_lattice, outstream, 1, 
                                           defected, coordType='pfractional')
            writeRegion(struc.getRegionIIAtoms(), cell_lattice, outstream, 2, 
                                            defected, coordType='pfractional')
        else: # cartesian
            writeRegion(struc.getRegionIAtoms(), cell_lattice, outstream, 1, 
                                                                    defected)
            writeRegion(struc.getRegionIIAtoms(), cell_lattice, outstream, 2,
                                                                     defected)
    else:
        # write lattice vectors
        preamble(outstream, do_relax=do_relax, relax_type=relax_type, prop=prop) 
        writeVectors(struc.getLattice(), outstream)
        
        if relax_type is None:
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
    name_form = re.compile('(?P<base>.+)\.gin')
    basename = name_form.match(outstream.name).group('base')
    
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
                crystalStruc.setVector(cellVectors[j], j)
        elif line.strip() in 'pcell':
            # reading in a 1D-periodic cluster -> record the cell height
            cell_height = float(gulp_lines[i+1].strip())
            crystalStruc.setC(np.array([0., 0., cell_height]))
        else:
            foundAtoms = atomLine.match(line)
            if foundAtoms:
                if atomsNotFound:
                    # record that we are now reading atom info
                    atomsNotFound = False

                # extract atom info to <allAtoms>
                extractAtom(foundAtoms, allAtoms)

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

def extractAtom(atomRegex, atomsDict):
    '''Extracts atom info found in <atomRegex> to existing dictionary of atoms
    <atomsDict>.
    '''

    atomicSymbol = atomRegex.group(1)

    if atomicSymbol in atomsDict:
        typeOfAtom = atomRegex.group(2)
        tempCoords = atomRegex.group(3)
        tempCoords = tempCoords.split()
        atomCoords = np.array([float(eval(x)) for x in tempCoords])
        if typeOfAtom in 'shell':
            index = atomsDict[atomicSymbol]['shells']
            atomsDict[atomicSymbol]['atoms'][index].addShell(atomCoords)
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
            newAtom = GulpAtom(atomicSymbol, atomCoords)
            atomsDict[atomicSymbol]['atoms'].append(newAtom)

    return

def cellToCart(parameters):
    '''Converts 6 cell parameters to lattice vectors.
    For the moment, assume that we are working with a cell
    whose lattice vectors are orthogonal. ### NEED TO GENERALIZE THIS###.
    '''

    # extract the unit cell parameters
    [a, b, c, alp, bet, gam] = parameters

    x1 = a*cry.e(1)
    x2 = b*cry.e(2)
    x3 = c*cry.e(3)

    return cry.Lattice(x1, x2, x3)

def writeRegion(region_basis, lattice, outstream, regionNumber, disloc,
                                  use_cart=True, coordType='cartesian'):
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
            atom.write(outstream, lattice, defected=disloc, to_cart=use_cart)
    elif regionNumber == 2:
        outstream.write("{} region 2 rigid\n".format(coordType))
        for atom in region_basis:
            atom.clusterOrder()
            atom.write(outstream, lattice, defected=disloc, to_cart=use_cart)
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

def extractRegions(cluster_file, rIBasis, rIIBasis):
    '''Given a gulp output (for a cluster calculation), extracts a list of
    atoms in region I and region II.
    Precondition: <rIBasis> and <rIIBasis> have been initialised and contain
    no atoms.
    '''

    cluster = util.read_file(cluster_file)

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
                extractAtom(foundAtom, dictRI)
            elif inRII and foundAtom:
                extractAtom(foundAtom, dictRII)

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

def run_gulp(gulp_exec, basename):
    '''Runs a gulp calculation (using the gulp executable <gulp_exec>) taking
    <basename>.gin -> <basename>.gout.
    '''

    gin = open('{}.gin'.format(basename))
    gout = open('{}.gout'.format(basename), 'w')
    subprocess.call(gulp_exec, stdin=gin, stdout=gout)

    gin.close()
    gout.close()
    return
    
### ROUTINES FOR CALCULATING DEFECT ENERGY SURFACES IN A CYLINDER

def cluster_from_grs(filename, rI, rII, r=None):
    '''Reads in the .grs file output after a successful cluster-based dislocation
    simulation and use it to construct a <TwoRegionCluster>.
    '''
    
    # read in the .grs file and convert it to cartesian coordinates
    grs_struc = cry.Crystal()
    sysinfo = parse_gulp(filename, grs_struc)
    
    for atom in grs_struc:
        # arrange coords in order x, y, z
        atom.from_cluster()
        # convert z from pcell units to length units
        atom.to_cart(grs_struc)
        
    # create the cluster
    height = norm(grs_struc.getC())        
    new_cluster = rs.TwoRegionCluster(R=rI, regionI=rI, regionII=rII, height=height,
                                            periodic_atoms=grs_struc)                            
    
    return new_cluster, sysinfo

def calculateImpurity(sysinfo, gulpcluster, radius, defect, gulpexec='./gulp',
                   constraints=None, minimizer='bfgs', maxcyc=100, noisy=True, 
                                                                do_calc=False):
    '''Iterates through all atoms in <relaxedCluster> within distance <radius>
    of the dislocation line, and sequentially replaces one atom of type 
    <replaceType> with an impurity <newType>. dRMin is the minimum difference
    between <RI> and <radius>. Ensures that the impurity is not close to region
    II, where internal strain would not be relaxed. <constraints> contains any 
    additional tests we may perform on the atoms, eg. if the thickness is > 1,
    we may wish to restrict substituted atoms to have z (x0) coordinates in the
    range [0,0.5) ( % 1). The default algorithm used to relax atomic coordinates
    is BFGS but, because of the N^2 scaling of the memory required to store the 
    Hessian, other solvers (eg. CG or numerical BFGS) should be used for large
    simulation cells.
    
    Tests to ensure that radius < (RI - dRMin) to be performed in the calling 
    routine (ie. <impurityEnergySurface> should only be called if radius < 
    (RI-dRMin) is True. 
    '''
    
    # dummy variables for lattice and toCart. Due to the way the program
    # is set up, disloc is set equal to false, as the atoms are displaced 
    # and relaxed BEFORE we read them in
    lattice = np.identity(3)
    toCart = False
    disloc = False
    coordType = 'pcell'
    
    # test to see if <defect> is located at a single site
    if type(defect) is mutate.Impurity:
        pass
    else:
        raise TypeError('Invalid impurity type.')
        
    use_indices = []    
    for i, atom in enumerate(gulpcluster):
        # check conditions for substitution:
        # 1. Is the atom to be replaced of the right kind?
        if atom.getSpecies() != defect.getSite():
            continue
        # 2. Is the atom within <radius> of the dislocation line?
        if norm(atom.getCoordinates()[:-1]) > radius:
            continue  
              
        # check <constraints>
        if constraints is None:
            pass
        else:
            for test in constraints:
                useAtom = test(atom) 
                if not useAtom:
                    print("Skipping atom {}.".format(i))
                    break
            if not useAtom:
                continue       
        if noisy:        
            print("Replacing atom {} (index {})...".format(str(atom), i)) 
        
        # record that atom <i> is to be replaced      
        use_indices.append(i)        
        
    for i in use_indices:        
        # set the coordinates and site index of the impurity
        atom = gulpcluster[i]
        defect.site_location(atom)
        defect.set_index(i)
        
        # create .gin file for the calculation
        coords = atom.getCoordinates()
        #outname = '{}.{}.{:.6f}.{:.6f}.{:.6f}'.format(defect.getName(), defect.getSite(),
        #                                                  coords[0], coords[1], coords[2])
        outname = '{}.{}.{}'.format(defect.getName(), defect.getSite(), defect.get_index())
        outstream = open(outname+'.gin','w')
        
        # write structure to output file, including the coordinates of the 
        # impurity atom(s)
        write_gulp(outstream, gulpcluster, sysinfo, defected=False, to_cart=False, 
                                                                 impurities=defect)
                                                                 
        # run calculation, if requested by user
        if do_calc:
            run_gulp(gulpexec, outname)
                    
    return
    
def calculateCoupledImpurity(sysInfo, regionI, regionII, radius, defectCluster,
                                        gulpExec='./gulp', constraints=None):
    '''As above, but for an impurity cluster with defects at multiple sites.
    '''
    
    # dummy variables for lattice and toCart. Due to the way the program
    # is set up, disloc is set equal to false, as the atoms are displaced 
    # and relaxed BEFORE we read them in
    lattice = np.identity(3)
    toCart = False
    disloc = False
    coordType = 'pfractional'
    
    # test to see if <defect> is located at a single site
    if type(defectCluster) is CoupledImpurity:
        ### NOT YET IMPLEMENTED ###
        print('Coupled defects have not been implemented yet.')
        pass
    else:
        raise TypeError('Invalid impurity type.')
                   
    return
