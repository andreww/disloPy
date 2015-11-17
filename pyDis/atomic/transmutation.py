#!/usr/bin/env python
'''Contains functions required to introduce defects into a 1D-periodic 
cluster.
'''
from __future__ import print_function

import numpy as np
import numpy.linalg as L
import sys
import os

import crystal as cry
import gulpUtils as gulp
import re

class Impurity(cry.Basis):
    '''An impurity. May be a single defect (eg. a Ca atom in Mg2SiO4
    forsterite) or a cluster (eg. four protons in the hydrogarnet defect).
    This class supports impurities at a single site. Currently set up for 
    GULP
    '''
    
    def __init__(self, site, defect_name, sitecoords=None):
        '''<site> tells us which site this Impurity occupies. <defectName>
        is the name used to identify this defect type (eg. hydrogarnet for
        4 hydrogens in a silicon vacancy).
        '''
        
        cry.Basis.__init__(self)
        self.__site = site
        self.__name = defect_name
        # by default, the defect is not assigned to a specific site
        self._sitecoords = sitecoords 
        
    def write_impurity(self, outstream, lattice=np.identity(3), to_cart=False,
                                                        add_constraints=False):
        '''Writes all atoms in an <Impurity> centred at a specific <site>.
        Assume that impurity coordinates are already in the correct order 
        and units.
        '''
        
        for atom in self:
            atom.write(outstream, lattice, defected=True, toCart=toCart)
            
        return 
        
    def getSite(self):
        return self.__site
        
    def setSite(self,newSite): 
        self.__site = newSite  
        
    def getName(self):
        return self.__name
        
    def setName(self,newDefectName):
        self.__name = newDefectName

    def site_location(self, new_location, use_displaced=True):
        '''Sets the origin of the <Impurity> to <new_location>. Typically, this
        will be the coordinates of the <Atom> for which the <Impurity> is 
        substituting. If <use_displaced>, use the displaced coordinates of the 
        atom for which the impurity is substituting. If not, use its coordinates
        in the perfect crystal (note: applies only when type(new_location) <: Atom)
        '''

        if isinstance(new_location, cry.Atom):
            if use_displaced:
                # use the displaced coordinates of the provided atom
                self._sitecoords = new_location.getDisplacedCoordinates()
            else: # use perfect coordinates
                self._sitecoords = new_location.getCoordinates()
        else:
            self._sitecoords = np.copy(new_location)

        # set displaced coordinates equal to coordinates in actual structure
        for atom in self:
            atom.setDisplacedCoordinates(atom.getCoordinates() + 
                                         np.copy(self._sitecoords))
        
class CoupledImpurity(object):
    '''Impurity at two or more sites. Relationship with class <Impurity> is
    #HAS-A# not #IS-A#. NOTE: NOT FINISHED YET!
    '''
    
    def __init__(self,maxDistance=10.):
        '''<maxDistance> is the maximum diameter of the coupled defect.
        '''
        
        self.__impurities = []
        self.__diameter = maxDistance
        
    def addImpurity(self,newImpurity,impurityName):
        self.__impurities.append(newImpurity)
        
    def getSites(self):
        return self.__impurities
        
    def nSites(self):
        '''Outputs number of impurities in the defect cluster.
        '''
        
        return len(self.__impurities)
        
    def isEmpty(self):
        '''Tells user if any impurities have been defined -> note that a 
        vacancy is a object of type <Impurity> to which no atoms have been 
        added.
        '''
        
        if self.__impurities:
            return True
        else:
            return False


### IMPURITIES IN CLUSTERS (IE. 1D-PERIODIC CELLS) ###
    
def calculateImpurity(sysInfo,regionI,regionII,radius,defect,gulpExec='./gulp',
                                   constraints=None,minimizer='bfgs',maxcyc=100):
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
    coordType = 'pfractional'
    
    # test to see if <defect> is located at a single site
    if type(defect) is Impurity:
        pass
    else:
        raise TypeError('Invalid impurity type.')
        
    for atom in regionI:
        # check conditions for substitution:
        # 1. Is the atom to be replaced of the right kind?
        if atom.getSpecies() != defect.getSite():
            continue
        # 2. Is the atom within <radius> of the dislocation line?
        if L.norm(atom.getCoordinates()[1:]) > radius:
            continue  
              
        # check <constraints>
        if constraints is None:
            pass
        else:
            for test in constraints:
                useAtom = test(atom) 
                if not useAtom:
                    print("Skipping atom.")
                    break
            if not useAtom:
                continue           
            
        print("Replacing atom %s..." % str(atom))
        
        # need to work out some schema for the outstream
        coords = atom.getCoordinates()
        outName = '%s.%s.%.6f.%.6f.%.6f' % (defect.getName(),defect.getSite(),
                                                  coords[0],coords[1],coords[2])
        outStream = open(outName+'.gin','w')
        
        # write header information
        #outStream.write('opti conv qok eregion bfgs\n')
        # for testing, do single point calculations
        outStream.write('opti conv qok eregion %s\n' % minimizer)
        outStream.write('maxcyc %d\n' % maxcyc)
        outStream.write('pcell\n')
        outStream.write(sysInfo[1])
            
        # record that the atom should not be written to output, and retrieve
        # coordinates of site
        atom.switchOutputMode()
        
        # write non-substituted atoms + defect to outStream
        gulp.writeRegion(regionI,lattice,outStream,1,disloc,toCart,coordType)
        defect.write_impurity(outStream, atom)
        gulp.writeRegion(regionII,lattice,outStream,2,disloc,toCart,coordType)
        
        # switch output mode of <atom> back to <write>
        atom.switchOutputMode()
        
        # finally, write interatomic potentials, etc. to file
        for line in sysInfo[2:]:
            outStream.write(line)
        
        # specify output of relaxed defect structure
        gulp.restart(outName, outStream)
            
        # close output file and run calculation
        outStream.close()
        gulp.run_gulp(gulpExec,outName)
                    
    return
    
def calculateCoupledImpurity(sysInfo,regionI,regionII,radius,defectCluster,
                                        gulpExec='./gulp',constraints=None):
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

    
# POSSIBLE CONSTRAINTS

def heightConstraint(zMin,zMax,atom):
    '''Ensures that the z-coordinate of <atom> (in pfractional units) is in the 
    range [zMin,zMax).
    '''
    
    # z coordinate of atom, constrained to be in [0,1)
    atomicZ = atom.getCoordinates()[0] % 1
    
    useAtom = in_range(atomicZ,zMin,zMax)
    return useAtom

def plane_constraint(atom, i, xmin=-np.inf, xmax=np.inf, use_polymer=True,
                                                          tolerance=1e-12):
    '''Allows substitution iff coordinate of <atom> > xmin(-tolerance). The
    inclusion of a tolerance factor accounts for small deviations away from a
    symmetry plane, generally due to the use of CG or numerical BFGS. We assume
    symmetry about x_{i} == 0. <xmax> (<xmin>) defaults to inf (-inf); ie. 
    unbounded above (below).
    '''

    if use_polymer:
        # coordinates given in polymer order, ie. 3 1 2
        i = (i+1) % 3

    coord = atom.getCoordinates()[i] 
    use_atom = in_range(coord,xmin-tolerance,xmax)
    return use_atom
    
def azimuthConstraint(thetaMin,thetaMax,atom):
    '''Constraints impurity energies to be calculated in a finite range of angles.
    Useful when the defect (eg. screw dislocation) has some rotational symmetry.
    '''
    
    # atomic x and y coordinates
    atomicX,atomicY = atom.getCoordinates()[1],atom.getCoordinates()[2]
    # atomic theta
    atomicTheta = np.arctan2(atomicY,atomicX)
    
    useAtom = in_range(atomicTheta,thetaMin,thetaMax)
    return useAtom
        
def in_range(value,rangeMin,rangeMax):
    '''Tests to see if rangeMin <= value < rangeMax.
    '''
    
    # if <rangeMin> > <rangeMax>, assume that <rangeMin> is actually a negative
    # value modulo 1, so that -0.05 would be entered as 0.95. For example, this
    # allows us to select all atoms near the basal plane (z==0.0) by entering 
    # <rangeMin> = (1-d), <rangeMax> = d, 0 < d < 0.5
    if rangeMin > rangeMax:
        rangeMin -= 1.0
    
    if (value - rangeMin) > -1e-12 and value < rangeMax:
        return True  
    else:
        return False
        
# GLOBAL DICTIONARY OF CONSTRAINTS

constraintDictionary = {'height':heightConstraint,'azimuth':azimuthConstraint,
                         'plane':plane_constraint}
        
def readConstraints(constraintName):
    '''Reads constraints from an input parameter file. Each line of this file 
    has the form:
    
    %constraint_type min_value max_value
    
    Returns constraints as a list.
    '''
    
    # open constraint file
    try:
        constraintFile = open(constraintName,'r')
    except IOError:
        print('File %s not found.' % constraintName,end=' ')
        print('Do you wish to specify a different file (yes/no)?',end=' ')
        rename = raw_input()
        while rename.lower() not in ['yes','no']:
            rename = raw_input('Please enter yes/no: ')
        
        if rename.lower() == 'yes':
            constraintName = raw_input('Enter name of constraint file: ')
            try:
                constraintFile = open(constraintName,'r')
            except IOError:
                print('File does not exist. Defaulting to constraints = None.')
                constraints = None
            else:
                constraints = parseConstraints(constraintFile)
        else:
            print('Continuing without constraints.')
            constraints = None
    else:
        constraints = parseConstraints(constraintFile)
    
    return constraints
    
def parseConstraint(constraintFile):
    '''Parses constraintFile to extract valid constraints.
    '''
    
    # regex to find a constraint line
    constraintLine = re.compile('\%(?P<name>\w+)\s+(?P<min>-?\d+\.?\d*)' +
                                                '\s+(?P<max>-?\d+\.?\d*)')
    
    lines = [line.strip() for line in constraintFile if line.rstrip()]
    # make sure that lines is not empty; return None if it is
    if not lines:
        return None
    
    for constraint in lines:
        if line.startswith('#'):
            # comment line
            continue
        #else
        cons = re.search(constraintLine,constraint)
        try:
            typeOfConstraint = cons.group('name')
        except NameError:
            continue
        else:
            minVal,maxVal = cons.group('min'),cons.group('max')
         
            typeOfConstraint = typeOfConstraint.lower()
            try:
                constraintFunction = constraintDictionary[typeOfConstraint]
            except:
                continue
            else:
                pass
            
                       
# GULP specific routines

def readGulpCluster(cluster_name):
    '''Reads in the relaxed GULP cluster in <clusterName>.grs.
    '''
    
    # Basis objects to hold RI and RII
    regionI = cry.Basis()
    regionII = cry.Basis()
    
    # extract atomic coordinates to <Basis> objects
    gulp.extractRegions(cluster_name+'.grs',regionI,regionII)
    
    return regionI,regionII
    
# LAMMPS specific routines -> not implemented yet

# DL_POLY specific routines -> not implemented yet
            
