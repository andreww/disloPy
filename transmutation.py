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
    
    def __init__(self,site,defectName):
        '''<site> tells us which site this Impurity occupies. <defectName>
        is the name used to identify this defect type (eg. hydrogarnet for
        4 hydrogens in a silicon vacancy).
        '''
        
        cry.Basis.__init__(self)
        self.__site = site
        self.__name = defectName
        
    def writeImpurity(self,outStream,replacedAtom):
        '''Writes all atoms in an <Impurity> centred at a specific <site>.
        Assume that impurity coordinates are already in the correct order 
        and units.
        '''
        
        # dummy variables for lattice and toCart -> do need to convert to
        # pfractional, though
        lattice = np.identity(3)
        toCart = False
        
        for atom in self.getAtoms():
            defectCoords = replacedAtom.getCoordinates()+atom.getCoordinates()
            atom.setDisplacedCoordinates(defectCoords)
            atom.write(outStream,True,lattice,toCart)
            
        return 1   
        
    def getSite(self):
        return self.__site
        
    def setSite(self,newSite): 
        self.__site = newSite  
        
    def getName(self):
        return self.__name
        
    def setName(self,newDefectName):
        self.__name = newDefectName
        
class CoupledImpurity:
    '''Impurity at two or more sites. Relationship with class <Impurity> is
    #HAS-A# not #IS-A#.
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
    
def calculateImpurity(sysInfo,regionI,regionII,radius,defect,gulpExec='./gulp',
                                                             constraints=None):
    '''Iterates through all atoms in <relaxedCluster> within distance <radius>
    of the dislocation line, and sequentially replaces one atom of type 
    <replaceType> with an impurity <newType>. dRMin is the minimum difference
    between <RI> and <radius>. Ensures that the impurity is not close to region
    II, where internal strain would not be relaxed. <constraints> contains any 
    additional tests we may perform on the atoms, eg. if the thickness is > 1,
    we may wish to restrict substituted atoms to have z (x0) coordinates in the
    range [0,0.5) ( % 1).
    
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
        
    for atom in regionI.getAtoms():
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
                    break
            if not useAtom:
                continue           
            
        # need to work out some schema for the outstream
        coords = atom.getCoordinates()
        outName = '%s.%s.%.6f.%.6f.%.6f' % (defect.getName(),defect.getSite(),
                                                  coords[0],coords[1],coords[2])
        outStream = open(outName+'.gin','w')
        
        # write header information
        #outStream.write('opti conv qok eregion bfgs\n')
        # for testing, do single point calculations
        outStream.write('opti conv qok eregion bfgs\n')
        outStream.write('pcell\n')
        outStream.write(sysInfo[1])
            
        # record that the atom should not be written to output, and retrieve
        # coordinates of site
        atom.switchOutputMode()
        
        # write non-substituted atoms + defect to outStream
        gulp.writeRegion(regionI,lattice,outStream,1,disloc,toCart,coordType)
        defect.writeImpurity(outStream,atom)
        gulp.writeRegion(regionII,lattice,outStream,2,disloc,toCart,coordType)
        
        # switch output mode of <atom> back to <write>
        atom.switchOutputMode()
        
        # finally, write interatomic potentials, etc. to file
        for line in sysInfo[2:]:
            outStream.write(line)
            
        outStream.write('dump %s.grs\n' % outName)
            
        # close output file and run calculation
        outStream.close()
        os.system('%s < %s.gin > %s.gout' % (gulpExec,outName,outName))
                    
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
    
    useAtom = inRange(atomicZ,zMin,zMax)
    return useAtom
        
def azimuthConstraint(thetaMin,thetaMax,atom):
    '''Constraints impurity energies to be calculated in a finite range of angles.
    Useful when the defect (eg. screw dislocation) has some rotational symmetry.
    '''
    
    # atomic x and y coordinates
    atomicX,atomicY = atom.getCoordinates()[1],atom.getCoordinates()[2]
    # atomic theta
    atomicTheta = np.arctan2(atomicY,atomicX)
    
    useAtom = inRange(atomicTheta,thetaMin,thetaMax)
    return useAtom
        
def inRange(value,rangeMin,rangeMax):
    '''Tests to see if rangeMin <= value < rangeMax.
    '''
    
    if abs(value - rangeMin) > -1e-12 and value < rangeMax:
        return True  
    else:
        return False
        
# GLOBAL DICTIONARY OF CONSTRAINTS

constraintDictionary = {'height':heightConstraint,'azimuth':azimuthConstraint}
        
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

def readGulpCluster(clusterName):
    '''Reads in the relaxed GULP cluster in <clusterName>.grs.
    '''
    
    # Basis objects to hold RI and RII
    regionI = cry.Basis()
    regionII = cry.Basis()
    
    # open dislocation file
    disFile = gulp.readGulpFile(clusterName + '.grs')
    
    # extract atomic coordinates to <Basis> objects
    gulp.extractRegions(disFile,regionI,regionII)
    
    return regionI,regionII
    
# LAMMPS specific routines -> not implemented yet

# DL_POLY specific routines -> not implemented yet
            
