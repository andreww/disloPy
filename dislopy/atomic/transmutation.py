#!/usr/bin/env python
'''Contains functions required to introduce defects into simulation cells.
'''
from __future__ import print_function, absolute_import

import numpy as np
import re
import sys

from  numpy.linalg import norm

from dislopy.atomic import crystal as cry

class Impurity(cry.Basis):
    '''An impurity. May be a single defect (eg. a Ca atom in Mg2SiO4
    forsterite) or contain several atoms/ions (eg. four protons in the hydrogarnet 
    defect). This class supports impurities at a single site. 
    '''
    
    def __init__(self, site, defect_name, sitecoords=None, use_displaced=True, 
                                            updateatoms=False, site_index=None):
        '''<site> tells us which site this Impurity occupies (eg. Mg.). 
        <defectName>, while <site_index> gives its index.
        is the name used to identify this defect type (eg. hydrogarnet for
        4 hydrogens in a silicon vacancy).
        '''
        
        cry.Basis.__init__(self)
        self.__site = site
        self.__name = defect_name
        
        if not (sitecoords is None):
            # set the location of the impurity
            self.site_location(sitecoords, use_displaced=use_displaced, 
                                                updateatoms=updateatoms)
        else:
            self._sitecoords = None
            
        self.site_index = site_index
        
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
        
    def setSite(self, newSite): 
        self.__site = newSite  
        
    def getName(self):
        return self.__name
        
    def setName(self, newDefectName):
        self.__name = newDefectName
        
    def __str__(self):
        thisstr = ''
        for atom in self:
            thisstr += '{}\n'.format(atom)
        return thisstr
        
    def to_cell_coords(self, lattice):
        '''Converts the coordinates of all atoms to lattice units.
        '''
        
        for atom in self:
            atom.to_cell(lattice)

    def site_location(self, new_location, use_displaced=True, updateatoms=True):
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
        elif isinstance(new_location, cry.Basis) or isinstance(new_location, cry.Crystal):
            # ONLY WORKS IF <site_index> is defined
            if self.site_index is None:
                raise AttributeError("Cannot use <Crystal> or <Basis> object to set" +
                                 " the site location if <site_index> is undefined.")
                                 
            # otherwise, use <site_index> to extract site location
            new_loc = new_location[self.site_index]
            if use_displaced:
                # use displaced coordinates of provided atom
                self._sitecoords = new_loc.getDisplacedCoordinates()
            else: # perfect coords.
                self._sitecoords = new_loc.getCoordinates()
        else:
            self._sitecoords = np.copy(new_location)

        if updateatoms:
            # set displaced coordinates equal to coordinates in actual structure
            self.atomic_site_coords()
                                             
    def atomic_site_coords(self):
        '''Calculate coordinates of atoms in terms of the simulation cell using
        the coordinates of the defect site.
        '''
        
        if self._sitecoords is None:
            raise TypeError("Cannot calculate site coordinates if site is <None>.")
        else:
            for atom in self:
                atom.setDisplacedCoordinates(atom.getCoordinates() + 
                                            np.copy(self._sitecoords)) 
        
        return
                                         
    def copy(self, updateatoms=False):
        '''Creates a copy of the <Impurity> object.
        '''
        
        new_imp = Impurity(self.__site, self.__name, sitecoords=self._sitecoords,
                                    site_index=self.site_index)
        
        # add atoms to the new Impurity
        for atom in self:
            new_imp.addAtom(atom)
            
        if updateatoms and not (self._sitecoords is None):
            # calculate defect coordinates for a specific site
            self.atomic_site_coords()
        
        return new_imp
        
    def set_index(self, new_site_index):
        '''Sets the index of the specific site in a <Lattice> or <Crystal>
        that the <Impurity> replaces.
        '''
            
        self.site_index = new_site_index
            
    def get_index(self):
        '''Retrieves the index of the site to be replaced.
        '''
                
        return self.site_index
                                         
class CoupledImpurity(object):
    '''Several impurity atoms/vacancies located at multiple 
    crystallographic sites.
    '''
    
    def __init__(self, impurities=None, sites=None):
        '''List of impurities and the indices of the sites into which
        they subsitute.
        '''
        
        # if no impurities/sites have been provided, default to empty
        if impurities is None and sites is None:
            self.impurities = []
            self.sites = []
        elif (not(sites is None) and not(impurities is None)) and (len(sites) != len(impurities)):
            raise ValueError()
        else:
            self.impurities = []
            for i, imp in enumerate(impurities):
                new_imp = imp.copy()
                # set site index, if provided
                if not (sites is None):
                    imp.set_index(sites[i])
                    
                self.impurities.append(new_imp) 
            
        self.currentindex = 0   
    
    def add_impurity(self, impurity, site=None):
        '''Append a new impurity to the defect cluster.
        '''
        
        new_imp = impurity.copy() 
        
        # if the index of the site differs from that of <impurity>, change index
        if not (site is None):
            new_imp.set_index(site)
            
        self.impurities.append(new_imp)
            
    def __len__(self):
        '''Counts the total number of impurity atoms in the defect cluster.
        '''
        
        natoms = 0
        # count atoms in each Impurity
        for impurity in self:
            natoms += len(impurity)
        
        return natoms
    
    def nsites(self):
        '''Gives the total number of sites in the structure which are replaced
        with a defect.
        '''
        
        return len(self)
        
    def __getitem__(self, key):
        return self.impurities[key]
        
    def __str__(self):
        thisstr = ''
        for impurity in self:
            thisstr += '{}\n{}'.format(impurity.get_index(), impurity)
        return thisstr
    
    def site_locations(self, simulation_cell, use_displaced=True):
        '''Sets the site locations for each impurity in the defect
        cluster by extracting the appropriate coordinates from 
        <simulation_cell>, using the coordinates in the undeformed cell
        if <use_displaced> is False.
        '''
        
        for impurity in self:
            impurity.site_location(simulation_cell[impurity.get_index()], 
                                            use_displaced=use_displaced)
                                            
        return
                                            
    def atomic_site_coords(self):
        '''Calculates the atomic coordinates of each impurity atom in the 
        cluster using the impurity coordinates and the coordinates of the site
        at which each impurity is located. Note that, if the coordinates of the 
        impurity atoms relative to their associated defect sites have been given
        in angstroms/bohr/stadia, they must first be converted to crystal 
        coordinates using <self.to_cell_coords()>.
        '''
        
        for imp in self:
            imp.atomic_site_coords()
        
        return
                                        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.currentindex >= len(self.impurities):
            # reset iteration state
            self.currentindex = 0
            raise StopIteration
        else:
            self.currentindex += 1
            return self.impurities[self.currentindex-1]
            
    def next(self):
        return self.__next__()
            
    def to_cell_coords(self, lattice):
        '''Converts the coordinates of all atoms contained in the 
        impurities that make up <cluster> from cartesian to cell
        coordinates.
        '''
        
        for imp in self:
            imp.to_cell_coords(lattice)
            
        return
            
def merge_coupled(*defect_clusters):
    '''Creates a single <CoupledImpurity> from a collection of <CoupledImpurity>s.
    '''
    
    new_cluster = CoupledImpurity()
    
    # add all single-site defects from each cluster to new_cluster
    for cluster in defect_clusters:
        for impurity in cluster:
            new_cluster.add_impurity(impurity.copy())
    
    return new_cluster
            
### FUNCTIONS TO INSERT IMPURITIES INTO A STRUCTURE ###

def cell_defect(simcell, defect, use_displaced=True):
    '''Inserts the provided defect (+ site) into the simulation cell.
    '''
    
    # safety check to make sure that <defect> is not an <CoupledImpurity>
    if isinstance(defect, CoupledImpurity): 
        cell_defect_cluster(simcell, defect, use_displaced=True)
        return
        
    # else...
    # switch off atom to be replaced
    simcell[defect.get_index()].switchOutputMode()
    
    # set coordinates of defect
    defect.site_location(simcell, use_displaced=use_displaced)
    
    if len(defect) == 0:
        # impurity is empty => vacuum
        pass
    else:
        for atom in defect:
            # make displaced coordinates the base coordinates, for compatibility
            new_atom = atom.copy()
            new_atom.setCoordinates(new_atom.getDisplacedCoordinates())
            simcell.addAtom(new_atom)
            
    return
    
def cell_defect_cluster(simcell, defect_cluster, use_displaced=True):
    '''Inserts the provided defect cluster into the supercell.
    '''
    
    # safety check to make sure that <defect_cluster> is not an <Impurity>
    if isinstance(defect_cluster, Impurity): 
        cell_defect(simcell, defect_cluster, use_displaced=use_displaced)
    else:
        for defect in defect_cluster:
            cell_defect(simcell, defect, use_displaced=use_displaced)
        
    return
    
def undo_defect(simcell, defect_thingy):
    '''Removes all trace of a defect from a simulation cell by resetting the
    output mode of atoms that are in the perfect material and deleting any
    impurity atoms that may have been added to the <Crystal>/<Basis> object.
    <defect_thingy> may be an <Impurity> or a <CoupledImpurity>.
    '''
    
    # delete all impurity atoms
    for i in range(len(defect_thingy)):
        del simcell[-1]
        
    # switch original atoms in the defect sites back on
    if is_single(defect_thingy):
        simcell[defect_thingy.get_index()].switchOutputMode()
    else: # coupled cluster
        for dfct in defect_thingy:
            simcell[dfct.get_index()].switchOutputMode()
    
    return
    
def find_replaceable(supercell, defect):
    '''Finds indices of all sites occupied by a species for which <defect> may
    substitute. Works only for single impurities (ie. those that substitute at
    a single lattice site, although substitution at nearby interstitial sites
    is permitted).
    '''
    
    # make sure that the defect is an Impurity object
    if not is_single(defect):
        raise TypeError("Defect must be an <Impurity> object.")
    
    # find atoms for which <defect> may substitute
    indices = []
    for i, atom in enumerate(supercell):
        if atom.getSpecies() == defect.getSite():
            indices.append(i)
            
    return indices
    
def is_single(testobject):
    '''Tests to see if <testobject> is an single impurity.
    '''
    
    if type(testobject) == Impurity:
        return True
    else:
        return False
        
def is_coupled(testobject):
    '''Tests to see if <testobject> is a coupled impurity.
    '''
    
    if type(testobject) == CoupledImpurity:
        return True
    else:
        return False

    
# POSSIBLE CONSTRAINTS

def heightConstraint(zMin, zMax, atom, period=1, use_disp=False, index=-1):
    '''Ensures that the z-coordinate of <atom> (in pfractional units) is in the 
    range [zMin,zMax). <period> is the periodicity of the atomic structure along
    the dislocation line.
    '''
    
    # z coordinate of atom, constrained to be in [0,1)
    if use_disp:
        atomicZ = atom.getCoordinates()[index] % period
    else:
        atomicZ = atom.getDisplacedCoordinates()[index] % period
    
    useAtom = in_range(atomicZ, zMin, zMax)
    return useAtom

def plane_constraint(atom, i, xmin=-np.inf, xmax=np.inf, use_polymer=True,
                                                          tolerance=1e-1):
    '''Allows substitution iff coordinate of xmax(+tol) > <atom> > xmin(-tol). 
    The inclusion of a tolerance factor accounts for small deviations away from 
    symmetry plane, generally due to the use of CG or numerical BFGS. We assume
    a symmetry about x_{i} == 0. <xmax> (<xmin>) defaults to inf (-inf); ie. 
    unbounded above (below).
    '''

    if use_polymer:
        # coordinates given in polymer order, ie. 3 1 2
        i = (i+1) % 3

    coord = atom.getCoordinates()[i] 
    use_atom = in_range(coord, xmin-tolerance, xmax)
    return use_atom
    
def azimuthConstraint(thetaMin, thetaMax, atom, tol=1e-2, scale_by_r=True):
    '''Constraints impurity energies to be calculated in a finite range of angles.
    Useful when the defect (eg. screw dislocation) has some rotational symmetry.
    <scale_by_r> reduces the tolerance as r increases. 
    '''
    
    # atomic x and y coordinates
    atomicX, atomicY = atom.getCoordinates()[0], atom.getCoordinates()[1]
    # atomic angle
    atomicTheta = np.arctan2(atomicY, atomicX)
    
    if scale_by_r:    
        # calculate tolerance for this atom
        tol = tol/np.sqrt(atomicX**2+atomicY**2)

    # test to see if atom is in specified range, taking care to note the
    # periodicity of theta
    if thetaMax > thetaMin:
        if atomicTheta > thetaMax + tol:
            # calculate angular distance to the negative y axis
            atomicTheta = -2*np.pi + atomicTheta
            
        useAtom = in_range(atomicTheta, thetaMin, thetaMax, tol=tol)
    else: # thetaMax < thetaMin
        useAtom = not(in_range(atomicTheta, thetaMax, thetaMin, tol=tol))
    
    return useAtom
        
def in_range(value, rangeMin, rangeMax, tol=1e-2):
    '''Tests to see if rangeMin <= value < rangeMax.
    '''
    
    # if <rangeMin> > <rangeMax>, assume that <rangeMin> is actually a negative
    # value modulo 1, so that -0.05 would be entered as 0.95. For example, this
    # allows us to select all atoms near the basal plane (z==0.0) by entering 
    # <rangeMin> = (1-d), <rangeMax> = d, 0 < d < 0.5
    if rangeMin > rangeMax:
        rangeMin -= 1.0
    
    if value > (rangeMin-tol) and value < (rangeMax+tol):
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
            minVal, maxVal = cons.group('min'), cons.group('max')
         
            typeOfConstraint = typeOfConstraint.lower()
            try:
                constraintFunction = constraintDictionary[typeOfConstraint]
            except:
                continue
            else:
                pass    
                    
