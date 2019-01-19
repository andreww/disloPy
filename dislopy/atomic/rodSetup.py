#!/usr/bin/env python
'''Functions to construct 1D periodic cylinders, possibly with multiple regions.
'''
from __future__ import absolute_import

import numpy as np
import numpy.linalg as L
import sys

from numpy.random import uniform

from dislopy.atomic import crystal as cry
from dislopy.atomic import circleConstruct as grid

class PeriodicCluster(cry.Basis):
    '''A one-dimensionally periodic cluster of atoms.'''
    
    def __init__(self, unitCell=None, centre=np.zeros(2), R=None, thickness=1, 
                                              periodic_atoms=None, height=None):
        '''Given a <unitCell> (crystal) and <centre> (ie. point in that 
        unit cell that must lie at the origin), tiles the entire cluster,
        out to given <radius>. <thickness> gives the number of unit cells
        along the dislocation line. If <periodic_atoms> is provided, simply copy
        its contents into the <PeriodicCluster>.
        '''
        
        cry.Basis.__init__(self)
        
        # set cluster variables
        if R is None:
            raise AttributeError("Radius <R> not defined.")
        else:
            self._r = R
            self._thickness = thickness
        
        if unitCell is None:
            if periodic_atoms is None:
                raise Warning("No atoms included in <PeriodicCluster>.")
            else:
                # add atoms to <PeriodicCluster>
                for atom in periodic_atoms:
                    self.addAtom(atom)
                    
                self._height=height*thickness
                self._baseCell = cry.Lattice()
        else:    
            if not (periodic_atoms is None):
                raise Warning("Disregarding <periodic_atoms> and relying on " +
                                "specified <unitCell>.")
            # if a basecell has been provided, construct cluster using provided 
            # params
            self._baseCell = unitCell.copy()
            self._height = L.norm(unitCell.getC())*self._thickness
            
            # get the tiling of <unitCell> required to cover the cylinder cross-
            # section
            self._indexGuide, self._usedCells = grid.constructCluster(unitCell.getLattice(), 
                                                                                   R, centre)
                                                                                   
            self._dimensions = (len(self._usedCells[0]), len(self._usedCells))
            
            
            # populate the cluster with atoms
            self.constructCluster(centre)
        
    def getHeight(self):
        '''Returns the repeat distance along the axis of the cluster.
        '''
        
        return self._height

    def getBaseCell(self):
        '''Returns the lattice vectors for the unit cell of the perfect 
        (ie. dislocation free) crystal.
        '''
        
        return self._baseCell.copy() 
       
    def constructCluster(self, centre):
        '''Creates the cluster by placing one unit cell at each non-zero
        grid point in <self.tile>.
        '''
        
        for atom in self._baseCell:
            self._placeAllOfType(atom, centre)
       
    def _placeAllOfType(self, atom, centre):
        '''Places all copies of <atom> in the derived cluster. Private so that
        user cannot know precisely how atomic coordinates are chosen.
        '''
        
        lattice = self._baseCell.getLattice()
        # c cell parameter of material
        z = L.norm(lattice[-1])
        dim1 = self._dimensions[0]
        dim2 = self._dimensions[1]
        for i in range(dim2):
            for j in range(dim1):
                if self._usedCells[i, j] == 1:
                    for k in range(0, self._thickness):
                        # calculate m and n from <self.indexGuide>
                        m = self._indexGuide[i][j][0]
                        n = self._indexGuide[i][j][1]
                        newAtom = atom.copy()
                        # translate the new atom
                        newAtom.translateAtom(np.array([m, n, 0]))
                        # Next, calculate position of atom in cartesian coords
                        cartCoords = cry.fracToCart(newAtom.getCoordinates(),
                                                                        lattice)
                        # Finally, translate the atoms so that <eta> is at the
                        # origin of the cell. This ensures that the dislocation
                        # can be set up symmetrically.
                        cartCoords -= (centre[0]*lattice[0]+centre[1]*lattice[1])
                        # displace along c-axis
                        cartCoords[-1] = cartCoords[-1] + k*z
                        newAtom.setCoordinates(cartCoords)
                        newAtom.setDisplacedCoordinates(cartCoords)
                        newAtom.setSpecies(newAtom.getSpecies())
                        self.addAtom(newAtom)  
        return
        
class TwoRegionCluster(PeriodicCluster):
    '''Constructs a cluster for Gulp input. Begins by creating a periodic
    cluster and then partitioning into region I and region II. 
    '''
    
    def __init__(self, unitCell=None, centre=np.zeros(2), R=None, regionI=None, 
                  regionII=None, thickness=1, periodic_atoms=None, height=None,
                                                         rI_centre=np.zeros(2)):
        '''Initializes a 1D periodic GULP cluster.
        
        <rIcentre> specifies the coordinates of the region I axis. This defaults
        to the centre of the cluster (ie [0, 0]), but may be off-centre for 
        certain types of calculation, such as those in which dislocation-point
        defect interactions are being investigated.
        '''
        
        if R is None:
            R = regionII
        
        PeriodicCluster.__init__(self, unitCell=unitCell, centre=centre, R=R, 
                           thickness=thickness, periodic_atoms=periodic_atoms,
                                                                 height=height)
        if regionI is None:
            raise AttributeError("Region I radius not specified.")
        else:
            self._RI = regionI
            
        # if RII < RI, assume that the user has entered RII as the difference
        # between RI and the actual RII
        if regionII is None:
            raise AttributeError("Region II thickness/radius not specified.")
        elif regionII > regionI:
            self._RII = regionII
        else:
            self._RII = regionI + regionII
        
        # Create list of atoms in region I
        self._r1Atoms = cry.Basis()
        self._r2Atoms = cry.Basis()
        self._rI_centre = rI_centre.copy()
        self.specifyRegions(rI_centre=rI_centre) # exp.
        
    def getRI(self):
        '''Returns the region I radius
        '''
        
        return self._RI
        
    def getRII(self):
        '''Returns the region II radius.
        '''
        
        return self._RII
        
    def setRI(self, newRIBoundary):
        '''Changes the region I radius to <newRIBoundary>.
        '''
        
        self._RI = newRIBoundary
        self.specifyRegions()
        return 
        
    def setRII(self, newRIIBoundary):
        '''Changes the region II radius to <newRIIBoundary>. As before, if 
        <newRIIBoundary> < RI, assume that the user means RII =  RI +
        <newRIIBoundary>.
        '''
        
        if newRIIBoundary > self._RI:
            self._RII = newRIIBoundary
        else:
            self._RII = self._RI + newRIIBoundary
        
        # redetermine which regions the atoms are in    
        self.specifyRegions()
        return
        
    def specifyRegions(self, gulp_ordered=False, rI_centre=np.zeros(2)): # experimental
        '''Specifies which atoms are in region I. If <gulp_ordered> is True,
        the x and y coordinates are, respectively, elements -1 and -2 of the
        atomic position vector.
        '''
        
        self._r1Atoms.clearBasis()
        self._r2Atoms.clearBasis()
        for atom in self:
            # distance from the axis of the cluster
            Rxy = L.norm(atom.getDisplacedCoordinates()[:2])
            # distance from the axis of region I
            Rxy_I = L.norm(atom.getDisplacedCoordinates()[:2]-rI_centre)
            
            # determine to which region the atom belongs
            if Rxy_I < self._RI:
                # If the computed radial distance is less than the specified
                # region 1 radius, add atom to list of refinable atoms. Note 
                # that this requires that the dislocation line be at the 
                # origin (which the constructor should guarantee).
                self._r1Atoms.addAtom(atom)
            elif Rxy < self._RII:
                # set appropriate constraints
                newatom = atom.copy()
                newatom.set_constraints(np.zeros(3))
                try:
                    if newatom.hasShell():
                        newatom.shell_fix()
                except AttributeError:
                    pass
                    
                self._r2Atoms.addAtom(newatom)
                
    def getRegionIAtoms(self):
        '''Returns a list of atoms in region I.
        '''
        
        return self._r1Atoms.copy()
        
    def getRegionIIAtoms(self):
        '''Returns a list of atoms in region II
        '''
        
        return self._r2Atoms.copy()
        
    def access_regionI(self):
        '''Provides access to the basis representing atoms in region I.
        '''
        
        return self._r1Atoms
        
    def applyField(self, field_type, dis_cores, dis_burgers, Sij=0.5, branch=[0,-1],
                            branch_thresh=0.5, use_branch=True, centre_thresh=1e-10, 
                   at_dip=False, use_dip=False, centre_line=0., species_no_merge=[], 
                   randomise=False, random_r=5., random_amp=0.01):
        '''Applies field to cluster and then updates list of RI
        and RII atoms. Default branch cut is appropriate for the displacement
        field corresponding to a pure edge dislocation in isotropic elasticity.
        For anisotropic elasticity, we recommend setting branch=[-1, 0]
        '''      
        
        # calculate displaced coordinates for each atom in the cluster
        super(TwoRegionCluster, self).applyField(field_type, dis_cores, dis_burgers,
                                                        Sij, use_dip=use_dip, at_dip=at_dip)
                                                  
        # indices of the axes normal and parallel to the branch cut
        br_i = branch_index(branch)
        br_j = (br_i-1) % 2
            
        # check for overlapping atoms. Remove atoms that cross the branch cut.
        # NEED TO GENERALIZE FOR ARBITRARY BRANCH CUTS
        if use_branch:
            for i, atom in enumerate(self._atoms):
                # if atom is not going to be written to output, skip all tests
                if not(atom.writeToOutput()):
                    continue
                
                x0 = atom.getCoordinates()
                xi = atom.getDisplacedCoordinates()
                # if atom is near the centre of the dislocation, likewise skip tests
                if L.norm(xi[:-1]) < centre_thresh:
                    continue
                    
                threshold(x0)
                threshold(xi)

                if xi[br_j] <= 0.:
                    # atom remains in the half of the crystal defined by the branch cut
                    # NOTE: need to check if it matters whether or not it started in the 
                    # lower half of the simulation cell
                    if np.sign(centre_line-x0[br_i]) != np.sign(centre_line-xi[br_i]):
                        if np.sign(x0[br_i]-centre_line) < 0. and abs(centre_line-xi[br_i]) < 1e-12:#!
                            # atom was in the left half of the crystal and intersects 
                            # the branch cut -> will delete its counterpart from 
                            # the right half of the crystal
                            continue
                        else:
                            # atom passes over the branch cut and is deleted
                            # if it is within THRESH of the branch cut, will test to
                            # see if there are overlapping atoms 
                            if abs(xi[br_i]-centre_line) < branch_thresh:                     
                                for j, other_atom in enumerate(self._atoms):  
                                    if atom.getSpecies() in species_no_merge:
                                        continue
                                    elif i == j or not(other_atom.writeToOutput()):
                                        continue
                                    elif other_atom.getSpecies() != atom.getSpecies():
                                        # don't delete dissimilar atoms
                                        continue 
                                    else:
                                        xj = other_atom.getDisplacedCoordinates()
                                        delta = L.norm(xi-xj)
                                        if delta < 2*branch_thresh:
                                            atom.switchOutputMode()
                                            new_coords = other_atom.getDisplacedCoordinates()
                                            new_coords[br_i] = centre_line
                                            other_atom.setDisplacedCoordinates(new_coords)
                                            break 
                            else:
                                atom.switchOutputMode()
                                
                    # check for overlaps at the branch cut
                    elif x0[br_i] > centre_line and abs(centre_line-xi[br_i]) < branch_thresh:
                        for j, other_atom in enumerate(self._atoms):
                            if atom.getSpecies() in species_no_merge:
                                continue
                            elif i == j or not(other_atom.writeToOutput()):
                                continue
                            elif other_atom.getSpecies() != atom.getSpecies():
                                # don't delete dissimilar atoms
                                continue
                            else:
                                xj = other_atom.getDisplacedCoordinates()
                                #calculate distance between <atom> and <other_atom>
                                delta = L.norm(xi-xj)
                                if delta < 2*branch_thresh:
                                    # atoms deemed to be overlapping, and <atom> 
                                    # should be removed from output
                                    atom.switchOutputMode()                                    
                                    
                                    # "merge" atoms
                                    threshold(xj)
                                    new_coords = xj
                                    new_coords[br_i] = centre_line
                                    other_atom.setDisplacedCoordinates(new_coords)
                                                                
                                    break
        else:
            # no branch cut
            pass

        # randomise the coordinates of atoms < <random_r> from the dislocation line
        # if the option <randomise> is True    
        if randomise:
            for atom in self._atoms:
                x = atom.getDisplacedCoordinates()
                r = L.norm(x[:-1])
                if r < random_r:
                    # add some numerical noise to the coordinates
                    dx = uniform(low=-random_amp, high=random_amp, size=3)
                    atom.setDisplacedCoordinates(x+dx)
                                                       
        self.specifyRegions()
        return
        
# FUNCTIONS TO MANIPULATE CLUSTERS

def extend_cluster(base_cluster, new_thickness):
    '''Creates a new cluster by stacking <new_thickness> layers of <base_cluster>
    on top of one another.
    '''
    
    # create new cluster
    new_height = new_thickness*base_cluster.getHeight()
    new_cluster = TwoRegionCluster(regionI=base_cluster._RI,  regionII=base_cluster._RII,
                                height=new_height, periodic_atoms=base_cluster)

    # if thickness == 1, no need to add additional atoms
    if new_thickness == 1:
        return new_cluster
        
    # axial length vector
    disp_vec = np.array([0., 0., base_cluster.getHeight()])
    
    # add periodic images of atoms in <base_cluster>
    for atom in base_cluster:
        for i in range(new_thickness-1):
            new_atom = atom.copy()
            
            # calculate coordinates of new atom
            x = atom.getCoordinates()
            xd = atom.getDisplacedCoordinates()
            new_x = x + (i+1)*disp_vec
            new_xd = xd + (i+1)*disp_vec
            new_atom.setCoordinates(new_x)
            new_atom.setDisplacedCoordinates(new_xd)
            new_cluster.addAtom(new_atom)
    
    # assign atoms to region I/region II
    new_cluster.specifyRegions()
    return new_cluster

                                
'''utility functions for rotating vectors in R^3 (about the x3 axis)
'''

def rotation_matrix(theta):
    '''Creates the 3x3 matrix corresponding to a rotation about the x3-axis
    (in R^3) by theta radians.
    '''
    
    r_matrix = np.zeros((3, 3))
    r_matrix[0, 0] = np.cos(theta)
    r_matrix[0, 1] = -np.sin(theta)
    r_matrix[1, 0] = -r_matrix[0, 1]
    r_matrix[1, 1] = r_matrix[0, 0]
    r_matrix[2, 2] = 1.
    
    return r_matrix
    
def inverse_rotation(R):
    '''Creates the matrix corresponding to the inverse rotation of R.
    '''
    
    R_inverse = np.copy(R)
    R_inverse[0, 1] = -R[0, 1]
    R_inverse[1, 0] = -R[1, 0]
    return R_inverse
    
def threshold(x):
    '''Threshold parameter to align atoms which are very slightly off the 
    branch cut.
    '''
    
    for i in range(len(x)):
        if abs(x[i]) < 1e-12:
            x[i] = 0

def rotation_angle(branch_cut):
    '''Calculates the rotation angle between the specified <branch_cut> and the
    -y axis.
    '''
    
    theta_base = np.arctan2(branch_cut[1], branch_cut[0])
    # shift theta so that it is in the range [0, 2*pi]
    theta_base %= (2*np.pi)
    theta = 3*np.pi/2. - theta_base
    return theta
    
def branch_index(br):
    if br[0] != 0 and br[1] != 0:
        raise ValueError("Branch cut must lie along one axis.")
    elif br[0] == 0:
        return 0
    else:
        # branch cut lies along x axis
        return 1
