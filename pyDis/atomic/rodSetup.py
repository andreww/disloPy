#!/usr/bin/env python

import numpy as np
import numpy.linalg as L
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic import crystal as cry
from pyDis.atomic import circleConstruct as grid

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
                        cartCoords -= centre[0]*lattice[0]-centre[1]*lattice[1]
                        # displace along c-axis
                        cartCoords[-1] = cartCoords[-1] + k*z
                        newAtom.setCoordinates(cartCoords)
                        newAtom.setSpecies(newAtom.getSpecies())
                        self.addAtom(newAtom)  
        return
        
class TwoRegionCluster(PeriodicCluster):
    '''Constructs a cluster for Gulp input. Begins by creating a periodic
    cluster and then partitioning into region I and region II. 
    '''
    
    def __init__(self, unitCell=None, centre=np.zeros(2), R=None, regionI=None, 
                   regionII=None, thickness=1, periodic_atoms=None, height=None):
        '''Initializes a 1D periodic GULP cluster.
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
        self.specifyRegions()
        
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
        
    def specifyRegions(self, gulp_ordered=False):
        '''Specifies which atoms are in region I. If <gulp_ordered> is True,
        the x and y coordinates are, respectively, elements -1 and -2 of the
        atomic position vector.
        '''
        
        self._r1Atoms.clearBasis()
        self._r2Atoms.clearBasis()
        for atom in self:
            Rxy = L.norm(atom.getDisplacedCoordinates()[:2])
            if Rxy < self._RI:
                # If the computed radial distance is less than the specified
                # region 1 radius, add atom to list of refinable atoms. Note 
                # that this requires that the dislocation line be at the 
                # origin (which the constructor should guarantee).
                self._r1Atoms.addAtom(atom)
            elif Rxy < self._RII:
                self._r2Atoms.addAtom(atom)
                
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
                                                        THRESH=0.5, use_branch=True):
        '''Applies field to cluster and then updates list of RI
        and RII atoms. Default branch cut is appropriate for the displacement
        field corresponding to a pure edge dislocation in isotropic elasticity.
        For anisotropic elasticity, we recommend setting branch=[-1, 0]
        '''      
        
        # calculate displaced coordinates for each atom in the cluster
        super(TwoRegionCluster, self).applyField(field_type, dis_cores,
                                                           dis_burgers, Sij)
                                                           
        # set up rotation matrix
        theta = rotation_angle(branch)
        R = rotation_matrix(theta)
        Rinv = inverse_rotation(R)
            
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
                if L.norm(xi[:-1]) < THRESH:
                    continue
                # rotate coordinates to the basis with the branch cut along -y
                x0tilde = np.dot(R, x0)
                xitilde = np.dot(R, xi)
                threshold(x0tilde)
                threshold(xitilde)
                #if x0tilde[1] < 0. and xitilde[1] < 0.:
                if xitilde[1] < 0.:
                    # atom remains in the half of the crystal defined by the branch cut
                    # NOTE: need to check if it matters whether or not it started in the 
                    # lower half of the simulation cell
                    if np.sign(x0tilde[0]) != np.sign(xitilde[0]):
                        if np.sign(x0tilde[0]) < 0. and abs(xitilde[0]) < 1e-12:
                            # atom was in the left half of the crystal and intersects 
                            # the branch cut -> will delete its counterpart from 
                            # the right half of the crystal
                            continue
                        else:
                            # atom passes over the branch cut and is deleted
                            # if it is within THRESH of the branch cut, will test to
                            # see if there are overlapping atoms
                            if abs(xitilde[0]) < THRESH:                           
                                for j, other_atom in enumerate(self._atoms):
                                    if i == j or not(other_atom.writeToOutput()):
                                        continue
                                    else:
                                        xj = other_atom.getDisplacedCoordinates()
                                        delta = L.norm(xi-xj)
                                        if delta < 2*THRESH:
                                            atom.switchOutputMode()
                                            break 
                            else:
                                atom.switchOutputMode()
                                
                    # check for overlaps at the branch cut
                    elif x0tilde[0] > 0. and abs(xitilde[0]) < THRESH:
                        for j, other_atom in enumerate(self._atoms):
                            if i == j or not(other_atom.writeToOutput()):
                                continue
                            else:
                                xj = other_atom.getDisplacedCoordinates()
                                #calculate distance between <atom> and <other_atom>
                                delta = L.norm(xi-xj)
                                if delta < 2*THRESH:
                                    # atoms deemed to be overlapping, and <atom> 
                                    # should be removed from output
                                    atom.switchOutputMode()
                                    
                                    
                                    # "merge" atoms
                                    xjtilde = np.dot(R, xj)
                                    threshold(xjtilde)
                                    # project onto branch cut
                                    #xjtilde[0] = 0.
                                    # rotate back to original coordinate system
                                    #xj_unrot = np.dot(Rinv, xjtilde)
                                    #other_atom.setDisplacedCoordinates([xj_unrot[0],
                                    #                       xj_unrot[1], xj_unrot[2]])
                                                                
                                    break
        else:
            # no branch cut
            pass
                                                       
        #super(GulpCluster, self).applyField(fieldType, disCores, disBurgers, Sij)
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
