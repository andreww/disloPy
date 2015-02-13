#!/usr/bin/env python

import numpy as np
import numpy.linalg as L

import crystal as cry
import circleConstruct as grid

class PeriodicCluster(cry.Basis):
    '''A one-dimensionally periodic cluster of atoms.'''
    
    def __init__(self,unitCell,centre,R,thickness=1):
        '''Given a <unitCell> (crystal) and <centre> (ie. point in that 
        unit cell that must lie at the origin), tiles the entire cluster,
        out to given <radius>. <thickness> gives the number of unit cells
        along the dislocation line
        '''
        
        cry.Basis.__init__(self)
        # set cluster variables
        self.__r = R
        self.__baseCell = unitCell.copy()
        self.__indexGuide,self.__usedCells = grid.constructCluster(
                                 unitCell.getLattice(),R,centre)
        self.__dimensions = (len(self.__usedCells[0]),len(self.__usedCells))
        self.__thickness = thickness
        # make the cluster
        self.constructCluster(centre)
        
    def getHeight(self):
        '''Returns the repeat distance along the axis of the cluster.
        '''
        
        return L.norm(self.__baseCell.getC())

    def getBaseCell(self):
        '''Returns the lattice vectors for the unit cell of the perfect 
        (ie. dislocation free) crystal.
        '''
        
        return self.__baseCell.copy() 
       
    def constructCluster(self,centre):
        '''Creates the cluster by placing one unit cell at each non-zero
        grid point in <self.tile>.
        '''
        
        for atom in self.__baseCell.getAtoms():
            self.__placeAllOfType(atom,centre)
        
    def __placeAllOfType(self,atom,centre):
        '''Places all copies of <atom> in the derived cluster. Private so that
        user cannot know precisely how atomic coordinates are chosen.
        '''
        
        lattice = self.__baseCell.getLattice()
        # c cell parameter of material
        z = L.norm(lattice[-1])
        dim1 = self.__dimensions[0]
        dim2 = self.__dimensions[1]
        for i in range(dim2):
            for j in range(dim1):
                if self.__usedCells[i,j] == 1:
                    for k in range(0,self.__thickness):
                        # calculate m and n from <self.indexGuide>
                        m = self.__indexGuide[i][j][0]
                        n = self.__indexGuide[i][j][1]
                        newAtom = atom.copy()
                        # translate the new atom
                        newAtom.translateAtom(np.array([m,n,0]))
                        # Next, calculate position of atom in cartesian coords
                        cartCoords = cry.fracToCart(newAtom.getCoordinates(),
                                                                        lattice)
                        # Finally, translate the atoms so that <eta> is at the
                        # origin of the cell. This ensures that the dislocation
                        # can be set up symmetrically.
                        cartCoords = cartCoords - centre[0]*lattice[0] \
                                                    -centre[1]*lattice[1]
                        # displace along c-axis
                        cartCoords[-1] = cartCoords[-1] + k*z
                        newAtom.setCoordinates(cartCoords)
                        newAtom.setSpecies(newAtom.getSpecies())
                        self.addAtom(newAtom)  
        return
        
class GulpCluster(PeriodicCluster):
    '''Constructs a cluster for Gulp input. Begins by creating a periodic
    cluster and then partitioning into region I and region II. 
    '''
    
    def __init__(self,unitCell,centre,R,regionI,regionII,thickness=1):
        '''Initializes a 1D periodic GULP cluster.
        '''
        
        PeriodicCluster.__init__(self,unitCell,centre,R,thickness)
        self.__RI = regionI
        self.__RII = regionII
        # Create list of atoms in region I
        self.__r1Atoms = cry.Basis()
        self.__r2Atoms = cry.Basis()
        self.specifyRegions()
        
    def getRI(self):
        '''Returns the region I radius
        '''
        
        return self.__RI
        
    def setRI(self,newRIBoundary):
        '''Changes the region I radius to <newRIBoundary>.
        '''
        
        self.__RI = newRIBoundary
        self.specifyRegions()
        return 
        
    def setRII(self,newRIIBoundary):
        '''Changes the region II radius to <newRIIBoundary>.
        '''
        
        self.__RII = newRIIBoundary
        self.specifyRegions()
        return
        
    def specifyRegions(self):
        '''Specifies which atoms are in region I.
        '''
        
        self.__r1Atoms.clearBasis()
        self.__r2Atoms.clearBasis()
        for atom in self.getAtoms():
            Rxy = L.norm(atom.getDisplacedCoordinates()[:2])
            if Rxy < self.__RI:
                # If the computed radial distance is less than the specified
                # region 1 radius, add atom to list of refinable atoms. Note 
                # that this requires that the dislocation line be at the 
                # origin (which the constructor should guarantee).
                self.__r1Atoms.addAtom(atom)
            elif Rxy < self.__RII:
                self.__r2Atoms.addAtom(atom)
                
    def getRegionIAtoms(self):
        '''Returns a list of atoms in region I.
        '''
        
        return self.__r1Atoms.copy()
        
    def getRegionIIAtoms(self):
        '''Returns a list of atoms in region II
        '''
        
        return self.__r2Atoms.copy()
        
    def applyField(self,fieldType,disCores,disBurgers,Sij=0.5):
        '''Applies field to cluster and then updates list of RI
        and RII atoms.
        '''
        
        super(GulpCluster,self).applyField(fieldType,disCores,disBurgers,Sij)
        self.specifyRegions()
        return
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
