#!/usr/bin/env python

import numpy as np
import numpy.linalg as L

def ceiling(x):
    '''Substract a small number so that integers won't be changed.
    '''
    
    return (int(x-1e-12) + 1)

def grid(radius,cellParams):
    '''Grid of points to completely cover a circle of radius <R>. NOT the 
    minimal grid.
    '''

    a = L.norm(cellParams[0])
    b = L.norm(cellParams[1])
    
    Nx = ceiling(float(radius)/a)
    Ny = ceiling(float(radius)/b)
    return np.ones((2*Ny+1,2*Nx+1))
    
def centre(x1,x2):
    x1 = x1 % 1
    x2 = x2 % 1
    return np.array([x1,x2])

def origin(Nx,Ny):
    return np.array([Nx/2,Ny/2])

def coordinateCentre(x0,xCentre):
    return x0 + xCentre
    
def d(x,y):
    d1 = (x[0] - y[0])**2
    d2 = (x[1] - y[1])**2
    return np.sqrt(d1+d2)
    
class Square:
        
    def __init__(self,m,n,L=1):
        self.locX = m
        self.locY = n
        self.sideLength = L
        
    def minDistance(self,eta):
        '''Calculate the minimum distance between this square
        and an arbitary point eta.
        '''
        if self.locX == 0 and self.locY == 0:
            # The point eta is inside the square -> distance is 0
            self.dmin = 0
        else:
            # Determine the x and y coordinates of the closest point
            minX = self.coordOfMin(self.locX,eta[0])
            minY = self.coordOfMin(self.locY,eta[1])
            
            self.dmin = self.dist(eta,minX,minY)
                
    def nonEmptyIntersect(self,eta,R):
        self.minDistance(eta)
        if self.dmin < R:
            return True
        else:
            return False
            
    def dist(self,x0,x,y):
        '''Internal distance function - permits us to work with
        shapes in the abstract.
        '''
        return self.sideLength*d(x0,np.array([x,y])) 
            
    def coordOfMin(self,coordIndex,centreCoord):
        '''Determines a coordinate of square closest to centreCoord.
        '''
        if coordIndex == 0:
            minimumCoord = centreCoord
        elif coordIndex > 0:
            minimumCoord = coordIndex
        else:
            minimumCoord = coordIndex + 1
         
        return minimumCoord
        
class Rectangle(Square):
    
    def __init__(self,m,n,x,y):
        Square.__init__(self,m,n)
        self.sideX = x
        self.sideY = y
        
    def dist(self,x0,x,y):
        '''Distance function -> uses the fact that sides have nonequal
        lengths.
        '''
        distMinPoint = np.array([self.sideX*x,self.sideY*y])
        x0 = np.array([self.sideX,self.sideY])*x0
        return d(x0,distMinPoint)
        
def constructCluster(cellVectors,R,eta):
    '''Given a specified <unitCell>, tiles the unit cell so that a circle of
    radius <R> is completely covered. <R> will, in general, be substantially
    larger than the desired outer radius RII of the cluster simulation, ensuring
    that the cluster will still be filled after the displacement field (which
    may displace atoms radially, not just azimuthally or axially) has been 
    applied.
    '''
    
    a = L.norm(cellVectors[0])
    b = L.norm(cellVectors[1])
    baseGrid = grid(R,cellVectors)
    dim1 = len(baseGrid[0])
    dim2 = len(baseGrid)
    
    useOrigin = origin(dim1,dim2)    
    useCentre = centre(eta[0],eta[1])    
    disLine = coordinateCentre(useOrigin,useCentre)
    
    indexTracker = np.zeros((dim2,dim1))
    indexTracker = np.ndarray.tolist(indexTracker)
    for i in range(dim2):
        for j in range(dim1):
            indexTracker[i][j] = [j-useOrigin[0],useOrigin[1]-i]
    
    return indexTracker,baseGrid
                
def printGrid(gridArray,gridFile,ratio):
    for i in range(len(gridArray)):
        for k in range(ratio):
            for j in range(len(gridArray[0])):        
                gridFile.write('%d ' % gridArray[i,j])
        
            gridFile.write('\n')
    gridFile.close()
    
    
    
        
        
        
            
        
