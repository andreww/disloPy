#!/usr/bin/env python

import numpy as np
import numpy.linalg as L

# Define helper functions

def ei(i):
    '''Constructs orthonormal basis vector with i-th element = 1.
    '''
    ei = np.zeros(3)
    ei[i-1] = 1
    return ei
    
def fracToCart(fracCoords,cellVectors):
    '''Converts the atoms in a basis to have Cartesian coordinates.
    DOES NOT OVERWRITE ORIGINAL DATA.
    '''
    newCoords = np.zeros(3)
    for i in range(3):
        newCoords += fracCoords[i] * np.copy(cellVectors[i])
    return newCoords
    
def cartToFrac(cartCoords,cellVectors):
    '''Converts cartesian coordinates to atomic coordinates. 
    Generalise later.
    '''
    newCoords = np.zeros(3)
    for i in range(3):
        numer = np.dot(cartCoords,cellVectors[i])
        denom = np.dot(cellVectors[i],cellVectors[i])
        newCoords[i] = numer/denom
    return newCoords
    
def cellToCart(cellParameters):
    '''Converts list of six cell parameters (a,b,c,alpha,beta,gamma)
    to three lattice vectors x1,x2,x3, with x3 || z.
    '''
    
    pass
    
# define classes

class Lattice(object):
    '''Defines an abstract lattice with no atoms. Provides
    functionality for coordinate transforms, etc.
    '''
    
    def __init__(self,x=ei(1),y=ei(2),z=ei(3)):
        # <x>, <y>, <z> are 3*1 np arrays. If no arguments are given,
        # set all lattice vectors to zero.
        self.__a = np.copy(x)
        self.__b = np.copy(y)
        self.__c = np.copy(z)
        
    def rotate(self,ax=np.array([0,0,1]),theta=0.0):
        # rotates around <ax> (a numpy array) by <theta> (in degrees).
        # Currently just a Dummy function
        pass
        
    def scale(self,scaleFactor):
        # scales all lattice vectors by <scaleVector>. Useful for 
        # isostructural materials
        self.__a = scaleFactor*self.__a
        self.__b = scaleFactor*self.__b
        self.__c = scaleFactor*self.__c
        
    def getA(self):
        return np.copy(self.__a)
                    
    def getB(self):
        return np.copy(self.__b) 
               
    def getC(self):
        return np.copy(self.__c)
               
    def setA(self,newA):
        self.__a = np.copy(newA)
        
    def setB(self,newB):
        self.__b = np.copy(newB)  
              
    def setC(self,newC):
        self.__c = np.copy(newC)
        
    def setVector(self,newVector,i):
        '''Can be used to set or change the value of the i-th vector to 
        newVector.
        '''
        if i == 1:
            self.setA(newVector)
        elif i == 2:
            self.setB(newVector)
        elif i == 3:
            self.setC(newVector)
        else:
            print "Error: invalid cell index."
            # add actual exception later.
            
    def getVector(self,i):
        '''Can be used to retrieve the value of the i-th vector. Provided 
        '''
        if i == 1:
            self.getA()
        elif i == 2:
            self.getB()
        elif i == 3:
            self.getC()
        else:
            print "Error: invalid cell index."
            # add actual exception later.
            
    def getLattice(self):
        return np.array([self.getA(),self.getB(),self.getC()])
        
    def copy(self):
        '''Copy constructor for the <Lattice> class.
        '''
        return Lattice(self.getA(),self.getB(),self.getC())
        
    def printLattice(self):
        print '%.4f %.4f %.4f' % (self.__a[0],self.__a[1],self.__a[2])
        print '%.4f %.4f %.4f' % (self.__b[0],self.__b[1],self.__b[2])
        print '%.4f %.4f %.4f' % (self.__c[0],self.__c[1],self.__c[2])
        
    def setLattice(self,otherLattice):
        '''Sets <self> equal to some <otherLattice>.
        '''
        self.__a = otherLattice.getA()
        self.__b = otherLattice.getB()
        self.__c = otherLattice.getC()
        
class Atom(object):
    '''Contains the (non-program specific) information for a single
    atom in the basis. Provides functionality for rotating.
    '''
    
    def __init__(self,atomicSymbol,coordinates):
        # Assume atomicSymbol correct, allows for more general names
        self.__species = atomicSymbol
        self.__coordinates = np.copy(coordinates)
        # displaced coordinates stores any displacement due to an elastic
        # field. Initialise to u(x,y,z) = 0 for all x,y,z, but can modify.
        # Used so that edge dislocation core energies can be defined properly.
        self.__displacedCoordinates = np.copy(coordinates)
        # member variable .__write tells us whether to write atom to output.
        # <True> for most applications, but may be <False> when inserting 
        # impurities.
        self.__write = True
        
    def rotate(self,ax=np.array([1,0,0]),theta=0.0):
        # See above, but for atoms. May define a new function in the 
        # namespace of this module to handle the actual rotating.
        # CURRENTLY DUMMY
        pass
        
    def setSpecies(self,newSpecies):
        self.__species = newSpecies
        
    def setCoordinates(self,newCoordinates):
        self.__coordinates = np.copy(newCoordinates)
        
    def setDisplacedCoordinates(self,newCoordinates):
        self.__displacedCoordinates = np.copy(newCoordinates)
        
    def normaliseCoordinates(self,cellA=1,cellB=1,cellC=1):
        '''Normalise coordinates by dividing by given factors. Useful for
        setting up supercell calculations.'''
        normalisationFactor = np.array([float(cellA),float(cellB),
                                        float(cellC)])
        self.__coordinates = self.__coordinates/normalisationFactor
        
    def normaliseDisplacedCoordinates(self,cellA=1,cellB=1,cellC=1):
        '''Normalise coordinates by dividing by given factors. Useful for
        setting up supercell calculations.'''
        normalisationFactor = np.array([float(cellA),float(cellB),
                                        float(cellC)])
        self.__displacedCoordinates = (self.__displacedCoordinates 
                                                / normalisationFactor )
        
    def translateAtom(self,x):
        '''Translates atom along x. No periodic BCs enforced.
        '''
        self.__coordinates = (self.__coordinates + x)
        
    def getSpecies(self):
        return self.__species
        
    def getCoordinates(self):
        return np.copy(self.__coordinates)
       
    def getDisplacedCoordinates(self):
        return np.copy(self.__displacedCoordinates)
        
    def writeToOutput(self):
        '''Tells us whether or not to write <atom> to output.
        '''
        
        return self.__write
        
    def switchOutputMode(self):
        '''If self.__write (ie. output mode) is True, changes to False,
        and vice versa.
        '''
        
        self.__write = (not self.__write)
                                    
    def __str__(self):
        atomString = '%s %.4f %.4f %.4f' % (self.__species,
        self.__coordinates[0],self.__coordinates[1],self.__coordinates[2])
        return atomString
                                    
    def copy(self):
        # copy over everything but the displacement field
        newAtom = Atom(self.getSpecies(),self.getCoordinates())
        # ...then copy the displaced coordinates
        newAtom.setDisplacedCoordinates(self.getDisplacedCoordinates())        
        return newAtom
        
class Basis(object):
    '''Define a basis -> ie. a set of atoms to serve as basic repeating
    chemical unit
    '''
    
    def __init__(self):
        # <__atoms> is an array of Atom objects.
        self.__atoms = []
        self.__numberOfAtoms = 0
        
    def clear(self):
        '''Delete all atoms from <Basis> instance.
        '''
        
        self.__init__()
        
    def addAtom(self,newAtom):
        '''Appends the <Atom> object <newAtom> to the basis.
        '''
        self.__atoms.append(newAtom.copy())
        self.__numberOfAtoms += 1
        
    def getAtoms(self):
        return self.__atoms
        
    def numberOfAtoms(self):
        return self.__numberOfAtoms
        
    def removeAtom(self,i):
        '''Removes i-th atom.
        '''
        del self.__atoms[i]
        self.__numberOfAtoms -= 1
        
    def clearBasis(self):
        '''Clears the basis.
        '''
        self.__init__()
            
    def __str__(self):
        basisString = ''
        for atom in self.__atoms:
            basisString += (str(atom) +'\n')
        return basisString            
            
    def copy(self):
        '''Copy constructor for <Basis> class. Creates a new <Basis> object 
        and then copies atoms in <self> over individually. 
        '''
        # Temporary basis to copy elements of self into
        tempBasis = Basis()
        for atom in self.getAtoms():
            tempBasis.addAtom(atom)
        return tempBasis
        
    def setBasis(self,otherBasis):
        '''Sets basis equal to input basis. Although unnecessary for working
        with bases, this function is needed to define a robust copy 
        constructor for the <Crystal> class.
        '''
        # begin by clearing the present basis
        self.clearBasis()
        
        # copy the atoms of <otherBasis> over to the present Basis
        for atom in otherBasis.getAtoms():
            self.addAtom(atom)
        
    def applyField(self,fieldType,disCores,disBurgers,Sij=0):
        '''Total displacement due to many dislocations.
        '''
        for atom in self.__atoms:
            self.totalDisplacement(fieldType,atom,disCores,disBurgers,Sij)
            
    def totalDisplacement(self,u,atom,disCores,disBurgers,Sij=0):
        '''Sums the displacements at x due to each dislocation in <disCores>, with
        Burgers vectors given in <disBurgers> and <Sij> are the relevant elasticity
        parameters.
        '''
        if len(disCores) != len(disBurgers):
            print 'Error: number of cores and number of given Burgers vectors do ',
            print 'not match. Exiting...'
            sys.exit(1)
            
        x = atom.getCoordinates() 
                                     
        uT = np.array([0.,0.,0.])
        for i in range(len(disCores)):
            uT += u(x,disBurgers[i],disCores[i],Sij)
            
        # Apply displacement to atomic coordinates
        x += uT
        # ...and modify the member variable using the appropriate accessor function
        atom.setDisplacedCoordinates(x)

        
class Crystal(Lattice,Basis):
    '''Crystal is defined as a lattice with a basis.
    '''
    
    def __init__(self,a=ei(1),b=ei(2),c=ei(3)):
        Lattice.__init__(self,a,b,c)
        Basis.__init__(self)
        
    def copy(self):
        '''Copy constructor for the <Crystal> class. As for Basis,
        this is made more involved by the fact that member variables do
        not match the input format.
        '''
        newCrystal = Crystal()
        newCrystal.setBasis(self)
        newCrystal.setLattice(self)
        return newCrystal
        
    def applyField(self,fieldType,disCores,disBurgers,Sij=0):
        '''Total displacement due to many dislocations. Use the first two 
        rings of quadrupoles around the central quadrupole (ie. periodic images
        out to m=n=2), which leads to mostly converged field values. Unlike for
        <Basis>, we thus needs to add extra cores beyond the boundary, along 
        with their associated Burgers vectors
        '''
        
        N = 2
        # may need to test N for new systems, but N=2 likely sufficient
        newCores = []
        newBurgers = []
        for n in range(-N,N+1):
            for b in disBurgers:
                newBurgers.append(b)
            for x in disCores:
                newX = np.array([x[0]+n,x[1]+n])
                newCores.append(newX)
                
        newBurgers = np.array(newBurgers)
        newCores = np.array(newCores)
        
        for atom in self.getAtoms():
            self.totalDisplacement(fieldType,atom,newCores,newBurgers,Sij)
        
    def totalDisplacement(self,u,atom,disCores,disBurgers,Sij=0):
        '''Sums the displacements at x due to each dislocation in <disCores>, with
        Burgers vectors given in <disBurgers> and <Sij> are the relevant elasticity
        parameters. Normalises displacements by ratio of sidelengths to Burgers vector
        length, noting that burgers vectors are in units of c-lengths.
        '''
        
        if len(disCores) != len(disBurgers):
            print 'Error: number of cores and number of given Burgers vectors do ',
            print 'not match. Exiting...'
            sys.exit(1)
            
        cartCoords = fracToCart(atom.getCoordinates(),self.getLattice()) 
                                     
        uT = np.array([0.,0.,0.])
        for i in range(len(disCores)):
            # Must convert the dislocation core locations to cartesian 
            # coordinates. Note that dislocation core locations are given
            # in terms of the crystallographic directions.
            disCoresCart = (disCores[i,0]*self.getA()
                            + disCores[i,1]*self.getB())[:2]
            uT += u(cartCoords,disBurgers[i],disCoresCart,Sij)
            
        # Apply displacement to atomic coordinates
        cartCoords += uT
        # change to fractional coords
        fracCoords = cartToFrac(cartCoords,self.getLattice())
        # ...and modify the member variable using the appropriate accessor function
        atom.setDisplacedCoordinates(fracCoords)
        
# define functions that act only on the crystal structure
        
def superConstructor(baseCell,Nx=1,Ny=1,Nz=1):
    '''Given unit cell <baseCell> and multiplicities in x, y, and z
    (Nx, Ny, and Nz, respectively), creates supercell.
    '''
    
    # construct supercell vectors from unit cell vectors
    superA = Nx*baseCell.getA()
    superB = Ny*baseCell.getB()
    superC = Nz*baseCell.getC()
    
    superCell = Crystal(superA,superB,superC)
    
    # Define the incremental displacement lengths
    dx = 1./Nx
    dy = 1./Ny
    dz = 1./Nz
    
    # Copy all of the atoms from <
    for basisAtom in baseCell.getAtoms():
        
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    superAtom = basisAtom.copy()
                    superAtom.normaliseCoordinates(Nx,Ny,Nz)
                    superAtom.translateAtom(np.array([i*dx,j*dy,k*dz]))
                    superCell.addAtom(superAtom)                    
                                                                                                                                                         
    return superCell 
    
def extractDistinctSpecies(inputBasis):
    '''Given a basis, constructs a list of distinct atomic species present.
    Returns as list
    '''
    atSpecies = []
    
    atomList = inputBasis.getAtoms()
    nAtoms = len(atomList)
    
    for i in range(nAtoms):
        atomName = atomList[i].getSpecies()
        if i == 0:
            # First atom is always distinct
            atSpecies.append(atomName)
        else:
            # Check to see if atomic species has occurred before.
            if atomName in atSpecies:
                continue
            else:
                atSpecies.append(atomName)
             
    return atSpecies
    
    

    



    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
