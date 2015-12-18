#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import numpy.linalg as L

# Define helper functions

def ei(i,usetype=float):
    '''Constructs orthonormal basis vector with i-th element = 1.
    '''
    ei = np.zeros(3,dtype=usetype)
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

class Atom(object):
    '''Contains the (non-program specific) information for a single
    atom in the basis. Provides functionality for rotating.
    '''
    
    def __init__(self,atomicSymbol,coordinates=np.zeros(3)):
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
        # set constraints -> defaults to free optimization
        self._constraints = np.ones(3,dtype=int)
        self.has_constraints = False
        # index is a dummy which can be used if weird indices need to be assigned
        # to atoms (see eg. the format for constraints in CASTEP)        
        self.index = None
        return
    
    def set_constraints(self,new_constraint):
        '''Sets optimization constraints, defaulting to optimize freely.
        '''
        
        self._constraints = np.copy(new_constraint)
        self.has_constraints = True
        
    def get_constraints(self):
        return np.copy(self._constraints)
        
    def __str__(self):
        return '%s %.3f %.3f %.3f' % (self.__species,self.__coordinates[0],
                                    self.__coordinates[1],self.__coordinates[2])
        
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
        
    def normaliseCoordinates(self, cellA=1, cellB=1, cellC=1):
        '''Normalise standard and displaced coordinates by dividing by 
        given factors. Useful for setting up supercell calculations.
        '''

        normalisationFactor = np.array([float(cellA),float(cellB),
                                        float(cellC)])
        self.__coordinates = self.__coordinates/normalisationFactor
        self.__displacedCoordinates = (self.__displacedCoordinates 
                                                / normalisationFactor )
        
    def translateAtom(self, x, reset_disp=True, modulo=False):
        '''Translates atom along x, without enforcing any boundary conditions.
        Note that, by default, this routine resets the displaced coordinates to
        be equal to the new base coordinates (useful for proper setup of supercells).
        if reset_disp is False, translate the displaced coordinates instead
        '''

        self.__coordinates = (self.__coordinates + x)
        if modulo:
            self.__coordinates = self.__coordinates % 1.
        if reset_disp:
            # print("Resetting displaced coordinates...")
            self.__displacedCoordinates = np.copy(self.__coordinates)
        else:
            self.__displacedCoordinates = (self.__displacedCoordinates + x)
            if modulo:
                self.__displacedCoordinates = self.__displacedCoordinates % 1.
        
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
                                    
    def copy(self):
        # copy over everything but the displacement field
        new_atom = Atom(self.getSpecies(),self.getCoordinates())
        # ...then copy the displaced coordinates
        new_atom.setDisplacedCoordinates(self.getDisplacedCoordinates())
        # check to see if atom is writable to output
        if self.writeToOutput():
            pass
        else:
            new_atom.switchOutputMode()      
        return new_atom
        
    def write(self,write_function,defected=True,add_constraints=False):
        '''Writes the coordinates of an atom to <write_function>, which is 
        typicall either the print function or an I/O stream.
        '''
        
        atomFormat = '%s %.6f %.6f %.6f'
        
        # test to see if atom should be output
        if not self.writeToOutput():
            return # otherwise, write atom to <outStream>
     
        # get coordinates of atomic core. Note that these will already be in
        # the correct units thanks to the routines in <crystal> and <rodSetup>.
        if defected:
            coords = self.getDisplacedCoordinates()
        else:
            coords = self.getCoordinates()
            
        write_function(atomFormat % (self.getSpecies(),coords[0],coords[1],coords[2]))

        # add constraints, if necessary
        if add_constraints:
            write_function(' %d %d %d\n' % (self._constraints[0],self._constraints[1],
                                                                      self._constraints[2]))
        else:
            write_function('\n')
                                                                     
        return
        

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
               
    def setA(self, newA):
        self.__a = np.copy(newA)
        
    def setB(self, newB):
        self.__b = np.copy(newB)  
              
    def setC(self, newC):
        self.__c = np.copy(newC)
        
    def setVector(self, newVector, i):
        '''Can be used to set or change the value of the i-th vector to 
        newVector.
        '''
        if i%3 == 0:
            self.setA(newVector)
        elif i%3 == 1:
            self.setB(newVector)
        elif i%3 == 2:
            self.setC(newVector)
        else:
            raise ValueError("{} is not a valid cell index.".format(i))
            
    def getVector(self, i):
        '''Can be used to retrieve the value of the i-th vector. Provided 
        '''
        if i%3 == 0:
            return self.getA()
        elif i%3 == 1:
            return self.getB()
        elif i%3 == 2:
            return self.getC()
        else:
            print("Error: invalid cell index.")
            # add actual exception later.
            
    def getLattice(self):
        return np.array([self.getA(),self.getB(),self.getC()])
        
    def copy(self):
        '''Copy constructor for the <Lattice> class.
        '''
        return Lattice(self.getA(),self.getB(),self.getC())
        
    def writeLattice(self,write_function):
        if write_function == print:
            end = ''
        else:
            end = '\n'
        write_function('%.4f %.4f %.4f%s' % (self.__a[0],self.__a[1],self.__a[2],end))
        write_function('%.4f %.4f %.4f%s' % (self.__b[0],self.__b[1],self.__b[2],end))
        write_function('%.4f %.4f %.4f%s' % (self.__c[0],self.__c[1],self.__c[2],end))
        
    def setLattice(self, otherLattice):
        '''Sets <self> equal to some <otherLattice>.
        '''
        self.__a = otherLattice.getA()
        self.__b = otherLattice.getB()
        self.__c = otherLattice.getC()
        
class Basis(object):
    '''Define a basis -> ie. a set of atoms to serve as basic repeating
    chemical unit
    '''
    
    def __init__(self):
        # <_atoms> is an array of Atom objects.
        self._atoms = []
        self.numberOfAtoms = 0
        self._currentindex = 0
        
    def clear(self):
        '''Delete all atoms from <Basis> instance.
        '''
        
        self.__init__()
        
    def addAtom(self,newAtom):
        '''Appends the <Atom> object <newAtom> to the basis.
        '''
        self._atoms.append(newAtom.copy())
        self.numberOfAtoms += 1
        
    def getAtoms(self):
        return self._atoms
        
    def removeAtom(self,i):
        '''Removes i-th atom.
        '''
        del self._atoms[i]
        self.numberOfAtoms -= 1
        
    def clearBasis(self):
        '''Clears the basis.
        '''
        self.__init__()
            
    def __str__(self):
        if self.numberOfAtoms == 0:
            return "empty"
        # else
        basisString = str(self[0])
        if self.numberOfAtoms > 1:
            for atom in self[1:]:
                basisString += ('\n' + str(atom))
        return basisString

    def __getitem__(self,key):
        '''Note that if key is a slice, the returned object will only have copies
        of atoms in its range, rather than references. In contrast, if key is an
        integer then it will return the object self._atoms[key].
        '''

        if isinstance(key,slice):
            # return a <Basis> object
            sub_basis = Basis()
            for atom in self._atoms[key]:
                sub_basis.addAtom(atom)
            return sub_basis
        else:
            return self._atoms[key]

    def __delitem__(self,key):
        '''Deletes atoms from the Basis. Particularly useful for point defect 
        calculations.
        '''

        del self._atoms[key]
        self.numberOfAtoms -= 1

    def __iter__(self):
        return self

    def next(self):
        if self._currentindex >= self.numberOfAtoms:
            self._currentindex = 0            
            raise StopIteration
        else:
            self._currentindex += 1
            return self[self._currentindex-1]

    def __len__(self):
        return self.numberOfAtoms
            
    def copy(self):
        '''Copy constructor for <Basis> class. Creates a new <Basis> object 
        and then copies atoms in <self> over individually. 
        '''
        # Temporary basis to copy elements of self into
        tempBasis = Basis()
        for atom in self:
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
        for atom in otherBasis:
            self.addAtom(atom)

    def translate_cell(self, disp_vec, reset_disp=True, modulo=False):
        '''Translates all atoms in <cell> by <disp_vec>. Particularly useful
        for setting up stacking fault calculations, as <disp_vec> can be set
        to -<current_slipz>*cry.ei(3) to shift the cell so that the desired
        slip plane is at z = 0. Set <reset_disp> to False if you want to translate 
        (rather than reset) the displaced atomic coordinates.
        '''
        
        for atom in self:
            atom.translateAtom(disp_vec, reset_disp, modulo)
        return    
        
    def applyField(self,fieldType,disCores,disBurgers,Sij=0):
        '''Total displacement due to many dislocations.
        '''
        for atom in self:
            self.totalDisplacement(fieldType,atom,disCores,disBurgers,Sij)
            
    def totalDisplacement(self,u,atom,disCores,disBurgers,Sij=0):
        '''Sums the displacements at x due to each dislocation in <disCores>, with
        Burgers vectors given in <disBurgers> and <Sij> are the relevant elasticity
        parameters.
        '''
        if len(disCores) != len(disBurgers):
            print('Error: number of cores and number of given Burgers vectors do ',)
            print('not match. Exiting...')
            sys.exit(1)
            
        x = atom.getCoordinates() 
        
        # apply displacements due to each dislocation                              
        uT = np.array([0.,0.,0.])
        for i in range(len(disCores)):
            uT += u(x,disBurgers[i],disCores[i],Sij)
            
        # Apply displacement to atomic coordinates
        x += uT
        # ...and then set this to be the displaced coordinate of the atom
        atom.setDisplacedCoordinates(x)

    def write(self, outstream, defected=True, add_constraints=False):
        '''Writes atoms in basis to output stream.
        '''

        for atom in self:
            atom.write(outstream.write,defected,add_constraints)

        return
        
class Crystal(Basis, Lattice):
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
            for m in range(-N,N+1):
                for b in disBurgers:
                    newBurgers.append(b)
                for x in disCores:
                    newX = np.array([x[0]+n,x[1]+m])
                    newCores.append(newX.copy())
                
        newBurgers = np.array(newBurgers)
        newCores = np.array(newCores)
        
        for atom in self:
            self.totalDisplacement(fieldType,atom,newCores,newBurgers,Sij)
        
    def totalDisplacement(self,u,atom,disCores,disBurgers,Sij=0):
        '''Sums the displacements at x due to each dislocation in <disCores>, with
        Burgers vectors given in <disBurgers> and <Sij> are the relevant elasticity
        parameters. Normalises displacements by ratio of sidelengths to Burgers vector
        length, noting that burgers vectors are in units of c-lengths.
        '''
        
        if len(disCores) != len(disBurgers):
            print('Error: number of cores and number of given Burgers vectors do ',)
            print('not match. Exiting...')
            sys.exit(1)
            
        cartCoords = fracToCart(atom.getCoordinates(),self.getLattice()) 
                                     
        uT = np.array([0.,0.,0.])
        for i in range(len(disCores)):
            # Must convert the dislocation core locations to cartesian 
            # coordinates. Note that dislocation core locations are given
            # in terms of the crystallographic directions.
            disCoresCart = (disCores[i][0]*self.getA()
                            + disCores[i][1]*self.getB())[:2]
            uT += u(cartCoords,disBurgers[i],disCoresCart,Sij)
            
        # Apply displacement to atomic coordinates
        cartCoords += uT
        # change to fractional coords
        fracCoords = cartToFrac(cartCoords,self.getLattice())
        # set displaced coordinates of the atom
        atom.setDisplacedCoordinates(fracCoords)
        
# define functions that act only on the crystal structure
        
def superConstructor(baseCell, dims=np.ones(3), reset_disp=False):
    '''Given unit cell <baseCell> and multiplicities in x, y, and z
    (Nx, Ny, and Nz, respectively), creates supercell.
    '''
    
    # construct supercell vectors from unit cell vectors
    superA = dims[0]*baseCell.getA()
    superB = dims[1]*baseCell.getB()
    superC = dims[2]*baseCell.getC()
    
    #superCell = Crystal(superA,superB,superC)
    # Let's try to make this work for subclasses of <Crystal> (eg. CastepCrystal)
    superCell = type(baseCell)(superA,superB,superC)
    
    # Define the incremental displacement lengths
    dx = 1./dims[0]
    dy = 1./dims[1]
    dz = 1./dims[2]
    
    # Copy all of the atoms from the unit cell
    for basisAtom in baseCell:
        
        for i in range(int(dims[0])):
            for j in range(int(dims[1])):
                for k in range(int(dims[2])):
                    superAtom = basisAtom.copy()
                    superAtom.normaliseCoordinates(dims[0], dims[1], dims[2])
                    superAtom.translateAtom(np.array([i*dx,j*dy,k*dz]), reset_disp)
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
    
def cell2cart(a, b, c, alpha, beta, gamma):
    if abs(alpha - 90.0) < 1e-12:
        sina = 1.0
        cosa = 0.0
    else:
        sina = np.sin(np.radians(alpha))
        cosa = np.cos(np.radians(alpha))
    if abs(beta-90.0) < 1e-12:
        sinb = 1.0
        cosb = 0.0
    else:
        sinb = np.sin(np.radians(beta))
        cosb = np.cos(np.radians(beta))
    if abs(gamma-90.0) < 1e-12:
        sing = 1.0
        cosg = 0.0
    else:
        sing = np.sin(np.radians(gamma))
        cosg = np.cos(np.radians(gamma))
        
    c_x = 0.0
    c_y = 0.0
    c_z = c
        
    b_x = 0.0
    b_y = b*sina
    b_z = b*cosa
        
    a_z = a*cosb
    a_y = a*(cosg - cosa*cosb)/sina
    trm1 = a_y/a
    a_x = a*np.sqrt(1.0 - cosb**2 - trm1**2)
        
    return np.array([[a_x, a_y, a_z],    
                     [b_x, b_y, b_z],
                     [c_x, c_y, c_z]])
