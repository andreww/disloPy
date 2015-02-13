#!/usr/bin/env python
'''This program parses an input gulp file and extracts all information
that is required to run a simulation, including cell parameters, atomic
coordinates, species, potentials, etc.
'''

import numpy as np
import re
import sys

import crystal as cry
import rodSetup as rs

    
# Functions to write different GULP input files

class GulpAtom(cry.Atom):
    '''Same as <cry.Atom>, but contains an additional variable to record
    whether it has a shell or breathing shell.
    
    ### NOT FINISHED ###
    '''
    
    def __init__(self,atomicSymbol,coordinates):
    	'''Greates an atom object with additional properties relating to the
    	realization of atoms in GULP.
    	'''

        cry.Atom.__init__(self,atomicSymbol,coordinates)
        self.__hasShell = False
       	# records whether the atom has a breathing shell
        self.__isBSM = False
        return
        
    def addShell(self,shelCoords,shellType='shel'):
    	'''Gives the atom a polarizable shell, with coordinates <shelCoords>. 
    	May add position checking later.
    	'''
    	
    	self.__hasShell = True
    	self.__shelCoords = self.getShellDistance(shelCoords)
    	
    	if shellType in 'bshe':
    		# using a breathing shell model
    		self.__isBSM = True
    	
    	return
    	
    def copyShell(self,shelDisplacement,shellType='shel'):
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
    		print 'Error: atom does not have a shell.'
    		return None
    	
    def getShellDistance(self,shelCoords):
    	'''Returns the distance between the centre of the shell and the 
    	(undisplaced) coordinates of the ionic core.
    	'''
    	
    	return (shelCoords - self.getCoordinates())
    	
    def setShell(self,newShell):
    	'''Changes coordinates of the polarizable shell to <newShell>.
    	'''
    	
    	self.__shelCoords = np.array(newShell).copy()
    	return
    	
    def normaliseShelCoordinates(self,cellA=1,cellB=1,cellC=1):
        '''Normalise shell coordinates by dividing by given factors. Useful for
        setting up supercell calculations.'''
        
        normalisationFactor = np.array([float(cellA),float(cellB),
                                        float(cellC)])
        self.__shelCoords = self.__shelCoords/normalisationFactor
        return
        
    def clusterOrder(self):    
        '''Permutes indices so that the position vector reads (z,x,y), the
        order used in a GULP polymer cell calculation.
        '''
        
        # coords in undislocated cell
        atomCoords = self.getCoordinates()
        atomCoords = np.array([atomCoords[2],atomCoords[0],atomCoords[1]])
        self.setCoordinates(atomCoords)
        
        # coords in dislocated cell
        uCoords = self.getDisplacedCoordinates()
        uCoords = np.array([uCoords[2],uCoords[0],uCoords[1]])
        self.setDisplacedCoordinates(uCoords)
        
        # shell coordinates
        if self.hasShell():
            shelCoords = self.getShell()
            shelCoords = np.array([shelCoords[2],shelCoords[0],shelCoords[1]])
            self.setShell(shelCoords)
            
        return
        
    def write(self,outStream,disloc,lattice,toCart=True):
        '''Writes an atom (including Dick-Overhauser or BSM polarizable shell,
        if present) to <outStream>.
        '''
        
        atomFormat = '%s %s %.5f %.5f %.5f\n'
        
        # test to see if atom should be output
        if not self.writeToOutput():
            return # otherwise, write atom to <outStream>
     
    	# get coordinates of atomic core. Note that these will already be in
    	# the correct units thanks to the routines in <crystal> and <rodSetup>.
        if disloc == True:
            coords = self.getDisplacedCoordinates()
        else:
            coords = self.getCoordinates()
            
        outStream.write(atomFormat % (self.getSpecies(),'core',coords[0],
                                                 coords[1],coords[2]))
        # test to see if the atom has a shell
        if self.hasShell():
        	# calculate coordinates of shell in deformed material
        	if toCart:
        	    # convert to cartesian coordinates
        	    shellCoreDist = cry.fracToCart(self.getShell(),lattice)
        	else:
        	    shellCoreDist = self.getShell()
        	    
        	newShellCoords = coords + shellCoreDist
        	# determine type of shell (ie. Dick-Overhauser shell model or BSM
        	shellType = ('bshe' if self.isBreathingShell() else 'shel')
        	outStream.write(atomFormat % (self.getSpecies(),shellType,
        			newShellCoords[0],newShellCoords[1],newShellCoords[2]))
        	
        return
            
    def copy(self):
    	'''Creates a copy of the atom.
    	'''
    	
        # copy over everything but the displacement field
        newAtom = GulpAtom(self.getSpecies(),self.getCoordinates())
        
        # ...then copy the displaced coordinates and the shell coordinates (if
        # the atom has a shell)
        newAtom.setDisplacedCoordinates(self.getDisplacedCoordinates()) 
        if self.hasShell():      	
        	newAtom.copyShell(self.getShell(),'bshe' if self.__isBSM
        													 else 'shel') 
    	else:
    		# core only
    		pass
    		      
        return newAtom
      
def writeSuper(cell,sysInfo,outFile,relax='conv',coordinateType='frac'):
    '''Writes file to <outFile> for supercell (3D periodic) calculation.
    '''
    
    perOutStream = open(outFile + '.gin','w')
    disOutStream = open('ndf.' + outFile + '.gin','w')
    
    for stream in [perOutStream,disOutStream]:
        stream.write('opti %s conjugate\n' % relax)
        writeVectors(cell.getLattice(),stream)
        if stream == perOutStream:
            # do not use displaced coordinates
            for atom in cell.getAtoms():
                atom.write(stream,False,cell.getLattice(),toCart=False)
        elif stream == disOutStream:
            # use displaced coordinates
            for atom in cell.getAtoms():
                atom.write(stream,True,cell.getLattice(),toCart=False)
        # write the species list, potentials and other system information to
        # the output stream.
        for line in sysInfo:
            stream.write(line + '\n')
            
        # switch to bfgs when gnorm drops below a certain (system-dependent) size
        if cell.numberOfAtoms() >= 100:
            gnormSwitch = 1./((cell.numberOfAtoms()/100) * 100)
            gnormSwitch = max(gnormSwitch,0.001)
        else:
            gnormSwitch = 0.1
        
        stream.write('switch nume gnorm %.6f\n' % gnormSwitch)
        stream.write('dump every 10 %s.grs' % outFile)
        
        stream.close()
    return
    
def writeSlab():
    '''Write a slab simulation cell to file.
    '''
    
    pass
    
def write1DCluster(clusterCell,sysInfo,outFile):
    '''Writes 1D-periodic simulation cell to file. Always write an accompanying
    undislocated cluster, so that dislocation energy can be calculated
    '''
    
    # Get list of atoms in each region
    rIBasis = clusterCell.getRegionIAtoms()
    rIIBasis = clusterCell.getRegionIIAtoms()
    
    
    # For some strange reason, gulp places dislocation line along e(1),
    # so we have to permute the coordinates of all atoms
    for atom in rIBasis.getAtoms():
        atom.clusterOrder()
        
    for atom in rIIBasis.getAtoms():
        atom.clusterOrder()
        
    perOutStream = open('ndf.' + outFile + '.gin','w')
    disOutStream = open('dis.' + outFile + '.gin','w')
        
    perOutStream.write('qok eregion\n')
    disOutStream.write('opti conv qok eregion conjugate\n')
    
    height = clusterCell.getHeight()
    
    for stream in [perOutStream,disOutStream]:
        stream.write('pcell\n')  
        stream.write('%.6f 0\n' % height)
        
        # fourth variable tells <writeRegion> which coordinates to use
        if stream == perOutStream:
            disloc = False
        else:
            disloc = True
            
        baseCell = clusterCell.getBaseCell().getLattice()
            
        writeRegion(rIBasis,baseCell,stream,1,disloc)
        writeRegion(rIIBasis,baseCell,stream,2,disloc)
            
        for line in sysInfo:
            stream.write(line + '\n')
    
    # switch to bfgs when gnorm drops below a certain (system-dependent) size
    gnormSwitch = 1./((rIBasis.numberOfAtoms()/100) * 100)
    gnormSwitch = max(gnormSwitch,0.001)
    disOutStream.write('switch nume gnorm %.6f\n' % gnormSwitch)
        
    # tell gulp to produce a restart file at the end
    perOutStream.write('dump %s.grs' % ('ndf.' + outFile))
    disOutStream.write('dump every 10 %s.grs' % ('dis.' + outFile))
    
    perOutStream.close()
    disOutStream.close()
    
    return
    
def writeDefectCluster():
    '''Sets up cluster calculation (eg. Mott-Littleton).
    '''
    pass

# Master level functions for opening and parsing GULP input

atomLine = re.compile(r'([A-Z][a-z]?\d*)\s+(core|c|shel|s|bshe|bcor)' + \
                                                    '((\s+-?\d+\.\d+){3})')
#speciesLine = re.compile('^([A-Z][a-z]?\d*?)\s+(core|c|shel|s|bshe|bcor)' + \
#                                                          '\s+-?\d+\.\d+\s*$')

def readGulpFile(filename,path='./'):
    '''Reads a GULP file and prepares it for parsing.
    '''
    
    inputFile = open(path+filename,'r')
    lines = [line.rstrip() for line in inputFile.readlines()]
    inputFile.close()
    
    return lines
    
def parseGulpFile(gulpLines,crystalStruc):
    '''Parses the file <line> (which has been extracted and
    formatted by <readGulpFile>) to extract structural information,
    atomic potentials, and control parameters. <crystalStruc> must be 
    initialised (with dummy cell parameters) prior to use.
    '''

    nLines = len(gulpLines)
    
    # <systemInformation> stores species, potentials, etc.
    sysInfo = []    
    atomsNotFound = True
    allAtoms = dict()
    
    i = 0
    
    for i,line in enumerate(gulpLines):
        if line.strip() in 'vectors':
            #can read in vectors directly
            for j in range(3):
                temp = gulpLines[i+1+j].split()
                cellVector = np.array([float(temp[k]) for k in range(3)])               
                crystalStruc.setVector(cellVector,j+1)                                                   
        elif line.strip() in 'cell':
            #different stuff
            cellParameters = gulpLines[i+1].split()[:6]
            cellParameters = [float(a) for a in cellParameters]
            # reformat cell vectors
            cellVectors = cry.cellToCart(cellParameters)
            for j in range(3):
                crystalStruc.setVector(cellVectors[j],j)
        else:
            foundAtoms = atomLine.match(line)
            if foundAtoms:
                if atomsNotFound:
                    # record that we are now reading atom info
                    atomsNotFound = False
                
                # extract atom info to <allAtoms>    
                extractAtom(foundAtoms,allAtoms)
                        
            elif (not atomsNotFound) and (not foundAtoms):
                # Locates the end of the atomic line section
                if ('dump' not in line) and ('switch' not in line):
                    sysInfo.append(line)  
        
    for element in allAtoms:
        for atom in allAtoms[element]['atoms']:
            crystalStruc.addAtom(atom) 
    	
    # delete list of atoms
    del allAtoms     
    
    return sysInfo
    
# Utility functions used to parse specific GULP input 
            
def extractAtom(atomRegex,atomsDict):                   
    '''Extracts atom info found in <atomRegex> to existing dictionary of atoms
    <atomsDict>.
    '''
    
    atomicSymbol = atomRegex.group(1)
               
    if atomicSymbol in atomsDict:
        typeOfAtom = atomRegex.group(2)
        tempCoords = atomRegex.group(3)
        tempCoords = tempCoords.split()
        atomCoords = np.array([float(x) for x in tempCoords])
        if typeOfAtom in 'shell':
            index = atomsDict[atomicSymbol]['shells']
            atomsDict[atomicSymbol]['atoms'][index].addShell(atomCoords)
            atomsDict[atomicSymbol]['shells'] += 1
        else:
            newAtom = GulpAtom(atomicSymbol,atomCoords)
            atomsDict[atomicSymbol]['atoms'].append(newAtom)
    else:
        atomsDict[atomicSymbol] = dict()
        atomsDict[atomicSymbol]['shells'] = 0
        atomsDict[atomicSymbol]['atoms'] = []
        if atomRegex.group(2) in 'shell' or atomRegex.group(2) in 'bshe':
            print 'Error'
        else:
            tempCoords = atomRegex.group(3)
            tempCoords = tempCoords.split()
            atomCoords = np.array([float(x) for x in tempCoords])
            newAtom = GulpAtom(atomicSymbol,atomCoords)
            atomsDict[atomicSymbol]['atoms'].append(newAtom)
            
    return 
    
def cellToCart(parameters):
    '''Converts 6 cell parameters to lattice vectors.
    For the moment, assume that we are working with a cell
    whose lattice vectors are orthogonal. ### NEED TO GENERALIZE THIS###.
    '''
    
    # extract the unit cell parameters
    [a,b,c,alp,bet,gam] = parameters
    
    x1 = a*cry.e(1)
    x2 = b*cry.e(2)
    x3 = c*cry.e(3)
                 
    return cry.Lattice(x1,x2,x3) 
    
def writeRegion(regionBasis,lattice,outStream,regionNumber,disloc,
                                        toCart=True,coordType='cartesian'):
    '''Outputs all atoms in <regionBasis> to <outStream>, preceded by
    "cart region 1 if <regionNumber> == 1 and with all refinement flags set
    to 1, or cart region 2 rigid and all refinement flags == 0 if 
    <regionNumber> == 2. <disloc> tells us whether to use displaced coords.
    '''
    
    #atomList = regionBasis.getAtoms()
    if regionNumber == 1:
        outStream.write("%s region 1\n" % coordType)
        for atom in regionBasis.getAtoms():
            atom.write(outStream,disloc,lattice,toCart)
    elif regionNumber == 2:
        outStream.write("%s region 2 rigid\n" % coordType)
        for atom in regionBasis.getAtoms():
            atom.write(outStream,disloc,lattice,toCart)
    else:
        raise ValueError('%s is not a valid region.')
        
    return
    
def writeVectors(cellVectors,outStream):
    '''Writes <cellVectors to output file <outStream>.
    '''
    
    outStream.write('vectors\n')
    for i in range(3):
        outStream.write('%.5f %.5f %.5f\n' % (cellVectors[i][0],
                             cellVectors[i][1],cellVectors[i][2]))
    return
    
def extractRegions(cluster,rIBasis,rIIBasis):
    '''Given a gulp output (for a cluster calculation), extracts a list of 
    atoms in region I and region II. 
    Precondition: <rIBasis> and <rIIBasis> have been initialised and contain
    no atoms.
    ''' 
    
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
            foundAtom = re.search(atomLine,line)
            if inRI and foundAtom:
                extractAtom(foundAtom,dictRI)
            elif inRII and foundAtom:
                extractAtom(foundAtom,dictRII)
                
    # copy atoms from the temporary, element-sorted dictionaries into regions 
    # and II            
    for element in dictRI:
        for atom in dictRI[element]['atoms']:
            rIBasis.addAtom(atom)
     
    for element in dictRII:
        for atom in dictRII[element]['atoms']:
            rIIBasis.addAtom(atom) 
            
    del dictRI, dictRII
                       
    return
        
        
        
        
        
        
    
