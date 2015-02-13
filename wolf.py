#!/usr/bin/env python

import numpy as np
import re

def readGULP(filename):
    '''Reads in a gulp output file.
    '''
    
    inFile = open(filename,'r')
    lines = [line.rstrip() for line in inFile.readlines()]
    inFile.close()
    return lines
    
def findProp(filename):
    '''Finds the total lattice energy of a GULP simulation (in eV). Returns
    <nan> and a warning if no energy is found.
    '''
    
    ELine = re.compile(r'^\s*Total lattice energy\s+=\s+' +
                '(?P<energy>-?\d+\.\d+)\s*eV')
                
    VLine = re.compile('Primitive cell volume =\s+(?P<vol>\d+\.\d+)')
    gulpFile = readGULP(filename)
    for line in gulpFile:
        EFound = re.search(ELine,line)
        VFound = re.search(VLine,line)
        if EFound:
            energy = float(EFound.group('energy'))
        if VFound:
            volume = float(VFound.group('vol'))
            
    try:
        return energy, volume
    except NameError:
        print 'Warning: Properties not found'
        return np.nan
        
def optimumWolf(material,dampRange,cutRange):
    '''Finds the damping parameter and distance cutoff that gives the best 
    agreement between the energy found using an Ewald sum and the Wolf summation
    method.
    '''
    
    # energy with ewald sum
    ewaldE,ewaldV = findProp('%s.ewald.gout' % material)
    ewald = [ewaldE,ewaldV]
    
    delta = []
    prop = []
    
    for d in dampRange:
        for c in cutRange:
            wolfEnergy,wolfVolume = findProp('%s.wolf.%.2f.%.1f.gout'  
                            % (material,d,c))
            dE = (wolfEnergy - ewaldE)/ewaldE*100
            dV = (wolfVolume - ewaldV)/ewaldV*100
            delta.append([d,c,dE,dV])
            prop.append([wolfEnergy,wolfVolume])
                
    return delta,prop,ewald
    
def runWolf(baseFile,dampRange,cutRange):
    '''Runs a series of single-point calculations in GULP with Wolf summation
    parameters given in dampRange and cutRange. Extracts optimum parameters.
    '''
    
    pass

