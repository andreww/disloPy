#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

from numpy.linalg import norm

from pyDis.atomic import crystal as cry
from pyDis.atomic import qe_utils as qe

def periodic_distance(atom1, atom2, lattice, perfect=True):
    '''Calculates the smale
    '''
    
    # cell lengths
    scale = np.zeros(3)
    for i in range(3):
        scale[i] = norm(lattice[i])
    
    # extract appropriate coordinates
    if perfect:
        x1 = scale*atom1.getCoordinates()
        x2 = scale*atom2.getCoordinates()
    else: 
        # use defected coordinates
        x1 = scale*atom1.getDisplacedCoordinates()
        x2 = scale*atom2.getDisplacedCoordinates()
    
    # calculate distance between atom 1 and the closest periodic image of atom 2
    mindist = np.inf
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                d = np.array([i, j, k])
                d = scale*d
                dist = np.linalg.norm(x1-(x2 + d))
                if dist < mindist:
                    mindist = dist
                    
    return mindist
    
def locate_bonded(site, siteindex, bondatom, supercell, nbonds):
    bondlist = []
    bonddist = []
    for i, atom in enumerate(supercell):
        if atom.getSpecies() != bondatom:
            continue
        elif i == siteindex:
            continue
        # else
        dist = distance2(site, atom)
        if len(bondlist) < nbonds:
            # list of bonded atoms not yet full
            bondlist.append(i)
            bonddist.append(dist)
            if len(bondlist) == nbonds:
                # get sorted indices
                idx = np.argsort(bonddist)
                # now sort the index and distance arrays simultaneously
                bondlist = list(np.array(bondlist)[idx])
                bonddist = list(np.array(bonddist)[idx])
        else: # len(bondlist) >= nbonds
            # check to see if <atom> is closer than any previously
            # seen by the loop
            pass


# code fragments

if __name__ == "__main__":
    notclose = True
    for i, x in enumerate(xi[-1::-1]):
        if z <= x and notclose:
            notclose = False
        elif z > x and notclose:
            print("Not bonded.")
            break
        elif z <= x:
            continue
        elif z > x and not notclose:
            print(i, xi[-1-i])
            break
            
    minbonddist = np.inf
    for atom in testsuper:
        if atom.getSpecies() != 'O':
            continue
        else: 
            dist = ms.periodic_distance(testsuper[imp_index], atom)
            if dist < minbonddist:
                minbonddist = dist
    
    maxratio = 1.1            
    nbonded = 0
    for atom in testsuper:
        if atom.getSpecies() != 'O':
            continue
        else:
            dist = ms.periodic_distance(testsuper[imp_index], atom,
                                        testsuper.getLattice())
            if dist < maxratio*minbonddist:
               nbonded += 1



