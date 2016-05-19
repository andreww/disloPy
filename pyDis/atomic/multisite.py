#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

from numpy.linalg import norm

from pyDis.atomic import crystal as cry
from pyDis.atomic import qe_utils as qe
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import castep_utils as castep
from pyDis.atomic import transmutation as mutate

def periodic_distance(atom1, atom2, lattice, use_displaced=True):
    '''Calculates the smallest distance between <atom1> and any periodic image
    of <atom2> (including the one defined by the lattice vector (0, 0, 0))
    '''
    
    # cell lengths
    scale = np.zeros(3)
    for i in range(3):
        scale[i] = norm(lattice[i])
    
    # extract appropriate coordinates
    if use_displaced:
        x1 = scale*atom1.getDisplacedCoordinates()
        x2 = scale*atom2.getDisplacedCoordinates()
    else: 
        # use defected coordinates
        x1 = scale*atom1.getCoordinates()
        x2 = scale*atom2.getCoordinates()
    
    # calculate distance between atom 1 and the closest periodic image of atom 2
    mindist = np.inf
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                d = np.array([i, j, k])
                d = scale*d
                dist = norm(x1-(x2 + d))
                if dist < mindist:
                    mindist = dist
                    
    return mindist
    
def closest_atom_oftype(atomtype, atom, supercell, use_displaced=True):
    '''Locates the closest atom of species <atomtype> to <atom> in the 
    provided <supercell>. Primarily useful for locating hydroxyl oxygens.
    '''
    
    lattice = supercell.getLattice()
    
    mindist = np.inf
    index = -1
    
    for i, atom2 in enumerate(supercell):
        if atom2.getSpecies() != atomtype:
            # wrong species, carry on
            continue
        # else
        dist = periodic_distance(atom, atom2, lattice, use_displaced=use_displaced)

        if dist < mindist:
            mindist = dist
            index = i

    return index
    
def hydroxyl_oxygens(hydrous_defect, site, supercell, hydroxyl_str,
                                program='gulp', oxy_str='O'):
    '''Locate the hydroxyl oxygens for molecular mechanics simulations
    and create hydroxyl oxygens (species <hydroxyl_str>) to insert into
    the simulation cell. 
    '''
    
    hydrous_defect.site_location(supercell[site])

    hydroxyls = []
    hydroxyl_indices = []
    
    for H in hydrous_defect:
        # add an impurity of type <hydroxyl_str> to <hydroxyls>
        if program.lower() == 'gulp':
            hydroxyls.append(gulp.GulpAtom(hydroxyl_str))
        else:
            hydroxyls.append(cry.Atom(hydroxyl_str))
        
        # locate index of site containing nearest oxygen atom
        hydroxyl_indices.append(closest_atom_oftype(oxy_str, H,
                                                     supercell))

    # create <CoupledImpurity> containing all hydroxyl ions
    hydroxyl_O = mutate.CoupledImpurity(impurities=hydroxyls, sites=hydroxyl_indices)
    
    # merge the hydrogen atoms in the defect with the hydroxyl defects
    full_defect = mutate.merge_coupled(hydrous_defect, hydroxyl_O)

    return full_defect
    
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



