#!/usr/bin/env python
from __future__ import print_function

import sys
import os
sys.path.append(os.environ['PYDISPATH'])

import numpy as np
import re

from numpy.linalg import norm

from pyDis.atomic import crystal as cry
from pyDis.atomic import qe_utils as qe
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import castep_utils as castep
from pyDis.atomic import transmutation as mutate

def periodic_distance(atom1, atom2, lattice, use_displaced=True, oned=False,
                                                to_cart=True):
    '''Calculates the smallest distance between <atom1> and any periodic image
    of <atom2> (including the one defined by the lattice vector (0, 0, 0))
    '''
    
    if type(lattice) == cry.Crystal:
        lattice = lattice.getLattice()
        
    # cell lengths
    if oned:
        # construct z-aligned vector with length equal to the cell height
        scale = np.array([0., 0., lattice.getHeight()])
    else:
        # scale is the simulation cell lengths
        scale = np.zeros(3)
        for i in range(3):
            scale[i] = norm(lattice[i])
    
    # extract appropriate coordinates
    if isinstance(atom1, np.ndarray):
        x1 = atom1
    elif use_displaced:
        x1 = atom1.getDisplacedCoordinates()
    else: 
        # use non-defected coordinates
        x1 = atom1.getCoordinates()
        
    if isinstance(atom2, np.ndarray):
        x2 = atom2
    elif use_displaced:
        x2 = atom2.getDisplacedCoordinates()
    else: 
        # use non-defected coordinates
        x2 = atom2.getCoordinates()
        
    # if <to_cart> has been specified (typically only for supercells), multiply 
    # coordinates by <scale>
    if to_cart:
        if oned:
            sc = np.array([1., 1., lattice.getHeight()])
            x1 = sc*x1
            x2 = sc*x2
        else:
            x1 = scale*x1
            x2 = scale*x2
    
    # calculate distance between atom 1 and the closest periodic image of atom 2
    mindist = np.inf
    if oned:
        # 1D-periodic cluster - look only along the axis of the cylinder
        for i in [-1, 0, 1]:
            d = np.array([0., 0., i])
            d = scale*d
            vec = x1 - (x2+d)
            dist = norm(vec)
            if dist < mindist:
                mindist = dist
                minvec = vec
    else: 
        # 3D-periodic supercell - look at periodic images in all directions
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    d = np.array([i, j, k])
                    d = scale*d
                    vec = x1-(x2 + d)
                    dist = norm(vec)
                    if dist < mindist:
                        mindist = dist
                        minvec = vec
                    
    return mindist, minvec
    
def as_unit(vector):
    '''Returns a unit vector with the same orientation as <vector>.
    '''
    
    vector = np.array(vector)
    # make sure that <vector> is not the zero vector
    if abs(norm(vector)) < 1e-6:
        raise ValueError("Direction of zero vector undefined.")
        
    return vector/norm(vector)
    
def closest_atom_in_direction(atomtype, site, supercell, direction, oned=False,
                     use_displaced=True,  to_cart=True, phitol=1e-6, dtol=1e-6):
    '''Finds the closest atom in the specified direction.
    '''
    
    # convert direction vector to unit length
    unit_dir = as_unit(direction)
    
    if isinstance(site, int):
        # assume that <site> is a site index
        atomindex = site
        site = supercell[site]
    else:
        atomindex = np.nan
    
    mindist = np.inf
    closest_index = np.nan
    for i, atom in enumerate(supercell):
        # check that <atom> is not the same as the one at <atomindex>, and that
        # it is of the correct species
        if i == atomindex or atom.getSpecies() != atomtype:
            continue
        
        # calculate the direction and magnitude of the shortest distance between
        # atom <atomindex> and any periodic repeat of <atom>
        dist, vec = periodic_distance(atom, site, supercell, to_cart=to_cart,
                                                                    oned=oned)
        unit_vec = as_unit(vec)
        
        # check to see if <atom> is the closest atom (along <direction>) so far
        # <phitol> specifies how closely the interatomic separation must align
        # with the specified direction
        if dist-dtol < mindist and abs(np.dot(unit_vec, unit_dir) - 1) < phitol:
                mindist = dist
                closest_index = i

    return closest_index 
    
def closest_atom_oftype(atom, supercell, atomtype, use_displaced=True, oned=False,
                                                       to_cart=True, skip_atoms=[]):
    '''Locates the closest atom of species <atomtype> to <atom> in the 
    provided <supercell>. Primarily useful for locating hydroxyl oxygens. 
    <skip_atoms> is a list indices for atoms which should NOT be considered.
    '''
    
    if type(supercell) == cry.Crystal:
        lattice = supercell.getLattice()
    
    mindist = np.inf
    index = -1
    
    for i, atom2 in enumerate(supercell):
        if atom2.getSpecies() != atomtype or i in skip_atoms:
            # wrong species, carry on
            continue

        dist, vec = periodic_distance(atom, atom2, supercell, use_displaced=use_displaced,
                                                                oned=oned, to_cart=to_cart)

        if dist < mindist:
            mindist = dist
            index = i

    return index
    
def hydrogens_index(coupled_defect):
    '''Checks a coupled defect to find the index of the hydrogen-containing site.
    This is relevant specifically for cases where the hydrous defect is not neutral,
    and is being charge-balanced by another chemical impurity (eg. {Ti}.._{Mg} 
    with {2H}''_{Si}, forming the titanoclinohumite defect. We assume that the 
    use has supplied a <CoupledImpurity>, and that only one site contains hydrogen
    '''
    
    # regex for Hydrogen, which may be numbered
    h_reg = re.compile(r'H\d*')
    
    # find index of H containing site    
    for i, site in enumerate(coupled_defect):
        for atom in site:
            is_h = h_reg.match(atom.getSpecies())
            if is_h:
                return i
    
    # return NaN, which can then be handled by the user            
    return np.nan
    
def hydroxyl_oxygens(hydrous_defect, supercell, hydroxyl_str, program='gulp',
                     oxy_str='O', oned=False, to_cart=True, shell_model=False):
    '''Locate the hydroxyl oxygens for molecular mechanics simulations and 
    create hydroxyl oxygens (species <hydroxyl_str>) to insert into the simulation 
    cell. We assume that the coordinates of the hydrogen atoms have been set. If
    <shell_model> is True, then the polarizability of the hydroxyl oxygens is
    represented using a Dick-Overhauser shell model.
    '''

    hydroxyl_oxys = []
    
    # if <hydrous_defect> is a <CoupledImpurity>, locate the site containing
    # hydrogen
    if mutate.is_coupled(hydrous_defect):
        hydrous_defect.site_locations(supercell)
        hydrogen_site = hydrous_defect[hydrogens_index(hydrous_defect)] 
    else: # <Impurity> object
        hydrous_defect.site_location(supercell)
    
    # find hydroxyl oxygens, making sure that no oxygen for part of more than
    # one hydroxyl group 
    already_replaced = []
    for H in hydrous_defect:
        # add an impurity of type <hydroxyl_str> to <hydroxyls>
        new_hydrox = mutate.Impurity(oxy_str, 'hydroxyl oxygen')
        if program.lower() == 'gulp':
            new_atom = gulp.GulpAtom(hydroxyl_str)
            if shell_model:
                # add a polarizable shell with the coordinates of the atom
                new_atom.addShell(np.zeros(3))
                
            new_hydrox.addAtom(new_atom)        
        else:
            new_hydrox.addAtom(cry.Atom(hydroxyl_str))
               
        # locate index of site containing nearest oxygen atom
        site_index = closest_atom_oftype(H, supercell, oxy_str, oned=oned, 
                             to_cart=to_cart, skip_atoms=already_replaced)

        new_hydrox.set_index(site_index)
        new_hydrox.site_location(supercell)
        already_replaced.append(site_index)

        hydroxyl_oxys.append(new_hydrox.copy())

    # create <CoupledImpurity> to hold hydroxyl oxygens
    full_defect = mutate.CoupledImpurity()
    
    # ...which we now add to <full_defect>
    for imp in hydroxyl_oxys:
        full_defect.add_impurity(imp)
    
    # finally, add the hydrogen atoms in the defect with the hydroxyl defects
    if mutate.is_single(hydrous_defect):
        full_defect.add_impurity(hydrous_defect)
    else: # coupled defect, eg. Ti-clinohumite defect
        full_defect = mutate.merge_coupled(hydrous_defect, full_defect)

    return full_defect
    
def locate_bonded(site, siteindex, bondatom, supercell, nbonds):
    '''Locates atoms bonded to a particular site.
    '''
    
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

def calculate_hydroxyl(sysinfo, gulpcluster, radius, defect, gulpexec='./gulp',
                    constraints=None, minimizer='bfgs', maxcyc=100, noisy=False, 
                              oh_str='Oh', o_str='O', centre_on_impurity=False,
                                                       do_calc=False, tol=1e-1):
    '''Similar to the function <calculateImpurity> in <gulpUtils>, but with 
    the ability to replace oxygen atoms bonded to H atoms with their hydroxyl
    counterparts.
    '''
    
    # file to keep track of defect sites and IDs
    idfile = open('{}.{}.id.txt'.format(defect.getName(), defect.getSite()), 'w')
    idfile.write('# site-id x y z\n')
    # record base name of simulation files
    idfile.write('# {}.{}\n'.format(defect.getName(), defect.getSite()))
    
    # dummy variables for lattice and toCart. Due to the way the program
    # is set up, disloc is set equal to false, as the atoms are displaced 
    # and relaxed BEFORE we read them in
    lattice = np.identity(3)
    toCart = False
    disloc = False
    coordType = 'pcell'
    
    # test to see if <defect> is located at a single site
    if type(defect) is mutate.Impurity:
        pass
    else:
        raise TypeError('Invalid impurity type.')
        
    use_indices = []    
    for i, atom in enumerate(gulpcluster):
        # check conditions for substitution:
        # 1. Is the atom to be replaced of the right kind?
        if atom.getSpecies() != defect.getSite():
            continue
        # 2. Is the atom within <radius> of the dislocation line?
        if norm(atom.getCoordinates()[:-1]) > (radius+tol):
            continue  
              
        # check <constraints>
        if constraints is None:
            pass
        else:
            for test in constraints:
                useAtom = test(atom) 
                if not useAtom:
                    # if atom fails to satisfy ANY constraint, skip over it
                    if noisy:
                        print("Skipping atom {}.".format(i))
                    break
            if not useAtom:
                continue       
        if noisy:        
            print("Replacing atom {} (index {})...".format(str(atom), i)) 
        
        # record that atom <i> is to be replaced      
        use_indices.append(i)    
        
    # record all indices in <idfile> if not running calculations at the time of
    # execution. This facilitates later automation of calculations
    idfile.write('#')
    for i in use_indices:
        idfile.write(' {}'.format(i))
    idfile.write('\n')    
        
    # write atomic site coords to <idfile> for later use
    for i in use_indices:
        coords = gulpcluster[i].getCoordinates()
        idfile.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i, coords[0], 
                                       coords[1], coords[2], norm(coords[:-1])))
            
    idfile.close()
      
    # construct input files and, if requested by the user, run calculations    
    for i in use_indices:        
        # set the coordinates and site index of the impurity
        atom = gulpcluster[i]
        defect.site_location(atom)
        defect.set_index(i)
        
        # if the defect contains hydrogen, replace normal oxygen atoms with 
        # hydroyl oxygen atoms
        full_defect = hydroxyl_oxygens(defect, gulpcluster, oh_str, oxy_str=o_str,
                                                        oned=True, to_cart=False)          
        
        # create .gin file for the calculation
        coords = atom.getCoordinates()
        outname = '{}.{}.{}'.format(defect.getName(), defect.getSite(), defect.get_index())
        outstream = open(outname+'.gin','w')
       
        # write structure to output file, including the coordinates of the 
        # impurity atom(s)
        if centre_on_impurity:
            # use coordinates of Impurity in the plane
            rI_centre = coords[:-1]
        else:
            # centre on cylinder axis
            rI_centre = np.zeros(2)           
            
        gulp.write_gulp(outstream, gulpcluster, sysinfo, defected=False, to_cart=False,
                            impurities=full_defect, rI_centre=rI_centre, relax_type='',
                                                                  add_constraints=True)
                                                                 
        # run calculation, if requested by user
        if do_calc:
            print('Relaxing structure with defect at site {}...'.format(i))
            gulp.run_gulp(gulpexec, outname)
                    
    return

