#!/usr/bin/env python
from __future__ import print_function, absolute_import

import sys

import re
import numpy as np

from numpy.linalg import norm

from dislopy.atomic import crystal as cry
from dislopy.atomic import gulpUtils as gulp

### END OF IMPORT SECTION ###

def make_slab(unit_cell, num_layers, vacuum=0.0, d_fix=5., free_atoms=[], axis=-1):
    '''Makes a slab for GSF calculation, with <num_layers> atomic layers, a 
    vacuum buffer of thickness <vacuum is added to the top, and <constraints>
    (a list of functions testing specific constraints, such as proximity to the
    buffer, atom type, etc.) are applied.
    
    ### NEEDS TO BE GENERALIZED ###
    '''
    
    # set up slab, without the vacuum layer
    new_dimensions=np.ones(3)
    new_dimensions[axis] = num_layers
    slab = cry.superConstructor(unit_cell, dims=new_dimensions)
    
    # total height of cell, including vacuum layer
    old_height = norm(slab.getVector(axis)) 
    new_height = old_height + vacuum
    
    # test to see if there is a vacuum layer, in which case a proximity
    # constraint will be applied
    if vacuum > 1e-10:
        print('Non-zero vacuum buffer. Proximity constraint to be used ' +
                  'with d_fix = {:.1f}.'.format(d_fix))
        use_vacuum = True
    else:
        print('3D-periodic boundary conditions. Proximity constraints' +
                                                   ' will not be applied.')
        use_vacuum = False
    
    # apply constraints
    for atom in slab.getAtoms():
        # fix atom if it is within d_fix of the slab-vacuum interface, provided
        # that vacuum thickness is non-zero -> notify user if this constraint is
        # set
        if use_vacuum:
            near_interface = proximity_constraint(atom, old_height, d_fix, axis)           
        else:
            near_interface = False
            
        if not near_interface:
            if atom.getSpecies() in free_atoms or free_atoms == 'all':
                # atom allowed to relax freely
                pass
            else:
                # relax normal to the slip plane
                atom.set_constraints(cry.ei(3, usetype=int))
            
    # add vacuum to the top of the slab
    # begin by computing coordinates of all atoms in the vacuum-buffered slab
    if use_vacuum:
        for atom in slab:
            coords = atom.getCoordinates()
            new_length = coords[axis] * old_height/new_height
            new_coords = np.copy(coords)
            new_coords[axis] = new_length
            atom.setCoordinates(new_coords)
            
            coords_disp = atom.getDisplacedCoordinates()
            length_disp = coords_disp[axis] * old_height/new_height
            new_disp = np.copy(coords_disp)
            new_disp[axis] = length_disp
            atom.setDisplacedCoordinates(new_disp)
                                                                
        # increase the height of the slab
        slab.setVector(slab.getVector(axis)*new_height/old_height, axis)
    
    return slab
    
def write_gulp_slab(outStream, slab, disp_vec, system_info):
    '''Routine to write a GULP output file for a slab calculation, with the 
    atoms satisfying x.e3 > 0.5 displaced by <disp_vec>.
    '''
    
    # write the header line and lattice vectors
    outStream.write('opti\n')
    outStream.write('vectors\n')
    for vector in slab.getLattice():
        outStream.write('{:.6f} {:.6f} {:.6f}\n'.format(vector[0], vector[1],
                                                                   vector[2]))
                        
    # add constraints to fix all cell vectors
    outStream.write('0 0 0 0 0 0\n')
    outStream.write('frac\n')
        
    # displace atoms in the top half of the simulation cell, and write to file
    for atom in slab.getAtoms():
        if atom.getCoordinates()[-1] < 0.5:
            atom.write(outStream, slab.getLattice(), defected=False, toCart=False,
                                                             add_constraints=True)
        else:
            x0 = atom.getCoordinates()
            atom.setDisplacedCoordinates((x0+disp_vec) % 1)
            atom.write(outStream, slab.getLattice(), toCart=False, add_constraints=True)
    
    # write the atomic charges and interatomic potentials to file            
    for line in system_info:
        outStream.write(line + '\n')
    
    return

def insert_gsf(slab, disp_vec, vacuum=0., eps=1e-6):
    '''Inserts generalised stacking fault with stacking fault vector <disp_vec>
    into the provided <slab>. If <vacuum> == 0., the stacking fault is inserted
    at z = 0.5, otherwise, we insert it at 0.5*(z-vacuum)/z (z := slab height). 
    '''
    
    height = norm(slab.getC())

    # disp_vec may be entered in 2D, but atomic coordinates are 3D
    if len(disp_vec) == 3:
        disp_vec = np.array(disp_vec)
    elif len(disp_vec) == 2:
        print("Converting displacement vector in R^{2} into a vector in R^{3}")
        temp = np.copy(disp_vec)
        disp_vec = np.array([temp[0], temp[1], 0.])
    else:
        raise ValueError("<disp_vec> has invalid (%d) # of dimensions" % len(disp_vec))

    # find the middle of the slab of atoms
    middle = 0.5*(height - vacuum)/norm(height)
    
    # record whether or not a small perturbation has been applied
    perturbed = False

    for atom in slab.getAtoms():
        if 0. <= atom.getCoordinates()[-1] < middle-eps:
            # atom in the lower half of the cell -> no slip
            continue
        else:
            # displace atom
            x0 = atom.getCoordinates()
            if not perturbed:
                new_x = (x0+disp_vec+cry.ei(3)*0.01) % 1
            else:
                new_x = (x0+disp_vec) % 1
            atom.setDisplacedCoordinates((x0 + disp_vec) % 1)
    
    return 
  
def ceiling(real_number):
    '''Ceiling function
    Input: x in R (real number)
    Output: the smallest integer >= x
    '''
    
    test_int = int(real_number)
    if abs(real_number-test_int) < 1e-11:
        # <real_number> is an integer, return its value
        return test_int
    else:
        if real_number > 0.:
            return int(real_number+1)
        else:
            return int(real_number)   
    
def gs_sampling(lattice, resolution=0.25, limits=[1., 1.]):
    '''Determine the number of samples along [100] and [010] required to 
    achieve specified resolution.
    '''
    
    Nx = ceiling(abs(limits[0])*norm(lattice[0])/resolution)
    Ny = ceiling(abs(limits[1])*norm(lattice[1])/resolution)
    
    # make sure that Nx and Ny are even
    Nx = int(Nx)
    Ny = int(Ny)   
    if Nx % 2 == 1:
        Nx = Nx + 1
        print("Incrementing Nx to make value even. New value is {}.".format(Nx))
    if Ny % 2 == 1:
        Ny = Ny + 1
        print("Incrementing Ny to make value even. New value is {}.".format(Ny))
    return Nx, Ny
    
def gl_sampling(lattice, resolution=0.25, vector=cry.ei(1), limits=1.):
    '''Determine the number of samples along <vector> required to achieve
    specified <resolution>.
    '''
    

    # transform Burgers vector in cartesian coordinates
    burgers = np.zeros(3)
    for i in range(3):
        burgers += vector[i] * lattice[i]

    # get minimum number of samples required for desired resolution   
    N = ceiling(abs(limits)*norm(burgers)/resolution)
    N = int(N)
    # Make sure that N is an even integer
    if N % 2 == 1:
        N = N + 1
        print("Incrementing N to make value even. New value is {}.".format(N))
    
    return N

def gamma_line(slab, line_vec, resolution, write_fn, sys_info, limits=1.0, 
               vacuum=0., basename='gsf', suffix='in', mkdir=False, relax=None):
    '''Sets up GULP input files required to compute energies along a gamma
    line oriented along <line_vec>, with a sampling density of <N>.
    '''
    
    N = gl_sampling(slab.getLattice(), resolution=resolution, vector=line_vec, 
                                                                limits=limits)
    
    # iterate over displacement vectors along the given Burgers vector direction
    for n in range(0, N+1):
        gsf_name = '{}.{}'.format(basename, n)
        # calculate displacement vector and insert into slab
        disp_vec = line_vec * n/float(N)*limits
        insert_gsf(slab, disp_vec, vacuum=vacuum) 
             
        # create input file appropriate for the atomic simulation code used.
        if mkdir: # make directory
            if not os.path.exists(gsf_name):
                os.mkdir(gsf_name)
            outstream = open('{}/{}.{}'.format(gsf_name, gsf_name, suffix), 'w')
        else: # use current directory   
            outstream = open('{}.{}'.format(gsf_name, suffix), 'w')
            
        write_fn(outstream, slab, sys_info, to_cart=False, defected=True, do_relax=True,
                                     add_constraints=True, relax_type=relax, prop=False)
        
    return  

def gamma_surface(slab, resolution, write_fn, sys_info, basename='gsf',
            suffix='in', limits=(1, 1), vacuum=0., mkdir=False, relax=None):
    '''Sets up gamma surface calculation with a sampling density of <N> along
    [100] and <M> along [010]. 
    
    Note: if N and M have been determined using <gs_sampling>, they will already
    be even but, for transferability, we nevertheless check that this condition
    is met.
    '''
    
    # using <gs_sampling>, calculate the number of increments along x and y 
    # required to give *at least* the specified <resolution>.
    N, M = gs_sampling(slab.getLattice(), resolution, limits)
        
    # iterate over displacement vectors in the gamma surface (ie. (001))
    for n in range(0, N+1):
        for m in range(0, M+1):
            gsf_name = '{}.{}.{}'.format(basename, n, m)
            # insert vector into slab
            disp_vec = cry.ei(1)*n*limits[0]/float(N) + cry.ei(2)*m*limits[1]/float(M)
            insert_gsf(slab, disp_vec, vacuum=vacuum)

            # write to code appropriate output file
            if mkdir: # make directory
                if not os.path.exists(gsf_name):
                    os.mkdir(gsf_name)
                outstream = open('{}/{}.{}'.format(gsf_name, gsf_name, suffix), 'w')
            else: # use current directory   
                outstream = open('{}.{}'.format(gsf_name, suffix), 'w')

            write_fn(outstream, slab, sys_info, to_cart=False, defected=True,
             do_relax=True, add_constraints=True, relax_type=relax, prop=False)

    return

# functions to set constraints

def proximity_constraint(atom, slab_thickness, d_fixed, axis, eps=1e-2):
    '''Tests to see if <atom> is within distance <d_fixed> of the top or bottom
    of a slab of height <slab_thickness>.
    '''
    
    # z coordinate of atom
    cell_coord = slab_thickness*atom.getCoordinates()[axis]
    
    # Test to see if atom is in the fixed region near the edge of the slab. Note
    # that, because of the finite precision of <d_fixed>, we subtract a small
    # value from it when testing if an atom is near the top of the slab.
    if (cell_coord < d_fixed) or (cell_coord >= (slab_thickness-d_fixed-eps)):
        # set all constraints to zero, and return True to indicate that the 
        # constraint has been applied
        atom.set_constraints(new_constraint=np.zeros(3))
        return True
    else:
        return False
