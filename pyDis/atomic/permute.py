#!/usr/bin/env python
'''Routines to permute the order of a specified crystal's lattice vectors.
'''
from __future__ import print_function

import numpy as np
import argparse
import warnings
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic import crystal as cry
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import castep_utils as castep
from pyDis.atomic import qe_utils as qe
#from pyDis.atomic import lammps_utils as lammps

def input_options():
    '''Options to specify the cell to be permuted, whether to overwrite
    the input file, the name of the output file (if required), and the required
    ordering of the indices.
    '''
    
    options = argparse.ArgumentParser()
    
    options.add_argument('-i', '--input', type=str, dest='input_struc',
                         help='Name of the file containing the unit cell.')
    options.add_argument('-o', '--output', type=str, dest='output_name', default='',
                         help='Destination for structure with permutated lattice vector.s')
    options.add_argument('-p', '-permutation', nargs=3, type=int, dest='perm',
                         default=np.array([0, 1, 2]), help='New order of indices')
    options.add_argument('-prog', '--program', type=str, dest='prog',
                         help='Name of atomistic simulation code used.')
                         
    return options
    
def check_permutation(permutation):
    '''Checks that <permutation> is a valid ordering of the cell vectors. Prints
    a warning if the multiplicity of the permutation is odd (which may be an 
    issue for some low symmetry structures).
    '''
    
    # check that all lattice indices are represented in <permutation>
    if (not 0 in permutation) or (not 1 in permutation) or (not 2 in permutation):
        raise ValueError("Not a valid permutation.")
    
    # test for multiplicity and throw a warning if it is odd
    print_warning = False
    if ((permutation[0] == 0 and permutation[1] != 1) or 
        (permutation[0] == 1 and permutation[1] != 2) or 
        (permutation[0] == 2 and permutation[1] != 0)):
        
        warnings.warn('The permutation is of odd multiplicity.')
    
    return
    
def permute_vectors(incell, permutation):
    '''Permutes the lattice vectors of <incell>. The length 3 vector <permutation>
    gives the new order of the lattice directions. NOTE: This is given using 
    PYTHON numbering, so the first vector should be labelled 0.
    ''' 

    vecs = [incell.getA(), incell.getB(), incell.getC()]
    for i, p in enumerate(permutation):
        incell.setVector(vecs[p][permutation], i)
    
    return
    
def permute_atomic_coords(incell, permutation, shell_allowed=False):
    '''Permutes the order of the coordinates for the atoms in <incell>. The
    variable <shell_allowed> records whether or not the specific atomistic
    simulation method permits the use of polarizable shells (eg. GULP)
    '''
    
    for atom in incell:
        # extract atomic coordinates and permute their order
        x = atom.getCoordinates()
        xd = atom.getDisplacedCoordinates()
       
        atom.setCoordinates(x[permutation])
        atom.setDisplacedCoordinates(xd[permutation])
                      
        # permute shell coordinates, if the atom has a polarizable shell
        if shell_allowed:
            if atom.hasShell():
                xs = atom.getShell()
                atom.setShell(xs[permutation])
        
    return
    
def permute_kgrid(kgrid, permutation):
    '''Permutes the order of the specified k-point grid spacing.
    '''
    
    new_grid = []
    try:
        use_grid = kgrid['old_spacing']
    except KeyError:
        kgrid['old_spacing'] = kgrid['spacing']
        use_grid = kgrid['spacing']
        
    for p in permutation:
        new_grid.append(use_grid[p])
        
    kgrid['spacing'] = new_grid
    return
    
def permute_cell(base_struc, program, permutation):
    '''Applies the permutation to a unit cell.
    '''
    
    permute_vectors(base_struc, permutation)
    if program == 'gulp':
        # permute shell coordinates as well
        permute_atomic_coords(base_struc, permutation, shell_allowed=True)
    else:
        permute_atomic_coords(base_struc, permutation)
    return
    
def main():
    '''Read in and permute structure.
    '''
    
    options = input_options()
    args = options.parse_args()
    
    base_struc = cry.Crystal()
    ab_initio = False
    if args.prog == 'gulp':
        read_fn = gulp.parse_gulp
        write_fn = gulp.write_gulp
    elif args.prog == 'qe':
        read_fn = qe.parse_qe
        write_fn = qe.write_qe
        ab_initio = True
    elif args.prog == 'castep':
        read_fn = castep.parse_castep
        write_fn = castep.write_castep
        ab_initio = True
    else:
        raise ValueError("{} is not a supported atomistic simulation code".format(args.prog))
                
    sys_info = read_fn(args.input_struc, base_struc)
    
    # check permutation and permute coordinates
    check_permutation(args.perm)
    permute_cell(base_struc, args.prog, args.perm)
        
    if ab_initio:
        # permute order of k-points
        permute_kgrid(sys_info['cards']['K_POINTS'], args.perm)
        
    # write to output
    if args.output_name:
        ostream = open(args.output_name, 'w')
    else: # overwrite input file
        ostream = open(args.input_struc, 'w')
    
    write_fn(ostream, base_struc, sys_info, to_cart=False)
        
    return
    
if __name__ == "__main__":
    main()
