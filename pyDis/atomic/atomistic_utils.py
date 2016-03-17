#!/usr/bin/env python
'''A module to hold miscellaneous functions/classes/etc. that are generally

useful (eg. scaling k-point grids and reading input files), but have no obvious
home in any other module and are not substantial enough to form the basis of

their own modules. If you want to implement a minor helper function, this is the

module to do it in.
'''

from __future__ import print_function, division

import sys
sys.path.append('/home/richard/code_bases/dislocator2/')
from numpy.linalg import norm
import numpy as np

from pyDis.atomic import crystal as cry

def read_file(filename,path='./', return_str=False):
    '''Reads a file and prepares it for parsing. Has the option to return the

    output as a single string (with newline characters included), which can be
    useful if the structure of the input file makes regex easy (eg. CASTEP, QE)
    '''

    lines = []
    with open('%s%s' % (path, filename)) as input_file:
        for line in input_file:
            temp = line.rstrip()
            if temp:
                lines.append(temp)

    if return_str:
        all_lines = ''
        # stitch all elements of lines together
        for line in lines:
            all_lines += line + '\n'
        lines = all_lines

    return lines

def ceiling(x):
    '''Returns the smallest integer >= x.
    '''

    if abs(int(x) - x) < 1e-12:
        # if x has integer value (note: NOT type), return x
        return float(int(x))
    else:
        return float(int(x + 1.0))

def scale_kpoints(kgrid, sc_dimensions):
    '''Scales the k-point grid to reflect new supercell dimensions.
    '''

    new_grid = []
    for k, dim in zip(kgrid['spacing'], sc_dimensions):
        new_grid.append(max(int(ceiling(k / dim)), 1))

    kgrid['spacing'] = new_grid

def write_kgrid(write_fn, kgrid):
    '''Writes k-point grid.
    '''

    write_fn('%s ' % kgrid['preamble'])
    for k in kgrid['spacing']:
        write_fn(' %d' % k)
    write_fn('\n')
    return

def isiter(x):
    '''Tests to see if x is an iterable object whose class is NOT 
    <Basis> or any class derived from <Basis> (eg. <Crystal>, 
    <TwoRegionCluster>, etc.).
    '''
    
    if isinstance(x, (cry.Basis, cry.Atom)):
        # NOTE: need to implement __getitem__ for <cry.Atom>
        return False
    # else
    try:
        a = iter(x)
        return True
    except TypeError:
        return False

def write_xyz(input_crystal, filename, defected=False, description='xyz file',
                                                       to_cart=False, r=np.inf):
    '''Writes the atoms in <input_basis> to the specified .xyz file. 
    '''
    

    xyz_lines = ''
    natoms = 0
    
    for atom in input_crystal:
        # check that atom is to be written to output
        if norm(atom.getDisplacedCoordinates()[:-1]) > r:
            continue
        
        #else
        natoms += 1
            
        # write coordinates in deformed crystal if <defected> is True
        if defected:
            x = atom.getDisplacedCoordinates()
        else:
            x = atom.getCoordinates()
            
        # convert to Cartesian coordinates if <to_cart> is True. This is of 
        # particular relevance when the <input_crystal> is 3D periodic, in which
        # case the atomic coordinates are likely expressed in fractional coordinates
        if to_cart:
            x = cry.fracToCart(x, input_crystal.getLattice())
        
        xyz_lines += '{} {:.6f} {:.6f} {:.6f}\n'.format(atom.getSpecies(), x[0],
                                                                     x[1], x[2])
                                                                     
    xyz_file = open(filename, 'w')
    xyz_file.write('{}\n'.format(natoms))
    xyz_file.write('{}\n'.format(description))
    xyz_file.write(xyz_lines)
        
    xyz_file.close()
    
def to_bool(in_str):
    '''Routine to convert <in_str>, which may take the values "True" or "False", 
    a boolean (needed because bool("False") == True)
    '''
    
    bool_vals = {"True":True, "False":False}
    
    try:
        new_value = bool_vals[in_str]
    except KeyError:
        raise ValueError("{} is not a boolean value.".format(in_str))
        
    return new_value
