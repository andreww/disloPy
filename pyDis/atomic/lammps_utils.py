#!/usr/bin/env python
'''Contains utilities required for interfacing with the molecular-mechanics
massively parallel MD code LAMMPS.
'''
from __future__ import print_function

import re
import numpy as np
import sys
sys.path.append('/home/richitensor/programs/pyDis/')

from pyDis.atomic import crystal as cry
from pyDis.atomic import atomistic_utils as atm

def parse_lammps(basename, unit_cell, use_data=False, datafile=None):
    '''Parses LAMMPS file (FILES?) contained in <basename>, extracting geometrical
    parameters to <unit_cell> and simulation parameters to <sys_info>.
    '''
    
    # check that the user has passed a data.* file if <use_data> is True
    if use_data and datafile == None:
        raise NameError("No data.* file defined")

    sys_info = None
    return sys_info

def write_lammps(outstream, lmp_struc, sys_info, atom_ids, defected=True, 
                   to_cart=False, add_constraints=False, impurities=None,  
                          relax_type=None, use_data=False, datafile=None):
    '''Writes structure contained in <lmp_struc> to <outstream>. If <use_data>
    is True, write the atomic coordinates to a separate data.* file
    '''
    
    if use_data:
        # check that a data.* filename has been provided
        if datafile == None:
            raise NameError("No data.* file defined.")
            
    datastream = open(datafile, 'w')
            
    for i, atom in enumerate(lmp_struc):
        if use_data:
            datastream.write('{} {} {:.6f} {:.6f} {:.6f}'.format(i, atom_ids[atom.getSpecies()],
                                    atom.getCoordinates()

    pass
    
def read_atomic_data(datalines, thiscrystal, atom_ids):
    '''Reads in the data. file containing the atomic coordinates, if the user
    has chosen to specify them this way. <atom_ids> is a dictionary mapping the 
    numbers n the data. file to a specific atomic species (eg. 1 <==> Mg).
    '''
    
    atom_form = re.compile(r'\d+\s+(?P<typ>\d+)(?P<coords>(?:\s+-?\d+\.\d+){3})')
    
    for line in datalines:
        atom_match = atom_form.match(line)
        if not atom_match:
            # no atom found on this line, move along
            continue
            
        # extract and format the atomic coordinates
        coordinates = np.array([float(x) for x in atom_match.group('coords').split]) 
        thiscrystal.addAtom(cry.Atom(atom_match.group('sym'), coordinates))
    
    return
