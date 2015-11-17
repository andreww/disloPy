#!/usr/bin/env python
'''Contains utilities required for interfacing with the molecular-mechanics
massively parallel MD code LAMMPS.
'''
from __future__ import print_function

import re
import numpy as np
import sys
sys.path.append('/home/richitensor/programs/pyDis/')

import crystal as cry
import atomistic_utils as util

def parse_lammps(basename, unit_cell):
    '''Parses LAMMPS file (FILES?) contained in <basename>, extracting geometrical
    parameters to <unit_cell> and simulation parameters to <sys_info>.
    '''

    sys_info = None
    return sys_info

def write_lammps(outstream, lmp_struc, sys_info, defected=True, to_cart=False,
                                       add_constraints=False, relax_type=None):
    '''Writes structure contained in <lmp_struc> to <outstream>.
    '''

    pass
