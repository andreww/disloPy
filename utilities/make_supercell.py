#!/usr/bin/env python
from __future__ import print_function

import sys
import os
sys.path.append(os.environ['PYDISPATH'])
import numpy as np
import argparse

from pyDis.atomic import qe_utils as qe
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import castep_utils as castep
from pyDis.atomic import atomistic_utils as atm
from pyDis.atomic import crystal as cry

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

def input_options():
    '''Parse command line options to determine base structure, supercell size,
    and the atomic simulation code being used.
    '''
    
    options = argparse.ArgumentParser()
    
    options.add_argument('-u', '--unitcell', type=str, dest='unitcell',
                         help='Name of file containing unit cell.')
    options.add_argument('-p', '--program', choices=['gulp', 'qe', 'castep'],
                         type=str, dest='prog', help='Name of atomistic code.' +
                        ' Valid options are: \'qe\', \'gulp\', and \'castep\'.')
    options.add_argument('-d', '--dimensions', nargs=3, type=int, dest='dims',
          default=np.ones(3, dtype=int), help='Supercell size (length 3 array)')
    options.add_argument('-o', '--output_file', type=str, dest='supercell_name',
                         default='supercell.in', help='Name of output file.')
    options.add_argument('-r', '--relax', type=to_bool, dest='relax', default=True,
                         help='Used for fixed cell gulp calculation.')
    options.add_argument('-c', '--calc', type=str, dest='calc_type', default=None,
                         help='Specify calculation type (eg. relax, scf, etc.)') 
    options.add_argument('-prop', '--properties', type=to_bool, dest='prop', 
                         default=False, help='Calculate cell properties in GULP.')  
                         
    return options 

def main():
    '''Make supercell. This is just a utility script for personal use, so we'll
    hard code all of the parameters.
    
    CURRENT_VERSION: GULP
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
                
    sys_info = read_fn(args.unitcell, base_struc)
    if ab_initio:
        atm.scale_kpoints(sys_info['cards']['K_POINTS'], np.array(args.dims))
                              
    supercell = cry.superConstructor(base_struc, np.array(args.dims))
    
    outstream = open(args.supercell_name, 'w')
    write_fn(outstream, supercell, sys_info, defected=False, do_relax=args.relax,
                                        relax_type=args.calc_type, prop=args.prop)

if __name__ == "__main__":
    main()
