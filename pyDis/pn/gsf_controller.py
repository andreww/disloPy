#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys
import argparse
import subprocess
sys.path.append('/home/richard/code_bases/dislocator2/')

import gsf_setup as gsf

from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import qe_utils as qe
from pyDis.atomic import castep_utils as castep

def command_line_options():
    '''Parse command line options to enable optional features and change
    default values of parameters.
    
    Example use:
    
    $: ./gsf_controller -u example_file.gin -sn example -n 10 -g $GULP/Src/./gulp
    
    -> defaults to a gamma surface calculation with a minimum resolution of 1 node
    per 0.25 x 0.25 square (in \AA, bohr, or whatever other distance units are 
    chosen.
    '''
    
    options = argparse.ArgumentParser()
    options.add_argument('-c', '--control-file', type=str, dest='control',
                    default='', help='File containing simulation parameters')
    options.add_argument('-u', '--unit-cell', type=str, dest='gulp_name',
                         help='Name of GULP file containing the unit cell')
    options.add_argument('-sn', '--name', type=str, dest='sim_name', default='gsf',
                         help='Base name for GSF calculation input files')
    options.add_argument('-n', '--num-layers', type=int, dest='n', default=2,
                         help='Thickness of simulation slab in unit-cells.')
    options.add_argument('-v', '--vacuum', type=float, dest='vac', default=0.0,
                         help='Thickness of the vacuum layer.')
    options.add_argument('-p' '--prog', type=str, dest='prog', default=None,
                         help='Name of atomistic program used. Options:\n' +
                                                'GULP\n' +
                                                'Quantum Espresso (QE)\n' 
                                                'CASTEP')
    options.add_argument('-exe', '--executable', type=str, dest='progexec', default=None,
                         help='Path to the executable for the atomistic code.')
    options.add_argument('-t', '--type', type=str, choices=['gline', 'gsurface'],
                         dest='simulation_type', default='gsurface',
                         help='Choose whether to calculate the PES of a gamma' +
                              ' line or a gamma surface.\n\nDefault is gamma ' +
                              ' surface.')
    options.add_argument('-d', '--direction', nargs=3, type=float, dest='line_vec',
                       default=np.array([1, 0, 0]), help='Direction of gamma line.')
    options.add_argument('-r', '--resolution', type=float, dest='res', default=0.25,
                         help='Sampling resolution of the gamma line/surface')
    options.add_argument('-f', '--dfix', type=float, dest='d_fix', default=5.0,
                         help='Thickness of static region in vacuum-buffered slab.')
    options.add_argument('-s', '--shift', type=float, dest='shift', default=0.0,
                         help='Shifts origin of unit cell.')
                                    
    return options
    
def manual_options():
    '''Prompt user to input options manually.
    '''
    
    # ask user if a control file exists. If so, read input from this file
    # otherwise, prompt user for simulation parameters
    print('Not implemented')
    pass

def read_control(control_file):
    '''Read the control file. STILL NEED TO IMPLEMENT!
    '''
    
    lines = []
    
    with open(control_file) as f:
        for line in f:
            if line:
                lines.append(line)    

def main():
    '''Controller program.
    '''
    
    if not sys.argv[1:]:
        # test to see if command line options have been passed. If they have
        # not, start interactive prompt
        args = manual_options()
    elif len(sys.argv[1:]) == 1:
        # assume that the user is providing the name of an input file
        args = read_control(sys.argv[1])
    else:
        # parse arguments
        options = command_line_options()
        args = options.parse_args()
        if args.control:
            # read simulation parameters from control file
            args = read_control(args.control)
        else:
            pass
    
    # extract unit cell and GULP run parameters from file <gulp_name>
    unit_cell = gsf.cry.Crystal()
    system_info = gsf.gulp.parse_gulp(args.gulp_name, unit_cell)
    
    # shift origin of cell
    unit_cell.translate_cell(np.array([0., 0., -1*args.shift]), modulo=True)
    
    # select output mode appropriate for the atomistic calculator used
    if 'gulp' == args.prog.lower():
        write_fn = gulp.write_gulp
    elif 'qe' == args.prog.lower()
        write_fn = qe.write_qe
    elif 'castep' == args.prog.lower():
        write_fn = castep.write_castep
    else:
        raise ValueError("{} is not a supported atomistic simulation code." +
                         "Supported codes are: GULP; QE; CASTEP.")
    
    # make the slab and construct the gamma surface/line
    new_slab = gsf.make_slab(unit_cell, args.n, args.vac, d_fix=args.d_fix, free_atoms=[])
    if args.simulation_type == 'gsurface':
        increments = gsf.gs_sampling(new_slab.getLattice(), args.res)
            
        gsf.gamma_surface(new_slab, increments, write_fn, system_info, suffix='gin',
                                       basename=args.sim_name, vacuum=args.vac)    
        
        # run the calculations, if an executable has been provided. Otherwise,
        # assume that that the input files will be transferred to another machine
        # and run by the user.
        if args.progexec != None:
            for n in xrange(0, increments[0]+1):
                for m in xrange(0, increments[1]+1):
                    print("Relaxing cell with generalized stacking fault vector" +
                            " ({}, {})...".format(n, m), end="")
                    basename = '%s.%d.%d' % (args.sim_name, n, m)
                    if 'gulp' == args.prog:
                        gulp.run_gulp(args.progexec, basename) 
                    elif 'qe' == args.prog.lower():
                        qe.run_qe(args.progexec, basename)
                    elif 'castep' == args.prog.lower():
                        castep.run_castep(args.progexec, basename)
                    
                    print("complete.") 
        else:
            pass
    elif args.simulation_type == 'gline':
        increments = None
        
        gsf.gamma_line(new_slab, args.res)
        
        if args.progexec != None:
            # run calculations
        else:
            pass
    else:
        pass          
    
if __name__ == "__main__":
    main()

    
    
