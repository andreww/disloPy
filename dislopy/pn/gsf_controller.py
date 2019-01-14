#!/usr/bin/env python
from __future__ import print_function, absolute_import

import sys

import numpy as np
import argparse
import subprocess

from dislopy.pn import gsf_setup as gsf
from dislopy.atomic import gulpUtils as gulp
from dislopy.atomic import qe_utils as qe
from dislopy.atomic import castep_utils as castep
from dislopy.atomic import crystal as cry
from dislopy.utilities import atomistic_utils as atm

### END IMPORT SECTION

# list of atomic simulation codes currently supported by dislopy
supported_codes = ('qe', 'gulp', 'castep')

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
    options.add_argument('-u', '--unit-cell', type=str, dest='cell_name',
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
                       default=cry.ei(1), help='Direction of gamma line.')
    options.add_argument('-r', '--resolution', type=float, dest='res', default=0.25,
                         help='Sampling resolution of the gamma line/surface')
    options.add_argument('-s', '--shift', type=float, dest='shift', default=0.0,
                         help='Shifts origin of unit cell.')

    # limit the gamma surface/line calculation to one sector of the slip plane.
    # Only x and y limits are available through the command line and manual input.
    # For all other limits (in particular by angle), use the input file (not yet
    # implemented).

    options.add_argument('-x', '--xmax', type=float, dest='max_x', default=1.0,
                         help='Maximum displacement vector along x.')
    options.add_argument('-y', '--ymax', type=float, dest='max_y', default=1.0,
                         help='Maximum displacement vector along y.')
                         
                         
    # list of contraints    
    options.add_argument('-fx', '--dfix', type=float, dest='d_fix', default=5.0,
                         help='Thickness of static region in vacuum-buffered slab.')
    options.add_argument('-fr', '--free', type=str, nargs = '*', dest='free_atoms',  
                help='List of atomic species allowed to relax without constraint.',
                                                                    default=[])
                                    
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
                
    options = dict()
                
    return options

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
            
    # make sure that the atomic simulation code specified by the user is supported
    if args.prog.lower() in supported_codes:
        pass
    else:
        raise ValueError("{} is not a supported atomistic simulation code." +
                         "Supported codes are: GULP; QE; CASTEP.")

    # extract unit cell and GULP run parameters from file <cell_name>
    ab_initio = False   
    if 'gulp' == args.prog.lower():
        unit_cell = cry.Crystal()
        parse_fn = gulp.parse_gulp
    elif 'qe' == args.prog.lower():
        unit_cell = cry.Crystal()
        parse_fn = qe.parse_qe
        ab_initio = True
    elif 'castep' == args.prog.lower():
        unit_cell = castep.CastepCrystal()
        parse_fn = castep.parse_castep
        ab_initio = True
        
    sys_info = parse_fn(args.cell_name, unit_cell)

    # if the calculation uses an ab initio solver, scale the k-point grid
    if ab_initio:
        if 'qe' == args.prog.lower():
            atm.scale_kpoints(sys_info['cards']['K_POINTS'], 
                              np.array([1., 1., args.n]))
            qe.scale_nbands(sys_info['namelists']['&system'], np.array([1., 1., args.n]))
        elif 'castep' == args.prog.lower():
            atm.scale_kpoints(sys_info['mp_kgrid'], np.array([1., 1., args.n]))
    
    # shift origin of cell
    unit_cell.translate_cell(np.array([0., 0., -1*args.shift]), modulo=True)
    
    # select output mode appropriate for the atomistic calculator used, together
    # with the correct suffix for the input files (may be better to let the 
    # output functions determine the suffix?)
    if 'gulp' == args.prog.lower():
        write_fn = gulp.write_gulp
        suffix = 'gin'
        relax = None
    elif 'qe' == args.prog.lower():
        write_fn = qe.write_qe
        suffix = 'in'
        relax = 'relax'
    elif 'castep' == args.prog.lower():
        write_fn = castep.write_castep
        suffix = 'cell'
        relax = None
    
    # make the slab and construct the gamma surface/line
    new_slab = gsf.make_slab(unit_cell, args.n, args.vac, d_fix=args.d_fix, 
                                                 free_atoms=args.free_atoms)
    if args.simulation_type == 'gsurface':
        limits = (args.max_x, args.max_y)          
        gsf.gamma_surface(new_slab, args.res, write_fn, sys_info, suffix=suffix,
                          limits=limits, basename=args.sim_name, vacuum=args.vac,
                                                                    relax=relax)    
        
        # run the calculations, if an executable has been provided. Otherwise,
        # assume that that the input files will be transferred to another machine
        # and run by the user.
        if not (args.progexec is None):
            # extract increments
            N, M = gsf.gs_sampling(new_slab.getLattice(), args.res, limits)
            for n in range(0, N+1):
                for m in range(0, M+1):
                    print("Relaxing cell with generalized stacking fault vector" +
                            " ({}, {})...".format(n, m), end="")
                    basename = '{}.{}.{}'.format(args.sim_name, n, m)
                    if 'gulp' == args.prog.lower():
                        gulp.run_gulp(args.progexec, basename) 
                    elif 'qe' == args.prog.lower():
                        qe.run_qe(args.progexec, basename)
                    elif 'castep' == args.prog.lower():
                        castep.run_castep(args.progexec, basename)
                    
                    print("complete.") 
        else:
            pass
    elif args.simulation_type == 'gline':      
        gsf.gamma_line(new_slab, np.array(args.line_vec), args.res, write_fn,  
                           sys_info, suffix=suffix, limits=args.max_x,  
                           basename=args.sim_name, vacuum=args.vac, relax=relax)
        
        if not (args.progexec is None):
            # extract limits
            N = gsf.gl_sampling(new_slab.getLattice(), resolution=args.res, 
                            vector=np.array(args.line_vec), limits=args.max_x)
            # run calculations
            for n in range(0, N+1):
                print("Relaxing cell {}...".format(n), end="")
                basename = '{}.{}'.format(args.sim_name, n)
                if 'gulp' == args.prog.lower():
                    gulp.run_gulp(args.progexec, basename)
                elif 'qe' == args.prog.lower():
                    qe.run_qe(args.progexec, basename)
                elif 'castep' == args.prog.lower():
                    castep.run_castep(args.progexec, basename)

                print("complete.")
        else:
            pass
    else:
        pass          
    
if __name__ == "__main__":
    main()

    
    
