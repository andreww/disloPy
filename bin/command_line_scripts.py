#!/usr/bin/env python
'''Contains functions allowing Atomistic, Segregation, and Peierls-Nabarro
simulations to be run from the command line.
'''
from __future__ import absolute_import

import argparse
import sys

from pydis.atomic import _atomic_control, _segregation_control
from pydis.pn import _pn_control

def main_generic(simtype):
    '''Runs a simulation of the specified kind.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, nargs='?', dest='filename', default='0')
    
    args = parser.parse_args()
    
    if args.filename != '0':
        filename = args.filename
    else:
        # read in filename from the command line
        if sys.version_info.major == 2:
            filename = raw_input('Enter name of input file: ')
        elif sys.version_info.major == 3:
            filename = input('Enter name of input file: ')
            
    # run specified simulation type
    if simtype == 'atomistic':
        _atomic_control.AtomisticSim(filename)
    elif simtype == 'pn':
        _pn_control.PNSim(filename)
    elif simtype == 'segregation':
        _segregation_control.SegregationSim(filename)
    else:
        raise ValueError("{} is not a valid simulation in pydis.".format(simtype))
            
def main_atomistic():
    '''Runs an Atomistic simulation.
    '''

    main_generic('atomistic')
    
def main_pn():
    '''Runs a Peierls-Nabarro simulation.
    '''

    main_generic('pn')
    
def main_segregation():
    '''Runs a segregation calculation.
    '''

    main_generic('segregation')
