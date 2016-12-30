#!/usr/bin/env python
'''Scripts to process and plot the output from point defect-dislocation 
calculations. At present, only the lattice statics program GULP is supported,
but support for LAMMPS is planned as a future development.
'''
from __future__ import print_function

import re
import argparse
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

import numpy as np

from pyDis.atomic.atomistic_utils import extract_energy, to_bool

def parse_control(basename):
    '''Parses the control file to extract a list of sites together with their
    position relative to the dislocation line.
    '''
    
    # open control file and extract indices, etc.
    site_info = []
    with open('{}.id.txt'.format(basename)) as f:
        for line in f:
            if line.startswith('#'):
                # comment line -> skip
                continue
            #else
            site_line = line.rstrip().split()
            # index of site
            i = int(site_line[0])
            # position of site relative to dislocation line. ignores z-coordinate
            x = np.array([float(x) for x in site_line[1:3]])
            
            # calculate distance of site from dislocation line, and its azimuthal
            # location
            r = np.linalg.norm(x)
            phi = np.arctan2(x[1], x[0])
            
            site_info.append([i, x, r, phi])
            
    return site_info
    
def get_energies(basename, site_info, program='gulp', suffix='gout'):
    '''Extracts energies of defected simulation cells from GULP output files.
    '''
    
    # extract energy from GULP output files corresponding to each index in 
    # <site_info>
    energies = []    
    for site in site_info:
        simname = '{}.{}.{}'.format(basename, site[0], suffix)
        E, units = extract_energy(simname, program)
        energies.append(E)
        
    return energies
    
def defect_excess_energy(defect_energies, E0, n):
    '''Calculates the excess energy of the defected simulation cells relative
    to a point-defect free cluster with energy E0. <n> refers to the height
    of the point-defect bearing cluster relative to the base cluster.
    '''
    
    e_excess = []
    for E in defect_energies:
        de = E - n*E0
        e_excess.append(de)
        
    return e_excess
    
def segregation_energy(excess_energy, e_bulk):
    '''Calculates the energy difference between a point defect in the bulk and
    one that has been segregated to a dislocation core.
    '''
    
    segregation_energies = []
    for e_disloc in excess_energy:
        e_seg = e_disloc - e_bulk
        segregation_energies.append(e_seg)
    
    return segregation_energies 
    
def reflect_atoms(site_info, e_excess, e_seg, axis, tol=1e-1):
    '''Reflects all atoms about the specified <axis> (which must take the values
    0 or 1), excepting only those that are within <tol> of the mirror axis.
    '''
    
    if axis != 0 and axis != 1:
        raise ValueError("Axis must be either 0 (x) or 1 (y)")
     
    # list to hold info for full region
    sites_full = []
    e_excess_full = []
    e_seg_full = [] 
       
    for site_i, Ei, dEi in zip(site_info, e_excess, e_seg):
        # add atom to the full region
        sites_full.append(site_i)
        e_excess_full.append(Ei)
        e_seg_full.append(dEi)
        
        # check to see if atom is on or near the mirror plane
        x = site_i[1]
        if abs(x[(axis+1) % 1]) < tol:
            continue
            
        # otherwise, reflect atom about axis
        if axis == 0:
            new_x = [x[0], -x[1]]
        else:
            new_x = [-x[0], x[1]]
            
        phi_new = np.arctan2(new_x[1], new_x[0])            
        new_site = [site_i[0], new_x, site_i[2], phi_new]
        
        sites_full.append(new_site)
        e_excess_full.append(Ei)
        e_seg_full.append(dEi)
    
    return sites_full, e_excess_full, e_seg_full
    
def write_energies(outname, site_info, e_excess, e_seg):
    '''Writes the excess and segregation energies for defects at each site to
    the specified output file.
    ''' 
    
    outfile = open(outname, 'w')
    outfile.write('# site-id x y r phi E_excess E_segregation\n')
    
    for i, site in enumerate(site_info):
        line_fmt = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'
        outfile.write(line_fmt.format(site[0], site[1][0], site[1][1], site[2],
                                site[3], e_excess[i], e_seg[i]))   

    outfile.close()

def command_line_options():
    '''Options to control parsing of the output from point-defect dislocation 
    interaction simulations.
    '''
    
    options = argparse.ArgumentParser()
    options.add_argument('-b', type=str, dest='basename', help='Base name of ' +
                         'simulation files.')
    options.add_argument('-e0', type=float, dest='E0', help='Energy of cell ' +
                         'containing only an isolated dislocation')
    options.add_argument('-de', type=float, dest='dE0', help='Point defect energy ' +
                         'in the bulk crystal.')
    options.add_argument('-n', type=int, dest='n', help='Height of point-defect ' +
                         'bearing cells, in units of the dislocation line vector length.')
    options.add_argument('-m', type=to_bool, default='False', dest='mirror',
                         help='Tells the program to reflect the atoms about an axis')
    options.add_argument('-mboth', type=to_bool, default='False', dest='mirror_both',
                         help='Reflect atoms about the x and y axes.')
    options.add_argument('-ax', type=int, default=0, dest='axis', 
                         help='Axis about which to reflect atoms')
                         
    return options                     

def main():
    
    options = command_line_options()
    args = options.parse_args()
    
    # read in control file
    site_info = parse_control(args.basename)
    
    # calculate excess and segregation energies
    e_calc = get_energies(args.basename, site_info)
    e_excess = defect_excess_energy(e_calc, args.E0, args.n)
    e_seg = segregation_energy(e_excess, args.dE0)
    
    # reflect atoms about specified axis. Do so twice if atoms are to be reflected
    # about both x and y
    if (args.mirror and args.axis == 0) or args.mirror_both:
        site_info, e_excess, e_seg = reflect_atoms(site_info, e_excess, e_seg, 0)
    if (args.mirror and args.axis == 1) or args.mirror_both: 
        site_info, e_excess, e_seg = reflect_atoms(site_info, e_excess, e_seg, 1)
    
    # write to output file
    outname = '{}.energies.dat'.format(args.basename)
    write_energies(outname, site_info, e_excess, e_seg)

if __name__ == "__main__":
    main()    
