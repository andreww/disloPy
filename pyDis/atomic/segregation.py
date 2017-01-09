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
from scipy.optimize import curve_fit
try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
except ImportError:
    print("Module <matplotlib> not found. Do not use plotting functions.")

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
            x = float(site_line[1])
            y = float(site_line[2])
            
            # calculate distance of site from dislocation line, and its azimuthal
            # location
            r = np.sqrt(x**2+y**2)
            phi = np.arctan2(y, x)
            
            site_info.append([i, x, y, r, phi])
            
    return np.array(site_info)
    
def get_energies(basename, site_info, program='gulp', suffix='gout'):
    '''Extracts energies of defected simulation cells from GULP output files.
    '''
    
    # extract energy from GULP output files corresponding to each index in 
    # <site_info>
    energies = []    
    for site in site_info:
        simname = '{}.{}.{}'.format(basename, int(site[0]), suffix)
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
    
def fit_seg_energy(e_seg, sites, min_r=2.):
    '''Fits the calculated segregation energy to a function of the form 
    A0*sin(theta)/r + A1/r**2 (see de Batist p. 29-30), with the first term
    representing between interaction a spherical defect and the dislocation,
    and the second an interaction energy arising from the inhomogeneity effect.
    '''
    
    def energy_function(x, A0, A1):
        r = np.sqrt(x[0]**2+x[1]**2)
        th = np.arctan2(x[1], x[0])
        return A0*np.sin(th)/r + A1/r**2
    
    # determine indices of sites that are not in the immediate vicinity of the 
    # dislocation core (where inelastic interactions dominate)
    i = np.where(sites[:, 3] > min_r)[0]
    e_seg = np.array(e_seg)
    # fit point defect-dislocation binding energy    
    par, pcov = curve_fit(energy_function, (sites[:, 1][i], sites[:, 2][i]), e_seg[i])
    perr = np.sqrt(np.diag(pcov))
    return par, perr
    
def reflect_atoms(site_info, e_excess, e_seg, axis, tol=1.):
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
        x = site_i[1:3]
        if abs(x[(axis+1) % 1]) < tol:
            continue
            
        # otherwise, reflect atom about axis
        if axis == 0:
            new_x = [x[0], -x[1]]
        else:
            new_x = [-x[0], x[1]]
            
        phi_new = np.arctan2(new_x[1], new_x[0])            
        new_site = [site_i[0], new_x[0], new_x[1], site_i[3], phi_new]
        
        sites_full.append(new_site)
        e_excess_full.append(Ei)
        e_seg_full.append(dEi)
    
    return np.array(sites_full), e_excess_full, e_seg_full
    
def write_energies(outname, site_info, e_excess, e_seg, pars=None):
    '''Writes the excess and segregation energies for defects at each site to
    the specified output file.
    ''' 
    
    outfile = open(outname, 'w')
    outfile.write('# site-id x y r phi E_excess E_segregation\n')
    if pars is not None:
        outfile.write(pars)
    
    for i, site in enumerate(site_info):
        line_fmt = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'
        outfile.write(line_fmt.format(int(site[0]), site[1], site[2], site[3], 
                                               site[4], e_excess[i], e_seg[i]))   

    outfile.close()
    
### PLOTTING FUNCTIONS

def plot_energies_contour(sites, e_seg, figname, cmtype='coolwarm', refine=T,
                                       units='eV', figformat='tif', levels=25):
    '''Produces a contour plot of the segregation energy at sites around a 
    dislocation. Use <levels> to control the number of contours.
    '''
    
    # create contour plot of segregation energies using a Delauney triangulation
    # scheme
    x = sites[:, 1]
    y = sites[:, 2]
    triang = tri.Triangulation(x, y)
    
    fig = plt.figure()
    plt.gca().set_aspect('equal')
    
    # plot energy contours
    if refine:
        # refine data for improved high-res plot
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, es_refi = refiner.refine_field(e_seg, subdiv=3)
        plt.tricontourf(tri_refi, es_refi, levels, cmap=plt.get_cmap(cmtype))
    else:
        # use raw data to produce contours
        plt.tricontourf(triang, e_seg, levels, cmap=plt.get_cmap(cmtype))
        
    plt.xlabel('x ($\AA$)', size='x-large', family='serif')
    plt.ylabel('y ($\AA$)', size='x-large', family='serif')
    
    cb = plt.colorbar()
    cb.set_label('E ({})'.format(units), size='x-large', family='serif')

    # add points to mark the locations of the atomic sites
    plt.scatter(x, y, c='k', s=40)
        
    plt.xlim(x.min()-2, x.max()+2)
    plt.ylim(y.min()-2, y.max()+2)
    
    plt.savefig('{}.{}'.format(figname, figformat))
    plt.close()    
    
    return
    
def plot_energies_scatter(sites, e_seg, figname, cmtype='coolwarm', units='eV',
                                                             figformat='tif'):
    '''Produces a scatterplot showing all of the sites for which segregation 
    energies were calculated, with each point coloured according to the 
    segregation energy at that site.
    '''

    x = sites[:, 1]
    y = sites[:, 2]

    fig = plt.figure()
    plt.gca().set_aspect('equal')
    plt.scatter(x, y, c=e_seg, cmap=plt.get_cmap(cmtype), s=100)
    
    plt.xlim(x.min()-2, x.max()+2)
    plt.ylim(y.min()-2, y.max()+2)
    plt.xlabel('x ($\AA$)', size='x-large', family='serif')
    plt.ylabel('y ($\AA$)', size='x-large', family='serif')
    
    cb = plt.colorbar()
    cb.set_label('E ({})'.format(units), size='x-large', family='serif')
    plt.savefig('{}.{}'.format(figname, figformat))
    plt.close()
    
    return


### END PLOTTING FUNCTIONS

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
    options.add_argument('-ps', type=to_bool, default='False', dest='plot_scatter',
                         help='Create scatter plot of segregation energies.')
    options.add_argument('-pc', type=to_bool, default='False', dest='plot_contour', 
                         help='Create contour plot of segregation energies')
    options.add_argument('-pn', default=None, dest='plotname', help='Name to be '+
                         'used for all figures produced.')
    options.add_argument('-f', default='tif', dest='figformat', help='Image format.')
    options.add_argument('-fit', type=to_bool, default='False', dest='fit',
                         help='Fit the form of the calculated segregation energies.')
                         
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
        
    # fit segregation energies to obtain homogeneous and inhomogeneous contributions
    if args.fit:
        par, perr = fit_seg_energy(e_seg, site_info)
        fit_str = '# A0 = {:.4f} +/- {:.4f}; A1 = {:.4f} +/- {:.4f}\n'.format(par[0],
                                                        perr[0], par[1], perr[1])
    else:
        fit_str = None 
    
    # write to output file
    outname = '{}.energies.dat'.format(args.basename)
    write_energies(outname, site_info, e_excess, e_seg, pars=fit_str)
    
    # create plots showing segregation energies
    if args.plotname:
        plotname = args.plotname
    else:
        plotname = args.basename
        
    if args.plot_scatter:
        plot_energies_scatter(site_info, e_seg, plotname+'.scatter', 
                                            figformat=args.figformat)
    if args.plot_contour:
        plot_energies_contour(site_info, e_seg, plotname+'.contour', 
                                            figformat=args.figformat)

if __name__ == "__main__":
    main()    
