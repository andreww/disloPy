#!/usr/bin/env python
''' Given a numerical gamma surface (eg. calculated using DFT or lattice 
statics), performs a Fourier series decomposition to provide a continuous,
low cost, gsf energy function for use esp. in Peierls-Nabarro modelling
of dislocation misfit profiles.
'''

import numpy as np
import scipy
import re
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline, interp1d

from pyDis.atomic import atomistic_utils as atm

def read_numerical_gsf(filename):
    '''Reads in a grid of energy values for a gamma line or gamma surface,
    including the units in which those energies are expressed.
    '''
    
    gsf = []
    units = None
    units_line = re.compile('#\s+units\s+(?P<units>\w+)')

    with open(filename) as gsf_file:
        for line in gsf_file:
            unit_match = units_line.match(line)
            if unit_match:
                units = unit_match.group('units')
            data = line.rstrip().split()
            if not(data): # empty line
                continue
            # otherwise, format data
            gsf.append([float(value) for value in data])
            
    # check that units were contained in file
    if unit_match == None:
        print("Warning: no energy units specified. Defaulting to eV")
        units = 'ev'
            
    return np.array(gsf), units
    
def mirror1d(gline):
    '''Reflects a 1D gamma line about 0.5*a.
    '''
    
    n = len(gline)
    newlength = 2*n-1
    new_gl = np.zeros((newlength, 2))
    
    for i in range(n):
        for j in (0, 1):
            new_gl[i, j] = gline[i, j]
            new_gl[newlength-1-i] = (newlength-1-i, gline[i])[j]
            
    return new_gl

def mirror2d(gsurf, axis=(0, 1)):
    '''Reflects <gsf> about the provided symmetry <axis>. <axis> can 
    take the values 0 or 1. Can also provide axis = (0, 1) to mirror 
    about the x and y axes.
    '''

    if atm.isiter(axis):
        temp_gs = mirror2d(gsurf, axis[0])
        new_gs = mirror2d(temp_gs, axis[1])
        return new_gs
    # else
    if axis != 0 and axis != 1:
        raise ValueError("Invalid axis. Are you a Fortran programmer?")
    grid_shape = list(np.shape(gsurf))
    nx = grid_shape[axis]
    grid_shape[axis] = 2*nx-1
    new_gs = np.zeros(grid_shape)
    for i in range(np.shape(gsurf)[0]):
        for j in range(np.shape(gsurf)[1]):
            for k in range(3):
                new_gs[i, j, k] = gsurf[i, j, k]
            
            if axis == 0:
                if i == nx-1:
                    pass
                else:
                    new_gs[2*nx-2-i, j, 0] = 2*nx-2-i
                    new_gs[2*nx-2-i, j, 1] = j
                    new_gs[2*nx-2-i, j, 2] = gsurf[i, j, 2]
            else: # axis == 1
                if j == (nx - 1):
                    pass
                else:
                    new_gs[i, 2*nx-2-j, 0] = i
                    new_gs[i, 2*nx-2-j, 1] = 2*nx-2-j
                    new_gs[i, 2*nx-2-j, 2] = gsurf[i, j, k]

    return new_gs
    
def spline_fit1d(num_gsf, a, b, angle=np.pi/2., two_planes=True, units='ev'):
    '''Fits a bivariate spline to a numerical gsf energies along a line (ie.
    fits a gamma line).
    '''

    # extract gsf vectors and energies from <num_gsf>
    x_vals = num_gsf[:, 0]
    E_vals = num_gsf[:, 1]
    
    # convert energies to eV/\AA^2
    if units.lower() == 'ev':
        pass
    elif units.lower() in 'rydberg':
        E_vals *= 13.6057
    E_vals -= E_vals.min()
    E_vals /= a*b*abs(np.sin(angle))
    if two_planes:
        E_vals /= 2.
    
    # fit gamma line and create periodic function
    g_spline = interp1d(x_vals, E_vals, kind='cubic')

    def gamma(x):        
        return g_spline(x % a)  
          
    return gamma

def spline_fit2d(num_gsf, a, b, angle=np.pi/2., two_planes=True):
    '''extract coordinates of calculations, and gs-energies at each point
    grid values are given in integer values -> need to convert to \AA
    '''
    
    #!!! Need to change function calls in other modules to accommodate 1D 
    #!!! and 2D spline fits.
    x_vals = get_axis(num_gsf, 0, a)
    y_vals = get_axis(num_gsf, 1, b)
   
    # import energies and convert to eV/\AA^{2}
    E_vals = num_gsf[:, 2].reshape((len(x_vals), len(y_vals)))
    E_vals -= E_vals.min()
    E_vals /= a*b*abs(np.sin(angle))
    if two_planes:
        # two slip planes present in simulation cell -> set <two_planes> to False
        # if a vacuum layer is used.
        E_vals *= 0.5
    
    g_spline = RectBivariateSpline(x_vals, y_vals, E_vals)
    
    if int(scipy.__version__.split(".")[1]) < 14:
        def gamma(x, y):        
            return g_spline.ev(x % a, y % b)
    else:
        def gamma(x, y):
            return g_spline(x % a, y % b, grid=False)
               
    return gamma

def get_axis(gsf_array, index, length):
    values = set(gsf_array[:, index])
    values = np.array([x for x in values])
    values *= length/float(values.max())
    return values
    
# ROUTINES TO TRANSFORM GAMMA SURFACE INTO COORDINATES WITH \xsi || y
    
def remap_input(f_orig, x_new, y_new):
    '''Remaps input to gamma surface <f_orig> to conform with the coordinate
    system used in PN modelling. For example, suppose we have computed a gamma 
    surface defined by vectors a1 = <100> and a2 = <010>, but have a burgers 
    vector b = 1/2<110>. If ux corresponds to displacement || to b and uy is 
    displacement perpendicular to b, then x = (ux + uy)/sqrt(2) and
    y = (ux - uy)/sqrt(2).
 
    To generate a gamma surface function that returns the correct energy for
    any given (ux, uy), we have
    
    new_gsf = remap_input(old_gsf, lambda x, y: (x+y)/sqrt(2),
                                    lambda x, y: (x-y)/sqrt())
                                    
    E = new_gsf(ux, uy) # gives gsf energy at ux, uy
    '''
    
    f_new = lambda x, y: f_orig(x_new(x, y), y_new(x, y))
    return f_new

def create_lambda(in_str):
    '''From a provided input string matching the specified format, constructs
    a lambda expression mapping coordinate systems to one another.
    '''
    
    # extract the input function form
    map_format = re.compile('\s*function\s*:\s*\(' +
            '(?P<x1>\w.*),\s*(?P<x2>\w.*)\)\s*-\>\s*(?P<func>.*)\s*')
    matched_form = re.search(map_format, in_str)

    x1 = matched_form.group('x1')
    x2 = matched_form.group('x2')
    func = matched_form.group('func')
    remapped_coord = eval('lambda {}, {}: {}'.format(x1, x2, func))
    return remapped_coord
    
def new_gsf(gsf_calc, x_form, y_form):

    x_new = create_lambda(x_form)
    y_new = create_lambda(y_form)
    
    transformed_gsf = remap_input(gsf_calc, x_new, y_new)
    
    return transformed_gsf
    
def projection(gsf_func, const=0, axis=0):
    '''Extracts a specific gamma line from a calculated gamma surface. 
    <axis> specifies the direction of the gamma line, while <const> gives 
    the intersection of the gamma line with the constant axis. As the most common
    usage of this will be to calculate displacement fields for pure edge (or 
    screw) dislocations, <const> defaults to 0.
    '''
    
    # check that the specified axis label is valid
    if axis == 0 or axis == 1:
        pass
    else:
        raise ValueError("{} is an invalid axis label.".format(axis))
    
    if axis == 0:
        g = lambda x: gsf_func(x, const)
    elif axis == 1: 
        g = lambda x: gsf_func(const, x)
    else:
        raise ValueError("{} is an invalid axis to project onto.".format(axis))
    
    return g
    
# Plotting functions

def gamma_surface3d(X, Y, Z, xlabel, ylabel, zlabel, size):
    '''Plots a gamma surface in 3D.
    '''
    
    fig = plt.figure(figsize=size)
    ax = fig.gca(projection='3d')
    surf  = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.3, shade=True, alpha=1)
    fig.colorbar(surf, shrink=0.7, aspect=12)
    
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel(xlabel, family='serif', size=12)
    ax.set_ylabel(ylabel, family='serif', size=12)
    ax.set_zlabel(zlabel, family='serif', size=12)
       
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=20., azim=-15)
    
    plt.tight_layout(pad=0.5)
    plt.show()
    return fig, ax
    
def contour_plot(X, Y, Z, xlabel, ylabel, size):
    '''Two dimensional contour plot of gamma surface height.
    '''
    
    fig = plt.figure(figsize=size)
    ax = plt.subplot()
    cont = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("afmhot"))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(cont)
    return 
