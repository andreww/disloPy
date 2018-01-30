#!/usr/bin/env python
''' Given a numerical gamma surface (eg. calculated using DFT or lattice 
statics), performs a Fourier series decomposition to provide a continuous,
low cost, gsf energy function for use esp. in Peierls-Nabarro modelling
of dislocation misfit profiles.
'''
from __future__ import print_function, absolute_import

import numpy as np
import scipy
import re
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import RectBivariateSpline, interp1d, interp2d

try:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
except ImportError:
    print("Module <matplotlib> not found. Do not use plotting functions.")

from pydis.utilities import atomistic_utils as atm
from pydis.pn import fourier

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
                continue
                
            data = line.rstrip().split()
            if not(data): # empty line
                continue
            # otherwise, format data
            gsf.append([float(value) for value in data])
            
    # check that units were contained in file
    if units is None:
        print("Warning: no energy units specified. Defaulting to eV")
        units = 'ev'
            
    return np.array(gsf), units
   
def spline_fit1d(num_gsf, a, b, angle=np.pi/2., hasvac=False, units='ev',
                                         do_fourier_fit=False, n_order=2):
    '''Fits a bivariate spline to a numerical gsf energies along a line (ie.
    fits a gamma line). <a> is the length of the cell axis along the gamma line
    (typically the Burgers vector), and <b> is the length of the third axis 
    (after <a> and the normal vector).
    '''
    '''
    # extract gsf vectors and energies from <num_gsf>
    x_vals = num_gsf[:, 0]
    x_vals = b*x_vals/x_vals.max()
    E_vals = num_gsf[:, 1]
    '''
    
    # extract gsf vectors and energies from <num_gsf>
    x_vals = get_axis(num_gsf, 0, a)
    E_vals = num_gsf[:, 1]
    
    # convert energies to eV/\AA^2
    if units.lower() == 'ev':
        pass
    elif units.lower() in 'rydberg':
        E_vals *= 13.6057
           
    E_vals -= E_vals.min()
    E_vals /= a*b*abs(np.sin(angle))
    if hasvac:
        pass
    else:
        # 3D periodic -> cell contains two stacking faults
        E_vals /= 2.
    
    # fit gamma line and create periodic function
    g_spline = interp1d(x_vals, E_vals, kind='cubic')

    def gamma(x):        
        return g_spline(x % a)  
        
    # fit a fourier series, if specified by the user
    if do_fourier_fit:
        gamma_n = fourier.fourier_series1d(gamma, n_order, a)#gline_fourier(gamma, n_order, a)
    else:
        gamma_n = gamma
          
    return gamma_n

def spline_fit2d(num_gsf, a, b, angle=np.pi/2., hasvac=False, units='ev',
                                         do_fourier_fit=False, n_order=2,
                                                               m_order=2):
    '''extract coordinates of calculations, and gs-energies at each point
    grid values are given in integer values -> need to convert to \AA
    '''
    
    # extract values of the intervals along the x and y axes
    x_vals = get_axis(num_gsf, 0, a)
    y_vals = get_axis(num_gsf, 1, b)
    
    # import energies and convert to eV/\AA^{2}
    E_vals = num_gsf[:, 2].reshape((len(x_vals), len(y_vals)))
    E_vals -= E_vals.min()
    E_vals /= a*b*abs(np.sin(angle))
    if units.lower() == 'ev':
        pass
    elif units.lower() in 'rydberg':
        E_vals *= 13.6057
        
    if hasvac:
        pass
    else:
        # 3D periodic -> cell contains two stacking faults
        E_vals /= 2.
    
    g_spline = RectBivariateSpline(x_vals, y_vals, E_vals)#, kx=1, ky=1)
    
    if int(scipy.__version__.split(".")[1]) < 14:
        def gamma(x, y):        
            return g_spline.ev(x % a, y % b)
    else:
        def gamma(x, y):
            return g_spline(x % a, y % b, grid=False)
    
    # fit a fourier series, if specified by the user
    if do_fourier_fit:
        gamma_n = fourier.fourier_series2d(gamma, n_order, m_order, a, b)#gsurf_fourier(gamma, n_order, m_order, a, b)
    else:
        gamma_n = gamma
                  
    return gamma_n

def get_axis(gsf_array, index, length):
    '''Returns the energy of the generalised stacking faults along the specified
    axis.
    '''
    
    values = set(gsf_array[:, index])
    values = np.array([x for x in values])
    values *= length/float(values.max())
    return values
    
def gsurf_fourier(gsurf_func, n_order, m_order, T1, T2):
    '''Fits a fourier series of order <n_order> to <gsurf_func>, the gamma
    surface energy function, which is periodic in x and y (with periods <T1> and
    <T2>, respectively).
    '''
    
    fourier_func = fourier.fourier_series2d(gsurf_func, n_order, m_order, T1, T2)
    
    return fourier_func
    
def gline_fourier(gline_func, n_order, T):
    '''Fits a fourier series of order <n_order> to the gamma line energy function
    <gline_func>, which is periodic with period T.
    '''
    
    fourier_func = fourier.fourier_series1d(gline_func, n_order, T)

    return fourier_func
    
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
    '''Remaps <gsf_calc> so that the x and y directions correspond to those 
    given in <x_form> and <y_form>.
    '''

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
    
    if axis == 0:
        g = lambda x: gsf_func(x, const)
    elif axis == 1:
        g = lambda x: gsf_func(const, x)  
    else:
        raise ValueError("{} is an invalid axis label.".format(axis))       
    
    return g
    
### Plotting functions ###

def gamma_line(X, E, xlabel, ylabel, size):
    '''Plots a gamma line.
    '''
    
    fig = plt.figure(figsize=size)
    ax = plt.subplot()
    
    ax.plot(X, E, 'r--')
    ax.plot(X, E, 'rs', markersize=6)
    
    ax.set_xlim(X.min(), X.max())
    
    pad = (E.max() - E.min())/20.
    ax.set_ylim(E.min()-pad, E.max()+pad)
    
    ax.set_xlabel(xlabel, family='serif', weight='bold', size=14)
    ax.set_ylabel(ylabel, family='serif', weight='bold', size=14)
    
    plt.tight_layout(pad=0.5)
    plt.show()

    return fig, ax

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
    
def plot_gamma_surface(gsf_file, a, b, plotname='gamma_surface.tif', 
                                             angle=np.pi/2., hasvac=False):
    
    # read in gridded data and units                                                
    gsf_grid, units = read_numerical_gsf(gsf_file)
    
    gsf_func = spline_fit2d(gsf_grid, a, b, angle=angle, hasvac=hasvac)
    
    #!!! Should make the number of increments adaptive; still this probably
    #!!! provides sufficient sampling density for display purposes
    x = np.linspace(0., a, 100)
    y = np.linspace(0., b, 100)
    
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(np.shape(X))
    
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i, j] = gsf_func(X[i, j], Y[i, j])
            
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    #cset = ax.contourf(X, Y, Z, 50, zdir='z', cmap=cm.coolwarm, offset=-0.1)
    
    ax.plot_surface(X, Y, Z, cstride=2, rstride=2, cmap=cm.coolwarm)
    ax.set_zlim(Z.min(), Z.max()+0.05*(Z.max()-Z.min()))
    ax.set_xlim(0., a)
    ax.set_ylim(0., b)
    plt.show()
    
    return Z
