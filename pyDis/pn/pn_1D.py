#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import numpy.random as rand
from numpy.linalg import norm

from scipy.optimize import fmin_slsqp, curve_fit
import matplotlib.pyplot as plt

# suppress divide by zero Runtime warnings
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

def simple_gamma(u):
    return 0.5*(1-np.cos(2*np.pi*u))

def test_gamma(u):
    a = 0.5*(1-np.cos(2*np.pi*u))
    return 1/0.14*(a-0.5*(np.exp(1.09*a)-1))

def generate_A(N):
    A = rand.uniform(size=N)
    A /= A.sum()
    return A
    
def generate_x(N, A, spacing, sigma=1.):
    '''Generates midpoints for N dislocations. If N == 1, we simply locate the
    dislocation at the origin.
    '''
    if N == 1:
        return np.zeros(1)
    else:
        x = rand.normal(0., sigma, N)
        x_mean = (A*x).sum()
        return (x-(x_mean-(x_mean % spacing)))
    
def generate_c(N):
    c = rand.lognormal(0, 1, N)
    return c
        
def gen_inparams(n_funcs, spacing):
    A = generate_A(n_funcs)
    x0 = generate_x(n_funcs, A, spacing)
    c = generate_c(n_funcs)
    input_parameters = list(A)+list(x0)+list(c)
    return input_parameters

def gen_symmetric(n_funcs, spacing, scale=1.):
    '''Generates <n_funcs> partial dislocations that are symmetric about the 
    origin. <scale> exists so that the A parameters can be scaled when <n_funcs>
    is odd and > 1 (so that sum(A) == 1).
    '''

    if n_funcs == 1:
        return gen_inparams(n_funcs, spacing)
    elif n_funcs % 2 == 0: # ie. even
        sym_funcs = n_funcs/2
        
        # generate a starting set of asymmetric parameters
        asymm_pars = gen_inparams(sym_funcs, spacing)
        A = [a/2.*scale for a in asymm_pars[:sym_funcs]]
        x0 = [x for x in asymm_pars[sym_funcs:2*sym_funcs]]
        c = [w for w in asymm_pars[2*sym_funcs:]]
        
        # symmetrise parameters
        A, x0, c = symmetrise(A, x0, c, spacing)
    else: # n_funcs % 2 == 1 and n_funcs > 1
        # generate the central dislocation
        A1 = [rand.uniform(0.1, 0.9)]
        x01 = [0.]
        c1 = list(generate_c(1))
        
        # generate a symmetric distribution of dislocation about the central
        # dislocation, scale so that the integrated Burgers vector density is
        # correct.
        sympar = gen_symmetric(n_funcs-1, spacing, scale=1-A1[0])        
        A2 = sympar[:n_funcs-1]
        x02 = sympar[n_funcs-1:2*(n_funcs-1)]
        c2 = sympar[2*(n_funcs-1):]
        
        # combine the central and flanking dislocations
        A = A1 + A2
        x0 = x01 + x02
        c = c1 + c2
    
    return (A+x0+c)
        
def symmetrise(A, x0, c, spacing, normalize=False):
    '''Convert the asymmetric input parameters into a set of dislocations with 
    a total dislocation density distribution that is symmetric about the origin.
    '''
    
    A = A + A
    c = c + c
    x0 = x0 + [-x for x in x0]
    
    # normalize A, so that sum(A) = 1
    if normalize:
        A = [a/sum(A) for a in A]
        
    # shift all elements of x0 so that <x> != 0.
    shift = rand.uniform()
    x0 = [(x+shift) for x in x0]
    x_mean = sum([ai*xi for ai, xi in zip(A, x0)])
    x0 = [(x-(x_mean-(x_mean % spacing))) for x in x0]
    return A, x0, c
    
def generate_input(N, spacing, use_sym=False):
    '''Generate a trial set of parameters {Ai; xi; ci} for a 1D dislocation
    distribution.
    '''

def u_field(x, A, x0, c, b, bc=0.5):
    u = 0.
    for Ai, xi, ci in zip(A, x0, c):
        u += 1/np.pi*(Ai*np.arctan((x-xi)/ci))
    return b*(u - bc)
    
def get_u1d(parameters, b, spacing, N, bc=0.5):
    '''Calculate displacement field from a list of fit parameters
    '''
    
    n_funcs = len(parameters)/3
    r = spacing*np.arange(-N, N)
    
    A = parameters[:n_funcs]
    x0 = parameters[n_funcs:2*n_funcs]
    c = parameters[2*n_funcs:]
    
    u = u_field(r, A, x0, c, b, bc)
    
    return u
    
def rho(u, r):
    rho_vals = (u[1:] - u[:-1])/(r[1:]-r[:-1])
    return rho_vals
    
def center_of_mass(rho, x, b):
    return 1/b*(rho*(x[1:]**2 - x[:-1]**2)).sum()
    
def rho_form(x, a, xsi, x0):
    return a*xsi**2/((x-x0)**2+xsi**2)
    
def dislocation_width(rho_vals, r):
    '''Note: at present, works only when rhox has a single, well defined
    maximum value. Will implement analysis function for dislocations composed 
    of multiple partials later.
    '''
    
    fit_par, cov = curve_fit(rho_form, r[1:], rho_vals)
    xsi = fit_par[1]
    return xsi 
    
def max_rho(rho, spacing):
    return rho.max()/spacing
    
### PLOTTING FUNCTIONS ###
    
def plot_rho(ax, rho, x, colour='b', width=1., a_val=0.40):
    '''Width should usually be equal to the spacing between atomic planes.
    '''
    
    #rho_disc = ax.bar(x[:-1], rho, width, color=colour, align='center', alpha=a_val,  
    #                                         edgecolor=colour, label=r'$\rho(r)$')
    rho_disc = ax.plot(x[:-1], rho, '{}D'.format(colour), label=r'$\rho(r)$')
    ax.plot(x[:-1], rho, '{}{}'.format(colour, '-.'))
    return rho_disc
    
def plot_u(ax, u, x, colour='r', shape='s', linestyle='-.'):
    u_disc = ax.plot(x, u, '{}{}'.format(colour, shape), label='$u(r)$')
    ax.plot(x, u, '{}{}'.format(colour, linestyle))
    return u_disc 
    
def plot_both(u, x, b, spacing, rho_col='b', u_col='r', along_b=True, nplanes=30):
    fig, ax = plt.subplots()

    if along_b:
        u_norm = (u + b)/b # this is the main dislocation
    else:
        u_norm /= b # edge (screw) component of a screw (edge) dislocation
        
    u_disc = plot_u(ax, u_norm, x, colour=u_col)
    rho_vals = rho(u, x)
    
    # normalise <rho_vals> by the amount required so that a dislocation density
    # distribution with non-zero values at only a single point has a value (at 
    # that point) of 1. ie. divide by ||b||/spacing, since ||b|| = height*spacing
    rho_vals /= b/spacing
    
    mid_dist = (x[1]-x[0])/2.
    
    rho_disc = plot_rho(ax, rho_vals, x+mid_dist, colour=rho_col, width=spacing)
    plt.xlim(-1*nplanes*spacing, nplanes*spacing)
    plt.ylim(0, 1.05)
    plt.tick_params(axis='y', which='both', left='off', right='off',
                                 labelleft='off', labelright='off')
    plt.xlabel('r ($\AA$)', family='serif', size=18)
    plt.tight_layout()
    plt.legend(numpoints=1, loc='upper left', fontsize=18, frameon=False)
    return fig, ax
    
### ENERGY TERMS ###
    
def elastic_energy(A, x0, c, b=1, K=1):
    E = 0.
    for Ai, xi, ci in zip(A, x0, c):
        for Aj, xj, cj in zip(A, x0, c):
            E += Ai*Aj*np.log((xi-xj)**2+(ci+cj)**2)
    return -0.5*K*E*b**2
    
def misfit_energy(A, x0, c, N, energy_function, b, spacing, shift=0.):

    r = spacing*(np.arange(-N, N))
    u = u_field(r, A, x0, c, b)
    Em = energy_function(u).sum()*spacing
    return Em
    
def total_energy(A, x0, c, N, energy_function, b, spacing, K=1):
    '''Return the total elastic and inelastic (ie. misfit) energy of the 
    dislocation.
    '''
    
    Em = misfit_energy(A, x0, c, N, energy_function, b, spacing)
    E_elast = elastic_energy(A, x0, c, b, K)
    return Em + E_elast
    
def total_optimizable(params, *args):
    n_funcs = args[0]
    N = args[1]
    energy_function = args[2]
    b = args[3]
    spacing = args[4]
    K = args[5]
        
    # extract parameters and symmetrise, if required
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    c = params[2*n_funcs:]
            
    return total_energy(A, x0, c, N, energy_function, b, spacing, K)
    
def make_limits(n_funcs, max_x):
    unbound = (-np.inf, np.inf)
    spatial_bounds = (-max_x/2., max_x/2.)
    non_negative = (0, 100.)
    positive = (1e-4, 100.)
    
    A_bounds = [non_negative for i in xrange(n_funcs)]
    c_bounds = [non_negative for i in xrange(n_funcs)]
    x_bounds = [spatial_bounds for i in xrange(n_funcs)]
    
    lims = A_bounds + x_bounds + c_bounds
    return lims
    
def cons_func(inparams, *args):
    n_funcs = args[0]
    A = inparams[:n_funcs]
    return 1.-sum(A)

def mc_step1d(N, max_x, energy_function, lims, noopt, use_sym, b, spacing, K):
    '''Single monte carlo step for dislocation structure.
    '''
    
    if use_sym:
        params = gen_symmetric(N, spacing)
    else:
        params = gen_inparams(N, spacing)

    if noopt:
        pass
    else:
        try:
            new_par = fmin_slsqp(total_optimizable, params, eqcons=[cons_func],
                                  args=(N, max_x, energy_function, b, spacing, K),
                                             bounds=lims, iprint=0, acc=1e-14)
                       
            E = total_optimizable(new_par, N, max_x, energy_function, b, spacing, K)
        except RuntimeError:
            new_par = None
            E = np.inf

    return E, new_par
        
def run_monte1d(n_iter, N, K, max_x=100, energy_function=test_gamma, noopt=False,
                                   use_sym=False, b=1., spacing=1., noisy=False):
    Emin = 1e6
    x_opt = None
    lims = make_limits(N, max_x)
    
    for i in xrange(n_iter):
        if i % 100 == 0:
            print("Starting iteration {}...".format(i))

        E, x_try = mc_step1d(N, max_x, energy_function, lims, noopt, use_sym, b, spacing, K)
        is_valid = check_parameters1d(x_try, N, lims)
 
        if is_valid and (E < Emin):
            Emin = E
            x_opt = x_try[:]
            
            # if noisy mode has been selected, print current best solution
            if noisy:
                print("Current best solution: ".format(x_opt))
                print("Energy: {:.6f}\n".format(Emin))
            
    return Emin, x_opt
    
def check_parameters1d(x_try, n_funcs, limits):
    '''Checks that the x0 and c parameters in x_try for a particular dislocation
    density distribution are near to their limits. This is used to detect 
    unstable solutions.
    '''
    
    dist = 1e-1
    
    # extract parameters and limits
    A0 = x_try[:n_funcs]
    limits_A = limits[0]
    x0 = x_try[n_funcs:2*n_funcs]
    limits_x = limits[n_funcs]
    c0 = x_try[2*n_funcs:]
    limits_c = limits[2*n_funcs]
    
    # check that none of the x0 have reached the bounds of the spatial region
    # to which dislocations are constrained.
    for x in x0:
        if abs(x-limits_x[0]) < dist or abs(x-limits_x[-1]) < dist:
            return False
            
    # check c parameters. 
    for c in c0:
        if abs(c-limits_c[-1]) < dist:
            return False
        elif c < limits_c[0]:
            return False
    
    return True

def contained_in(element, open_set):
    if element > open_set[0] and element < open_set[1]:
        return True
    else:
        return False    
