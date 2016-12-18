#!/usr/bin/env python

import sys
import os
sys.path.append(os.environ['PYDISPATH'])

import numpy as np

from pyDis.pn import pn_1D as pn1

from scipy.optimize import fmin_slsqp
from numpy.random import uniform

# suppress divide by zero Runtime warnings
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

def opposing_partials(N):
    '''Returns coefficients A[:] st. sum(A[:]) == 0
    '''
    
    if N == 1:
        # if only one partial dislocation is used for the component of the 
        # displacement perpendicular to the dislocation, then it must be 
        # identically zero everywhere; ie. A == 0.
        A_test =np.array([0.])
    else:
        A_test = uniform(low=-1, high=1, size=N)
        # shift A[:] so that it sums to 0.
        A_test -= A_test.sum()/N
    return A_test

def generate_input(N, disl_type, spacing, use_sym=False):
    '''disl_type in ['edge', 'screw'].
    '''
    
    # check to make sure that there are equal numbers of x and y dislocations
    if  N % 2:
        raise AttributeError("Invalid number of partials.")
    else:
        # number of partial dislocations / direction
        n_part = N/2
   
    # generate starting parameters || b
    if use_sym:
        dis1 = pn1.gen_symmetric(n_part, spacing)
    else:
        dis1 = pn1.gen_inparams(n_part, spacing)
    
    A1 = dis1[:n_part]
    x1 = dis1[n_part:2*n_part]
    c1 = dis1[2*n_part:]
    
    # generate starting parameters for the component perpendicular to the burgers
    # vector  
    c2 = pn1.generate_c(n_part) 
    A2 = opposing_partials(n_part)
    x2 = pn1.generate_x(n_part, A2, spacing)
    
    if disl_type.lower()  == 'edge':
        return list(A1) + list(A2) + list(x1) + list(x2) + list(c1) + list(c2)
    else: # screw dislocation
        return list(A2) + list(A1) + list(x2) + list(x1) + list(c2) + list(c1)
    
def unzip_parameters(A, x0, c):
    '''Extract parameters for the x and y partial dislocations.
    ''' 
        
    n_disloc = len(A)/2
    A1 = A[:n_disloc]
    A2 = A[n_disloc:]
    x01 = x0[:n_disloc]
    x02 = x0[n_disloc:]
    c1 = c[:n_disloc]
    c2 = c[n_disloc:]
    
    return A1, x01, c1, A2, x02, c2

def elastic_energy2d(A, x0, c, b=1, K=[1, 1]):
    '''Calculate the elastic energy for the 2D dislocation density distribution
    defined by <A>, <x0>, and <c>.
    '''
    
    # K = [K_edge, K_shear]
    Ke = K[0]
    Ks = K[1]
    
    A1, x01, c1, A2, x02, c2 = unzip_parameters(A, x0, c)
    
    E = pn1.elastic_energy(A1, x01, c1, b, Ke) + pn1.elastic_energy(A2, x02, c2, b, Ks)
    return E   
    
def misfit_energy2d(A, x0, c, N, energy_function, b, spacing, shift=[0., 0.5], 
                                                                translate=0.):
    '''Defaults shift corresponds to screw dislocation.
    
    0 <= xc <= 1
    '''
    
    A1, x01, c1, A2, x02, c2 = unzip_parameters(A, x0, c)
    
    r = spacing*(np.arange(-N, N))+translate
    ux = pn1.u_field(r, A1, x01, c1, b, bc=shift[0])
    uz = pn1.u_field(r, A2, x02, c2, b, bc=shift[1])
    Em = energy_function(ux, uz).sum()*spacing

    return Em
    
def total_energy2d(A, x0, c, N, energy_function, K, b, spacing, shift=[0., 0.5], 
                                                                   translate=0.):
    '''Calculates the total energy of the 2D dislocationd density distribution
    defined by <A>, <x0>, and <c>. Defaults to screw dislocation: K = [Ke, Ks]
    '''
    
    Em = misfit_energy2d(A, x0, c, N, energy_function, b, spacing, shift=shift,
                                                           translate=translate)
    E_el = elastic_energy2d(A, x0, c, b, K)
    return Em + E_el
    
def total_optimizable2d(params, *args):
    '''A form of the total energy functional that can be used with the slsqp
    function from scipy.optimize.
    '''
    
    n_funcs = args[0]
    N = args[1]
    energy_function = args[2]
    K = args[3]
    shift = args[4]
    b = args[5]
    spacing = args[6]
    
    # extract parameters and symmetrise, if required
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    c = params[2*n_funcs:]
            
    return total_energy2d(A, x0, c, N, energy_function, K, b, spacing, shift)
    
def make_limits2d(n_funcs, max_x, disl_type):
    '''Construct limits on the fit parameters for the misfit profile.
    '''
    
    unbound = (-np.inf, np.inf)
    spatial_bounds = (-max_x/2., max_x/2.)
    non_negative = (0, np.inf)
    positive = (1e-4, 100.)
    
    A_1 = [non_negative for i in xrange(n_funcs/2)]
    A_0 = [unbound for i in xrange(n_funcs/2)]
    c_bounds = [positive for i in xrange(n_funcs)]
    x_bounds = [spatial_bounds for i in xrange(n_funcs)]
    
    if disl_type in 'edge':
        lims = A_1 + A_0 + x_bounds + c_bounds
    else: # screw dislocation
        lims = A_0 + A_1 + x_bounds + c_bounds
        
    return lims
    
def dislocation2d(N, max_x, energy_function, lims, K, shift, constraints, use_sym,
                                                     disl_type, b, spacing, inpar=None):
    '''Optimise disregistry profile of dislocation with Burgers vector <b> and
    misfit energy defined by <energy_function>.
    '''
    
    if not (inpar is None):
        # use the user-provided dislocation parameters
        params = inpar 
        
        # check that <params> matches <N> and has the correct number of parameters
        # for each arctan function
        if len(params) % 3 != 0:
            raise ValueError("Each function must have three parameters: A, x0, and c")
        elif len(params)/3 != N:
            raise Warning("Number of supplied parameters does not match specified" +
                          " number of functions. Setting N = {}".format(len(params)/3))
            N = len(params)/3   
    else:
        # generate a trial input configuration
        params = generate_input(N, disl_type, spacing, use_sym)

    try:
        # attempt to optimize the disregistry field
        new_par = fmin_slsqp(total_optimizable2d, params, eqcons=constraints,
                        args=(N, max_x, energy_function, K, shift, b, spacing),
                                              bounds=lims, iprint=0, acc=1e-16)
        
        # calculate energy of optimized field                                                           
        E = total_optimizable2d(new_par, N, max_x, energy_function, K, shift, b, spacing)
    except RuntimeError:
        new_par = None
        E = np.inf

    return E, new_par
    
def supported_dislocation(disl_type):
    '''Check that <disl_type> is a supported dislocation type and return 
    the correct constraints. At present, only edge and screw dislocations are 
    supported. In the future, however, we will extend this to general dislocations.
    
    NOTE: This can be (partially) fooled by setting the "edge" and "screw" directions
    to be the b-parallel and b-perpendicular components of the disregistry.
    '''
    
    valid_dislocations = ['edge', 'screw']
    if not(disl_type.lower() in valid_dislocations):
        raise ValueError("Please enter \"edge\" or \"screw\"")
    else:
        if disl_type.lower() in 'edge':
            constraints = [full_edge, no_screw]
            shift = [0.5, 0.]
        else: # screw dislocation
            constraints = [no_edge, full_screw]
            shift = [0., 0.5] 
           
        return constraints, shift
      
### CONSTRAINTS ON THE A[:] COEFFICIENTS FOR A SCREW DISLOCATION

def full_screw(params, *args):
    '''Sum of the A coefficients must be 1 for the screw component of displacement
    of a pure screw dislocation.
    '''
    
    n_funcs = args[0]
    A = params[:n_funcs]
    A_screw = A[len(A)/2:]
    return 1. - sum(A_screw)
    
def no_edge(params, *args):
    '''Sum of the A coefficients must be 0 for the edge component of displacement
    of a pure screw dislocation.
    '''
    
    n_funcs = args[0]
    A = params[:n_funcs]
    A_edge = A[:len(A)/2]
    return sum(A_edge)

### CONSTRAINTS ON THE A[:] COEFFICIENTS FOR AN EDGE DISLOCATION   

def no_screw(params, *args):
    '''Sum of the A coefficients must be 0 for the screw component of displacement
    of a pure edge dislocation.
    '''
    
    n_funcs = args[0]
    A = params[:n_funcs]
    A_screw = A[len(A)/2:]
    return sum(A_screw)
    
def full_edge(params, *args):
    '''Sum of the A coefficients must be 1 for the edge component of displacement
    of a pure edge dislocation.
    '''
    
    n_funcs = args[0]
    A = params[:n_funcs]
    A_edge = A[:len(A)/2]
    return 1. - sum(A_edge)
    
def get_u2d(params, b, spacing, N, disl_type):
    '''Calculate specified component of the displacement field from a list of 
    fit parameters. 
    ''' 
    
    if disl_type.lower() in 'edge':
        shift = [0.5, 0.]
    elif disl_type.lower() in 'screw':
        shift = [0., 0.5]
    else:
        raise ValueError("Specified dislocation type not supported.")
        
    # extract A, x0, and c fit parameters for the edge and screw components
    n_funcs = len(params)/6
    # -> edge component
    Ax = params[:n_funcs]
    x0x = params[2*n_funcs:3*n_funcs]
    cx = params[4*n_funcs:5*n_funcs]
    # -> screw component
    Ay = params[n_funcs:2*n_funcs]
    x0y = params[3*n_funcs:4*n_funcs]
    cy = params[5*n_funcs:]
    
    # calculate displacement fields
    r = spacing*np.arange(-N, N) # lattice planes
    ux = pn1.u_field(r, Ax, x0x, cx, b, bc=shift[0])
    uy = pn1.u_field(r, Ay, x0y, cy, b, bc=shift[1])
    
    return ux, uy
    
def run_monte2d(n_iter, N, disl_type, K, max_x=100, energy_function=None,
                 use_sym=False, b=1, spacing=1, noisy=False, params=None):
    '''Runs a collection of dislocation energy minimization calculations with
    random dislocation configurations to find the optimum(ish) dislocation 
    configuration.
    '''
 
    # check that the specified number of functions matches <params>
    if not (params is None):
        if N != len(params)/3:
            N = len(params)/3 
                                                              
    # generate limits and constraints
    if not energy_function:
        raise ValueError("Energy function must be defined")

    constraints, shift = supported_dislocation(disl_type)              
    lims = make_limits2d(N, max_x, disl_type) 
    
    # run monte carlo simulation -> initialise optimimum parameters and 
    # minimum energy to suitable dummy values
    Emin = 1e6
    x_opt = None
    
    for i in xrange(n_iter):
        if noisy and i % 100 == 0:
            print("Starting iteration {}...".format(i))
        
        # generate a trial dislocation configuration and calculate its energy    
        E, x_try = dislocation2d(N, max_x, energy_function, lims, K, shift, constraints,
                                          use_sym, disl_type, b, spacing, inpar=params)
                                                  
        # check that the parameters are reasonable(ish)
        is_valid = check_parameters2d(x_try, N, lims, disl_type)
        
        if is_valid and (E < Emin):
            # check that the new energy is not crazy
            if Emin < 1.0 and abs(E-Emin) < 10. or Emin > 1.0:
                Emin = E
                x_opt = np.copy(x_try)
                
                # if noisy mode has been selected, print current best solution
                if noisy:
                    print("Current best solution: ".format(x_opt))
                    print("Energy: {:.6f}\n".format(Emin))
            else:
                # energy is not reasonable
                if noisy:
                    print("Unreasonable dislocation energy calculated...")
                else:
                    pass
            
    return Emin, x_opt
    
def check_parameters2d(x_try, n_funcs, limits, disl_type):
    '''Same as the check_parameters function in <pn_1D>, except for two 
    components of displacement.
    '''
    
    dist = 1e-1
    
    # extract edge and screw components
    Aboth = x_try[:n_funcs]
    x0both = x_try[n_funcs:2*n_funcs]
    limits_x = limits[n_funcs]
    cboth = x_try[2*n_funcs:]
    limits_c = limits[2*n_funcs]
    
    A1, x01, c1, A2, x02, c2 = unzip_parameters(Aboth, x0both, cboth)
    
      
    # check that none of the x0 have reached the bounds of the spatial region
    # to which dislocations are constrained.
    for x in (list(x01)+list(x02)):
        if abs(x-limits_x[0]) < dist or abs(x-limits_x[-1]) < dist:
            return False
            
    # check c parameters. 
    for c in list(c1)+list(c2):
        if abs(c-limits_c[-1]) < dist:
            return False
        elif c < limits_c[0]:
            return False
           
    # Check that the component of disregistry perpendicular to the Burgers vector
    # does not have large values for the coefficients A
    A_max = 10.
    if disl_type == 'edge':
        A_perp = A2
    else:
        A_perp = A1
        
    for A in A_perp:
        if abs(A) > A_max:
            return False
    
    return True
