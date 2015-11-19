#!/usr/bin/env python

import numpy as np
import pn_1D as pn1

from scipy.optimize import fmin_slsqp
#from test_2D_gamma import test_gamma_surf
from numpy.random import uniform

def opposing_partials(N):
    '''Returns coefficients A[:] st. sum(A[:]) == 0
    '''
    
    A_test = uniform(low=-1,high=1,size=N)
    # shift A[:] so that it sums to 0.
    A_test -= A_test.sum()/N
    return A_test

def generate_input(N,disl_type,spacing,use_sym=False):
    '''disl_type in ['edge','screw'].
    '''
    
    # check to make sure that there are equal numbers of x and y dislocations
    if  N % 2:
        raise AttributeError("Invalid number of partials.")
    else:
        # number of partial dislocations / direction
        n_part = N/2
   
    # generate starting parameters || b
    if use_sym:
        dis1 = pn1.gen_symmetric(n_part/2,spacing)
    else:
        dis1 = pn1.gen_inparams(n_part,spacing)
    
    A1 = dis1[:n_part]
    x1 = dis1[n_part:2*n_part]
    c1 = dis1[2*n_part:]
    
    # generate starting parameters for the edge component (which we assume,
    # for the moment, to satisfy ux(-inf) = ux(inf) = 0.0
    c2 = pn1.generate_c(n_part) 
    A2 = opposing_partials(n_part)
    x2 = pn1.generate_x(n_part,abs(A2)/abs(A2).sum(),spacing)
    if disl_type.lower()  == 'edge':
        return list(A1) + list(A2) + list(x1) + list(x2) + list(c1) + list(c2)
    else: # screw dislocation
        return list(A2) + list(A1) + list(x2) + list(x1) + list(c2) + list(c1)
    
def unzip_parameters(A,x0,c):
    '''Extract parameters for the x and y partial dislocations.
    ''' 
        
    n_disloc = len(A)/2
    A1 = A[:n_disloc]
    A2 = A[n_disloc:]
    x01 = x0[:n_disloc]
    x02 = x0[n_disloc:]
    c1 = c[:n_disloc]
    c2 = c[n_disloc:]
    
    return A1,x01,c1,A2,x02,c2

def elastic_energy2d(A,x0,c,b=1,K=[1,1]):
    # K = [K_edge,K_shear]
    Ke = K[0]
    Ks = K[1]
    
    A1,x01,c1,A2,x02,c2 = unzip_parameters(A,x0,c)
    
    E = pn1.elastic_energy(A1,x01,c1,b,Ke) + pn1.elastic_energy(A2,x02,c2,b,Ks)
    return E
    
def misfit_energy2d(A,x0,c,N,energy_function,shift,b,spacing,xc=0):
    '''Defaults shift corresponds to screw dislocation.
    
    0 <= xc <= 1
    '''
    
    A1,x01,c1,A2,x02,c2 = unzip_parameters(A,x0,c)
    
    r = spacing*(np.arange(-N,N))+xc
    ux = pn1.u_field(r,A1,x01,c1,b,bc=shift[0])
    uz = pn1.u_field(r,A2,x02,c2,b,bc=shift[1])
    Em = energy_function(ux,uz).sum()*spacing

    return Em
    
def total_energy2d(A,x0,c,N,energy_function,K,b,spacing,shift=[0.,0.5]):
    '''Defaults to screw dislocation. K = [Ke,Ks]
    '''
    
    Em = misfit_energy2d(A,x0,c,N,energy_function,shift,b,spacing)
    E_el = elastic_energy2d(A,x0,c,b,K)
    return Em + E_el
    
def total_optimizable2d(params,*args):
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
            
    return total_energy2d(A,x0,c,N,energy_function,K,b,spacing,shift)
    
def make_limits2d(n_funcs,max_x,disl_type):
    unbound = (-np.inf,np.inf)
    spatial_bounds = (-max_x,max_x)
    non_negative = (0,np.inf)
    
    A_1 = [non_negative for i in xrange(n_funcs/2)]
    A_0 = [unbound for i in xrange(n_funcs/2)]
    c_bounds = [non_negative for i in xrange(n_funcs)]
    x_bounds = [spatial_bounds for i in xrange(n_funcs)]
    
    if disl_type in 'edge':
        lims = A_1 + A_0 + x_bounds + c_bounds
    else: # screw dislocation
        lims = A_0 + A_1 + x_bounds + c_bounds
        
    return lims
    
def mc_step2d(N,max_x,energy_function,lims,K,shift,constraints,use_sym,
                                                    disl_type,b,spacing):
    params = generate_input(N,disl_type,spacing,use_sym)
    
    try:
        new_par = fmin_slsqp(total_optimizable2d, params, eqcons=constraints,
                        args=(N, max_x, energy_function, K, shift, b, spacing),
                                              bounds=lims, iprint=0, acc=1e-12)
                                                                    
        E = total_optimizable2d(new_par,N,max_x,energy_function,K,shift,b,spacing)
    except RuntimeError:
        new_par = None
        E = np.inf
    return E,new_par

def run_monte2d(n_iter,N,disl_type,K,max_x=100,energy_function=None,
                                                   use_sym=False,b=1,spacing=1):
                                                              
    # generate limits and constraints
    if not energy_function:
        raise ValueError("Energy function must be defined")

    constraints,shift = supported_dislocation(disl_type)              
    lims = make_limits2d(N,max_x,disl_type) 
    
    # run monte carlo simulation -> initialise optimimum parameters and 
    # minimum energy to suitable dummy values
    Emin = 1e6
    x_opt = None
    for i in xrange(n_iter):
        if i % 100 == 0:
            print("Starting iteration {}...".format(i))
        E,x_try = mc_step2d(N,max_x,energy_function,lims,K,shift,constraints,
                                                  use_sym,disl_type,b,spacing)
        is_valid = True
        
        for j,param in enumerate(x_try):
            if pn1.contained_in(param,lims[j]):
                continue
            else:
                is_valid = False
                break

        if is_valid and (E < Emin):
            Emin = E
            x_opt = np.copy(x_try)
            
    return Emin, x_opt
    
def supported_dislocation(disl_type):
    '''Check that <disl_type> is a supported dislocation type and return 
    the correct constraints.
    '''
    
    
    valid_dislocations = ['edge','screw']
    if not(disl_type.lower() in valid_dislocations):
        raise ValueError("Please enter \"edge\" or \"screw\"")
    else:
        if disl_type.lower() in 'edge':
            constraints = [full_edge,no_screw]
            shift = [0.5,0.]
        else: # screw dislocation
            constraints = [no_edge,full_screw]
            shift = [0.,0.5] 
           
        return constraints,shift
    
    
### CONSTRAINTS ON THE A[:] COEFFICIENTS FOR A SCREW DISLOCATION

def full_screw(params,*args):
    n_funcs = args[0]
    A = params[:n_funcs]
    A_screw = A[len(A)/2:]
    return 1. - sum(A_screw)
    
def no_edge(params,*args):
    n_funcs = args[0]
    A = params[:n_funcs]
    A_edge = A[:len(A)/2]
    return sum(A_edge)

### CONSTRAINTS ON THE A[:] COEFFICIENTS FOR AN EDGE DISLOCATION   

def no_screw(params,*args):
    n_funcs = args[0]
    A = params[:n_funcs]
    A_screw = A[len(A)/2:]
    return sum(A_screw)
    
def full_edge(params,*args):
    n_funcs = args[0]
    A = params[:n_funcs]
    A_edge = A[:len(A)/2]
    return 1. - sum(A_edge)
    
def get_u2d(params,b,spacing,N,disl_type):
    '''Calculate specified component of the displacement field from a list of 
    fit parameters. 
    ''' 
    
    if disl_type.lower() in 'edge':
        shift = [0.5,0.]
    elif disl_type.lower() in 'screw':
        shift = [0.,0.5]
    else:
        raise ValueError("Specified dislocation type not supported.")
        
    # extract A,x0, and c fit parameters for the edge and screw components
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
    r = spacing*np.arange(-N,N) # lattice planes
    ux = pn1.u_field(r,Ax,x0x,cx,b,bc=shift[0])
    uy = pn1.u_field(r,Ay,x0y,cy,b,bc=shift[1])
    
    return ux,uy
