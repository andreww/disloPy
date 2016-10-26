#!/usr/bin/env python

import numpy as np

import pn_1D as pn1
import pn_2D as pn2

from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt

atomic_to_GPa =  160.2176487 # convert from eV/Ang**3 to GPa

def stress_energy(tau, rho, x_vals):
    '''Calculate total stress energy.
    '''
    
    return -0.5*tau*(rho*(x_vals[1:]**2-x_vals[:-1]**2)).sum()
    
def total_stressed(A, x0, c, n_funcs, max_x, energy_function, K, b, spacing, 
                                           shift, tau, disl_type=None, dims=2):
    '''Calculate the (fully relaxed) energy of a 1- or 2-dimensional dislocation
    under the action of an applied stress <tau>.
    
    #!!! Probably need some routine to check that the dimensions of <shift>,
    <energy_function>, and <K> are compatible with <dims>
    '''
                                                            
    # determine (from <dims>) which functions should be used to calculate the 
    # elastic and inelastic (ie. misfit) components of the dislocation energy. 
    if dims == 1:
        el_func = pn1.elastic_energy
        misfit_func = pn1.misfit_energy
    elif dims == 2:
        el_func = pn2.elastic_energy2d
        misfit_func = pn2.misfit_energy2d
    
    # calculate elastic and misfit energies 
    E_el = el_func(A, x0, c, b, K)
    Em = misfit_func(A, x0, c, max_x, energy_function, b, spacing, shift)
    
    # calculate the component of the displacement field affected by the applied
    # stress, and the corresponding dislocation distribution.
    r = spacing*np.arange(-max_x, max_x)
    if dims == 1:
        u = pn1.u_field(r, A, x0, c, b, bc=shift)
    if dims == 2:
        if disl_type.lower() == 'edge':
            # stress acts on edge component
            u = pn1.u_field(r, A[:n_funcs/2], x0[:n_funcs/2], c[:n_funcs/2], b,
                                                                bc=shift[0])
        elif disl_type.lower() == 'screw':
            # stress acts on screw component
            u = pn1.u_field(r, A[n_funcs/2:], x0[n_funcs/2:], c[n_funcs/2:], b,
                                                                bc=shift[1])
        else: # will never evaluate inside this block if using <taup> function
            raise ValueError("{} not a valid dislocation type.".format(acts_on))  

    rho_vals = pn1.rho(u, r)
    
    # calculate the stress energy
    E_stress = stress_energy(tau, rho_vals, r)
    return (Em + E_el + E_stress)
   
def total_opt_stress(params, *args):
    '''Version of total_stressed that can be passed into the optimization 
    routines in <scipy.optimize>.
    '''
    
    # parse <args> to extract values for <total_stressed>. (Is there any reason
    # why this cannot be done as part of <total_stressed>? This function seems
    # largely redundant).
    n_funcs = args[0]
    max_x = args[1]
    energy_function = args[2]
    K = args[3]
    b = args[4]
    spacing = args[5]
    shift = args[6]
    tau = args[7]
    disl_type = args[8]
    dims = args[9]
    
    # extract dislocation parameters and calculate energy
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    c = params[2*n_funcs:]
    
    E = total_stressed(A, x0, c, n_funcs, max_x, energy_function, K, b, spacing, 
                                                      shift, tau, disl_type, dims)
    return E
    
    
def stressed_dislocation(params, n_funcs, max_x, energy_function, K, b, spacing,
                                           tau, disl_type=None, dims=1):
    '''Calculates the fully relaxed structure of a dislocation whose structure
    has previously been determined under unstressed conditions. 
    
    #!!! are <disl_type> and <acts_on> the same variable?
    '''
                                                          
    # check to make sure provided dislocation type (edge/screw) is supported
    #!!! How to generalise this so that it works for 1- and 2-dimensional dislocations?
    if dims == 1:
        constraints = [pn1.cons_func]
        shift = 0.5
        lims = pn1.make_limits(n_funcs, np.inf)
    elif dims == 2:
        constraints, shift = pn2.supported_dislocation(disl_type)
        lims = pn2.make_limits2d(n_funcs, np.inf, disl_type)
    
    in_args = (n_funcs, max_x, energy_function, K, b, spacing, shift, tau, 
                                                            disl_type, dims)
    # obtain the relaxed structure of the dislocation in the applied stress 
    # field
    new_par = fmin_slsqp(total_opt_stress, params, eqcons=constraints, args=in_args,
                                                     bounds=lims, iprint=0, acc=1e-14)
     
    # calculate total self-interaction + misfit + stress energy                                   
    E = total_opt_stress(new_par, *in_args)       
    
    return E, new_par
    
# Calculate the response of a dislocation to an applied stress

def approx_iss2d(gfunc, b, disl_type):
    '''Return the smallest stress with only 1 decimal place that is larger than
    the ideal shear stress for the given gamma surface.
    '''
    
    if disl_type.lower() == 'edge':
        gvalues = gfunc(np.linspace(0., b, 100), np.zeros(100))
    elif disl_type.lower() == 'screw':
        gvalues = gfunc(np.zeros(100), np.linspace(0., b, 100))
    
    g_max = gvalues.max()
    dist = b*np.where(gvalues == gvalues.max())[0][0]/100.
    
    return g_max/dist

def approx_iss1d(gfunc, b):
    '''Return the approximate value of the ideal shear strength.
    '''
    
    gvalues = gfunc(np.linspace(0., b, 100))
    g_max = gvalues.max()
    dist = b*np.where(gvalues == gvalues.max())[0][0]/100.
    return g_max/dist
    
def com(rho, r, b, n=1000):
    av_x = 0
    dr = 1./(n-1)
    for i, rhoi in enumerate(rho):
        av_x += (dr*np.linspace(r[i]+dr, r[i+1], n)*rhoi).sum()
    
    return av_x

    
def taup(dis_parameters, max_x, gsf_func, K, b, spacing,  dims=1, disl_type=None,
                                                 dtau=0.001, in_GPa=True, thr=0.5):
    '''Calculate positive and negative Peierls stresses from optimized
    dislocation misfit profiles.
    '''
    
    # <threshold> is the amount by which the difference between a stressed 
    # dislocation and an unstressed dislocation must exceed the difference
    # between adjacent unstressed dislocations
    threshold = thr*spacing
    
    
    # construct list of stresses to apply, using the gamma surface as a guide 
    # for the maximum possible value
    if dims == 1:
        s_max = 100*approx_iss1d(gsf_func, b)
    elif dims == 2:
        if disl_type.lower() != 'edge' and disl_type.lower() != 'screw':
            raise ValueError("{} not a valid dislocation type.".format(acts_on)) 
        else: # calculate maximum stress
            s_max = 100*approx_iss2d(gsf_func, b, disl_type)
            
    else:
        raise ValueError("Dislocation must be 1- or 2-dimensional.")
        
    stresses = np.arange(0., s_max, dtau)
    
    # response to positive and negative stress may be different
    # see (Gouriet et al, 2014) in Modelling Simul. Mater. Sci.
    # Eng.
    tau_p_minus = None
    tau_p_plus = None
    r = spacing*np.arange(-max_x, max_x)
       
    # find difference between the base dislocation and one that has been displaced
    # note: may have to change this later for complicated misfit profiles,
    # but this seems to work for edge dislocations in UO2 
    #!!! SHOULD WE USE THE DISLOCATION DENSITY?!
    shift = max(5, int(float(b)/spacing)) # distance a dislocation can move
    if dims == 1:
        # get displacement field of unstressed dislocation, compare with 
        # shifted dislocation
        u = pn1.get_u1d(dis_parameters, b, spacing, max_x)
    else: # dims == 2
        ux, uy = pn2.get_u2d(dis_parameters, b, spacing, max_x, disl_type)
        if disl_type.lower() == 'edge':
            u = ux
        else: # screw
            u = uy
    
    rho = pn1.rho(u, r)
    #d_max = (rho*r[1:]/rho.sum()).sum()
    cm0 = pn1.center_of_mass(rho, r, b)
    # apply stress to the dislocation, starting with the positive direction
    new_par = dis_parameters
    for s in stresses:
        Ed, new_par = stressed_dislocation(new_par, len(dis_parameters)/3, 
                            max_x, gsf_func, K, b, spacing, s, disl_type,  dims)
        
        # compare new displacement field to that of original (ie. unstressed)
        # dislocation
        if dims == 1:
            us = pn1.get_u1d(new_par, b, spacing, max_x)
        else: # two-dimensions
            uxs, uys = pn2.get_u2d(new_par, b, spacing, max_x, disl_type)
            if disl_type == 'edge':
                us = uxs
            else: # screw
                us = uys
        
        # calculate the stressed dislocation's centre            
        rhos = pn1.rho(us, r)        
        cm_new = pn1.center_of_mass(rhos, r, b) 
        # calculate distance of dislocation density c.o.m from the location of
        # the unstressed dislocation, recording the value if it exceeds specified
        # threshold
        if abs(cm_new - cm0) >= threshold:
            tau_p_plus = s
            break

    # apply negative stress
    new_par = dis_parameters
    for s in -stresses:
        Ed, new_par = stressed_dislocation(new_par, len(dis_parameters)/3, 
                             max_x, gsf_func, K, b, spacing, s, disl_type, dims)
        
        # compare with unstressed dislocation                     
        if dims == 1:
            us = pn1.get_u1d(new_par, b, spacing, max_x)
        else: # two-dimensions
            uxs, uys = pn2.get_u2d(new_par, b, spacing, max_x, disl_type)
            if disl_type == 'edge':
                us = uxs
            else: # screw
                us = uys
        
        # calculate centre of mass of the stressed dislocation
        rhos = pn1.rho(us, r)
        cm_new = pn1.center_of_mass(rhos, r, b)
        
        # calculate distance of dislocation density c.o.m from the location of
        # the unstressed dislocation, recording the value if it exceeds specified
        # threshold
        if abs(cm_new - cm0) >= threshold:
            tau_p_minus = -s
            break
                   
    peierls_stresses = [tau_p_minus, tau_p_plus]
    
    if in_GPa:
        # express Peierls stresses in GPa
        peierls_stresses = [atomic_to_GPa*tau for tau in peierls_stresses]
            
    peierls_stresses = [sig_figs(tau, 3) for tau in peierls_stresses]
    
    # calculate the average Peierls stress
    taup_av = sum(peierls_stresses)/2
    
    return peierls_stresses, taup_av
    
def peierls_barrier(taup, b, in_GPa=True):
    '''Calculate an approximate value for the Peierls barrier from the provided
    values for the Peierls stress using the expression:
    
    (max x in [0, b))(Wp(x)) = taup * b**2 / pi
    '''
    
    wp_max = taup*b**2/np.pi
    
    if in_GPa:
        wp_max = wp_max/atomic_to_GPa
        
    return wp_max

def sig_figs(value, n_figs):
    x = '{{:.{}f}}'.format(n_figs).format(value)
    return float(x)
    
### PEIERLS BARRIER BY TRANSLATION AND OPTIMIZATION ###

def com_constraint(par, *args):
    '''Constrains the location of the centre of mass of the 
    dislocation density distribution.
    '''
    
    # extract sim. parameters
    n_funcs = args[0]
    N = args[1]
    energy_function = args[2]
    b = args[3]
    spacing = args[4]
    K = args[5]
    fix_centre = args[6]
    dimensions = args[7]
    if dimensions == 2:
        disl_type = args[8]
    
    # calculate dislocation density
    if dimensions == 1:
        u = pn1.get_u1d(par, b, spacing, N)
    elif dimensions == 2:
        ux, uy = pn2.get_u2d(par, b, spacing, N, disl_type)
        if disl_type.lower() == 'edge':
            u = ux
        elif disl_type.lower() == 'screw':
            u = uy
     
    # calculate the dislocation density distribution
    r = np.arange(-N*spacing, N*spacing, spacing)
    rho = pn1.rho(u, r)
    
    # calculate dislocation density c.o.m.
    com = pn1.center_of_mass(rho, r, b)
    # return distance between current and desired centre of 
    # mass
    return (com - fix_centre)

