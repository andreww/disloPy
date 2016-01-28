#!/usr/bin/env python

import numpy as np

import pn_1D as pn1
import pn_2D as pn2

from scipy.optimize import fmin_slsqp

atomic_to_GPa =  160.2176487 # convert from eV/Ang**3 to GPa

### PEIERLS BARRIER IN 1D ###
    
def total_shift1d(A, x0, c, N, energy_function, b, spacing, xc, K=1):
    E_el = pn1.elastic_energy(A, x0, c, b, K)
    Em = pn1.misfit_energy(A, x0, c, N, energy_function, b, spacing, xc)
    return (Em + E_el)
    
def shift_opt1d(params, *args):
    n_funcs = args[0]
    N = args[1]
    energy_function = args[2]
    b = args[3]
    spacing = args[4]
    K = args[5]
    xc = args[6]
    
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    c = params[2*n_funcs:]
    
    return total_shift1d(A, x0, c, N, energy_function, b, spacing, xc, K)
    
def fixed_location1d(params, *args):
    n_funcs = args[0]
    x_mean = args[-1]
    
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    
    return (x_mean - (A*x0).sum())
    
def displace_disloc(params, n_funcs, max_x, energy_function, b=1., spacing=1., K=1, xc=0.):

    # calculate the mean position of the undisplaced dislocation
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    x_mean = (A*x0).sum()

    in_args = (n_funcs, max_x, energy_function, b, spacing, K, xc, x_mean)
    
    lims = pn1.make_limits(n_funcs, max_x)
    
    new_par = fmin_slsqp(shift_opt1d, params, eqcons=[pn1.cons_func, fixed_location1d],
                                        bounds=lims, args=in_args, iprint=0, acc=1e-16) 
                         
    E = shift_opt1d(new_par, *in_args)    
    return E, new_par 

### PEIERLS BARRIER IN 2D ###
### NOTE: DOES *NOT* WORK VERY WELL - THIS METHOD DOES NOT REALLY CAPTURE THE
### PHYSICS OF DISPLACING A DISLOCATION WITH EDGE AND SCREW COMPONENTS

def fix_location2d(params, *args):
    disl_type = args[-2].lower()
    N = args[1]
    n_funcs = args[0]
    b = args[5]
    spacing = args[6]
    x_mean = args[-3]
    xc = args[-1]
    
    r = spacing*(np.arange(-N, N)) + xc
    
    if disl_type == 'edge':
        A_b = params[:n_funcs/2]
        x_b = params[n_funcs:n_funcs+n_funcs/2]
        c_b = params[2*n_funcs:2*n_funcs+n_funcs/2]
    else: # screw dislocation
        A_b = params[n_funcs/2:n_funcs]
        x_b = params[n_funcs+n_funcs/2:2*n_funcs]
        c_b = params[2*n_funcs+n_funcs/2:]
        
    u = pn1.u_field(r, A_b, x_b, c_b, b)
    rho = pn1.rho(u, r)
    xm_current = pn1.center_of_mass(rho, r, b)
        
    return (x_mean-xm_current)
    
def total_shift2d(A, x0, c, N, energy_function, b, spacing, shift, xc, K=[1, 1]):
    E_el = pn2.elastic_energy2d(A, x0, c, b, K)
    Em = pn2.misfit_energy2d(A, x0, c, N, energy_function, shift, b, spacing, xc)
    return (Em + E_el)
    
def shift_opt2d(params, *args):
    n_funcs = args[0]
    N = args[1]
    energy_function = args[2]
    K = args[3]
    shift = args[4]
    b = args[5]
    spacing = args[6]
    xc = args[-1]
    
    # extract dislocation parameters and calculate energy
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    c = params[2*n_funcs:]
    
    E = total_shift2d(A, x0, c, N, energy_function, b, spacing, shift, xc, K)
    return E
    
def displace_disloc2d(params, n_funcs, max_x, energy_function, K, b, spacing,
                                                        disl_type, xc=0.):
                                                                   
    # check to make sure provided dislocation type (edge/screw) is supported
    constraints, shift = pn2.supported_dislocation(disl_type)
    
    # calculate the mean position (perp. to \xsi) of the dislocation
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    c = params[2*n_funcs:]
    if disl_type.lower() == 'edge':
        A_b = A[:n_funcs/2]
        x0_b = x0[:n_funcs/2]
        c_b = c[:n_funcs/2]
    else: # screw dislocation
        A_b = A[n_funcs/2:]
        x0_b = x0[n_funcs/2:]
        c_b = c[n_funcs/2:]
        
    r = spacing*np.arange(-max_x, max_x)
    u = pn1.u_field(r, A_b, x0_b, c_b, b)
    rho = pn1.rho(u, r)
    x_mean = pn1.center_of_mass(rho, r, b)
        
    constraints.append(fix_location2d)
    
    lims = pn2.make_limits2d(n_funcs, max_x, disl_type)
    
    # construct input arguments to <shift_opt2d>
    in_args = (n_funcs, max_x, energy_function, K, shift, b, spacing, x_mean,
                                                          disl_type, xc)
        
    new_par = fmin_slsqp(shift_opt2d, params, eqcons=constraints, args=in_args,
                                                        bounds=lims, iprint=0)
                                                                     
    E = shift_opt2d(new_par, *in_args)       
    return E, new_par
    
### CALCULATING PEIERLS STRESS BY CALCULATING WORK DONE BY DISPLACING A 
### DISLOCATION

def stress_energy(tau, rho, x_vals):

    return -0.5*tau*(rho*(x_vals[1:]**2-x_vals[:-1]**2)).sum()\

### IN 1-D

#def stressed1d(A, x0#

### TWO-DIMENSIONAL GAMMA SURFACE
    
def total_stressed(A, x0, c, n_funcs, max_x, energy_function, b, spacing, shift, tau,
                                                            acts_on, K=[1, 1]):
    E_el = pn2.elastic_energy2d(A, x0, c, b, K)
    Em = pn2.misfit_energy2d(A, x0, c, max_x, energy_function, shift, b, spacing)
    
    # calculate stress energy
    r = spacing*np.arange(-max_x, max_x)
    if acts_on.lower() == 'edge':
        # stress acts on edge component
        u = pn1.u_field(r, A[:n_funcs/2], x0[:n_funcs/2], c[:n_funcs/2], b,
                                                            bc=shift[0])
    elif acts_on.lower() == 'screw':
        # stress acts on screw component
        u = pn1.u_field(r, A[n_funcs/2:], x0[n_funcs/2:], c[n_funcs/2:], b,
                                                            bc=shift[1])
    else:
        raise ValueError("%s is not a valid dislocation type." % acts_on)  
    
    rho_vals = pn1.rho(u, r)
    E_stress = stress_energy(tau, rho_vals, r)
    return (Em + E_el + E_stress)
    
def total_opt_stress(params, *args):
    n_funcs = args[0]
    max_x = args[1]
    energy_function = args[2]
    K = args[3]
    shift = args[4]
    b = args[5]
    spacing = args[6]
    acts_on = args[7]
    tau = args[8]
    
    # extract dislocation parameters and calculate energy
    A = params[:n_funcs]
    x0 = params[n_funcs:2*n_funcs]
    c = params[2*n_funcs:]
    
    E = total_stressed(A, x0, c, n_funcs, max_x, energy_function, b, spacing, shift, tau,
                                                                      acts_on, K)
    return E
    
    
def stressed_dislocation(params, n_funcs, max_x, energy_function, K, b, spacing,
                                                   disl_type, acts_on, tau=0.):
                                                          
    # check to make sure provided dislocation type (edge/screw) is supported
    constraints, shift = pn2.supported_dislocation(disl_type)
    lims = pn2.make_limits2d(n_funcs, np.inf, disl_type)
    
    in_args = (n_funcs, max_x, energy_function, K, shift, b, spacing, acts_on, tau)

    new_par = fmin_slsqp(total_opt_stress, params, eqcons=constraints, args=in_args,
                                                        bounds=lims, iprint=0)
                                                                     
    E = total_opt_stress(new_par, *in_args)       
    return E, new_par
    
# Calculate the response of a dislocation to an applied stress

def approx_iss(g_max, spacing):
    '''Return the smallest stress with only 1 decimal place that is larger than
    the ideal shear stress for the given gamma surface.
    '''
    
    # work out whether to index or not. Weird bug in np.asarray() means that the
    # type of asarray(<int>) + <int> is int.
    if isinstance(g_max, np.ndarray) and not isinstance(g_max + 1, np.ndarray):
        iss = g_max/(0.5*spacing)
    else:
        iss = g_max[0]/(0.5*spacing)  
    
    return float(int(iss*10)+1)/10

def taup_2d(dis_parameters, max_x, gsf_func, K, b, spacing, disl_type, dtau=0.001,
                                                                      in_GPa=True):
    '''Calculate positive and negative Peierls stresses from optimized
    dislocation misfit profiles.
    '''
    
    # <threshold> is the amount by which the difference between a stressed 
    # dislocation and an unstressed dislocation must exceed the difference
    # between adjacent unstressed dislocations
    threshold = 10.
    
    if disl_type.lower() == 'edge':
        s_max = 100*approx_iss(gsf_func(b/2, 0.), spacing)
    else:
        s_max = 100*approx_iss(gsf_func(0., b/2), spacing)
        
    stresses = np.arange(0., s_max, dtau)
    
    # response to positive and negative stress may be different
    # see (Gouriet et al, 2014) in Modelling Simul. Mater. Sci.
    # Eng.
    tau_p_minus = None
    tau_p_plus = None
    ux, uy = pn2.get_u2d(dis_parameters, b, spacing, max_x, disl_type)
    
    shift = max(1, int(b/spacing)) # distance a dislocation can move
    uxn = np.array([ux[(i-shift) % len(ux)] for i in xrange(len(ux))])
    uyn = np.array([uy[(i-shift) % len(uy)] for i in xrange(len(uy))])
    
    # note: may have to change this later for complicated misfit profiles,
    # but this seems to work for edge dislocations in UO2
    if disl_type.lower() == 'edge':
        d_max = abs(ux-uxn).sum() 
    else: 
        d_max = abs(uy-uyn).sum()    
    # do positive stress first    
    for s in stresses:
        Ed, new_par = stressed_dislocation(dis_parameters, len(dis_parameters)/3, 
                        max_x, gsf_func, K, b, spacing, disl_type, disl_type, s)
        uxs, uys = pn2.get_u2d(new_par, b, spacing, max_x, disl_type)
        d_current = abs(ux-uxs).sum() if disl_type=='edge' else \
                             abs(uy-uys).sum()
        
        if d_current > threshold*d_max:
            tau_p_plus = s
            break
        
    # apply negative stress
    for s in -stresses:
        Ed, new_par = stressed_dislocation(dis_parameters, len(dis_parameters)/3, 
                        max_x, gsf_func, K, b, spacing, disl_type, disl_type, s)
        uxs, uys = pn2.get_u2d(new_par, b, spacing, max_x, disl_type)
        d_current = abs(ux-uxs).sum() if disl_type=='edge' else \
                    abs(uy-uys).sum()
        
        if d_current > threshold*d_max:
            tau_p_minus = -s
            break
                    
    peierls_stresses = [tau_p_plus, tau_p_minus]
    peierls_stresses.sort()
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
    x = ('%.' + ('%.df' % n_figs)) % value
    return float(x)


#!!! IN PROGRESS: 1D version of the dislocation displacement routine

def taup1d(dis_parameters, max_x, gsf_func, K, b, spacing, disl_type, dtau=0.001,
                                                                     in_GPa=True):
    '''Gradually apply stress to a dislocation with 1 component of displacement
    (although it need not be a pure edge or screw dislocation, see eg. 60 deg
    dislocations in Si).  
    '''
    
    # threshold for difference between stressed and unstressed dislocation. Used
    # to detect dislocation motion.
    threshold = 10.
    
    s_max = 100*approx_iss(gsf_func(b/2), spacing)    
    stresses = np.arange(0., s_max, dtau)
    
    taup_minus, taup_plus = None, None
    u = pn1.get_u1d(dis_parameters, b, spacing, N, bc=0.5)
    
    shift = max(2, int(b/spacing)) # distance a dislocation can move
    
    un = np.array([u[(i-shift) % len(u)] for i in xrange(len(u))])
    
    # overlap between input dislocation and dislocation shifted along x
    d_max = abs(u-un).sum()
    
