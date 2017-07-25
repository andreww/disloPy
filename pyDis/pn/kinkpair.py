#!/usr/bin/env python
'''Functions to determine the critical shape and energy of a kink-pair in a
general material.
'''
from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline as rbs

def x0(a, b, tau, taup):
    '''Equilibrium position of a dislocation with Peierls stress <taup>, spacing
    <a>, and Burgers vector <b>, under an applied stress of <tau>.
    '''
    
    return a/(2*np.pi)*np.arcsin(a/float(b)*tau/taup)


def DPeierls(Vp, h, w, a, b, tau, taup):
    '''Calculates the variation in Peierls energy induced by the nucleation of
    a kink-pair of height <h> and width <w>.
    
    <Vp>: a function mapping kink height to Peierls energy
    <h>: kink height (float)
    <w>: kink width (float)
    <a>: spacing between adjacent Peierls valleys (float)
    '''
    
    # calculate Peierls energy of the kink segments themselves, limiting the 
    # height of the kink to a, the spacing between adjacent Peierls valleys
    x0i = x0(a, b, tau, taup)
    Ek = 2*quad(Vp, x0i, x0i+h)[0]
    
    # calculate energy of dislocation segment
    Eseg = w*(Vp(x0i+h)-Vp(x0i))

    return Ek+Eseg
    
def simple_wp(wpmax, a):
    '''Generates a simple form for the Peierls potential (given on page 241 of
    Hirth and Lothe)
    
    <wpmax>: height of the Peierls barrier
    <a>: periodicity of the Peierls potential
    '''
    
    wp_func = lambda x: wpmax/2.*(1-np.cos(2*np.pi*x/a))
    
    return wp_func
    
def DElastic(Ke, Ks, h, w, b, rho, disl_type, in_GPa=True):
    '''Calculates the excess elastic energy incurred by the nucleation of a 
    new kink-pair for a dislocation of specified type.
    
    <Ke>: edge energy coefficient, in GPa if boolean <in_GPa> is True,
    otherwise in eV/angstrom**2 (float)
    <Ks>: screw energy coefficient, units as for <Ke> (float)
    <b>: Burgers vector, in angstroms (float)
    <rho>: parameter specifying the core size, typically ~0.05b (float)
    <disl_type>: character (edge or screw) of initial dislocation
    '''
    
    if in_GPa:
        # convert to eV/angstrom**2
        Ke /= 160.2
        Ks /= 160.2
    
    r = np.sqrt(w**2+h**2)

    # calculate elastic energy of kinks
    kink_energy = w-r+h*np.log((h+r)/w)-h*np.log(h/(np.e*rho))
    
    # calculate energy of thrown-out segment
    seg_energy = r-w-h+w*np.log(2*w/(w+r))

    if disl_type.lower() == 'edge':
        E_elast = b**2*(Ke*seg_energy-Ks*kink_energy)/(2*np.pi)
    elif disl_type.lower() == 'screw':
        E_elast = b**2*(Ks*seg_energy-Ke*kink_energy)/(2*np.pi)
    
    return E_elast
    
def work(stress, b, h, w, in_GPa=True):
    '''Calculates the work done by a <stress> on a dislocation with Burgers 
    vector <b> to create a kink pair of height <h> and width <w>.
    
    <stress>: applied stress (float). If boolean <in_GPa> is True, units are
    GPa, otherwise eV/angstrom**2
    '''
    
    if in_GPa:
        # convert to eV/angstrom**2
        stress /= 160.2
    
    return stress*b*h*w
    
def DH_kink_pair(h, w, disl_type, Ke, Ks, b, a, stress, rho, Vp, taup, in_GPa=True):
    '''Calculates the total enthalpy variation caused by nucleation of a 
    kink-pair, with parameters defined in <DPeierls>, <work>, and <DElastic>.
    '''
    
    DE = DElastic(Ke, Ks, h, w, b, rho, disl_type, in_GPa=in_GPa)
    DP = DPeierls(Vp, h, w, a,b, stress, taup)
    W = work(stress, b, h, w, in_GPa=in_GPa)

    return DE+DP-W
    
def DH_kink_pair_mappable(disl_type, Ke, Ks, b, a, rho, Vp, taup, in_GPa=True):
    '''Produces a function that converts the function <DH_kink_pair> into a 
    function mapping kink-pair height and width to the kink-pair energy for 
    a given applied stress.
    '''
    
    return lambda h, w, s: DH_kink_pair(h, w, disl_type, Ke, Ks, b, a, s, rho, 
                                                            Vp, taup, in_GPa=in_GPa)
    
def kocks_form(tau, dh0, p, q, taup): 
    '''Kocks classical formalism for the stress dependence of kink nucleation enthalpies.
    
    <tau>: stress
    <dh0>: critical nucleation enthalpy at 0K
    <taup>: the Peierls stress (float)
    <p>, <q>: empirical parameters giving the 
    '''
    
    return dh0*(1-(tau/taup)**p)**q
       
def kocks_fit(stress, enthalpy, taup):
    '''Fits the calculated stress-enthalpy curve to the Kocks equation.
    
    <stress>: list of stresses
    <enthalpy>: list of critical nucleation enthalpies, with same length as 
    <stress>
    '''
    
    # give Kocks equation and fit
    kocks = lambda tau, DH0, p, q: kocks_form(tau, DH0, p, q, taup)
    initial_guess = [enthalpy[0], 1., 2.]
    parameters, error = opt.curve_fit(kocks, stress, enthalpy, p0=initial_guess)
    
    return parameters, error 

def metastable_config(b, a, xsi, Ke, Ks, taup, disl_type, wpfunc=None, wmin=5,
                      wmax=1000, tau_frac_min=0.05, tau_frac_max=0.3, ntau=20, 
                                 in_GPa=True,  nh=100, nw=200, fit_kocks=True):
                                                        
    '''Finds the kink-pair geometry (h, w) at which the kink-pair energy is
    at a critical point, ie dH/dw = dH/dh = 0
    
    <xsi>: dislocation width (float)
    <wmin>: minimum kink-pair separation, in angstroms 
    <wmax>: maximum kink-pair separation
    <a>: dislocation spacing, in angstroms. If <None>, use the Burgers vector 
    <tau_frac_min>: smallest stress at which to calculate metastable shape, as
    a fraction of <taup>
    <tau_frac_max>: largest stress, as fraction of <taup>
    <ntau>; <nh>; <nw>: number of stress, height, and width increments to use
    '''
    
    if wpfunc is None:
        # construct a simple sine potential using the Peierls stress and
        # dislocation geometry
        wpmax = taup*b**2/np.pi
        if in_GPa:
            wpmax /= 160.2

        wpfunc = simple_wp(wpmax, a)
        
    # create kink-pair energy function
    rho = 0.05*xsi # core size
    kp_func = DH_kink_pair_mappable(disl_type, Ke, Ks, b, a, rho, wpfunc, taup)
    
    # calculate critical shape in specified range of stresses    
    critical_values = []
    stresses = np.linspace(tau_frac_min*taup, tau_frac_max*taup, ntau)
    w = np.linspace(wmin, wmax, nw)
    h = np.linspace(0.1, a, nh)
    for s in stresses:
        # calculate kink-pair energies at stress <s> for all kink-pair widths 
        # and heights        
        Ekp = np.zeros((len(w), len(h)))
        for i, wi in enumerate(w):
            for j, hj in enumerate(h):
                Ekp[i, j] = kp_func(hj, wi, s)
                
        # calculate width and height derivates of kink-pair energy
        dEdw, dEdh = np.gradient(Ekp)
        
        # construct spline-fits for derivative functions and find saddle point
        fw = rbs(w, h, dEdw)
        fh = rbs(w, h, dEdh)
        fwi = lambda x: abs(fw(*x)[0, 0])
        fhi = lambda x: abs(fh(*x)[0, 0])
        fci = lambda x: fhi(x)+fwi(x)
        optvals = opt.brute(fci, [[2*wmin, wmax], [0.2*a, a]])
        
        critical_values.append([s]+list(optvals)+[kp_func(optvals[1], optvals[0], s)])
        
    critical_values = np.array(critical_values)
    
    # fit the shape of the activation enthalpy, if specified by user
    if fit_kocks:
        par, err = kocks_fit(critical_values[:, 0], critical_values[:, -1], taup)
        return critical_values, par, err
    else:
        return critical_values, None, None

def velocity(tau, taup, b, dH0, p, q, wc, T, vD):
    '''Returns a value proportional to the velocity of a dislocation, with the 
    proportionality constant equal to the dislocation length.
    
    <vD>: the material's debye frequency
    <dH0>: critical nucleation enthalpy at 0K
    <wc>: critical nucleation width (typically at high stress)
    '''
    
    kb = 8.6e-5      
    prefactor = b**2*vD/wc**2*np.exp(-dH0/(kb*T))
    sinh_term = np.sinh(dH0*(1-(1-(tau/taup)**p)**q)/(kb*T))
    return prefactor*sinh_term

def strain_rate(rho, b, v):
    '''Calculates the strain rate due to a density of dislocations with Burgers
    vector <b> moving at velocity <v>.
    '''
    
    return rho*b*v
    
def make_rational(exponent):
    '''Converts the fitted value of <exponent> (either <p> or <q>) to a rational
    number. This is done because both exponents are assumed to be rational, and
    there is relatively little sensitivity of the fit to their exact value.
    '''
    
    # construct a table of reasonable values for the exponent
    e3 = np.arange(0., 16.)/3
    e4 = np.arange(0., 21.)/4
    e5 = np.arange(0., 26.)/5
    
    e3_unique = np.setdiff1d(e3, e4)
    e34 = np.append(e3_unique, e4)
    e5_unique = np.setdiff1d(e5, e34)
    rationals = np.append(e34, e5_unique)
    
    # calculate distance from <exponent> to each rational, and return closest
    dx = abs(exponent - rationals)
    best_exponent = rationals[(dx == dx.min())]
    if type(best_exponent) == float:   
        return best_exponent
    else:
        # take smaller value
        return best_exponent[0]
