#!/usr/bin/env python
'''Routines to calculate various properties relevant to the critical resolved
shear stress of a dislocation migrating via kink-pair nucleation. 
'''
from __future__ import print_function, division, absolute_import

import numpy as np
import re
from pydis.pn import kinkpair as kp

def parse_material_file(filename):
    '''Reads in a file containing a list of slip systems, together with the 
    material parameters that govern their mobility in the glide-controlled
    regime.
    '''
    
    # list of parameters used in CRSS calculation -> will parse this to construct
    # dictionary for each slip system, second list contains parameters <type>s
    par_list = ['pot', 'slip', 'disltype', 'b', 'a', 'Ke', 'Ks', 'taup', 'xsi',
                'wmin', 'wmax', 'taup_min', 'taup_max']
    par_types = [str, str, str, float, float, float, float, float, float, float,
                 float, float, float]
    npars = len(par_list)
    
    slip_systems = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                # comment line -> skip
                continue
            
            # extract parameters and check that line is non-empty and contains 
            # the correct number of parameters
            pars = line.rstrip().split()
            if not pars:
                # empty
                continue
            elif len(pars) == npars:
                pass
            else:
                error_msg = 'CRSS simulation requires {:.0f} parameters; you' + \
                            ' have provided only {:.0f} in line {:0f}.'
                raise ValueError(error_msg.format(npars, len(pars), i))
                
            # construct a dictionary containing parameters for this slip system
            slipsys = dict()             
            for i, par in enumerate(pars):
                slipsys[par_list[i]] = par_types[i](par)
                
            # convert elastic coefficients
            elastic_coeffs_to_gpa(slipsys)
            
            # calculate critical values and kocks parameters, and add to dict
            crits, kocks = calculate_kp_mobility(slipsys)
            slipsys['crits'] = crits.copy()
            slipsys['kocks_pars'] = kocks
          
            slip_systems.append(slipsys)
    
    return slip_systems
    
def elastic_coeffs_to_gpa(pardict):
    '''Converts the screw and edge elastic coefficients to GPa (and removes the
    factor of 4pi).
    '''
    
    conv = 4*np.pi*160.2   
    pardict['Ke'] = conv*pardict['Ke']
    pardict['Ks'] = conv*pardict['Ks']
    return
    
def calculate_kp_mobility(slipsys):
    '''Calculates the critical shape and enthalpy for kink-pair nucleation as
    a function of applied stress, and fits this to a Kocks equation to find
    both the stress-free nucleation enthalpy and the dependence of the critical
    enthalpy on the applied stress.
    '''
    
    # <crit_vals> are calculated critical shapes and enthalpies, <fitpars> the
    # Kocks parameters defining them
    crit_vals, fitpars, fiterrs = kp.metastable_config(slipsys['b'],
                                                       slipsys['a'],
                                                       slipsys['xsi'],
                                                       slipsys['Ke'],
                                                       slipsys['Ks'],
                                                       slipsys['taup'],
                                                       slipsys['disltype'],
                                                       wmin=slipsys['wmin'],
                                                       wmax=slipsys['wmax'],
                                                       tau_frac_min=slipsys['taup_min'],
                                                       tau_frac_max=slipsys['taup_max']
                                                      )
                                                      
    # convert the <p> and <q> exponents to rational values
    p_new = kp.make_rational(fitpars[1])
    q_new = kp.make_rational(fitpars[2])
    errors = np.sqrt(np.diag(fiterrs))
    kocks_pars = dict()
    kocks_pars['dH0'] = {'val': fitpars[0], 'err': errors[0]}
    kocks_pars['p'] = {'val': p_new, 'err': errors[1]}
    kocks_pars['q'] = {'val': q_new, 'err': errors[2]}
    
    return crit_vals, kocks_pars
    
def calc_dH(slipsys, niter=100):
    '''Calculates the critical enthalpy between 0 GPa and the Peierls stress 
    using fitted values for dH0, p, and q.
    '''
    
    stress = np.linspace(0., slipsys['taup'], niter)
    dH = kp.kocks_form(stress, slipsys['kocks_pars']['dH0']['val'], 
                               slipsys['kocks_pars']['p']['val'], 
                               slipsys['kocks_pars']['q']['val'], 
                               slipsys['taup'])
    
    return dH
    
def crss_at_T(T, dH0, p, q, taup, C=25):
    '''Calculates the critical resolved shear stress of a slip system with 0 GPa
    critical nucleation enthalpy <dH0>, stress dependence parameters <p> and <q>,
    and Peierls stress <taup>. <C> is the proportionality constant between the 
    temperature and the CRSS. Typical values are in the range 20-30, so the default
    is taken here as 25.
    '''
    
    kb = 8.6173e-5    
    tau = taup*np.power(1.-np.power(C*kb*T/dH0, 1./q), 1./p)
    
    return tau
    
def crss_from_slipsys(T, slipsys, C=25):
    '''As above, but starting from a <slip_system> dictionary.
    '''
    
    return crss_at_T(T, slipsys['kocks_pars']['dH0']['val'], 
                        slipsys['kocks_pars']['p']['val'], 
                        slipsys['kocks_pars']['q']['val'], slipsys['taup'], C=C)                                      
    
def write_critical_pars_calc(slipsys):
    '''Writes the calculated critical shape and enthalpy of the slip system to
    file.
    '''
    
    output_name = 'calc_{}_{}_{}.txt'.format(slipsys['pot'], slipsys['slip'], 
                                                        slipsys['disltype'])
    ostream = open(output_name, 'w')
    
    ostream.write('# stress (GPa) w_crit (\AA) h_crit (\AA) H_crit (eV)\n')
    for s in slipsys['crits']:
        ostream.write('{:.3f} {:.6f} {:.6f} {:.6f}\n'.format(s[0], s[1], s[2], s[3]))
        
    ostream.close()
    
    return
    
def write_critical_pars_fitted(slipsys, dH):
    '''Writes the FITTED critical shape and enthalpy of the slip system to file.
    '''
    
    # generate list of stresses from dH and the Peierls stress
    stress = np.linspace(0., slipsys['taup'], len(dH))
    
    output_name = 'fit_{}_{}_{}.txt'.format(slipsys['pot'], slipsys['slip'], 
                                                        slipsys['disltype'])
    ostream = open(output_name, 'w')
    
    # write fit parameters
    ostream.write('# dH0 {:.3f} +/- {:.3f};'.format(slipsys['kocks_pars']['dH0']['val'],
                                                    slipsys['kocks_pars']['dH0']['err'])) 
    ostream.write('p {:.2f} +/- {:.2f};'.format(slipsys['kocks_pars']['p']['val'],
                                                slipsys['kocks_pars']['p']['err'])) 
    ostream.write('q {:.2f} +/- {:.2f}\n'.format(slipsys['kocks_pars']['q']['val'],
                                                 slipsys['kocks_pars']['q']['err'])) 
    
    # write actual fitted values for the critical enthalpy
    for s, dHs in zip(stress, dH):
        ostream.write('{:.3f} {:.6f}\n'.format(s, dHs))
        
    ostream.close()
    
    return 
    
def write_crss(crss, T, slipsys):
    '''Writes the calculated critical resolved shear stress (as a function of 
    temperature, T), to file.
    '''
    
    output_name = 'crss_{}_{}_{}.txt'.format(slipsys['pot'], slipsys['slip'], 
                                                        slipsys['disltype'])                                    
    ostream = open(output_name, 'w')
    ostream.write('# T (K) CRSS (GPa)\n')
    for Ti, si in zip(T, crss):
        ostream.write('{:.1f} {:.6f}\n'.format(Ti, si))
        
    return
    
def read_kocks_parameters(kocks_fit_file):
    '''Parses a file containing fitted critical enthalpies for
    kink-pair nucleation to extract the Kocks parameters <dH0>,
    <p>, and <q>.
    '''
    
    # get comment line containing the Kocks fit parameters and read them in
    infile = open(kocks_fit_file, 'r')
    parline = infile.readline()
    infile.close()
    
    dH0 = re.search('dH0.+?(?P<val>-?\d+\.\d+)', parline).group('val')
    q = re.search('q.+?(?P<val>-?\d+\.\d+)', parline).group('val')
    p = re.search('p.+?(?P<val>-?\d+\.\d+)', parline).group('val')
    return float(dH0), float(p), float(q)
