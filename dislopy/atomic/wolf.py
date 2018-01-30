#!/usr/bin/env python
'''Functions to help optimize the value of the cutoff and damping parameters in
the Wolf formulation of the Coulomb energy.
'''
from __future__ import print_function, absolute_import

import numpy as np
import re
import subprocess

from dislopy.atomic import gulpUtils as gulp
from dislopy.atomic.circleConstruct import ceiling
from dislopy.utilities import atomistic_utils as util
    
def findProp(filename):
    '''Finds the total lattice energy of a GULP simulation (in eV). Returns
    <nan> and a warning if no energy is found.
    '''
    
    # regular expressions to find (in GULP output file) the lattice energy, volume
    # and shear modulus (constant stress, constant strain, average). In the last
    # case we return the difference between G_{voigt} and G_{reuss}
    ELine = re.compile(r'^\s*Total lattice (?:energy|enthalpy)\s+=\s+' +
                '(?P<energy>-?\d+\.\d+)\s*eV')                
    VLine = re.compile('Primitive cell volume =\s+(?P<vol>\d+\.\d+)')
    GLine = re.compile(r'\s*Shear\s+Modulus\s+\(GPa\)\s+=\s+(?P<reuss>-?\d+\.\d+)' +
                    '\s+(?P<voigt>-?\d+\.\d+)\s+(?P<hill>-?\d+\.\d+)')

    gulpFile = util.read_file(filename)
    for line in gulpFile:
        EFound = re.search(ELine,line)
        VFound = re.search(VLine,line)
        GFound = re.search(GLine,line)
        if EFound:
            energy = float(EFound.group('energy'))
        if VFound:
            volume = float(VFound.group('vol'))
        if GFound:
            g_hill = float(GFound.group('hill'))
            delta_g = float(GFound.group('voigt'))-float(GFound.group('reuss'))
            
    try:
        return energy, volume, g_hill, delta_g
    except NameError:
        print('Warning: Properties not found in {}'.format(filename))
        # set properties to have infinite value -> guarantees non-convergence
        return np.inf, np.inf, np.inf, np.inf
        
def optimumWolf(base_name,dampRange,cutRange):
    '''Finds the damping parameter and distance cutoff that gives the best 
    agreement between the energy found using an Ewald sum and the Wolf summation
    method.
    '''
    
    # energy with ewald sum
    ewaldE,ewaldV = findProp('%s.ewald.gout' % base_name)
    ewald = [ewaldE,ewaldV]
    
    delta = []
    prop = []
    
    for d in dampRange:
        for c in cutRange:
            wolfEnergy,wolfVolume = findProp('%s.wolf.%.2f.%.1f.gout'  
                            % (base_name,d,c))
            dE = (wolfEnergy - ewaldE)/ewaldE*100
            dV = (wolfVolume - ewaldV)/ewaldV*100
            delta.append([d,c,dE,dV])
            prop.append([wolfEnergy,wolfVolume])
                
    return delta,prop,ewald

def put_wolf(ostream,xi,rcut):
    '''Inserts the wolf summation line at the end of a GULP input file.
    '''
    
    wolf_line = 'qwolf %.2f %.2f\n' % (xi,rcut)
    ostream(wolf_line)
    return
    
    
def gulp_wolf(base_name, base_file, gulp_exec, xi_min=0.05, xi_max=0.40, rcut_min=10.0,
                                                   rcut_max=25.0, dxi=0.01, dr=0.1):
    '''Runs a series of single-point calculations in GULP with Wolf summation
    parameters given in dampRange and cutRange. Extracts optimum parameters.
    '''
                                         
    xi_range = np.arange(xi_min,xi_max+dxi,dxi)
    rcut_range = np.arange(rcut_min,rcut_max+dr,dr)
    
    for xi in xi_range:
        print('Setting \\xi = %.2f:' % xi)
        for rcut in rcut_range:
            print('r = %.1f' % rcut)
            # create GULP input file for this combination of damping parameter
            # \xi and cutoff radius r_{cut}
            root_name = '%s.wolf.%.2f.%.1f' % (base_name,xi,rcut)
            outstream = open('%s.gin' % root_name,'w')           
            for line in base_file:
                outstream.write('%s\n' % line)
                
            put_wolf(lambda in_line: outstream.write(in_line),xi,rcut)
            outstream.close()
            
            # perform calculation
            gulp.run_gulp(gulp_exec,root_name)
    return
    
def lim(seq):
    '''Returns "limit" of sequence, which we take to be the average value.
    '''
    
    return seq.sum()/len(seq)
    
def epsilon(seq):
    '''Calculates epsilon so that d(x1,x2) <= epsilon for all x1,x2.
    '''
    
    N = len(seq)
    width = seq.max() - seq.min()
    return width
    
def rcut_properties(base_name, xi, rcut_min=10.0, rcut_max=25.0, dr=0.1):
    '''Read properties (as a function of r_{cut}) from GULP output files
    corresponding to a specified damping parameter xi.
    '''
    
    rcut_range = np.arange(rcut_min,rcut_max+dr,dr)
    ewald_E,ewald_V,ewald_G,ewald_dG = findProp('%s.ewald.gout' % base_name)
    ewald_prop = np.array([ewald_E,ewald_V,ewald_G,ewald_dG])
    wolf_prop = []
    
    for rcut in rcut_range:
        wolf_E,wolf_V,wolf_G,wolf_dG = findProp('%s.wolf.%.2f.%.1f.gout' 
                                         % (base_name,xi,rcut))
        wolf_prop.append([wolf_E,wolf_V,wolf_G,wolf_dG])
        
    return ewald_prop,np.array(wolf_prop)
    
def converge(ew_val, wolf_seq,threshold=1e-6, setback=10):
    '''Tests to see if the sequence <wolf_seq> converges to within <threshold>.
    '''
    
    # calculate the normalization to use when testing for convergence
    if abs(ew_val) > 0.1:
        normalization = abs(ew_val)
    else: # actual value very small -> just use absolute error
        normalization = 1.
    
    for i in range(len(wolf_seq)-setback):
        width = epsilon(wolf_seq[i:])/normalization
        if width < threshold:
            return i,lim(wolf_seq)
    
    # does not converge      
    return -1,np.nan
    
def find_par(base_name, xi_range, rcut_range):
    '''Finds best Wolf summation parameters.
    '''
    
    n_damp = len(xi_range)
    err_v = []; index_v = []
    err_g = []; index_g = []
    err_dg = []; index_dg = []
    for xi in xi_range:
        ew_par,wolf_par = rcut_properties(base_name,xi)
        i_v,v = converge(ew_par[1],wolf_par[:,1])
        i_g,g = converge(ew_par[2],wolf_par[:,2])
        i_dg,dg = converge(ew_par[3],wolf_par[:,3])
        index_v.append(i_v)
        index_g.append(i_g)
        index_dg.append(i_dg)
        if v == v:
            err_v.append(abs((v-ew_par[1])/ew_par[1]))
        else:
            err_v.append(np.nan)
        if g == g:
            err_g.append(abs((g-ew_par[2])/ew_par[2]))
        else:
            err_g.append(np.nan)
        if dg == dg:
            err_dg.append(abs((dg-ew_par[3])/ew_par[3]))
        else:
            err_dg.append(np.nan)
            
    # determine best xi and rcut to use
    max_err = np.inf
    best_index = -1
    for i in range(len(xi_range)):
        if index_v[i] == -1 or index_g[i] == -1 or index_dg[i] == -1:
            continue
        # else
        err_i = max(err_v[i],err_g[i],err_dg[i])
        if err_i < max_err:
            max_err = err_i
            best_index = i
            
    # error at best parameters
    errors = (err_v[best_index],err_g[best_index],err_dg[best_index])
    print('Best parameters (rcut to next greatest integer): ')
    print('\\xi = %.2f; ' % xi_range[best_index],end='')
    print('r_{cut} = %.1f' % ceiling(rcut_range[best_index]))
    return xi_range[best_index],rcut_range[best_index],errors              



