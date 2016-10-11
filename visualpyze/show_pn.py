#!/usr/bin/env python
'''Routines to visualize the atomic structure of a dislocation whose core 
structure has been calculated using the Peierls-Nabarro method.
'''
from __future__ import print_function

import numpy as np
import sys
import os
sys.path.append(os.environ['PYDISPATH'])
import matplotlib.pyplot as plt

from pyDis.atomic import crystal as cry
from pyDis.atomic import aniso as an
from pyDis.atomic import atomistic_utils as atm
from pyDis.pn import pn_1D as pn1
from pyDis.pn import pn_2D as pn2

def densify(rho, r, dx=0.1):
    '''Upsample the dislocation density distribution <rho> to give values at 
    intervals of <dx>.
    '''
    
    # sample the spatial locations of the distribution densely 
    r_dense = np.arange(r.min(), r.max(), dx)
    rho_dense = np.zeros(len(r_dense))
    
    # sample the dislocation distribution
    for i, x in enumerate(r_dense):
        low_density_index = np.where(r > x)[0].min()
        rho_dense[i] = rho[low_density_index]
   
    return r_dense, rho_dense
    
def restrict_rho(rho, r, threshold_ratio=1e-2):
    '''Only use those parts of the dislocation distribution where the ratio of 
    the local dislocation density to the maximum density of the distribution 
    exceeds some threshold.
    '''
    
    above_threshold = np.where(rho/rho.max() > threshold_ratio)[0]
    # filter rho and r
    rhobig = rho[above_threshold]
    rbig = r[above_threshold]
    
    return rbig, rhobig


def integrate_field(x, rho, r, elasticfield):
    '''Calculate the total elastic displacment due to the dislocation 
    distribution <rho> (with elastic displacement field <elasticfield>.
    '''

    # width of a partial used in the integration
    dx = r[1] - r[0]
    
    # add displacement fields of all partials together
    displace = np.zeros(3)
    for xi, partial in zip(r, rho):
        xi = np.array([xi, 0., 0.])
        displace += partial*elasticfield(x, xi)*dx
        
    return displace

def inelastic_displacement(x, u, r, disl_type):
    '''Displaces atoms above the dislocation glide plane according to
    the calculated misfit.
    '''
    # 
    # only displace atoms above the glide plane
    if x[1] >= 0:
        return np.zeros(3)

    # if atom's x coordinate is > r_{i} and < r_{i+1}, displace the 
    # atom by amount u_{i}
    node = np.where(r <= x[0])[0].max()
    disp_amount = u[node]
    
    # construct inelastic displacement vector appropriate to the 
    # dislocation character (ie. edge or screw)
    if disl_type == 'edge':
        return np.array([disp_amount, 0., 0.])
    elif disl_type == 'screw':
        return np.array([0., 0., disp_amount])
    else: 
        raise ValueError("Dislocation type not supported.")

def pn_displacement(x, r, edgefield=None, screwfield=None, uedge=None,
                                                          uscrew=None):
    '''Calculates the total (ie. elastic + inelastic) displacement
    field at <x> due to the dislocation density distribution characterised
    by the edge and screw misfit profiles <uedge> and <uscrew>, located
    along the glide plane with normal [0, 1, 0].

    #!!! May be possible to merge definition for edge and screw
    '''

    # calculate inelastic displacement due to edge and screw components
    # of the displacement
    if not (uedge is None):
        # check that elastic displacement field form has been provided
        if edgefield is None:
            raise ValueError("Edge field cannot be <None>.")
            
        # calculate inelastic displacement
        inelast_e = inelastic_displacement(x, uedge, r, 'edge')
    else:
        inelast_e = np.zeros(3)

    if not (uscrew is None):
        # check that screw displacement field form has been provided
        if screwfield is None:
            raise ValueError("Screw field cannot be <None>.")
            
        # calculate inelastic displacement
        inelast_s = inelastic_displacement(x, uscrew, r, 'screw')
    else:
        inelast_s = np.zeros(3)

    # total inelastic displacement
    inelast = inelast_e + inelast_s
    
    # calculate the elastic displacement field
    # calculate dislocation density
    if not (uedge is None):
        rho_e = pn1.rho(uedge, r)
        #rd_e, rhod_e = densify(rho_e, r[1:])
        
        # restrict integration to region with substantial dislocation density
        rsig_e, rhosig_e = restrict_rho(rho_e, r)
        elast_e = integrate_field(x+inelast, rhosig_e, rsig_e, edgefield)
    else:
        elast_e = np.zeros(3)

    if not (uscrew is None):
        rho_s = pn1.rho(uscrew, r)
        #rd_s, rhod_s = densify(rho_s, r[1:], r[1]-r[0])
        # filter
        rsig_s, rhosig_s = restrict_rho(rhod_s, rd_s)
        elast_s = integrate_field(x+inelast, rhosig_s, rsig_s, screwfield)
    else:
        elast_s = np.zeros(3)
        
    elast = elast_e + elast_s
    return elast + inelast
    
def pn_core(input_crystal, r, xyzname, edgefield=None, screwfield=None, uedge=None,
             uscrew=None, normal_shift=0., display_radius=None):
    '''Displaces all atoms in the <input_crystal> by an amount specified by 
    an input Peierls-Nabarro solution, and outputs the result to a .xyz file. 
    <normal> is the shift that should be applied to the coordinates of all
    atoms (along the glide plane normal).
    '''
    
    # construct shift vector
    shift = np.array([0., normal_shift, 0.])
           
    for atom in input_crystal:
        x = atom.getCoordinates()
        x += shift
        
        # calculate the displacement due to the presence of finite dislocation
        # density
        dx = pn_displacement(x, r, edgefield=edgefield, screwfield=screwfield,
                                                   uedge=uedge, uscrew=uscrew)
        atom.setDisplacedCoordinates(x + dx)
        
    atm.write_xyz(input_crystal, xyzname, defected=True, r=display_radius)
    return
        
        
        
        
        
