#!/usr/bin/env python
'''Routines to calculate the elastic energy coefficient for a Peierls-Nabarro
calculation, using either isotropic elasticity or the anisotropic Stroh 
sextic eigenvalue formulation.
'''
from __future__ import absolute_import

from numpy.linalg import norm

import numpy as np
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic import aniso
from pyDis.atomic import crystal as cry

# conversion factor to take elastic properties from GPa to atomic units
GPa_To_Atomic = 160.2176

def isotropic_K(K, G, using_atomic=False):
    '''Calculate the shear and edge dislocation energy coefficient from the
    isotropic bulk and shear moduli.
    '''
    
    # calculate the isotropic Poisson's ratio for the material
    nu = (3*K-2*G)/(2*(3*K+G))
    return isotropic_nu(nu, G, using_atomic=using_atomic)
    
def isotropic_nu(nu, G, using_atomic=False):
    '''Calculate the shear and edge dislocation energy coefficients from the
    isotropic shear modulus and the Poisson's ratio.
    '''
    
    Ks = G/(4*np.pi)
    Ke = Ks/(1-nu)
    if using_atomic:
        pass
    else:
        Ks /= GPa_To_Atomic
        Ke /= GPa_To_Atomic

    return [Ke, Ks]
    
def anisotropic_K(Cij, b_edge, b_screw, normal, using_atomic=True):
    '''Calculate the energy coefficient for a dislocation with burgers vector
    <b> and sense vector (ie. -dislocation line vector) n <cross> m. As readCij
    outputs the elastic constants in atomic units, defaults to <using_atomic>.
    '''
    
    # use unit vectors in the direction of the burgers vector and the normal
    # to the slip plane (ie. <b_edge> and <normal>) when solving the sextic 
    # eigenvalue problem.
    p, A, L = aniso.solve_sextic(Cij, b_edge/norm(b_edge), normal/norm(normal))
    energy_tensor = aniso.tensor_k(L)
    
    # calculate the scalar edge and screw energy coefficients
    Ke = aniso.scalar_k(energy_tensor, b_edge)
    Ks = aniso.scalar_k(energy_tensor, b_screw)

    return [Ke, Ks]
    
def anisotropic_K_b(Cij, b, n=cry.ei(1), m=cry.ei(2), using_atomic=True):
    '''Caluculate the energy coefficient for a dislocation with a specific 
    Burgers vector.
    '''
    
    # solve sextic eigenvalue problem
    p, A, L = aniso.solve_sextic(Cij, n, m)
    energy_tensor = aniso.tensor_k(L)
    
    # calculate the scalar energy coefficient
    K  = aniso.scalar_k(energy_tensor, b)
    
    return K
    
def predefined(kpara, knorm, using_atomic=False):
    '''Casts <kpara> and <knorm> in the form (and units) required by the P-N
    routines in pyDis.
    '''
    
    k1 = kpara/(4*np.pi)
    k2 = knorm/(4*np.pi)
    
    # put in eV/ang.^3
    if not using_atomic:
        k1 /= GPa_To_Atomic
        k2 /= GPa_To_Atomic
        
    return [k1, k2]
