#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import numpy.linalg as lin
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

from ..atomic import aniso

def stroh_block(a,b,Cij):
    return aniso.strohMatrix(a,b,Cij)

def stroh_matrix(n,m,Cij):
    '''Construct the N matrix from Stroh theory using the method on
    page 468 of Hirth and Lothe's "Theory of Dislocations".
    '''
    # initialise the 6x6 N matrix
    N = np.zeros((6,6))
    # Calculate (nn), (nm), (mn), and (mm)
    nn = stroh_block(n,n,Cij)
    nm = stroh_block(n,m,Cij)
    mn = stroh_block(m,n,Cij)
    mm = stroh_block(m,m,Cij)
    # calculate the upper left (UL), upper right (UR), lower left (LL),
    # and lower right (LR) blocks of the N matrix
    UL = -np.dot(lin.inv(nn),nm)
    UR = -lin.inv(nn)
    LL = -(np.dot(np.dot(mn,lin.inv(nn)),nm)-mm)
    LR = -np.dot(mn,lin.inv(nn))
    # assign values to corresponding elements of N
    for i in range(3):
        for j in range(3):
            N[i,j] = UL[i,j]
            N[i,j+3] = UR[i,j]
            N[i+3,j] = LL[i,j]
            N[i+3,j+3]  = LR[i,j]
    return N


def solve_sextic(n,m,Cij):
    '''Finds eigenvalues and eigenvectors of the N matrix in the Stroh
    sextic eignevalue formulation of dislocations in anisotropic 
    elasticity. Rearranges roots p and vectors A and L to have roots
    with Im(p) > 0 listed first. 
    '''
    # index list used to list eigenvalues with Im(p) > 0 first
    ordered_indices = np.array([0,2,4,1,3,5])
    p,xsi = lin.eig(stroh_matrix(n,m,Cij))
    nEig = len(p)
    # reorder p
    p = np.array([p[i] for i in ordered_indices])
    # Extract (non-normalised) A and L vectors, in correct order
    ATilde = np.array([xsi[:3,i] for i in ordered_indices])
    LTilde = np.array([xsi[3:,i] for i in ordered_indices])
    # normalise ATilde and LTilde to obey the condition 2*(Ai.Li) = 1
    normalisation = np.array([1./cmath.sqrt(2*np.dot(ATilde[i],LTilde[i]))
                                for i in range(nEig)])
    A = np.array([normalisation[i]*ATilde[i] for i in range(nEig)])
    L = np.array([normalisation[i]*LTilde[i] for i in range(nEig)])
    return p,A,L

def dyadic(vec):
    n = len(vec)
    dyad = np.zeros((n,n),dtype=complex)
    for i in range(n):
        for j in range(n):
            dyad[i,j] = vec[i]*vec[j]
    return dyad
    
def tensor_k(L):
    K = np.zeros((3,3),dtype=complex)
    for i in range(3):
        K += dyadic(L[i])
        K -= dyadic(L[i+3])
    return (1j*K).real


