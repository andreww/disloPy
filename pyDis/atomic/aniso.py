#!/usr/bin/env python
'''Module to calculate the  displacement field and energy coefficient
for a dislocation along z (which must be a high symmetry direction) using
anisotropic elasticity.
'''

import numpy as np
import cmath
import numpy.linalg as lin
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic import crystal as cry

CONV_EV_TO_GPA = 160.2176487

def readCij(basename, path='./'):
    '''Reads the elastic constants matrix Cij (in eV/ang**3)
    from the file <basename.cij>. Note that this procedure is simplest
    if Cij is calculated in coordinate system with xsi (the dislocation line)
    oriented along the z-axis [001].
    '''
    
    suffix = '.cij'
    if suffix in basename:
        CijFile = open(basename,'r')
    else:
        CijFile = open(basename + '.cij','r')
           
    Cij = [row.rstrip().split() for row in CijFile.readlines()]
    
    CijFile.close()
    
    Cij = np.array([[float(constant) for constant in row] for row in Cij])    
    return Cij/CONV_EV_TO_GPA
        
def voightIndices(i, j):
    '''Given a pair of indices ij, work out the corresponding single index I
    in the Voight notation. Note that, since we are using python, i and j run
    from 0 to 2, while I runs from 0 to 5 (inclusive). Recall that ij and ji
    map to the same value.
    '''
    
    if i == 0 and j == 0:
        # 11 -> 1
        return 0
    elif i == 1 and j == 1:
        # 22 -> 2
        return 1
    elif i == 2 and j == 2:
        # 33 -> 3
        return 2
    elif (i == 1 and j == 2) or (i == 2 and j == 1):
        # (23, 32) -> 4
        return 3
    elif (i == 0 and j == 2) or (i == 2 and j == 0):
        # (13, 31) -> 5
        return 4
    elif (i == 0 and j == 1) or (i == 1 and j == 0):
        # (12, 21) -> 6
        return 5
    else:
        print 'Error: Index out of bounds. Exiting...'
        sys.exit(1)
    return        

def Cijkl(i, j, k, l, Cij):
    '''Given a particular element of an elasticity tensor Cijkl, works out
    the corresponding element of the elastic constants matrix <Cij>.
    '''
    
    I = voightIndices(i, j)
    J = voightIndices(k, l)
    return Cij[I, J]
    
def stroh_matrix(a, b, Cij):
    '''Construct the matrix (ab) where (ab)_{jk} = a_{i}C_{ijkl}b_{j}, with
    summation over identical indices implied.
    '''
    
    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            M[i, j] = stroh_element(a, b, i, j, Cij)
    return M
    
def stroh_element(a, b, j, k, Cij):
    '''Work out a single element of the Stroh matrix
    (ab)jk = ai Cijkl bl (Einstein summation implied).
    Note that a and b must be orthonormal to xsi, the 
    unit vector along the dislocation line, and hence the z
    component of both must be equal to zero.
    '''
    
    element = 0
    for i in range(3):
        for l in range(3):
            element += a[i]*Cijkl(i, j, k, l, Cij)*b[l]
    
    return element
    
def nn(Cij, phi=0.):
    return strohMatrix(n(phi), n(phi), Cij)    
    
def mm(Cij, phi=0.):
    return strohMatrix(m(phi), m(phi), Cij)       
    
def mn(Cij, phi=0.):
    return strohMatrix(m(phi), n(phi), Cij)       
    
def nm(Cij, phi=0.):
    return strohMatrix(n(phi), m(phi), Cij)  
    
# Sextic eigenvalue problem   
    
def NStroh(Cij, n, m):
    '''Construct the N matrix from Stroh theory using the method outlined
    on page 468 of Hirth and Lothe's "Theory of Dislocations.
    '''
    
    # Initialise the 6x6 N matrix
    N = np.zeros((6, 6))
    
    # construct the n and m matrices
    nn = stroh_matrix(n, n, Cij)
    mm = stroh_matrix(m, m, Cij)
    nm = stroh_matrix(n, m, Cij)
    mn = stroh_matrix(m, n, Cij)
    
    # Calculate the upper left (UL), upper right (UR), lower left (ll)
    # and lower right (lr) 3x3 blocks of the N matrix using the formulae
    # on page 468 of HL.
    UL = np.dot(lin.inv(nn), nm)
    UR = lin.inv(nn)
    LL = np.dot(mn, np.dot(lin.inv(nn), nm))-mm
    LR = np.dot(mn, lin.inv(nn))
    
    #...assign values to the 4 3x3 blocks in the N matrix
    for i in range(3):
        for j in range(3):
            N[i, j] = UL[i, j]
            N[i, j+3] = UR[i, j]
            N[i+3, j] = LL[i, j]
            N[i+3, j+3] = LR[i, j]
          
    # N should be negative the value determined above       
    N = -N            
    return N
    
def solve_sextic(Cij, n=cry.ei(1), m=cry.ei(2)):
    '''Finds the eigenvalues and eigenvectors of the N tensor in the Stroh
    sextic eigenvalue formulation of anisotropic elasticity. Rearranges
    roots p and vectors A and L to have roots with Im(p) > 0 listed first.
    '''
    
    # Index list used to list eigenvalues with Im(p) > 0 first
    orderedIndices = np.array([0, 2, 4, 1, 3, 5])
    
    p, xsi = lin.eig(NStroh(Cij, n, m))
    
    # number of eigenvalues of N
    nEig = len(p)    
    # reorder p
    p = np.array([p[i] for i in orderedIndices])
    
    # Extract the (non-normalised) A and L vectors, noting that numpy.linalg
    # returns the i-th eigenvector as xsi[:, i]. Ensure that order is same as 
    # for pOrdered (tilde denotes non-normalised)
    ATilde = np.array([xsi[:3, i] for i in orderedIndices])
    LTilde = np.array([xsi[3:, i] for i in orderedIndices])
                                          
    # normalise A and L so that 2*(Ai.Li) = 1
    # begin by working out values of 2*(ATilde_i . LTilde_i)
    normalisation = np.array([1./cmath.sqrt(2*np.dot(ATilde[i], LTilde[i])) 
                                                 for i in range(nEig)])
                                               
    # now multiply each of the ATilde_i and LTilde_i by a normalization factor
    A = np.array([normalisation[i]*ATilde[i] for i in range(nEig)])
    L = np.array([normalisation[i]*LTilde[i] for i in range(nEig)])
    
    return p, A, L

#!!! should move these into <fields.py>
   
def makeAnisoField(Cij, n=cry.ei(1), m=cry.ei(2)):
    '''Given an elastic constants matrix <Cij>, defines a function 
    <uAniso> that returns the displacement at <x> for a dislocation
    with Burgers vector <b>.
    '''
    
    # calculate eigenvalues <p>, and the <A> and <L> vectors defined on pg. 468
    # of Hirth and Lothe (1982).
    p, A, L = solve_sextic(Cij)
    
    def uAniso(x, b, x0, dummy1=0, dummy2=0):
        '''Dummy variables used to ensure that all displacement fields
        have the same form.
        '''
        
        u = 0j*np.zeros(3)
        dx = x[0] - x0[0]
        dy = x[1] - x0[1]
        
        rho2 = dx**2 + dy**2
        
        # make sure that we are not at the line singularity of the dislocation
        coreradius2 = 1e-10
        if (rho2 < coreradius2):
            # inside the dislocation core
            u[0] = 0.
            u[1] = 0.
            u[2] = 0.
        else:
            # calculate displacement associated with each conjugate pair of 
            # eigenvalues/eigenvectors.
            for i in range(3):
                posEig = A[i]*np.dot(L[i], b)*np.log(dx+p[i]*dy)
                negEig = A[i+3]*np.dot(L[i+3], b)*np.log(dx+p[i+3]*dy)
                u = u + (posEig - negEig).copy()
                
        # make real
        u *= 1/(2.*np.pi*1j)
        return u.real
       
    return uAniso   
    
def anisoWedgeDisclination(Cij):
    '''Given an elastic constants matrix <Cij>, defines a function <uAnisoW> 
    that returns the displacement at <x> for a disclination with Frank vector
    <Omega>.
    '''
    
    # calculate eigenvalues and the <A> and <L> vectors from Stroh theory
    p, A, L = solve_sextic(Cij)
    
    def uAnisoW(x, Omega, dummy1=0, dummy2=0):
        '''Dummy variables to ensure compatibility with existing format for
        displacement fields.
        '''
        
        # dislocation density
        b = np.array([0.,-Omega[-1], 0.])
        
        u = 0j*np.zeros(3)
        if abs(x[1]) > 1e-8 or x[0] > 1e-8:
            # calculate displacement for each conjugate pair of eigenvalues
            for i in range(3):
                etaPlus = (x[0]+p[i]*x[1])
                etaMin = (x[0]+p[i+3]*x[1])
                posEig = -A[i]*np.dot(L[i], b)*etaPlus*(np.log(etaPlus)-1)
                negEig = -A[i+3]*np.dot(L[i+3], b)*etaMin*(np.log(etaMin)-1)
                u = u + (posEig-negEig).copy()
            
            # make real
            u *= 1/(2.*np.pi*1j)
        
        else: # abs(x[0]) < 1e-8 and abs(x[1]) < 1e-8
            # calculate displacement after perturbing y coordinate slightly
            x1prime = 1e-7
            for i in range(3):
                etaPlus = (x[0]+p[i]*x1prime)
                etaMin = (x[0]+p[i+3]*x1prime)
                posEig = -A[i]*np.dot(L[i], b)*etaPlus*(np.log(etaPlus)-1)
                negEig = -A[i+3]*np.dot(L[i+3], b)*etaMin*(np.log(etaMin)-1)
                u = u + (posEig-negEig).copy()
            
            # make real
            u *= 1/(2.*np.pi*1j)
            
        return u.real
        
    return uAnisoW

### ENERGY COEFFICIENT FUNCTIONS ###
    
def dyadic(vec1, vec2=None):
    '''Take the dyad product of <vec1> and <vec2>. The dyad product is an 
    '''
    
    if vec2 is None:
        # take the dyad product of <vec1> with itself
        return dyadic(vec1, vec2=vec1)
    else:
        dyad = np.zeros((len(vec1), len(vec2)), dtype=complex)      
        for i in range(len(vec1)):
            for j in range(len(vec2)):
                dyad[i, j] = vec1[i]*vec2[j]
                
        return dyad
    
def tensor_k(L):
    '''Construct the energy coefficient tensor from the vectors <L> obtained by 
    solving the Stroh sextic eigenvalue problem, as described in eqn. 13-189
    of Hirth and Lothe (page 471).
    '''
    
    K = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        K += dyadic(L[i])
        K -= dyadic(L[i+3])
        
    return (1j*K).real
    
def scalar_k(K, b):
    '''Calculate the scalar energy coefficient for a dislocation with burgers
    vector <b>
    '''
    
    e_coeff = 0.
    for i in range(3):
        for j in range(3):
            e_coeff += b[i]*K[i, j]*b[j]          
    e_coeff /= (4*np.pi)
            
    # normalise to obtain the scalar energy coefficient K
    e_coeff /= lin.norm(b)**2
    return e_coeff

def get_isotropic(Cij):
    '''Calculates the isotropic bulk and shear moduli for the material with
    elastic constants matrix Cij. Returns only the VRH average values for each
    of these properties.
    '''
    
    # get elastic compliance matrix
    Sij = np.linalg.invert(Cij)
    
    # calculate Voigt and Reuss bounds on the bulk and shear moduli
    K_v = (Cij[0, 0]+Cij[1, 1]+Cij[2, 2]+2*Cij[0, 1]+2*Cij[0, 2]+2*Cij[1, 2])/9.
    G_v = (Cij[0, 0]+Cij[1, 1]+Cij[2, 2]-Cij[0, 1]-Cij[0, 2]-Cij[1, 2]+3*Cij[3, 3]
           +3*Cij[4, 4]+3*Cij[5, 5])/15.
    
    K_r = 1/(Sij[0, 0]+Sij[1, 1]+Sij[2, 2]+2*Sij[0, 1]+2*Sij[0, 2]+2*Sij[1, 2])
    G_r = 15/(4*Sij[0, 0]+4*Sij[1, 1]+4*Sij[2, 2]-4*Sij[0, 1]-4*Sij[0, 2]-4*Sij[1, 2]
              +3*Sij[3, 3]+3*Sij[4, 4]+3*Sij[5, 5])
    
    # calculate Voigt-Reuss-Hill averages
    K_vrh = 0.5*(K_v + K_r)
    G_vrh = 0.5*(G_v + G_r)
    
    return K_vrh, G_vrh
