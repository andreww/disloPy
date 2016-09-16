#!/usr/bin/env python
'''Various fourier series functions.
'''
from __future__ import print_function

import numpy as np

### GENERAL 1D FOURIER SERIES ####

def integrate1d(f, xmax, ngrid=250, xmin=0.):
    '''Integrate function of 1 variable, <f>, from <xmin> to <xmax>.
    '''
    
    x = np.linspace(xmin, xmax, ngrid)
    integrated_value = f(x).sum()/ngrid
    return integrated_value

def an(f, n, T):
    '''Calculates the coefficient for the n-th cosine component of the fourier
    series approximation to <f>.
    '''
    
    # calculate angular frequency of n-th mode
    wn = 2*np.pi*n/T
    g = lambda x: f(x)*np.cos(wn*x)
    
    # calculate coefficient
    a = 2*integrate1d(g, T)
    return a

def bn(f, n, T):
    '''Calculates the coefficient for the n-th sine component of the fourier
    series approximation to <f>.
    '''
    
    # calculate angular frequency of n-th mode
    wn = 2*np.pi*n/T
    g = lambda x: f(x)*np.sin(wn*x)
    
    # calculate coefficient
    b = 2*integrate1d(g, T)
    return b
    
def coeffs_1d(f, N, T):
    '''Calculates the fourier coefficients of <f> (up to order <n>).
    '''
    
    a_n = np.zeros(N+1)
    b_n = np.zeros(N+1)
    
    # calculate coefficients
    for n in range(N+1):
        a_n[n] = an(f, n, T)
        b_n[n] = bn(f, n, T)
        
    return a_n, b_n

def fourier_series1d(f, N, T):
    '''Calculate the fourier series approximation of order <N> to the function
    <f>, with period <T>.
    '''
    
    # calculate an and bn coefficients
    a_n, b_n = coeffs_1d(f, N, T)
    
    # base angular frequency
    w0 = 2*np.pi/T
    
    # construct fourier series approximation
    def fourier_approx(x):
        approx = a_n[0]/2.
        for n in range(1, N+1):
            approx += a_n[n]*np.cos(n*w0*x)
            approx += b_n[n]*np.sin(n*w0*x)
            
        return approx

    # shift baseline so that function is zero at lattice points
    f0 = fourier_approx(0)
    fourier_shifted = lambda x: (fourier_approx(x) -f0).real
    
    return fourier_shifted

### GENERAL 2D FOURIER SERIES ###

def integrate2d(f, xmax, ymax, ngrid=250, ngridy=None, xmin=0., ymin=0.):
    '''Integrates f(x, y) on the grid (xmin, xmax)x(ymin, ymax), with ngrid
    points along each axis.
    '''
    
    if ngridy == None:
        # set number of grid points along y equal to grid number along x
        ngridy = ngrid
    
    # construct integration grid
    x = np.linspace(xmin, xmax, ngrid)
    y = np.linspace(ymin, ymax, ngridy)
    X, Y = np.meshgrid(x, y)
    
    # note that this value is correct only if the answer is supposed to be
    # normalised by the integration area
    integrated_value = f(X, Y).sum()/(ngridy*ngrid)
    
    return integrated_value

def kappa(n, m):
    '''Returns the prefactor <kappa> for use in calculating the fourier coefficients
    in 2 dimensions.
    '''
    
    if n == 0 and m == 0:
        return 1
    elif n == 0 or m == 0:
        return 2
    else: # n != 0 and m != 0
        return 4

def alpha_nm(f, n, m, T1, T2):
    '''Calculate the coefficient of the (n,m)-th cos*cos term in a 2D fourier
    series approximation to the function f.
    '''
    
    w1 = 2*np.pi/T1
    w2 = 2*np.pi/T2
    int_func = lambda x, y: f(x, y)*np.cos(n*w1*x)*np.cos(m*w2*y)
    anm = kappa(n, m)*integrate2d(int_func, T1, T2)
    return anm

def beta_nm(f, n, m, T1, T2):
    '''Calculate the coefficient of the (n,m)-th cos*sin term in a 2D fourier
    series approximation to the function f. <gamma_nm> is calculated using the
    relation gamma_nm(f, n, m, T1, T2) = beta(f, m, n, T1, T2).
    '''
    
    w1 = 2*np.pi/T1
    w2 = 2*np.pi/T2
    int_func = lambda x, y: f(x, y)*np.cos(n*w1*x)*np.sin(m*w2*y)
    bnm = kappa(n, m)*integrate2d(int_func, T1, T2)
    return bnm
    
def delta_nm(f, n, m, T1, T2):
    '''Calculate the coefficient of the (n,m)-th cos*cos term in a 2D fourier
    series approximation to the function f.
    '''
    
    w1 = 2*np.pi/T1
    w2 = 2*np.pi/T2
    int_func = lambda x, y: f(x, y)*np.sin(n*w1*x)*np.sin(m*w2*y)
    dnm = kappa(n, m)*integrate2d(int_func, T1, T2)
    return dnm
    
def coeffs_2d(f, N, T1, T2):
    '''Calculates all fourier coefficients with n <= N and m <= N.
    '''
    
    anm = np.zeros((N+1, N+1))
    bnm = np.zeros((N+1, N+1))
    cnm = np.zeros((N+1, N+1))
    dnm = np.zeros((N+1, N+1))
    
    # start by calculating components of <anm>, <bnm>, and <dnm>
    for n in range(N+1):
        for m in range(N+1):
            anm[n, m] = alpha_nm(f, n, m, T1, T2)
            bnm[n, m] = beta_nm(f, n, m , T1, T2)
            dnm[n, m] = delta_nm(f, n, m, T1, T2)
            
    # populate <cnm> using the entries of <bnm>
    for n in range(N+1):
        for m in range(N+1):
            cnm[n, m] = bnm[m, n]
            
    return anm, bnm, cnm, dnm
    
def fourier_series2d(f, N, T1, T2):
    '''Constructs the order-N fourier series representation of the function <f>,
    which we assume to be 2D-periodic, with periods <T1> and <T2> in the x1 and
    x2 directions, respectively. 
    '''
    
    # base angular frequencies
    w1 = 2*np.pi/T1
    w2 = 2*np.pi/T2
    
    # calculate fourier series coefficients
    anm, bnm, cnm, dnm = coeffs_2d(f, N, T1, T2)
    
    # construct fourier series
    def fourier_approx(x, y):
        '''Fourier series representation of <f>.
        '''
        
        approx = 0.
        for n in range(N+1):
            for m in range(N+1):
                approx += anm[n, m]*np.cos(n*w1*x)*np.cos(m*w2*y)
                approx += bnm[n, m]*np.cos(n*w1*x)*np.sin(m*w2*y)
                approx += cnm[n, m]*np.sin(n*w1*x)*np.cos(m*w2*y)
                approx += dnm[n, m]*np.sin(n*w1*x)*np.sin(m*w2*y)  
                        
        return approx      
    
    # make sure that f(0, 0) == 0 (otherwise dislocation core energies will 
    # diverge as the integration region increases)
    f0 = fourier_approx(0., 0.)
    fourier_shifted = lambda x, y: fourier_approx(x, y) - f0
    
    return fourier_shifted    
