#!/usr/bin/env python
from __future__ import print_function

import numpy as np

### 1D stuff ####

def integrate_adaptive(f, x0, x1, eps=1e-6):
    s = trap_n(f, x0, x1, 0, 0)
    for n in range(1, int(1e3)):
        olds = s
        s = trap_n(f, x0, x1, s, n)
        if abs(s-olds) < eps*abs(olds):
            return s
    # if we get here, the integrand has not converged
    return np.nan

def trap_n(f, x0, x1, s, n):
    dx0 = x1-x0
    if i ==  0:
        s = 0.5*(f(x0) + f(x1))*dx0
    else:
        tnm = i
        dx = dx0/float(i)
        x0i = x0 + 0.5*dx
        add_sum = 0
        for j in range(i):
            add_sum += f(x0i + j*dx)
        s = 0.5*(s+add_sum*dx)
    return s

def an(f, T, w0, n, eps=1e-6):
    g = lambda t: f(t)*np.exp(n*w0*t*1j)
    an_val = 2./T*integrate_adaptive(g, 0, T, eps=eps)
    return an_val.real


def fourier_cosine(f, T, n):
    # make <an> coefficients
    w0 = 2*np.pi/T
    an_i = [an(f, T, w0, i) for i in range(n+1)]
    g = []
    g.append(lambda x: an_i[0]/2.)
    def fourier_approx(t):
        approx = an_i[0]/2.
        for i in range(1, n+1):
            approx += an_i[i]*np.exp(1j*i*w0*t)
        return approx

    # shift baseline
    f0 = fourier_approx(0)
    fourier_shifted = lambda x: (fourier_approx(x) -f0).real
    return fourier_shifted

### 2D stuff ###

def int_mc(f, xmax, ymax, npoints=int(1e5)):
    xsamp = np.random.uniform(0, xmax, size=npoints)
    ysamp = np.random.uniform(0, ymax, size=npoints)
    num = f(xsamp, ysamp).sum()
    return num/npoints

def normalise_2d(n, m, T1, T2):
    def norm_func(x, y):
        return np.cos(2*np.pi*x*n/T1)**2*np.cos(2*np.pi*y*m/T2)**2
    return norm_func

def convolve_with_cosines(h, n, m, T1, T2):
    def f_conv(x, y):
        return h(x, y)*np.cos(2*np.pi*x*n/T1)*np.cos(2*np.pi*y*m/T2)
    return f_conv

def fnm(f, n, m, T1, T2, xmax, ymax):
    integ = convolve_with_cosines(f, n, m, T1, T2)
    return int_mc(integ, xmax, ymax)

def gnm(n, m, T1, T2, xmax, ymax):
    integ = normalise_2d(n, m, T1, T2)
    return int_mc(integ, xmax, ymax)

def coeff(f, n, m, T1, T2, xmax, ymax):
    num = fnm(f, n, m, T1, T2, xmax, ymax)
    denom = gnm(n, m, T1, T2, xmax, ymax)
    return num/denom

def fourier_coeffs2d(f, N, T1, T2, xmax, ymax):
    '''Generate all fourier coefficients with n <= N and
    m <= M (ie. assume symmetry).
    '''
    coeffs = np.zeros((N+1, N+1))
    for n in range(N+1):
        for m in range(N+1):
            coeffs[n, m] = coeff(f, n, m, T1, T2, xmax, ymax)
    return coeffs

def fourier_cosine2d(coeffs, T1, T2):
    w01 = 2*np.pi/T1
    w02 = 2*np.pi/T2
    def fourier_approx(x, y):
        approx = 0.0
        for n in range(len(coeffs)):
            for m in range(len(coeffs[0, :])):
                approx += coeffs[n, m]*np.cos(n*w01*x)*np.cos(m*w02*y)
        return approx
    
    # shift energy value to be zero at lattice points
    f0 = fourier_approx(0, 0)
    fourier_shifted = lambda x, y: (fourier_approx(x, y)-f0)
    return fourier_shifted

