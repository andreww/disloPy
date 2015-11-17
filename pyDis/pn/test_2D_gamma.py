#!/usr/bin/env python

import numpy as np
from math import sqrt

def test_gamma_surf(ux,uy,alpha=0.1,U=1.):
    return U*((1-alpha)*E1(ux,uy) + alpha*E2(ux,uy))
    
def E1(ux,uy):
    upx = u_prime(ux)    
    return (f(2*upx) + f(upx+uy) + f(upx-uy))/3.

def E2(ux,uy):
    uppx = u_prime2(ux)
    return 0.5+4./(3*sqrt(3))*g(2*uppx)*g(uppx+uy)*g(uppx-uy)
    
def u_prime(ux):
    return ux/sqrt(3) - 2./3.
   
def u_prime2(ux):
    return ux/sqrt(3) - 1./3.

def f(x):
    num = np.exp(np.cos(2*np.pi*x))-1/sqrt(np.e)
    return num/(np.e-1/sqrt(np.e))
    
def g(x):
    return np.sin(np.pi*x)
