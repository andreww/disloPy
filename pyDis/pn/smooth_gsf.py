#!/usr/bin/env python

import numpy as np
from scipy.interpolate import RectBivariateSpline

def read_gamma_line():
    '''Reads the GULP energies for GSFs along a gamma line.
    '''
    
    return

def read_gamma_surface():
    '''Reads in the calculated energies for GSFs on a gamma surface.
    '''
    
    return
    
# Plotting functions -> Two options, 3D surface plot and contour plot

def 3d_gamma_surface(X,Y,Z,xlabel,ylabel,zlabel,*size):
    '''Plots a gamma surface in 3D.
    '''
    
    fig = plt.figure(figsize=mm2inch(size))
    ax = fig.gca(projection='3d')
    surf  = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,linewidth=0.3,
                cmap=cm.coolwarm,shade=True,alpha=1)
    fig.colorbar(surf,shrink=0.7,aspect=12)
    
    ax.set_xlim(X.min(),X.max())
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlabel(xlabel,family='serif',size=12)
    ax.set_ylabel(ylabel,family='serif',size=12)
    ax.set_zlabel(zlabel,family='serif',size=12)
       
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=20.,azim=-15)
    
    plt.tight_layout(pad=0.5)
    plt.show()
    return fig,ax
    
def contour_plot(X,Y,Z,xlabel,ylabel,size):
    '''Two dimensional contour plot of gamma surface height.
    '''
    
    fig = plt.figure(figsize=size)
    plt.contourf(X,Y,Z,100,cmap=cm.afmhot)
    plt.set_xlabel(xlabel,family='serif',size=12)
    plt.set_ylabel(ylabel,family='serif',size=12)
    plt.set_xticks([])
    plt.set_yticks([])
    plt.colorbar()
    return fig
    
    
    

