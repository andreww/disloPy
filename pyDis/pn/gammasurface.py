#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/home/richard/code_bases/dislocator2/pyDis/pn')
from fit_gsf import read_numerical_gsf, spline_fit2d

def plot_gamma_surface(gsf_file, a, b, plotname='gamma_surface.tif', 
                                                     angle=np.pi/2.):
    
    # read in gridded data and units                                                
    gsf_grid, units = read_numerical_gsf(gsf_file)
    print(gsf_grid)
    
    gsf_func = spline_fit2d(gsf_grid, a, b, angle=angle, two_planes=True)
    
    #!!! Should make the number of increments adaptive; still this probably
    #!!! provides sufficient sampling density for display purposes
    x = np.linspace(0., a, 100)
    y = np.linspace(0., b, 100)
    
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(np.shape(X))
    
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i, j] = gsf_func(X[i, j], Y[i, j])
            
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    #cset = ax.contourf(X, Y, Z, 50, zdir='z', cmap=cm.coolwarm, offset=-0.1)
    
    ax.plot_surface(X, Y, Z, cstride=2, rstride=2, cmap=cm.coolwarm)
    ax.set_zlim(0, 0.3)
    plt.ylabel(r'GSF energy')
    plt.show()
    
    fig.save_fig(plotname)
    
def main(argv):
    if len(argv) == 4:
        plotname = argv[3]
    else:
        plotname = 'gamma_surface.tif'
        
    plot_gamma_surface(argv[0], float(argv[1]), float(argv[2]), plotname=plotname)
    
if __name__ == "__main__":
    main(sys.argv[1:])
    
