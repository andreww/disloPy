#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('/home/richard/code_bases/')

from pyDis.atomic import wolf

def main(argv):

    xi_min=0.05
    xi_max=0.40
    rcut_min=10.0
    rcut_max=25.0
    dxi=0.01
    dr=0.1

    xi_range = np.arange(xi_min,xi_max+dxi,dxi)
    rcut_range = np.arange(rcut_min,rcut_max+dr,dr)
    
    basename = argv[0]
    
    wolf.find_par(basename, xi_range, rcut_range)
    
if __name__ == "__main__":
    main(sys.argv[1:])
