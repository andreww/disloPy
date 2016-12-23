#!/usr/bin/env python
from __future__ import print_function

import os
import sys
sys.path.append(os.environ['PYDISPATH'])

import numpy as np

from pyDis.atomic import crystal as cry
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import transmutation as mut
from pyDis.atomic import multisite as ms
from pyDis.atomic import rodSetup as rs

# path to GULP executable (set this to the path appropriate for your computer)
gulpexec = os.environ['GULPPATH']

def main(argv):

    # name of .grs file containing relaxed dislocation structure
    grsfile = argv[0]
    r1 = int(argv[1])
    r2 = int(argv[2])
    # height of simulation cell (line vector lengths)
    n = int(argv[3])
    new_rI = int(argv[4])
    
    # construct cluster
    base_clus, sysinfo = gulp.cluster_from_grs(grsfile, r1, r2, new_rI=new_rI)
    new_clus = rs.extend_cluster(base_clus, n)
    
    # construct hydrous defect
    dfct = mut.Impurity('Mg', 'hydrous_dfct')
    dfct.addAtom(gulp.GulpAtom('H', coordinates=np.array([0, 1, 0])))
    dfct.addAtom(gulp.GulpAtom('H', coordinates=np.array([0, -1, 0])))
    
    # construct height constraint function
    z = new_clus.getHeight()
    z_constraint = lambda atom: mut.heightConstraint(0, z/n, atom, period=z)
    phi_constraint = lambda atom: mut.azimuthConstraint(-np.pi/2, np.pi/2, atom)
    
    # construct defect-bearing clusters and calculate adsorption energies
    ms.calculate_hydroxyl(sysinfo, new_clus, 3, dfct, oh_str='O2', o_str='O1',
                     constraints=[z_constraint, phi_constraint], do_calc=True, 
					                centre_on_impurity=True, gulpexec=gulpexec)   

if __name__ == "__main__":
    main(sys.argv[1:])

