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
gulpexec = '/home/richard/programs/atomistic/gulp/Src/gulp'

def main(argv):

    # name of .grs file containing relaxed dislocation structure
    grsfile = argv[0]
    r1 = int(argv[1])
    r2 = int(argv[2])
    # height of simulation cell (line vector lengths)
    n = int(argv[3])
    
    # construct cluster
    base_clus, sysinfo = gulp.cluster_from_grs(grsfile, r1, r2)
    new_clus = rs.extend_cluster(base_clus, n)
    
    # construct hydrous defect
    dfct = mut.Impurity('Mg', 'hydrous_dfct')
    dfct.addAtom(gulp.GulpAtom('H', coordinates=np.array([0, 1, 0])))
    dfct.addAtom(gulp.GulpAtom('H', coordinates=np.array([0, -1, 0])))
    
    # construct height constraint function
    z = new_clus.getHeight()
    z_constraint = lambda atom: mut.heightConstraint(0, z/n, atom, period=z)
    
    # construct defect-bearing clusters and calculate adsorption energies
    ms.calculate_hydroxyl(sysinfo, new_clus, 3, dfct, constraints=[z_constraint],
                   oh_str='O2', o_str='O1', gulpexec=gulpexec, do_calc=True)   

if __name__ == "__main__":
    main(sys.argv[1:])

