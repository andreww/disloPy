#!/usr/bin/env python
from __future__ import print_function
import numpy as np

import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic import crystal as cry
from pyDis.atomic import qe_utils as qe
from pyDis.pn import gsf_setup as gsf
from pyDis.atomic import transmutation as imp
from pyDis.pn import slab_impurity as sl
from pyDis.atomic import atomistic_utils as atm
from pyDis.atomic import gulpUtils as gulp

def main(argv):

    basefile = argv[0]
    vacuum = 15.
    d_fix = 2.8
    
    struc = cry.Crystal()
    sys_info = gulp.parse_gulp(basefile, struc)
    new_struc = cry.superConstructor(struc, dims=np.array([2, 2, 1]))
    new_slab = gsf.make_slab(new_struc, 8, vacuum, d_fix=d_fix, free_atoms=['H'])
    #atm.scale_kpoints(sys_info["cards"]["K_POINTS"], np.array([2, 2, 6]))

    # replace here to make new impurity
    dfct = imp.Impurity('Mg', 'vac')
    #dfct.addAtom(cry.Atom('H', coordinates=np.array([0.125, 0.0, 0.03])))
    #dfct.addAtom(cry.Atom('H', coordinates=np.array([-0.125, 0.0, -0.03])))
    
    # find and replace appropriate atom
    i = sl.replace_at_plane(new_slab, dfct, vacuum=15.)[0]
    dfct.set_index(i)
    sl.impure_faults(new_slab, dfct, gulp.write_gulp, sys_info, 0.1, argv[1], 
                         dim=1, limits=0.25, vacuum=vacuum, suffix='gin')
    

if __name__ == "__main__":
    main(sys.argv[1:])

