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
from pyDis.atomic import multisite as ms

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
    
    # find and replace appropriate atom
    i = sl.replace_at_plane(new_slab, dfct, vacuum=15.)[0]
    dfct.set_index(i)
    
    # add hydrogen atoms and replace the appropriate oxygen atoms with hydroxyl
    # oxygens
    dfct.addAtom(gulp.GulpAtom('H', coordinates=np.array([-0.707, 0., -0.707])))
    dfct.addAtom(gulp.GulpAtom('H', coordinates=np.array([0.707, 0., 0.707])))
    dfct.to_cell_coords(new_slab)
    dfct.site_location(new_slab)
    
    fulldfct = ms.hydroxyl_oxygens(dfct, new_slab, 'O2', oxy_str='O1', oned=False,
                                    to_cart=False)
    
    # insert defect into slab and calculate generalised stacking fault energies
    sl.cluster_faults(new_slab, fulldfct, gulp.write_gulp, sys_info, 0.1, argv[1], 
                         dim=1, limits=0.25, vacuum=vacuum, suffix='gin')
    

if __name__ == "__main__":
    main(sys.argv[1:])

