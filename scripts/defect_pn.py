#!/usr/bin/env python

import numpy as np

import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

from pyDis.atomic import crystal as cry
from pyDis.atomic import qe_utils as qe
from pyDis.pn import gsf_setup as gsf
from pyDis.atomic import transmutation as imp
from pyDis.pn import slab_impurity as sl
from pyDis.atomic import atomistic_utils as atm

def main(argv):

    basefile = argv[0]
    vacuum = 10.
    d_fix = 2.5
    
    struc = cry.Crystal()
    sys_info = qe.parse_qe(basefile, struc)
    new_struc = cry.superConstructor(struc, dims=np.array([1, 2, 1]))
    new_slab = gsf.make_slab(new_struc, 6, vacuum, d_fix=d_fix, free_atoms=['H'])
    atm.scale_kpoints(sys_info["cards"]["K_POINTS"], np.array([1, 2, 6]))

    # replace here to make new impurity
    dfct = imp.Impurity('Mg', 'water')
    dfct.addAtom(cry.Atom('H', coordinates=np.array([0.125, 0.0, 0.03])))
    dfct.addAtom(cry.Atom('H', coordinates=np.array([-0.125, 0.0, -0.03])))
    
    # find and replace appropriate atom
    i = sl.replace_at_plane(new_slab, dfct, vacuum=10.)[0]
    sl.impure_faults(new_slab, dfct, i, qe.write_qe, sys_info, 0.2, argv[1], 
                         dim=1, limits=0.5, vacuum=vacuum, relax='relax')
    

if __name__ == "__main__":
    main(sys.argv[1:])

