#!/usr/bin/env python

import numpy as np

from pyDis.atomic import crystal as cry
from pyDis.atomic import castep_utils as cas
from pyDis.atomic import atomistic_utils as atm
from pyDis.pn import  gsf_setup as gsf

path_to_phased = './'
sc_length = 6 # Number of unit cells along z
              # must be an even number in this case.
resolution = 0.33 # Spacing of points in x and y

phaseD = cas.CastepCrystal()
sys_info = cas.parse_castep('phaseD.cell', phaseD,
             path=path_to_phased)
phaseD.translate_cell(0.02*cry.ei(3))

slab = gsf.make_slab(phaseD,sc_length, free_atoms=['Mg', 'O', 'H'])
atm.scale_kpoints(sys_info['mp_kgrid'],np.array([1,1,sc_length]))

gsf.gamma_surface(slab,resolution,cas.write_castep,sys_info,
                   basename='gsf_phaseD',suffix='cell', limits=(0.5, 0.5),
                   mkdir=True)
