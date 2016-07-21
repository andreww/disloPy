#!/usr/bin/env python
'''Inserts defects into a gsf cell to calculate influence of point defects on 
the Peierls stress.
'''
from __future__ import print_function

import numpy as np
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

from numpy.linalg import norm

from pyDis.atomic import crystal as cry
from pyDis.atomic import transmutation as mutate

from pyDis.pn import gsf_setup as gsf

def replace_at_plane(slab_cell, impurity, plane=0.5, vacuum=0.,
                        constraints=None, eps=1e-12, height=0.):
    '''Inserts an impurity on the <plane> in <slab_cell>
    '''
    # calculate location of plane, accounting for the possible presence
    # of a vacuum layer
    midpoint = plane*((norm(slab_cell.getC())-vacuum)/
                                norm(slab_cell.getC()))
    to_substitute = []
    for i, atom in enumerate(slab_cell):
        use_atom = False
        if atom.getSpecies() == impurity.getSite():
            # work out if the atom is on the <plane> and satisfies the 
            # <constraints>
            if -eps < atom.getCoordinates()[-1]-midpoint < eps+height:
                use_atom = True 
                if not constraints:
                    pass
                else:
                    for constraint in constraints:
                        if not constraint(atom):
                            use_atom = False
            if use_atom:
                to_substitute.append(i)

    return to_substitute

def impure_faults(slab_cell, impurity, write_fn, sys_info, resolution, 
                      basename, dim=2, limits=(1,1), mkdir=False, vacuum=0., 
                      suffix='in', line_vec=np.array([1., 0., 0.]), relax=None):
    '''Runs gamma surface calculations with an impurity (or impurity cluster)
    inserted at the specified atomic sites. <site> gives the index of the atom
    to be replaced by the impurity 
    '''

    mutate.cell_defect(slab_cell, impurity, use_displaced=True)
    
    # create input files for generalised stacking fault calculations
    if dim == 1:
        gsf.gamma_line(slab_cell, line_vec, resolution, write_fn, sys_info, 
                       limits=limits, vacuum=vacuum, basename=basename, suffix=suffix,
                       mkdir=False, relax=relax)
    else: # dim == 2
        gsf.gamma_surface(slab_cell, resolution, write_fn, sys_info, mkdir=mkdir,
                                 basename=basename, suffix=suffix, limits=limits)
                                 
    # return the slab to its original state
    mutate.undo_defect(slab_cell, impurity)
    
    return
    
def cluster_faults(slab_cell, defectcluster, write_fn, sys_info, resolution, 
                      basename, dim=2, limits=(1,1), mkdir=False, vacuum=0., 
                      suffix='in', line_vec=np.array([1., 0., 0.]), relax=None):
    '''Insert a defect cluster into a supercell and then calculate gamma
    line/surface energies.
    '''
    
    # insert the defect cluster into the simulation cell
    mutate.cell_defect_cluster(slab_cell, defectcluster, use_displaced=True)
      
    # create input files for generalised stacking fault calculations
    if dim == 1:
        gsf.gamma_line(slab_cell, line_vec, resolution, write_fn, sys_info, 
                       limits=limits, vacuum=vacuum, basename=basename, suffix=suffix,
                       mkdir=False, relax=relax)
    else: # dim == 2
        gsf.gamma_surface(slab_cell, resolution, write_fn, sys_info, mkdir=mkdir,
                                 basename=basename, suffix=suffix, limits=limits)
    
    # return the slab to its original state
    mutate.undo_defect(slab_cell, defectcluster)
    
    return
