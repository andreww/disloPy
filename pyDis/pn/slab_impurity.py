#!/usr/bin/env python
'''Inserts defects into a gsf cell to calculate influence of point defects on 
the Peierls stress.
'''
from __future__ import print_function

import numpy as np
import sys
import os
sys.path.append('/home/richard/code_bases/dislocator2/')

from numpy.linalg import norm

from pyDis.atomic import crystal as cry
from pyDis.atomic import transmutation as mutate

import gsf_setup as gsf

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

def insert_impurities(slab_cell, impurity, site_list, write_fn, sys_info, grid, 
                          prefix='gsf', suffix='in', limits=(1,1), topdir=True,
                                                                   mkdir=False):
    '''Runs gamma surface calculations with impurities inserted at the atomic 
    sites enumerated in <site_list>. <topdir> determines whether or not a new 
    directory is created for each possible impurity site.
    '''
    
    for index in site_list:
        if topdir:
            # create directory for impurity calculations
            dir_name = '%s.%d' % (prefix, index)
            os.mkdir('%s' % dir_name)
            os.chdir(dir_name)

        slab_cell[index].switchOutputMode()
    
        # calculate coordinates of the <impurity> atoms and insert them into the 
        # slab
        impurity.site_location(slab_cell[index])
        if len(impurity) == 0:
            # impurity contains no atoms => we are inserting a vacancy
            pass
        else:
            for atom in impurity:
                slab_cell.append(atom)

        # create input files for generalised stacking fault calculations
        gsf.gamma_surface(slab_cell, grid, write_fn, sys_info, mkdir=mkdir,
            basename='%s.%d' % (prefix, index), suffix=suffix, limits=limits)

        # return <index>th atom to slab and delete all impurity atoms
        slab_cell[index].switchOutputMode()
        for i in range(len(impurity)):
            del slab_cell[-1]

        if topdir:
            os.chdir('..')

    return


