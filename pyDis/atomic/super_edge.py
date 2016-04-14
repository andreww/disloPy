#!/usr/bin/env python
'''Constructs a multipole of edge dislocations in what we assume (for now) to be
a simulation cell with a and b cell parameters perpendicular to one another.
In a quadrupole configuration, the arrangement of dislocations in the
simulation cell

    +    -

    -    +

which corresponds to the following (where we use a plus sign to denote deleted
atoms):

----------------------------------
-       +               +        -
-       +               +        -
-       +               +        -
-                                -
-       +               +        -
-       +               +        -
-       +               +        -
----------------------------------

For a dipole configuration, meanwhile, the arrangement of dislocations can be

    +                           +
                or          
    -                       -

which corresponds to the following:

---------------------------
-            +            -
-            +            -
-                         -
-            +            -
-            +            -
---------------------------

Atoms on either side of the cut regions are then displaced according to some
(as yet undetermined) algorithm. Relax using eg. BFGS, possibly at finite
temperature
'''
from __future__ import print_function

import numpy as np
import sys
from numpy.linalg import norm

def compare(a, b):
    if abs(a) < 1 and abs(b) < 1:
        if abs(a-b) < sys.float_info.epsilon:
            return True
        else:
            return False
    elif abs((a-b)/(0.5*(a+b))) < sys.float_info.epsilon:
        return True
    else:
        return False

class EdgeDipole(object):
    '''Trial class to hold the cuts required to construct a dipole of
    edge dislocations with arbitrary ordering of positive and negative
    Burgers vector directions.
    '''

    def __init__(self, x_neg, x_pos, b):
        '''Set up an edge dislocation dipole with negative Burgers vector
        at x1 and positive Burgers vector at x2. <b> is the Burgers vector 
        magnitude, in units of fractional cell parameters (ie. in a 2x2 
        supercell with b || x, b = 0.5).
        '''

        self.neg = np.copy(x_neg)
        self.pos = np.copy(x_pos)
        if type(b) == float or type(b) == int:
            self.b = b
        else: # user has probably entered a 
            self.b = norm(b)
            
        # work out the orientation of the dislocation
        orient_x = False
        orient_y = False
        if not compare(self.neg[0], self.pos[0]):
            orient_x = True
        if not compare(self.neg[1], self.pos[1]):
            if orient_x:
                raise ValueError('Undefined orientation')
            orient_y = True # else

        if not orient_x and not orient_y:
            # dislocations are co-located
            raise ValueError('Not a valid dipole configuration')

        # we now need to work out whether the cut should be in the middle
        # of the simulation cell (ie. continuous in a single image) or
        # on the edges of the simulation cell (ie. discontinuous across
        # a single image, but continuous across adjacent images).
        if orient_x:
            self.cut_index = 0
            self.b_index = 1
        else:
            self.cut_index = 1
            self.b_index = 0

        if self.neg[self.cut_index] < self.pos[self.cut_index]:
            # delete row in the middle of the cell
            self.continuous = True
        else:
            self.continuous = False

    def incut(self, atom):
        '''Test to see if <atom> lies in the cut defined by x1 and x2.
        '''

        # if the <atom> is already set to not write to output, return
        # immediately
        if not atom.writeToOutput():
            print('Atom {} not written to output. Skipping...'.format(atom))
            return

        x = atom.getCoordinates()
        along_cut = x[self.cut_index]
        along_b = x[self.b_index]

        # test each of the coordinates in turn
        in_cut = False
        if self.continuous:
            if (along_cut > self.neg[self.cut_index] and
                     along_cut < self.pos[self.cut_index]):
                in_cut = True # provisional
        else: # discontinuous cut
            if (along_cut > self.neg[self.cut_index] or
                     along_cut < self.pos[self.cut_index]):
                in_cut = True # provisional

        if in_cut:
            # this part is the same for both continuous and discontinuous
            # cuts
            if (along_b >= self.neg[self.b_index]-self.b/2. and
                      along_b < self.neg[self.b_index]+self.b/2.):
                # in cut
                pass
            else: # not in cut
                in_cut = False

        # return whether or not atom is in edge cut. Note that, in general, this
        # will be used to determine whether or not to change the atom's output
        # mode.
        return in_cut

def displacement(atom, edge, period=1):
    '''Determines which periodic image of <edge> the <atom> is closest to and
    then returns an appropriate value for the displacement vector to give a
    sensible input atomic-scale configuration for the dislocation. This
    displacement varies in magnitude according to the distance from the
    dislocation.
    '''

    x0 = edge.pos[edge.b_index]
    x = atom.getCoordinates()[edge.b_index]

    # begin by noting that if x > x0, we only need to consider the image in the
    # positive direction while, if x < x0, only the negative image need be
    # considered. This implies that sgn(x-x0) gives the correct direction of the
    # closer repeat. Now we just need to compare this image with the cut in the
    # unit cell.

    image_dist = np.sign(x-x0)*period
    xmax = period/2.
    dist_base = abs(x-x0)
    dist_image = abs(x - (x0+image_dist))

    if abs(dist_base-dist_image) < 2*sys.float_info.epsilon:
        # atom halfway between images; do not displace
        u = 0.
    elif dist_base < dist_image:
        u = np.sign(x0-x)*edge.b/2.*(xmax-max(dist_base-edge.b/2., 0))/xmax
    else: # dist_image > dis_base
        u = np.sign(x0+image_dist-x)*edge.b/2.*(xmax-max(dist_image-
                                                        edge.b/2., 0))/xmax

    u_vec = np.zeros(3)
    u_vec[edge.b_index] = u
    return u_vec

def in_interval(atom, edge):
    '''Tests to see if <atom> is in the range defined by <edge>.
    '''

    y = atom.getCoordinates()[edge.cut_index]
    y0 = edge.neg[edge.cut_index]
    y1 = edge.pos[edge.cut_index]
    if edge.continuous:
        return y0 < y and y < y1
    else:
        return y > y0 or y < y1

def overlaps(dip1, dip2):
    '''Tests to see if two edge dislocation dipoles overlap (in the cut
    direction).
    '''

    # note that dip1 and dip2 cannot simultaneously be discontinuous
    i1 = dip1.cut_index
    i2 = dip2.cut_index

    # both dislocations discontinuous not allowed
    if (not dip1.continuous) and (not dip2.continuous):
        raise Exception("Cannot have two discontinuous cuts.")

    # deal with the case of two continuous (basically, an offset dipole)
    # note that, in this case, ranges will basically be [0.-eps, a] and
    # [1-a, 1+eps], eps > 0., so that atoms on the edge of the cell are also
    # removed
    if dip1.continuous and dip2.continuous:
        if dip1.neg[i1] < dip2.pos[i2] < dip1.pos[i1]:
            return True
        elif dip1.neg[i1] < dip2.neg[i2] < dip1.pos[i1]:
            return True
        elif dip1.neg[i1] > dip2.neg[i2] and dip1.pos[i1] < dip2.pos[i2]:
            return True
        else:
            return False

    # deal with one continuous and one discontinuous dipole
    if dip1.continuous and (not dip2.continuous):
        cont = dip1
        discont = dip2
        icont = i1
        idiscont = i2
    else:
        cont = dip2
        discont = dip1
        icont = i2
        idiscont = i1

    if (cont.neg[icont] < discont.pos[idiscont]) or (cont.pos[icont] >
                                                discont.neg[idiscont]):
        return True
    else:
        return False

def cut_supercell(cell, *edges):
    '''Given a simulation cell <cell>, inserts the edge dislocation dipoles
    specified in <*edges>. This entails removing atoms in the cut region for
    each dipole and displacing atoms on either side of each cut to remove the
    created voids.
    '''

    # begin by testing for overlap of the edge dislocation dipoles
    if len(edges) == 1:
        # only one cut -> overlap impossible
        pass
    else:
        # for the moment, we assume that the maximum number of dislocations is
        # four (ie. quadrupole configuration)
        if overlaps(edges[0], edges[1]):
            raise ValueError("Dipole cuts overlap.")

    # remove atoms that lie within any of the specified cut regions
    for atom in cell:
        if not atom.writeToOutput():
            continue
        for cut in edges:
            if cut.incut(atom):
                atom.switchOutputMode()
                break # don't need to check if the atom is in other cuts

    # displace atoms to create initial (unrelaxed) dislocation configuration
    for atom in cell:
        if not atom.writeToOutput():
            # no need to worry about atoms that aren't included!
            continue
        # else
        for cut in edges:
            if in_interval(atom, cut):
                # displace atom towards closest cut
                u = displacement(atom, cut) # assume period = 1
                new_coords = atom.getCoordinates() + u
                atom.setDisplacedCoordinates(new_coords)

    return

def compress_cell(supercell, b, n=1, bdir=0):
    '''Given a simulation cell <supercell> into which one or more (usually max
    2) edge dislocation dipoles has been inserted, compresses the cell parameter
    parallel to the burgers vector b to reflect the amount of material that has
    been extracted from the cell to create the edge dislocations.
    
    ### NEED TO GENERALISE FOR MORE COMPLICATED GEOMETRIES ###
    '''

    # axis to compress is the same as the burgers vector index
    lattice_vec = supercell.getVector(bdir)
    
    # determine scale factor
    scale = 1. - b 
    lattice_vec *= scale
    supercell.setVector(lattice_vec, bdir)
    return            
