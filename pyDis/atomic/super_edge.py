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
    
### stuff for diagonally oriented dipole -> special case => special section ###

def taxicab(x, x0):
    '''Determines the distance between x and x0 in the taxicab metric.
    '''
    
    dist = 0
    for xi, x0i in zip(x, x0):
        dist += abs(xi - x0i)
        
    return dist
    
def nearest_image_dist(x, x0, metric=taxicab):
    '''Calculates the minimum distance in the specified metric from <x> to any 
    periodic image of <x0>.
    ''' 
    
    min_dist = np.inf

    if type(x) == float or type(x) == int:
        for dx in [-1, 0, 1]:
            dist = abs(x+dx - x0)
            if dist < min_dist:
                min_dist = dist
    else:  # vector
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                dist = metric([x[0]+dx, x[1]+dy], x0[:2])
                if dist < min_dist:
                    min_dist = dist
    
    return min_dist

class EdgeDiagDipole(object):
    '''Constructs an edge dislocation dipole with the dislocations with positive
    and negative Burgers vectors located in opposing corners of the simulation
    cell. Particularly useful for explicit atomistic calculations of the Peierls
    stress.
    '''
    
    def __init__(self, b, axis=0):
        '''<b> is the Burgers vector (or its magnitude), and <axis> gives its
        orientation (ie. the slip plane).
        '''
        
        if type(b) == int or type(b) == float:
            self.b = b
        else: # assume that the Burgers vector has been supplied
            self.b = norm(b)
        
        # check that the supplied axis exists
        if not (axis in [0, 1]):
            raise ValueError("Burgers vector must be parallel to x or y axes.")
                
        self.axis = axis
        
    def displace(self, atom):
        '''Tests to see if <atom> is in a region affected by one of the dislocations.
        '''
        
        x = atom.getCoordinates()
        along_b = float(x[self.axis] % 1)
        along_cut = float(x[(self.axis - 1) % 2] % 1)
        if self.axis == 0:
            xi = [along_b, along_cut]
        else:
            xi = [along_cut, along_b]
            
        xp = [0.25, 0.25] # positive dislocation
        xm = [0.75, 0.75] # negative dislocation
            
        # check to see if atom is in the "middle region" where displacement is
        # zero
        u = np.zeros(3)
        if along_cut >= 0.25 and along_cut <= 0.75:
            # no displacement
            pass
        else: 
            # determine the type of the nearest dislocation 
            dp = nearest_image_dist(xi, xp, taxicab) 
            dm = nearest_image_dist(xi, xm, taxicab)
            
            # calculate magnitude of displacement vector
            if dp < dm:
                # closest to positive dislocation
                dx = -np.sign(along_b-0.25)*abs(0.5-(along_b-0.25))/0.5*self.b/2.
                u[self.axis] = dx
            elif dp > dm:
                # closest to negative dislocation
                dx = -np.sign(along_b-0.75)*abs(0.5-(along_b-0.75))/0.5*self.b/2.
                u[self.axis] = dx
            else:
                pass
    
        return u        
            
    def incut(self, atom):
        '''Tests to see if atom needs to be removed.
        '''
                
        x = atom.getCoordinates()
        along_b = x[self.axis] % 1
        along_cut = x[(self.axis - 1) % 2] % 1
        
        if along_cut < 0.25: # in positive dislocation cut
            if (along_b >= 0.25 - self.b/2. and  along_b < 0.25 + self.b/2.):
                return True
            else:    
                return False 
        elif along_cut > 0.75 or along_cut < 1e-6: # in negative dislocation cut
            if (along_b >= 0.75 - self.b/2. and  along_b < 0.75 + self.b/2.):
                return True
            else:
                return False
        else:
            return False       
                
def make_diag_dip(supercell, b, axis=0):
    '''Inserts an edge dislocation dipole (diagonal orientation) into the
    provided supercell. Usually assume that b || to the x-axis.
    '''
    
    # create dipole
    dipole = EdgeDiagDipole(b)
    
    # test to see if atoms need to be removed
    for atom in supercell:
        if dipole.incut(atom):
            atom.switchOutputMode()
            
    # displace remaining atoms        
    for atom in supercell: 
        if atom.writeToOutput():
            dx = dipole.displace(atom)
            atom.setDisplacedCoordinates(atom.getCoordinates() + dx)
    
    # compress cell to account for material removed        
    compress_cell(supercell, b, bdir=axis)
    return              
