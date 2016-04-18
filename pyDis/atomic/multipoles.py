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
from __future__ import print_function, division

import numpy as np
import sys
from numpy.linalg import norm
import re

supported_codes = ('qe', 'gulp', 'castep')

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
    
    #!!! NEED TO GENERALISE FOR MORE COMPLICATED GEOMETRIES
    '''
    
    # make sure that b is (a) not a vector, and (b) positive. Fix if necessary
    if type(b) == float or type(b) == int:
        b = abs(b)
    else: # assume vector
        b = norm(b)

    # axis to compress is the same as the burgers vector index
    lattice_vec = supercell.getVector(bdir)
    
    # determine scale factor
    scale = 1. - (n*b)/2. 
    lattice_vec *= scale
    supercell.setVector(lattice_vec, bdir)
    return  
    
### FUNCTIONS TO INSERT MULTIPOLES INTO SIMULATION CELLS ###

def edge_dipole(supercell, b, bdir=0):
    '''Inserts a pair of edge dislocations with Burgers vectors +- b into the 
    provided <supercell>.
    '''
    
    # normalise the Burgers vector
    bnorm = b / norm(supercell.getVector(bdir))
    
    if bdir == 0:
        xneg = [0.5, 0.25]
        xpos = [0.5, 0.75]
    elif bdir == 1:
        xneg = [0.25, 0.5]
        xpos = [0.75, 0.5]
    else:
        raise ValueError("b must be parallel to either x (0) or y (1).")
        
    new_dip = EdgeDipole(xneg, xpos, bnorm)
    
    # cut the simulation cell and compress it to account for removed material 
    cut_supercell(supercell, new_dip)
    compress_supercell(supercell, bnorm, n=1, bdir=bdir)
    return
    
def edge_quadrupole(supercell, b, bdir=0):
    '''Inserts four dislocations into the <supercell>, with Burgers vectors 
    +- b
    '''
  
    # normalise the Burgers vector
    bnorm = b / norm(supercell.getVector(bdir))
         
    if bdir == 0:
        xneg1 = [0.25, 0.25]
        xpos1 = [0.25, 0.75]
        xneg2 = [0.75, 0.75]
        xpos2 = [0.75, 0.25]
    elif bdir == 1:
        xneg1 = [0.25, 0.25]
        xpos1 = [0.75, 0.25]
        xneg2 = [0.75, 0.75]
        xpos2 = [0.25, 0.75]
    else:
        raise ValueError("b must be parallel to either x (0) or y (1).")
        
    # create dipoles
    dipole1 = EdgeDipole(xneg1, xpos1, bnorm)
    dipole2 = EdgeDipoel(xneg2, xpos2, bnorm)
    
    # cut the simulation cell and compress it to account for removed material
    cut_supercell(supercell, dipole1, dipole2)
    compress_supercell(supercell, bnorm, n=2, bdir=bdir) 
    return
    
def screw_dipole(supercell, b, screwfield, sij):
    '''Screw dislocation dipole in <supercell>. Note that setting <screwfield>
    to be something other than a screw dislocation, you can technically use this
    for an arbitrary displacement field.
    '''
    
    xneg = [0.5, 0.25]
    xpos = [0.5, 0.75]
    
    cores = np.array([xneg, xpos])
    burgers = np.array([-b, b])
    
    supercell.applyField(screwfield, cores, burgers, Sij=sij)
    
    return
    
def screw_quadrupole(supercell, b, screwfield, sij):
    '''Inserts a dislocation quadrupole into the <supercell>. As with 
    <screw_dipole>, can actually be used with an arbitrary displacement field.
    '''
    
    xneg1 = [0.25, 0.25]
    xpos1 = [0.25, 0.75]
    xneg2 = [0.75, 0.75]
    xpos2 = [0.75, 0.25]
    
    cores = np.array([xneg1, xpos1, xneg2, xpos2])
    burgers = np.array([-b, b, -b, b])
    
    supercell.applyField(screwfield, cores, burgers, Sij=sij)
    
    return    
    
### FUNCTIONS TO EXTRACT CORE ENERGY ### 

# dictionary containing regex to match final energies for a variety of codes
energy_lines = {"gulp": re.compile(r"\n\s*Final energy\s+=\s+" +
                                  "(?P<E>-?\d+\.\d+)\s*(?P<units>\w+)\s*\n"),
                "castep": re.compile(r"\n\s*BFGS:\s*Final\s+Enthalpy\s+=\s+" +
                              "(?P<E>-?\d+\.\d+E\+\d+)\s*(?P<units>\w+)\s*\n"),
                "qe": re.compile(r"\n\s*Final energy\s+=\s+(?P<E>-?\d+\.\d+)" +
                                  "\s+(?P<units>\w+)\s*\n")
               }
               
def extract_energy(cellname, program):
    '''Extracts the final energy of relaxed cell <cellname>, using the regular
    expression appropriate for the atomic scale simulation code used.
    
    #!!! Can perhaps merge this with <get_gsf_energy> in <read_gsf>
    '''
    
    if not (program in supported_codes):
        raise ValueError("{} is not a supported atomistic code.".format(program))
    else:
        energy_regex = energy_lines[program]
        
    # read in the output file from the atomistic code
    outfile = open(cellname)
    output_lines = outfile.read()
    matched_energies = re.findall(energy_regex, output_lines)
    
    E = np.nan
    units = ''
    if not matched_energies:
        print("Warning: No energy block found.")
        pass
    else:
        # use the last matched energy (ie. optimised structure)
        E = float(matched_energies[-1][0])
        units = matched_energies[-1][1]
            
    return E, units    
    
def gridded_energies(basename, program, suffix, i_index, j_index=None, 
                                                        gridded=False):
    '''Read in energies from several supercells with sizes specified by <i_array>
    and <j_index>. If <j_index> == None, use <i_index> for both indices (ie. x
    and y).
    '''
    
    energy_values = []
    if j_index == None:
        # set equal to <i_index>
        j_index = np.copy(i_index)
    elif type(j_index) == int:
        # if gridded is True, this axis will be of length one
        if gridded:
            j_index = [j_index]
        else: # not gridded -> supplying indices raw
            # zip needs an iterable object
            j_index = j_index*np.ones(len(i_index), dtype=int)
        
    # read in supercell energies
    if gridded:
        for i in i_index:
            for j in j_index:
                cellname = '{}.{}.{}.{}'.format(basename, i, j, suffix)
                Eij, units = extract_energy(cellname, program)
                # record supercell size and energy
                energy_values.append([i, j, Eij])
    else:
        for i, j in zip(i_index, j_index):
            cellname = '{}.{}.{}.{}'.format(basename, i, j, suffix)
            Eij, units = extract_energy(cellname, program)
            energy_values.append([i, j, Eij])
            
    return energy_values, units     
   
def multipole_energy(sides, K, Ecore, b, ):
    '''Function that gives the energy of a supercell with a dislocation multipole
    embedded in it. Used for curve fitting. The variable <sides> is equal to
    [a1, a2], ie. the side lengths of the supercell.
    '''
    
    a1, a2 = sides
    
    #!!! need to check factors of pi
    E = Ecore + K*b**2*(np.log(abs(a1)/(2*rcore))+A*abs(a1)/abs(a2))
    return E 
    
def fit_core_energy_mp(dEij, basestruc, units='ev'):
    '''Fits core energy of a dislocation using the excess energies <dEij> of 
    dislocations in simulation cells of varying sizes. 
    '''
    
    # extract the cell sizes (in \AA) and energies (in eV)
    dEij = np.array(dEij)
    
    energies = dEij[:, -1]
    if units.lower() == 'ev':
        pass
    elif 'ry' in units.lower(): # assume Rydberg, can add others later
        energies /= 13.60569172
         
    
    # extract cell dimensions
    dims = []
    for i in range(3):
        dims.append(norm(basestruc.getVector(i))
        
    # convert side lengths from lattice units to \AA
    sides = dEij[:, :2]
    sides = np.array([[dims[0]*x[0], dims[1]*x[1]] for x in sides])
    
    return
    
    
