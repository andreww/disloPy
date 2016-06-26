#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys
from numpy.linalg import norm

sys.path.append('/home/richard/code_bases/dislocator2/')

from pyDis.atomic import crystal as cry
from pyDis.atomic import qe_utils as qe
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import castep_utils as castep
from pyDis.atomic import atomistic_utils as atm


def permutation(i, j, k):
    '''The permutation symbol e_{ijk}, which has the value 0 if two indices are
    the same, 1 if they are an even permutation of 012 and -1 if they are an odd
    permutation.
    '''
    
    if i == j or j == k or i == k:
        return 0
    else:
        sign = int(i>j) + int(i>k) + int(j>k)
        return sign
        
def partition_lattice(struc, ndomains=None, domain_size=None):
    '''Partitions a atoms in a crystal into domains to facilitate efficient 
    search. If specified, <ndomains> should be a 3x1 array specifying the number 
    of domains along each dimension. <domain_size>, if specified, gives the 
    maximum side length of a domain (in whatever units are used to express the
    cell lengths). Priority is <ndomains> > <domain_size>.
    '''
    
    if ndomains != None:
        # use the specified domain decomposition
        pass
    elif domain_size != None:
        # generate domains from the specified domain size
        ndomains = np.ones(3, dtype=int)
        for x in struc.getLattice():
            ndomains = atm.ceiling(norm(x)/domain_size)
    else:
        raise AttributeError("<ndomains> or <domain_size> must be specified.")
    
    for atom in base_lattice:
        # test to determine to which region the atom should be assigned
        
        # assign atom to a particular region
        
    
    return

def coordination(atom, lattice):
    '''Locates atoms in coordination sphere.
    '''
    
    pass
    
def associate_bonds(Q, P, phi_max=np.pi/6):
    '''Create correspondence between radius vectors in the dislocated crystal (Q)
    with those in the perfect crystal (P). <phi_max> is the rejection threshold
    for angular deviation.
    '''
    
    #Q_ordered = 
    # compute angles between each of the bonds Q^{\gamma} in the dislocated
    # crystal and the bonds in the reference crystal
    angles = np.zeros((len(Q), len(Q)))
    for i, P_i in enumerate(P):
        for j, Q_j in enumerate(Q):
            phi = np.arccos(np.dot(Q_j, P_i)/(norm(Q_j)*norm(P_i)))
            angles[i, j] = phi
            
    # work out closest bond in perfect crystal for each bond in the dislocated
    # crystal
    closest_bonds = np.zeros(len(Q))
    for i, bond in enumerate(angles):
        min_index = np.argmin(bond)
        if angles[i, min_index] < phi_max:
            closest_bonds[i] = min_index
        else:
            closest_bonds[i] = np.nan # so that <closest_bonds> has single type
            
    # filter for bonds that correspond to the same bond in the perfect crystal
    match_index = np.ones(len(P))
    match_index.fill(np.nan)
    
    for i, P_i in enumerate(P):
        indices_i = np.where(closest_bonds == i)[0]
        if len(indices_i) == 0:
            # no match
            print("No match for bond P_{{{}}}".format(i))
        elif len(indices_i) == 1:
            print("Q_{{{}}} matches bond P_{{{}}}".format(indices_i[0], i))
            print("Q_{{{}}}: {}".format(indices_i[0], Q[indices_i[0]]))
            print("P_{{{}}}: {}".format(i, P[i]))
            match_index[i] = indices_i[0]
        else:
            # determine which of the Q_{j} is closest in length to P_{i}
            min_dist = np.inf
            best = -1
            P_length = norm(P_i)
            for j in indices_i:
                dist = P_length - norm(Q[j])
                if dist < min_dist:
                    min_dist = dist
                    best = j
            print("Bond Q_{{{}}} is the best match for P_{{{}}}.".format(j, i))
            match_index[i] = j
    
    return match_index

def order_bonds(Q, P, match_index):
    '''Given list of indices relating the deformed (Q) to the perfect (P) bonds,
    create ordered list of Q and P.
    '''
    
    Q_ord = []
    P_ord = []
    for i, P_i in enumerate(P):
        if match_index[i] != match_index[i]:
            # no matching bond was found
            continue
        else:
            P_ord.append(P_i)
            Q_ord.append(Q[match_index[i]])
    
    Q_ord = np.array(Q_ord, dtype=float)
    P_ord = np.array(P_ord, dtype=float)
            
    return Q_ord, P_ord
    
def find_correspondence(Q, P):
    '''From list of bonds in perfect (P) and dislocated (Q) crystal, create 
    lattice correspondence matrix G satisfying P = Q . G.
    '''
    
    # rearrange Q to associate vectors in Q with those in P. Note that some of 
    # the bonds in the undislocated crystal (P) may not match any of the bonds
    # in the dislocated crystal (Q)
    match_index = associate_bonds(Q, P)
    Q_ord, P_ord = order_bonds(Q, P, match_index)
    G = np.dot(L.pinv(Q_ord), P_ord)
    
    return G
    
def AIM(G_0, G_g):
    '''Computes the vector A(IM).
    '''
    
    pass
    
def nye_tensor(G_dis, G_ref):
    '''Construct the Nye tensor using the lattice correspondence tensor, G, for
    the dislocated and reference crystals.
    '''
    
    pass
    
def domain_decomp(crystal_structure, scale):
    '''Decomposes a crystal (ie. collection of atoms) into spatial domains.
    Reduces the number of operations required to find bonds for all atoms in an
    N atom cluster to O(N) (simply searching through <crystal_structure> would
    be O(N^2).
    
    scale takes the form [n_x, n_y, n_z], where each component gives the number
    of cells along a particular axis. Not yet sure how to implement this for a
    cluster-based calculation.
    '''
    
    # partition <crystal_structure> into domains defined by a characteristic length
    # <scale>. Note that adjacent atoms can be found in different, albeit neighbouring
    # domains.
    
   return 
    
def iterate_through_atoms(structure, P):
    '''Iterate through all atoms in the dislocated structure to compute variation
    of the Nye tensor \\alpha.
    '''
    '''
    # decompose into domains
    domain_decomp(structure, scale)
    
    # for each atom in <list_of_atoms>, find neighbours and construct G_{0} and
    # Q_{\\gamma}
    for atom in structure:
        # determine which domain <atom> is in
        domain = atom.domain
        # search only in neighbourhood of <atom>. ie, within the same domain as
        # <atom> and (if <atom> is close to a domain boundary), adjacent domains
    
    for each atom in list_of_atoms:
        subroutine(find_neighbours)
        map neighbours, P -> coordinate transformation G
        record G
        record neighbours
        assert G and neighbours have the same ordering
    
    # construct nye tensor
    for each atom in list_of_atoms:
        # DG(IM) = G(IM)^{\\gamma} - G(IM)^{0}
        retrieve G_0
        for each atom gamma bonded to atom 0, retrieve G_gamma
        for each component IM:
            construct DG[IM] = G_gamma[IM] - G_0[IM]
        AIM = np.dot(L.pinv(Q_ordered), DG[IM])
        map AIM[k] - T[kIM]
        # calculate nye tensor components
        nye_0[jk] = -1*permutation(j, i, m)T[imk] # Sum over repeated indices
        map nye_0[jk] -> value of nye tensor at atom 0
        '''
    
    
'''    
### IS IT BETTER NOT TO SPECIFY THE METRIC USED? 
### SPECIFYING THE METRIC MAKES IT MORE DIFFICULT
### TO APPLY THE ALGORITHM GENERALLY. IN PARTICULAR,
### I NOTICE THAT IT WOULD BE DIFFICULT TO DEFINE A
### "FIRST COORDINATION" SPHERE FOR eg. F^{-} IN 
### APATITE.  
MAX_BOND_DIST = 6.
MAX_ANGLE = np.pi/6.
count_bonds = [[] for i in range(len(Q))]
for i, q1 in enumerate(Q):
    for j, q2 in enumerate(Q[i+1:]):
        # test to make sure bond q1->q2 is shorter than some threshold
        # value
        if norm(q2-q1) > MAX_BOND_DIST:
            continue
        print("Testing pair ({}, {}): ".format(i, j+i+1))
        min_index = np.nan
        min_angle = np.inf
        for k, bond in enumerate(P):
            cos_phi = np.dot(q2-q1, bond)/(norm(q2-q1)*norm(bond))
            phi = abs(np.arccos(cos_phi))
            if phi < min_angle:
                min_angle = abs(phi)
                min_index = k
        if min_angle > MAX_ANGLE:
            print("No matching bonds.")
        else:
            print("Angle {} with bond {}.".format(min_angle, min_index))
            count_bonds[i].append({'pair': (i,j+i+1), 'bond': min_index,
                             'length': norm(q2-q1), 'vector':np.copy(q2-q1)})
            count_bonds[i+j+1].append({'pair': (j+i+1,i), 'bond': min_index,
                            'length': norm(q2-q1), 'vector': np.copy(q2-q1)})
    
Q_ord = []
P_ord = []
for site in count_bonds:
    Q_temp = []
    P_temp = []
    bonds = np.array([x['bond'] for x in site])
    vectors = [x['vector'] for x in site]
    pairs = [x['pair'] for x in site]
    for i, P_i in enumerate(P):
        indices_i = np.where(bonds == i)[0]
        if len(indices_i) == 0:
            # no match
            pass
        elif len(indices_i) == 1:
            P_temp.append(P_i)
            Q_temp.append(np.copy(vectors[indices_i[0]]))
        else:
            # multiple possible matches; choose the one closest in
            # length to P_i
            P_temp.append(P_i)
            min_index = 0
            min_dist = norm(vectors[indices_i[0]]-P_i)
            for j in indices_i[1:]:
                new_dist = norm(vectors[j]-P_i)
                if new_dist < min_dist:
                    min_index = j
                    min_dist = new_dist
            Q_temp.append(vectors[min_index])
    Q_ord.append(Q_temp)
    P_ord.append(P_temp)
'''    

dev = np.zeros((len(test_cry), 2))
mean_bl = np.zeros((len(test_cry), 2))
for i in range(len(test_cry)):
    atom1 = test_cry[i]
    nbonds = 0
    bonds = []
    for j in range(len(test_cry)):
        if i == j:
            continue
        atom2 = test_cry[j]
        dist12 = dist_polymer(atom1, atom2, pcell=pcell)
        for k, d in enumerate(dist12):
            if d < 2.:
                #print("{} {} + {}: {:.6f}".format(i, j, k-1, d))
                nbonds += 1
                bonds.append(d)
    r = norm(atom1.getCoordinates()[1:])
    dev[i, 0] = r
    dev[i, 1] = np.std(bonds)
    mean_bl[i, 0] = r
    mean_bl[i, 1] = np.mean(bonds)
    if nbonds != 4:
        if r < 15.:
            print("Atom {} ({}) is {}-fold coordinated.".format(i,
                                                        r, nbonds))

