#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm, inv

from pydis.atomic import crystal as cry
from pydis.atomic import gulpUtils as gulp

def perfect_bonds(cellname, atom_index, max_bond_length, bonded_type=None):
    '''Extracts all bonds between atom <atom_index> and adjacent atoms of type 
    <atom_type> out to the given 
    distance in the undeformed crystal <cellname>.
    '''
    
    # read in crystal structure   
    unit_cell = cry.Crystal()
    sinfo = gulp.parse_gulp(cellname, unit_cell)
    
    # extract the coordinates of the atom specified and convert to angstroms
    atom0 = unit_cell[atom_index]
    x0 = atom0.getCoordinates()
    lattice = unit_cell.getLattice()
    x0 = x0[0]*lattice[0]+x0[1]*lattice[1]+x0[2]*lattice[2]
    
    # if <bonded_type> is not given, assume that we are calculating the Nye 
    # tensor on the sublattice to which <atom_index> belongs
    if bonded_type is None:
        bonded_type = atom0.getSpecies()

    # extract bonds
    P = []
    for i in range(len(unit_cell)): 
        # check that atom i is of the correct species   
        if unit_cell[i].getSpecies() != bonded_type:
            continue
        
        # convert coordinates of atom <i> to angstroms
        y0 = unit_cell[i].getCoordinates()
        y0 = y0[0]*lattice[0]+y0[1]*lattice[1]+y0[2]*lattice[2]
        
        # iterate through all periodic images of the cell that share a face,
        # edge, or corner with the original cell
        for j in range(-1, 2):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # check that this is not the original atom
                    if i == atom_index and (j == k == l == 0):
                        continue
                    
                    # calculate and store the bond vector from <atom_index> to
                    # this site 
                    y = y0+j*lattice[0]+k*lattice[1]+l*lattice[2]
                    if norm(x0-y) < max_bond_length:
                        P.append(x0-y)
                        
    return P

def bond_candidates(dis_cell, atom_type, max_bond_length, R, RI, RII, 
                                                    bonded_type=None):
    '''Extracts candidate bonds for all sites of the specified type with radius
    R.
    '''
    
    if bonded_type is None:
        # calculate Nye tensor on the <atom_type> sublattice
        bonded_type = atom_type
    
    # read in file containing the relaxed dislocation structure, then extract
    # atoms in the relaxed region, as well as the cell thickness
    discluster, sinfo = gulp.cluster_from_grs(dis_cell, RI, RII)
    relaxed = discluster.getRegionIAtoms()
    H = discluster.getHeight()
    
    n = len(relaxed)

    # extract a list of potential bonds for atoms in the sublattice of interest
    Qpot= dict()
    for i in range(n):      
        # check that atom i belongs to the sublattice of interest (usually not
        # oxygen)
        atomspecies = relaxed[i].getSpecies()
        if atomspecies != atom_type:
            continue
            
        # ensure that atom i is within the specified region
        x = relaxed[i].getCoordinates()
        if (np.sqrt(x[0]**2+x[1]**2)) > R:
            continue
        
        Qpoti = []
        for j in range(n):
            # check that atom j is one whose bonds with atom i we care about
            if relaxed[j].getSpecies() != bonded_type:
                continue
            
            # extract coordinates of atom j and its periodic images
            y0 = relaxed[j].getCoordinates()
            ydown = y0-cry.ei(3)*H
            yup = y0+cry.ei(3)*H
            
            # calculate distance to atom i and check to see if bond length is 
            # below given maximum value
            if i != j and norm(x-y0) < max_bond_length:
                Qpoti.append([j, x-y0])
            if norm(x-yup) < max_bond_length:
                Qpoti.append([j, x-yup])
            if norm(x-ydown) < max_bond_length:
                Qpoti.append([j, x-ydown])
                
        Qpot[i] = [x, Qpoti]
        
    return Qpot

def associate_bonds(Qpot, P, phimax=0.5, scale=1.5):
    '''Rearranges the list of potentials atomic bonds <Qpot> so that they 
    correspond to the bonds in <P> for the specified site in an undeformed
    crystal. <phimax> is the maximum tolerable angular distance between a 
    deformed and undeformed bond. <scale> gives the maximum permissible increase
    in bond length, and 
    '''
    
    Qordered = dict()
    
    # iterate through sites
    for site in Qpot:
        x = Qpot[site][0]
        Qi = Qpot[site][1]
        mapping = []
        for Pj in P:
            phimin = np.inf
            Qmin = np.zeros(3)
            j_site = -1
            for j, Qij in Qi:
                # calculate angle <phi> between the deformed bond Qij and the 
                # undeformed bond <Pj>
                phi = abs(np.arccos(np.dot(Pj, Qij)/(norm(Pj)*norm(Qij))))
                if phi < phimin and norm(Pj)/scale < norm(Qij) < scale*norm(Pj):
                    # store values for new best bond candidate
                    phimin = phi
                    Qmin = Qij
                    j_site = j
            
            # check that calculated minimum angular distance is below specified
            # threshold        
            if phimin > phimax:
                continue
            else:
                mapping.append([j_site, Pj, Qmin])
                
        Qordered[site] = [x, mapping]
        
    return Qordered


def moore_penrose(M):
    '''Calculates the Moore-Penrose (ie. Generalized) inverse of the matric <M>
    '''
    
    return np.dot(inv(np.dot(M.T, M)), M.T)

def lattice_correspondence_G(Qordered):
    '''From the bond vectors in the dislocated (Q) and undislocated (P) crystal,
    calculate the lattice correspondence tensor at each atomic site using 
    equation (17) from Hartley and Mishin (2005).
    '''
        
    G = dict()
    for site in Qordered:
        Pi = []
        Qi = []
        bond_indices = []
        
        # extract the bond vectors
        for bond in Qordered[site][1]:
            Pi.append(bond[1])
            Qi.append(bond[2])
            bond_indices.append(bond[0])
            
        Pi = np.array(Pi)
        Qi = np.array(Qi)
        
        # calculate the generalised inverse and solve for <G>
        Qidagger = moore_penrose(Qi)
        Gi = np.dot(Qidagger, Pi)
        G[site] = {'G': Gi, 'Qdag': Qidagger, 'bonded': bond_indices}

    return G

def DGIM(G, site, I, M):
    '''Calculates the difference of the component <IM> of the lattice 
    correspondence tensor <G> between the specified <site> and all <sites> to 
    which it is bonded.
    '''
    
    # check that the lattice correspondence tensor G has been 
    # calculated for all atoms "bonded" to <site>
    bonded_atoms = G[site]['bonded']
    for atom_index in bonded_atoms:
        if atom_index in G.keys():
            continue
        else:
            return np.nan
            
    # extract the specified component at the site
    G0IM = G[site]['G'][I, M]
    
    # iterate through bonded atoms and extract component GIM at each site
    GgIM = np.zeros(len(bonded_atoms))
    for i, j in enumerate(bonded_atoms):
        GgIM[i] = G[j]['G'][I, M]
        
    return GgIM - G0IM

def AIM(DGIM, G, site):
    '''Calculate the <A> vector given in equation (19) of Hartley and Mishin
    2005.
    '''
    
    return np.dot(G[site]['Qdag'], DGIM)

def derivatives_G(G):
    '''Calculates the tensor T_{imk} whose components are the partial derivates
    of the components of the lattice correspondence tensor G, ie. T_{imk} =
    d_{k}G_{im}
    '''
         
    T = dict()
    for key in G.keys():
        Timk = np.zeros((3, 3, 3))
        for I in range(3):
            for M in range(3):
                # calculate change in G between bonded sites
                dgnm = DGIM(G, key, I, M)
                
                # test to make sure that difference is define
                if not (dgnm is np.nan):
                    # invert to obtain the A matrix
                    anm = AIM(dgnm, G, key)
                    for k in range(3):
                        Timk[I, M, k] = anm[k]
        T[key] = Timk
        
    return T

def calculate_nye(T, Q):
    '''Using the tensor T_{imk}, calculate the components of the Nye tensor. Q 
    is also provided as an argument as it contains the coordinates of the atomic
    sites. 
    '''
    
    nye_a = dict()
    for site in T.keys():
        a = np.zeros((3, 3))
        for j in range(3):
            for k in range(3):
                ajk = 0
                for i in range(3):
                    for m in range(3):
                        ajk += -permute_eps(j, i, m)*T[site][i, m, k]
                a[j, k] = ajk
                
        # store calculated Nye tensor for <site>
        nye_a[site] = dict()
        nye_a[site]['a'] = a
        nye_a[site]['x'] = Q[site][0]
        
    return nye_a
    
def permute_eps(i, j, k):
    '''The standard permutation tensor \eps_{ijk}, which is 1 if i, j, k are an
    even permutation of 123 (or 012 in python indexing) and -1 if they are an 
    even permutation. If any index is repeated, \eps_{ijk} is 0.
    '''
    
    # check that there are no invalid indices
    valid = [0, 1, 2]
    if i not in valid or j not in valid or k not in valid:
        raise ValueError("Indices must be in [0, 1, 2]")
    # check that there are no repeated indices
    if i == j or i == k or j == k:
        return 0.
    elif i == 0:
        if j == 1:
            return 1.
        else:
            return -1.
    elif j == 0:
        if i == 2:
            return 1.
        else:
            return -1.
    elif i == 1:
        return 1.
    else:
        return -1.
    
def unravel_nye(a):
    '''Unravels the Nye tensor <a>, casting it in a form that is suitable for 
    plotting.
    '''

    # holds the values of the Nye Tensor components at all studied sites
    ajk = dict()
    nye_components = ['a00', 'a01', 'a02', 'a10', 'a11', 'a12', 'a20', 'a21', 'a22']
    for c in nye_components:
        ajk[c] = [] 
        
    for site in a.keys():
        # extract x and y coordinates of the site
        x.append([a[site]['x'][0], a[site]['x'][1]])
        
        # extract Nye tensor components
        for c in nye_components:
            ajk[c].append(a[site]['a'][int(c[1]), int(c[2])])
        
    # convert components to array
    for c in nye_components:
        ajk[c] = np.array(ajk[c])
        
    return ajk

def scatter_nye(x, ajk):
    '''Create scatter plot showing specified component of the Nye tensor <ajk>
    '''
    
    fig = plt.figure() 
    plt.gca().set_aspect('equal')
    plt.scatter(x[:, 0], y[:, 1], c=ajk, cmap=plt.get_cmap('viridis'), s=150,
                     linewidth='2')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.xlabel('x ($\AA$)', size='x-large', family='serif')
    plt.ylabel('y ($\AA$)', size='x-large', family='serif')
    plt.colorbar(format='%.2f')
    plt.tight_layout()
