#!/usr/bin/env python
'''Module to calculate migration barriers for atoms diffusing along a dislocation
line.
'''
from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.linalg import norm
from numpy.random import random
import glob
import re

import sys

from dislopy.atomic.multisite import periodic_distance
from dislopy.atomic import gulpUtils as gulp
from dislopy.utilities import atomistic_utils as util
from dislopy.atomic import segregation as seg

def read_sites(sitefile):
    '''Reads in a list of vacancy sites.
    '''
    
    site_info = []
    with open('{}.id.txt'.format(sitefile)) as f: 
        for line in f:
            if line.startswith('#'):
                continue
            elif not line.rstrip():
                continue
            # else
            site_line = line.rstrip().split()
            site_info.append([float(x) for x in site_line])
            
    return np.array(site_info)

def adjacent_sites(dfct_site, cluster, species, threshold=5e-1):
    '''Generates a list of atomic sites in <cluster> that are 
    adjacent to the specified vacancy (<dfct_site>) in the direction
    of the dislocation line vector.
    '''

    adjacent_indices = []
        
    # get coordinates of dfct site
    x0 = dfct_site[1:4]
    r0 = x0[:-1]    
        
    # find sites adjacent to <dfct_site>
    for j, atom in enumerate(cluster):
        if not(species.lower() in atom.getSpecies().lower()):
            continue
        # else
        x = atom.getCoordinates()
        dr = norm(x[:-1]-r0)
        if dr < threshold:
            h = cluster.getHeight()
            # potentially adjacent site, calculate distance from site
            min_dist = min([norm(x-x0), norm(x+np.array([0, 0, h])-x0), 
                                        norm(x-np.array([0, 0, h])-x0)])
            adjacent_indices.append([j, min_dist])
            
    return adjacent_indices

def atom_to_translate(dfct_site, possible_sites, cluster, tol=5e-1, toldist=1e-1):
    '''Determines which of the possible adjacent sites to translate. 
    '''
    
    # location of vacancy
    x0 = dfct_site[1:4]
    z0 = x0[-1]
    
    H = cluster.getHeight()
    
    # to hold list of possible candidates for translation
    candidates = []
    min_dist = np.inf
    
    # find distance ALONG dislocation line to next site
    for atom_index, dist in possible_sites:
        # check that either the <atom> or its image is below the vacancy
        zi = cluster[atom_index].getCoordinates()[-1]
        if zi > z0:
            # calculate ratio of distances to base and image atoms
            if (abs(z0-(zi-H)) / abs(zi-z0)) > 2.:
                continue
        # else
        if dist < min_dist:
            # see if <atom> is closer to the vacancy than any previously seen
            min_dist = dist
            
    # create list of candidate sites to which vacancy might migrate
    for atom_index, dist in possible_sites:
        if dist < min_dist+tol:
            candidates.append(atom_index)
                
    # find site(s) below the vacancy
    if len(candidates) == 1:
        # only 1 candidate, we are done
        return candidates
    elif len(candidates) == 0:
        # no candidates, this is a problem
        raise ValueError("Number of sites should be >= 1.")
    else:
        below_site = []
        for i in candidates:
            zi = cluster[i].getCoordinates()[-1]
            
            # distance to candidate in same cell as vacancy
            base_dist = abs(z0-zi)
            
            # get distance to image of candidate nearest to vacancy
            if zi < z0:
                # get distance to image above the vacancy
                image_dist = abs(H+zi-z0)
            else:
                # get distance to image below the vacancy
                image_dist = abs(z0+(H-zi))
            
            if image_dist > base_dist+toldist:
                if zi < z0:
                    below_site.append(i)
                else: 
                    pass
            else: 
                # note, includes cases when image_dist == base_dist
                if zi > z0:
                    below_site.append(i)
                else:
                    pass
                
        return below_site
                
def z_dist(atom1, atom2, height):
    '''Return distance from <atom2> to <atom1> along the cluster axis.
    '''
    
    z1 = atom1.getCoordinates()[-1]
    z2 = atom2.getCoordinates()[-1]
    if z1 > z2:
        return z1-z2
    else:
        return z1+height-z2

def next_occupied_site(site, poss, cluster):
    '''Finds the next occupied site along the dislocation axis.
    '''
    
    next_site = -1
    min_dist = np.inf
    H = cluster.getHeight()
    for index, dist in poss:
        if site == index:
            # same site
            pass
        else:
            d = z_dist(cluster[index], cluster[site], H)
            if d < min_dist:
                min_dist = d
                next_site = index
                
    return next_site, min_dist

def disp_distance(cluster, n, intersite_dist):
    '''Using the height of cluster, which is <n> unit cells high, and the distance
    <site_dist> between sites on either site of the vacancy.   
    '''
    
    return intersite_dist - cluster.getHeight()/n
    
def scale_plane_shift(plane_shift, x, xmax, node=0.5):
    '''Scales the lateral displacement vector for a migration path according
    to the axial distance x traversed along that path.
    '''
    
    # fractional distance travelled along axis
    ratio = x/xmax

    if ratio <= node:
        # region of gradually increasing displacement
        scale = ratio/node
    else:
        scale = (1.-ratio)/(1.-node)
        
    return scale*plane_shift
    
def perturb(x, mag=0.01):
    '''Apply random perturbation to coordinates to reduce symmetry.
    '''
    
    return x+2*mag*random(3)-mag
    
def max_index(vec):
    '''Determine for which index the distance between vec1 and vec2 is greatest
    '''
    
    i = 0
    d = abs(vec[0])
    for j in range(1, len(vec)):
        if abs(vec[j]) > d:
            i = j
    return i
    
def parse_bonds(bondfile):
    '''Finds all site pairs in a file containing lists of sites for which 
    impurity energies are calculated, and the sites to which each is bonded. 
    Typically, these are created by the routine <sites_to_replace_neb> in
    module <multisite>.
    '''
    
    # open file containing list of bonds
    bondstream = open(bondfile, 'r')
    bondlines = bondstream.read()
    bondstream.close()
    
    # extract all bond-blocks
    site_str = '\d+(?:\s+-?\d+\.\d+){4}'
    block_re = re.compile('(?:{}\n)+{};'.format(site_str, site_str))
    blocklist = block_re.findall(bondlines)
    
    # create a list of site-ID pairs for each bond
    if len(blocklist) == 0:
        raise ValueError("Number of sites must be > 0.")
        
    # else
    site_re = re.compile(site_str)
    site_dict = dict()
    for site in blocklist:
        sites = site_re.findall(site)
        for i, match in enumerate(sites):
            site_index = int(match.rstrip().split()[0])
            if i == 0:
                site_dict[site_index] = []
                current_site = site_index
            else:
                site_dict[current_site].append(site_index)
                
    return site_dict
    
def path_endpoints(start_cluster, stop_cluster, thresh=1):
    '''Find the coordinates of the atoms present in only one of two vacancy-bearing
    dislocation clusters.
    '''
    
    start_cluster.specifyRegions()
    stop_cluster.specifyRegions()
    
    # extract list of atoms in region I for both clusters - reduces the number of
    # atoms to be compared
    r11 = start_cluster.getRegionIAtoms()
    r12 = stop_cluster.getRegionIAtoms()
    
    # get number of atoms in region I for the two clusters
    n = r11.numberOfAtoms
    if n != r12.numberOfAtoms:
        raise ValueError("Clusters must contain the same number of atoms.")
    
    # go through atoms to find which are present in only 1 cluster
    found1 = False
    found2 = False
    s1=0
    s2=0
    for i in range(n):
        x1 = r11[i+s1].getCoordinates()[:-1]
        x2 = r12[i+s2].getCoordinates()[:-1]
        d = norm(x1-x2)
        if d > thresh and not found1:
            x1p = r11[i+1].getCoordinates()[:-1]
            x2p = r12[i+1].getCoordinates()[:-1]
            if norm(x1p-x2) < thresh:
                s1 = 1
                index_a = i
            elif norm(x2p-x1) < thresh:
                s2 = 1
                index_b = i
            found1 = True
        elif d > thresh and not found2:
            if s2 == 1:
                s2 = 0
                index_a = i
            elif s1 == 1:
                s1 = 0
                index_b = i
            found2 = True
        else:
            pass

    # find indices of the initial and final sites for the diffusion path in each
    # cluster
    initial_coords = r11[index_a].getCoordinates()
    final_coords = r12[index_b].getCoordinates()
    
    for i, atom in enumerate(start_cluster):
        x = atom.getCoordinates()
        if norm(x-initial_coords) < 1e-3:
            start_i = i
            break
            
    for j, atom in enumerate(stop_cluster):
        x = atom.getCoordinates()
        if norm(x-final_coords) < 1e-3:
            start_j = j
            break
    
    return start_i, start_j
    
def scale_plane_shift(shift, i, npoints, node):
    '''Scales the lateral displacement vector for a migration path according
    to the relative distance traversed along the migration path.
    '''

    if i <= node:
        # region of gradually increasing displacement
        scale = i/float(node)
    else:
        scale = 1-(i-node)/float(npoints-node)
        scale = (1.-i/float(npoints))/(1.-node/float(npoints))
        
    return scale*shift
    
def displacement_vecs(start_cluster, stop_cluster, start_i, stop_i, npoints):
    '''Construct displacement vectors for all atoms in a cluster.
    '''
    
    n = start_cluster.numberOfAtoms
    
    # calculate displacement vector between initial and final sites along
    # the migration path
    final_coords = stop_cluster[stop_i].getCoordinates()
    initial_coords = start_cluster[start_i].getCoordinates()
    dx = final_coords-initial_coords

    dx_list = []
    if start_i == stop_i:
        for i in range(n):
            if i == start_i:
                dx_list.append(dx)
            else:
                dxi = stop_cluster[i].getCoordinates()-start_cluster[i].getCoordinates()
                dx_list.append(dxi)
    elif start_i < stop_i:
        # assumes that cluster_a > cluster_b
        for i in range(n):
            x0 = start_cluster[i].getCoordinates()
            if i == start_i:
                dx_list.append(dx)
                continue
            elif i > start_i and i <= stop_i:
                x = stop_cluster[i-1].getCoordinates()
            else:
                x = stop_cluster[i].getCoordinates()
            dxi = x-x0
            dx_list.append(dxi)
    elif stop_i < start_i:
        for i in range(n):
            x0 = start_cluster[i].getCoordinates()
            if i >= stop_i and i < start_i:
                x = stop_cluster[i+1].getCoordinates()
            elif i == start_i:
                dx_list.append(dx)
                continue 
            else:
                x = stop_cluster[i].getCoordinates()   
            dxi = x-x0
            dx_list.append(dxi)
    
    # determine direction to constrain
    constrain_index = max_index(dx)
    
    # create increment of update
    dxn_list = np.array(dx_list)/npoints
    
    return dxn_list, constrain_index
    
def migrate_sites_general(basename, rI, rII, bondlist, npoints, executable=None, 
                    noisy=False, plane_shift=np.zeros(3), node=0.5, threshold=1,
                    centre_on_impurity=False, do_perturb=False, newspecies=None, 
                                                                adaptive=False):
    '''Calculates migration barriers between all pairs of atoms that are deemed
    to be bonded.
    '''
                     
    bond_dict = parse_bonds('{}.bonds.txt'.format(basename))
    
    for i in bond_dict.keys():
        start, sysinfo = gulp.cluster_from_grs('{}.{}.grs'.format(basename, i), rI, rII)
        for j in bond_dict[i]:
            stop, sysinfo = gulp.cluster_from_grs('{}.{}.grs'.format(basename, j), rI, rII)
            
            start_i, stop_j = path_endpoints(start, stop, thresh=threshold)
            
            dxn_ij, constrain_index = displacement_vecs(start, stop, start_i, 
                                                              stop_j, npoints) 
            
            pair_name = '{}.{}.{}'.format(basename, start_i, stop_j)                                                  
            gridded_energies, Eh, Ed = make_disp_files_gen(start,
                                                           start_i,
                                                           pair_name,
                                                           dxn_ij,
                                                           rI_centre=rI_centre,
                                                           do_perturb=do_perturb,
                                                           constrain_index=constrain_index,
                                                           newspecies=newspecies
                                                          )
                                                          
            outstream = open('disp.{}.barrier.dat'.format(pair_name), 'w')    
            # write header, including full displacement vector and barrier height 
            xstart = start[start_i].getCoordinates()
            xstop = stop[stop_j].getCoordinates()
            outstream.write('# {:.0f} {:.0f}\n'.format(start_i, stop_j))
           
            # write energies to file if they have been calculated
            if gridded_energies:     
                # write energies along path
                for z, E in gridded_energies:
                    outstream.write('{} {:.6f}\n'.format(z, E))
                
                heights.append([start_i, stop_j, site[1], site[2], Eh, Ed])
            
            outstream.close()

    return heights                                                         

def make_disp_files_gen(start, start_i, basename, dxn_list, rI_centre=np.zeros(2), 
                            do_perturb=False, constrain_index=2, newspecies=None):
    '''Generates input files for a constrained optimization calculation of migration
    barriers along an arbitrary migration path.
    '''
    
    # set constraints
    constraint_vector = np.ones(3)
    constraint_vector[constrain_index] = 0
    cluster[start_i].set_constraints(constraint_vector)
    
    # change species of diffusing atom, if requested
    if newspecies is not None:
        oldspecies = cluster[start_i].getSpecies()
        cluster[start_i].setSpecies(newspecies)
                                                                           
    for i in range(npoints):       
        # update dislocation structure
        for j in range(start.numberOfAtoms):
            dxi = dxn_list[j]*i/(npoints-1)
            new_x = x + dxi
            if j == start_i and do_perturb == False:
                # add a small random perturbation to lift symmetry
                new_x = new_x + perturb()
                
            
        cluster[i].setDisplacedCoordinates(new_x)
                
        outstream = open('disp.{}.{}.gin'.format(i, basename), 'w')
        gulp.write_gulp(outstream, cluster, sysinfo, defected=True, to_cart=False,
                             rI_centre=rI_centre, relax_type='', add_constraints=True)
        outstream.close()
            
        # if an executable has been provided, run the calculation
        if executable is not None:
            gulp.run_gulp(executable, 'disp.{}.{}'.format(i, basename))  
                   
        E = util.extract_energy('disp.{}.{}.gout'.format(i, basename), 'gulp')[0]  
        grid.append(new_z)
        energies.append(E)
        
    # unset the constraints
    cluster[start_i].set_constraints(np.ones(3))
    if newspecies is not None:
        cluster[start_i].setSpecies(oldspecies)
                
    # if energies have been calculated, extract the maximum energy (relative to
    # the undisplaced atom), the barrier height, and the energy difference 
    # between the initial and final sites
    if grid:
        energies = np.array(energies)
        energies -= energies.min()
        barrier_height = get_barrier(energies)
        site_energy_diff = energies[-1]-energies[0]
        
        return [[z, E] for z, E in zip(grid, energies)], barrier_height, site_energy_diff
    else:
        # energies not calculated, return dummy values
        return [], np.nan, np.nan
    
def adaptive_construct(index, cluster, sysinfo, dz, nlevels, basename, 
                       executable, rI_centre=np.zeros(2), dx=np.zeros(2),
                                       plane_shift=np.zeros(2), node=0.5):
    '''Constructs input files for points along the atom migration path, with
    the points determined by a binary search algorithm. May fail for complex
    paths (eg. those with multiple local maxima).
    '''
    
    ## user MUST provide an executable
    #if executable is None:
    #    raise ValueError("A valid executable must be provided.")
    
    # starting coordinates
    x = cluster[index].getCoordinates()
    
    # lists to hold grid spacing and energies
    grid = []
    energies = []
    
    # do first level
    for i in range(3):
        new_z = i/2.*dz
        if executable is not None:
            # calculate the energy directly
            
            pln_x = i/2.*dx
            # calculate displacement of atom in the plane at this point
            shift = scale_plane_shift(plane_shift, new_z, dz, node)
            
            # update dislocation structure
            new_x = x + np.array([shift[0]+pln_x[0], shift[1]+pln_x[1], new_z])
            cluster[index].setCoordinates(new_x)
            outstream = open('disp.{}.{}.gin'.format(i, basename), 'w')
            gulp.write_gulp(outstream, cluster, sysinfo, defected=False,
                      to_cart=False, rI_centre=rI_centre, relax_type='',
                                                    add_constraints=True)
            outstream.close()
        
            gulp.run_gulp(executable, 'disp.{}.{}'.format(i, basename))
      
        # extract energy from GULP output file            
        E = util.extract_energy('disp.{}.{}.gout'.format(i, basename), 'gulp')[0]
        
        grid.append(new_z)
        energies.append(E)
    
    # refine grid subsequent levels 
    imax = 1
    E_max = energies[imax] 
    grid_max = grid[imax]  
    counter = 3 # keep track of number of structures calculated
    for level in range(nlevels-1):
        # nodes halfway between the current maximum and the adjacent nodes
        new_z_m1 = grid_max-0.5**(level+2)*dz
        new_z_p1 = grid_max+0.5**(level+2)*dz
        grid.insert(imax, new_z_m1)
        grid.insert(imax+2, new_z_p1)
        
        if executable is not None:
            # in-plane change associate with new nodes
            pln_x_m1 = new_z_m1/dz*dx
            pln_x_p1 = new_z_p1/dz*dx
                       
            shift_m1 = scale_plane_shift(plane_shift, new_z_m1, dz, node)
            shift_p1 = scale_plane_shift(plane_shift, new_z_p1, dz, node)
            
            # update dislocation structure
            new_x_m1 = x + np.array([shift_m1[0]+pln_x_m1[0], shift_m1[1]+pln_x_m1[1], 
                                                                            new_z_m1])
            new_x_p1 = x + np.array([shift_p1[0]+pln_x_p1[0], shift_p1[1]+pln_x_p1[1],
                                                                            new_z_p1])
            
            for i in range(2):
                if i == 0:
                    # do lower point
                    cluster[index].setCoordinates(new_x_m1)
                else:
                    # do upper point
                    cluster[index].setCoordinates(new_x_p1)
                    
                outstream = open('disp.{}.{}.gin'.format(counter, basename), 'w')           
                gulp.write_gulp(outstream, cluster, sysinfo, defected=False, to_cart=False,
                                 rI_centre=rI_centre, relax_type='', add_constraints=True)
                outstream.close()
                
                # calculate energies
                if executable is not None:
                    gulp.run_gulp(executable, 'disp.{}.{}'.format(counter, basename))     
                
        E = util.extract_energy('disp.{}.{}.gout'.format(counter, basename), 'gulp')[0]           
        energies.insert(imax+2*i, E)
            
        counter += 1    
            
        # determine new maximum energy
        if energies[imax] > energies[imax+1] and energies[imax] > energies[imax+1]:
            E_max = energies[imax]
        elif energies[imax+2] > energies[imax+1]:
            E_max = energies[imax+2]
            imax +=2
        else:
            # highest energy location unchanged
            imax += 1
            
    # shift energy so that undisplaced vacancy is as 0 eV
    energies = np.array(energies)
    energies -= energies.min()
    barrier_height = get_barrier(energies)
    site_energy_diff = energies[-1]-energies[0]
            
    return [[z, E] for z, E in zip(grid, energies)], barrier_height, site_energy_diff
        
def construct_disp_files(index, cluster, sysinfo, dz, npoints, basename, 
                       rI_centre=np.zeros(2), executable=None, node=0.5,
                               plane_shift=np.zeros(2),  dx=np.zeros(2)):
    '''Constructs input files for points along the atom migration path.
    '''
        
    x = cluster[index].getCoordinates()
    grid = []
    energies = []

    for i in range(npoints):
        # calculate new position of atom
        new_z = i/(npoints-1)*dz
        pln_x = i/(npoints-1)*dx
        # calculate displacement of atom in the plane at this point DUE TO PLANE
        # SHIFT
        shift = scale_plane_shift(plane_shift, new_z, dz, node)
        
        # update dislocation structure
        new_x = x + np.array([shift[0]+pln_x[0], shift[1]+pln_x[1], new_z])
        cluster[index].setCoordinates(new_x)
        outstream = open('disp.{}.{}.gin'.format(i, basename), 'w')
        gulp.write_gulp(outstream, cluster, sysinfo, defected=False, to_cart=False,
                         rI_centre=rI_centre, relax_type='', add_constraints=True)
        outstream.close()
        
        # if an executable has been provided, run the calculation
        if executable is not None:
            gulp.run_gulp(executable, 'disp.{}.{}'.format(i, basename))         
            E = util.extract_energy('disp.{}.{}.gout'.format(i, basename), 'gulp')[0]  
            grid.append(new_z)
            energies.append(E)
            
    # if energies have been calculated, extract the maximum energy (relative to
    # the undisplaced atom), the barrier height, and the energy difference 
    # between the initial and final sites
    if grid:
        energies = np.array(energies)
        energies -= energies.min()
        barrier_height = get_barrier(energies)
        site_energy_diff = energies[-1]-energies[0]
        
        return [[z, E] for z, E in zip(grid, energies)], barrier_height, site_energy_diff
    else:
        # energies not calculated, return dummy values
        return [], np.nan, np.nan    

def get_barrier(energy_values):
    '''Calculates the maximum energy barrier that has to be surmounted getting
    between sites.
    '''
    
    eb = energy_values.max()-energy_values.min()
    
    return eb    

def migrate_sites(basename, rI, rII, atom_type, npoints, executable=None, 
             noisy=False, plane_shift=np.zeros(2), node=0.5, adaptive=False,
                  threshold=5e-1, newspecies=None, centre_on_impurity=False):
    '''Constructs and, if specified by user, runs input files for migration
    of vacancies along a dislocation line. <plane_shift> allows the user to 
    migrate the atom around intervening atoms (eg. oxygen ions). <adaptive> tells
    the program to perform a binary search for the maximum (with <npoints> levels)
    '''
    
    # read in list of sites
    site_info = read_sites(basename)
    heights = []
    
    if noisy:
        print("Preparing to calculate migration barriers.")
    
    for site in site_info:
        sitename = '{}.{}'.format(basename, int(site[0]))
        
        if noisy:
            print("Calculating migration barrier for site {}...".format(int(site[0])), end='')
   
        cluster, sysinfo = gulp.cluster_from_grs('{}.grs'.format(sitename), rI, rII)
        
        # height of simulation cell
        H = cluster.getHeight()
                                                
        # find atom to translate                                       
        possible_sites = adjacent_sites(site, cluster, atom_type, threshold=threshold)
        translate_index = atom_to_translate(site, possible_sites, cluster)

        for ti in translate_index:       
            if noisy:
                print('...from site {}...'.format(ti), end='')
                
            x0 = cluster[ti].getCoordinates()

            # calculate translation distance
            z0 = x0[-1]
            z1 = site[3]

            if z1 > z0:
                dz = z1-z0
            else:
                dz = z1+(H-z0)

            # calculate the required change of axial position, r
            r0 = x0[:-1]
            r1 = np.array(site[1:3])
            
            dr = r1-r0

            # change constraints for atom -> can only relax in plane normal to dislocation line
            cluster[ti].set_constraints([1, 1, 0])
            
            # change the species of the migrating atom, if <newspecies> specified
            if newspecies is not None:
                oldspecies = cluster[ti].getSpecies()
                cluster[ti].setSpecies(newspecies)
                
            # determine centre of region I
            if centre_on_impurity:
                rI_centre=site[1:3]
            else:
                rI_centre=np.zeros(2)
            
            # construct input files and run calculation (if specified), recording
            # the index of the defect and the atom being translated
            sitepairname = '{}.{}'.format(sitename, ti)
            if not adaptive:
                gridded_energies, Eh, Ed = construct_disp_files(ti, 
                                                                cluster, 
                                                                sysinfo, 
                                                                dz, 
                                                                npoints, 
                                                                sitepairname,
                                                                rI_centre=rI_centre, 
                                                                executable=executable,
                                                                plane_shift=plane_shift, 
                                                                node=node,
                                                                dx=dr
                                                               )
            else:
                gridded_energies, Eh, Ed = adaptive_construct(ti, 
                                                              cluster, 
                                                              sysinfo, 
                                                              dz, 
                                                              npoints, 
                                                              sitepairname, 
                                                              rI_centre=rI_centre,
                                                              executable=executable, 
                                                              plane_shift=plane_shift, 
                                                              node=node,
                                                              dx=dr
                                                             )
                                        
            outstream = open('disp.{}.barrier.dat'.format(sitepairname), 'w')    
            # write header, including full displacement vector and barrier height 
            outstream.write('# {:.3f} {:.3f} {:.3f}\n'.format(dr[0], dr[1], dz))
           
            # write energies to file if they have been calculated
            if gridded_energies:     
                # write energies along path
                for z, E in gridded_energies:
                    outstream.write('{} {:.6f}\n'.format(z, E))
                
                heights.append([int(site[0]), ti, site[1], site[2], Eh, Ed])
            
            outstream.close()
                
            # undo any changes to atom <ti>
            cluster[ti].set_constraints([1, 1, 1])
            cluster[ti].setCoordinates(x0)
            if newspecies is not None:
                # revert the species of the translated atom
                cluster[ti].setSpecies(oldspecies)
        
            if noisy:
                print("done.")
                
        if noisy:
            print('done.')
            
    return heights
    
def reorder_path(energy_path):
    '''Changes the zero index of <energypath> so that it starts with the lowest
    energy point.
    '''
    
    # get index of lowest energy point
    imin = np.where(energy_path == energy_path.min())[0][0]
    
    shifted_path = np.array(list(energy_path[imin:])+list(energy_path[:imin]))
    return shifted_path
                                                        
def read_migration_barrier(sitename, npoints, program='gulp'):
    '''Extracts the energies for points along the migration path of an individual
    defect near a dislocation line. The variable <program> is future-proofing.
    '''
    
    energies = []
    for i in range(npoints):
        energies.append(util.extract_energy('disp.{}.{}.gout'.format(i, sitename),
                                                                      program)[0])
                                                                          
    # shift energies so that the undisplaced defect is at 0 eV
    energies = np.array(energies)
    energies -= energies.min()
    
    # reorder barrier so that the index of the minimum energy point is 0
    #energies = reorder_path(energies)
    return energies
    
def extract_barriers_even(basename, npoints, program='gulp'):
    '''Extracts migration barriers for ALL sites near the dislocation core when
    an evenly spaced grid has been used. 
    '''
    
    site_info = read_sites(basename)
    heights = []

    for site in site_info:
        i = int(site[0])

        # extract a list of all sites whose occupants have been migrated to 
        # site <i>
        gout_files = glob.glob('disp.*.{}.{}.*.gout'.format(basename, i))
        mig_indices = set()
        for outfl in gout_files:
            # match the second index
            ti = int(re.search(r'.+\.(?P<j>\d+)\.gout', outfl).group('j'))
            mig_indices.add(ti)

        # extract barrier for each pair of indices
        for j in mig_indices:    
            sitename = '{}.{}.{}'.format(basename, i, j)
            energy = read_migration_barrier(sitename, npoints, program)

            # extract migration distance
            barrier_info = open('disp.{}.barrier.dat'.format(sitename), 'r')
            header = barrier_info.readlines()[0].rstrip()
            dx = float(header.split()[3])   
            barrier_info.close() 
               
            # record migration energies
            outstream = open('disp.{}.barrier.dat'.format(sitename), 'w')
            outstream.write(header+'\n')
            npoints = len(energy)
            for k, E in enumerate(energy):
                outstream.write('{:.3f} {:.6f}\n'.format(k*dx/npoints, E))
            outstream.close()
            
            # record barrier height and maximum
            Eh = get_barrier(energy)
            Ed = energy[-1]-energy[0]
            heights.append([i, j, site[1], site[2], Eh, Ed])
            
    return heights     
    
def write_heights(basename, heights):
    '''Writes migration barrier heights to file, including both the difference
    between the initial and maximum energy, and between the lowest and highest
    energy.
    '''
    
    outstream = open('{}.barrier.dat'.format(basename), 'w')
    outstream.write('# site-index atom-index x y barrier-height net-energy-diff\n')
    for site in heights:
        outstream.write('{} {} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(site[0],  
                                  site[1], site[2], site[3], site[4], site[5]))   
    outstream.close() 
                                                        
def read_heights(basename):
    '''Reads migration barrier heights.
    '''

    barrier_info = []
    with open('{}.barrier.dat'.format(basename)) as f: 
        for line in f:
            if line.startswith('#'):
                continue
            elif not line.rstrip():
                continue
            # else
            site_line = line.rstrip().split()
            barrier_info.append([float(x) for x in site_line])
            
    return np.array(barrier_info)  
    
def transition_rfo(strucname, rI, rII, maxiter=2500, executable=None):
    '''Takes a defect structure produced by the constrained minimization routine
    and sets up a transition state calculation using the RFO algorithm to find the
    saddle point. <maxiter> is large because of the slow convergence of the 
    l-BFGS algorithm used by GULP to find the transition state. 
    '''
    
    # produce input file for transition state calculation
    cluster, sysinfo = gulp.cluster_from_grs('{}.grs'.format(strucname), rI, rII)     
    ostream = open('rfo.{}.gin'.format(strucname), 'w')
    gulp.write_gulp(ostream, cluster, sysinfo,  prop=False, maxiter=maxiter, 
                        transition=True, do_relax=False, add_constraints=True)
    
    # run the calculation if an executable has been provided                    
    if executable is not None:                    
        gulp.run_gulp(executable, 'rfo.{}'.format(strucname))
    
    return
    
def plot_barriers(heights, plotname, r, mirror_both=False, mirror=False, mirror_axis=1,
                                                         inversion=False, tolerance=1.):
    '''Plots the lowest energy migration path for each site around the 
    dislocation core. 
    ''' 
    
    heights = np.array(heights)
    
    # extract barrier heights for each site
    siteinfo = dict()
    for h in heights:
        i = int(h[0])
        if i in siteinfo.keys():
            if h[-1] < siteinfo[i]['E']:
                siteinfo[i]['E'] = h[-2]
                siteinfo[i]['x'] = h[2:4]
        else:
            siteinfo[i] = dict()
            siteinfo[i]['E'] = h[-2]
            siteinfo[i]['x'] = h[2:4]

    # construct positions and energies           
    sites = []
    barriers = []
    for k in siteinfo:
        sites.append([k, siteinfo[k]['x'][0], siteinfo[k]['x'][1]])
        barriers.append(siteinfo[k]['E'])
                
        if mirror:
            # check to make sure that the site does not lie on the mirror axis
            if abs(siteinfo[k]['x'][(mirror_axis+1) % 2]) < tolerance:
                continue
        elif inversion:
            # check that the atom is not at the origin
            if np.sqrt(siteinfo[k]['x'][0]**2+siteinfo[k]['x'][1]**2) < tolerance:
                continue
        
        # reflect the site, if required
        if (mirror and mirror_axis == 0) or mirror_both:
            sites.append([k, siteinfo[k]['x'][0], -siteinfo[k]['x'][1]]) 
            barriers.append(siteinfo[k]['E'])
        if (mirror and mirror_axis == 1) or mirror_both:
            sites.append([k, -siteinfo[k]['x'][0], siteinfo[k]['x'][1]]) 
            barriers.append(siteinfo[k]['E'])
        if mirror_both:
            sites.append([k, -siteinfo[k]['x'][0], -siteinfo[k]['x'][1]])
            barriers.append(siteinfo[k]['E'])
            
        # invert site, if requested
        if inversion:
            sites.append([k, -siteinfo[k]['x'][0], -siteinfo[k]['x'][1]])
            barriers.append(siteinfo[k]['E'])
        
    sites = np.array(sites)
    barriers = np.array(barriers)
    
    # make plot
    seg.plot_energies_contour(sites, barriers, plotname, r)
    
### ROUTINES FOR MIGRATION IN THE BULK ###

def adjacent_sites_3d(x0, simcell, species, direction=np.array([1., 0., 0.]),
                                                                 dottol=1e-3):
    '''Generates a list of atomic sites in a 3D-periodic lattice
    that are along <direction>, originating at the <dfct_site>.
    '''

    x0 = np.array(x0)

    # unit vector along <direction>
    unit_dir = direction/np.linalg.norm(direction)

    # find sites
    adjacent_indices = []
    for j, atom in enumerate(simcell):
        if not(species.lower() == atom.getSpecies().lower()):
            continue
        #else
        # calculate unit vector in the direction from the defect <site> to 
        # the <atom>
        x = atom.getCoordinates() % 1
        dx = x0-x
        
        if np.linalg.norm(dx) < 1e-6:
            # atom is on site
            continue
            
        dxn = dx/np.linalg.norm(dx)
        dp = abs(np.dot(dxn, unit_dir))
        if abs(dp - 1) < dottol:
            adjacent_indices.append([j, dx])
            
    return adjacent_indices

def mindist3d(possible_sites):
    '''Finds the atom in <possible_sites> closest to <site>.
    '''
    
    # holds info about closest site
    minindex = np.nan
    mindist = np.inf
    mindx = np.zeros(2)
    
    for i, dx in possible_sites:
        # absolute distance from atom to site
        delta = np.linalg.norm(dx)
        if delta < mindist:
            mindist = delta
            mindx = dx
            minindex = i
            
    return minindex, mindx
