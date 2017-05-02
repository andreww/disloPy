#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
from numpy.linalg import norm
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic.multisite import periodic_distance
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import atomistic_utils as util

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

def atom_to_translate(dfct_site, possible_sites, cluster, tol=5e-1):
    '''Determines which of the possible adjacent sites to translate. 
    '''
    
    # location of vacancy
    x0 = dfct_site[1:4]
    
    # to hold list of possible candidates for translation
    candidates = []
    min_dist = np.inf
    
    # find atoms closest to vacancy
    for atom_index, dist in possible_sites:
        if dist < min_dist+tol:
            candidates.append(atom_index)
            if dist < min_dist:
                # update minimum distance
                min_dist = dist
                
    # make sure that there are not too many candidates
    if len(candidates) > 3:
        raise ValueError("Should be fewer than 3 adjacent sites")
    # if only 1 candidate, we are done
    elif len(candidates) == 1:
        return candidates[0]
    else:
        # otherwise, determine which site is below the vacancy  
        z_site = x0[-1]
        i0, i1 = candidates
        z0 = cluster[i0].getCoordinates()[-1]
        z1 = cluster[i1].getCoordinates()[-1]
        if z0 < z_site and z0 < z1:
            return i0
        elif z1 < z_site and z1 < z0:
            return i1
        elif z0 > z_site and z1 > z_site: 
            if z0 < z1:
                return i1
            else: # z0 > z1
                return i0
        elif z0 < z_site and z1 < z_site:
            if z0 < z1:
                return i1
            else: # z0 > z1
                return i0
                
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
    
def adaptive_construct(index, cluster, sysinfo, dx, nlevels, basename, 
                                    executable, rI_centre=np.zeros(2),
                                    plane_shift=np.zeros(2), node=0.5):
    '''Constructs input files for points along the atom migration path, with
    the points determined by a binary search algorithm. May fail for complex
    paths (eg. those with multiple local maxima).
    '''
    
    # user MUST provide an executable
    if executable is None:
        raise ValueError("A valid executable must be provided.")
    
    # starting coordinates
    x = cluster[index].getCoordinates()
    
    # lists to hold grid spacing and energies
    grid = []
    energies = []
    
    # do first level
    for i in range(3):
        new_z = i/2.*dx
        # calculate displacement of atom in the plane at this point
        shift = scale_plane_shift(plane_shift, new_z, dx, node)
        
        # update dislocation structure
        new_x = x + np.array([shift[0], shift[1], new_z])
        cluster[index].setCoordinates(new_x)
        outstream = open('disp.{}.{}.gin'.format(i, basename), 'w')
        gulp.write_gulp(outstream, cluster, sysinfo, defected=False, to_cart=False,
                         rI_centre=rI_centre, relax_type='', add_constraints=True)
        outstream.close()
        
        # calculate energy and extract energy
        gulp.run_gulp(executable, 'disp.{}.{}'.format(i, basename))        
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
        new_z_m1 = grid_max-0.5**(level+2)*dx
        new_z_p1 = grid_max+0.5**(level+2)*dx
        
        grid.insert(imax, new_z_m1)
        grid.insert(imax+2, new_z_p1)
        
        shift_m1 = scale_plane_shift(plane_shift, new_z_m1, dx, node)
        shift_p1 = scale_plane_shift(plane_shift, new_z_p1, dx, node)
        
        # update dislocation structure
        new_x_m1 = x + np.array([shift_m1[0], shift_m1[1], new_z_m1])
        new_x_p1 = x + np.array([shift_p1[0], shift_p1[1], new_z_p1])
        
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
    energies -= energies[0]
    Emax = energies.max()
    barrier_height = energies.max()-energies.min()
            
    return [[z, E] for z, E in zip(grid, energies)], Emax, barrier_height
        
def construct_disp_files(index, cluster, sysinfo, dx, npoints, basename, 
                                 rI_centre=np.zeros(2), executable=None, 
                                      plane_shift=np.zeros(2), node=0.5):
    '''Constructs input files for points along the atom migration path.
    '''
        
    x = cluster[index].getCoordinates()
    for i in range(npoints):
        # calculate new position of atom
        new_z = i/(npoints-1)*dx
        # calculate displacement of atom in the plane at this point
        shift = scale_plane_shift(plane_shift, new_z, dx, node)
        
        # update dislocation structure
        new_x = x + np.array([shift[0], shift[1], new_z])
        cluster[index].setCoordinates(new_x)
        outstream = open('disp.{}.{}.gin'.format(i, basename), 'w')
        gulp.write_gulp(outstream, cluster, sysinfo, defected=False, to_cart=False,
                         rI_centre=rI_centre, relax_type='', add_constraints=True)
        outstream.close()
        
        # if an executable has been provided, run the calculation
        if executable is not None:
            gulp.run_gulp(executable, 'disp.{}.{}'.format(i, basename)) 

def migrate_sites(basename, n, r1, r2, atom_type, npoints, executable=None, 
                 noisy=False, plane_shift=np.zeros(2), node=0.5, adaptive=False):
    '''Constructs and, if specified by user, runs input files for migration
    of vacancies along a dislocation line. <plane_shift> allows the user to 
    migrate the atom around intervening atoms (eg. oxygen ions). <adaptive> tells
    the program to perform a binary search for the maximum (with <npoints> levels)
    '''
    
    # read in list of sites
    site_info = read_sites(basename)
    heights = []
    
    for site in site_info:
        if noisy:
            print("Calculating migration barrier at site {}...".format(int(site[0])), end='')
            
        cluster, sysinfo = gulp.cluster_from_grs('{}.{}.grs'.format(basename, 
                                                str(int(site[0]))), r1, r2)
                                                
        # find atom to translate                                       
        possible_sites = adjacent_sites(site, cluster, atom_type)
        translate_index = atom_to_translate(site, possible_sites, cluster)
        
        # calculate translation distance
        next_index, intersite_dist = next_occupied_site(translate_index, 
                                                possible_sites, cluster)
        dx = disp_distance(cluster, n, intersite_dist)
        
        # change constraints for atom -> can only relax in plane normal to dislocation line
        cluster[translate_index].set_constraints([1, 1, 0])
        
        # construct input files and run calculation (if specified)
        sitename = '{}.{}'.format(basename, int(site[0]))
        if not adaptive:
            construct_disp_files(translate_index, cluster, sysinfo, dx, npoints,
                           sitename, rI_centre=site[1:3], executable=executable,
                                             plane_shift=plane_shift, node=node)
        else:
            gridded_energies, Emax, Eh = adaptive_construct(translate_index, cluster, 
                             sysinfo, dx, npoints, sitename, rI_centre=site[1:3],
                       executable=executable, plane_shift=plane_shift, node=node)
                                        
            # write energies to file
            outstream = open('disp.{}.barrier.dat'.format(basenam), 'w')
            for z, E in gridded_energies:
                outstream.write('{} {:.6f}\n'.format(j, E))
            outstream.close()
            
            heights.append([int(site[0]), site[1], site[2], Emax, Eh])
        
        if noisy:
            print("done.")
            
    return heights
                                                        
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
    energies -= energies[0] 
    
    return energies
    
def extract_barriers_even(basename, npoints, program='gulp'):
    '''Extracts migration barriers for ALL sites near the dislocation core when
    an evenly spaced grid has been used. 
    '''
    
    site_info = read_sites(basename)
    heights = []
    
    for site in site_info:
        i = int(site[0])
        sitename = '{}.{}'.format(basename, i)
        energy = read_migration_barrier(sitename, npoints, program)
        
        # record migration energies
        outstream = open('disp.{}.barrier.dat'.format(sitename), 'w')
        for j, E in enumerate(energy):
            outstream.write('{} {:.6f}\n'.format(j, E))
        outstream.close()
        
        # calculate the difference between the minimum and maximum energies along
        # the path. Note: NOT the same as the difference between the maximum 
        # energy and the energy of the undisplaced defect (because of site hopping)
        heights.append([int(site[0]), site[1], site[2], energy.max(), 
                                          energy.max()-energy.min()])
            
    return heights            
