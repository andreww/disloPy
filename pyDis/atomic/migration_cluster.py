#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
from numpy.linalg import norm
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic.multisite import periodic_distance
from pyDis.atomic import gulpUtils as gulp

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

def atoms_to_translate(dfct_site, possible_sites, cluster, tol=1e-1):
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
    z2 = atom2.getCoordinates()[-2]
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

def disp_distance(cluster, n, site_dist):
    '''Using the height of cluster, which is <n> unit cells high, and the distance
    <site_dist> between sites on either site of the vacancy.   
    '''
    
    return site_dist - cluster.getHeight()/n

def construct_disp_files(index, cluster, sysinfo, dx, npoints, basename, 
                                  rI_centre=np.zeros(2), executable=None):
    '''Constructs input files for points along the atom migration path.
    '''
    
    # change constraints for atom -> can only relax in plane normal to dislocation line
    cluster[index].set_constraints([1, 1, 0])

    # create input files for each point on migration path
    x = cluster[index].getCoordinates()
    for i in range(npoints):
        # calculate new position of atom
        new_x = x + np.array([0, 0, i/(npoints-1)*dx])
        cluster[index].setCoordinates(new_x)
        #cluster.specifyRegions()
        outstream = open('disp.{}.{}.gin'.format(i, basename), 'w')
        gulp.write_gulp(outstream, cluster, sysinfo, defected=False, to_cart=False,
                         rI_centre=rI_centre, relax_type='', add_constraints=True)
        outstream.close()
        # if an executable has been provided, run the calculation
        if executable is not None:
            gulp.run_gulp(executable, 'disp.{}.{}'.format(i, basename))

def migrate_sites(basename, n, r1, r2, atom_type, npoints, executable=None):
    '''Constructs and, if specified by user, runs input files for migration
    of vacancies along a dislocation line.
    '''
    
    # read in list of sites
    site_info = read_sites(basename)
    
    for site in site_info:
        cluster, sysinfo = gulp.cluster_from_grs('{}.{}.grs'.format(basename, 
                                                str(int(site[0]))), r1, r2)
                                                
        # find atom to translate                                       
        possible_sites = adjacent_sites(site, cluster, atom_type)
        translate_index = atom_to_translate(site, possible_sites, cluster)
        
        # calculate translation distance
        next_index, intersite_dist = next_occupied_site(translate_index,
                                                    possible_sites, cluster)
        dx = disp_distance(cluster, n, intersite_dist)
        
        # construct input files and run calculation (if specified)
        construct_disp_files(translate_index, cluster, sysinfo, dx, npoints,
                 '{}.{}'.format(basename, int(site[0])), rI_centre=site[1:3],
                                                        executable=executable)
