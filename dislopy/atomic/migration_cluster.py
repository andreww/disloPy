#!/usr/bin/env python
'''Module to calculate migration barriers for atoms diffusing along a dislocation
line.
'''
from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.linalg import norm
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
        return [], np.nan    

def get_barrier(energy_values):
    '''Calculates the maximum energy barrier that has to be surmounted getting
    between sites.
    '''
    
    eb = energy_values.max()-energy_values.min()
    
    return eb    

def migrate_sites(basename, n, rI, rII, atom_type, npoints, executable=None, 
                 noisy=False, plane_shift=np.zeros(2), node=0.5, adaptive=False,
                                               threshold=5e-1, newspecies=None):
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
            
            dr = norm(r1-r0)
            print(dr)
            # change constraints for atom -> can only relax in plane normal to dislocation line
            cluster[ti].set_constraints([1, 1, 0])
            
            # change the species of the migrating atom, if <newspecies> specified
            if newspecies is not None:
                oldspecies = cluster[ti].getSpecies()
                cluster[ti].setSpecies(newspecies)
            
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
                                                                rI_centre=site[1:3], 
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
                                                              rI_centre=site[1:3],
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
            heights.append([i, j, site[1], site[2], get_barrier(energy)])
            
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
                                                        
def read_heights(basename, heights):
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
            site_info.append([float(x) for x in site_line])
            
    return np.array(site_info)  
    
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
                siteinfo[i]['E'] = h[-1]
                siteinfo[i]['x'] = h[2:4]
        else:
            siteinfo[i] = dict()
            siteinfo[i]['E'] = h[-1]
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
