#!/usr/bin/env python
'''Module to calculate migration barriers for atoms diffusing along a dislocation
line.
'''
from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.linalg import norm
from numpy.random import random
from multiprocessing import Pool

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
    
def perturb(mag=0.01):
    '''Apply random perturbation to coordinates to reduce symmetry.
    '''
    
    return 2*mag*random(3)-mag
    
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
            coords = [float(x) for x in match.rstrip().split()[1:-1]]
            if i == 0:
                site_dict[site_index] = dict()
                site_dict[site_index]['bonded_sites'] = dict()
                site_dict[site_index]['site_coords'] = np.array(coords)
                current_site = site_index
            else:
                site_dict[current_site]['bonded_sites'][site_index] = np.array(coords)
                
    return site_dict
    
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
    
def displacement_vecs_new(cluster, x0, xfinal, npoints):
    '''Construct displacement vectors for the diffusing atom in the cluster.
    '''

    H = cluster.getHeight()
    
    dx = xfinal - x0
    
    # check that the final element of dx is right
    dz = dx[-1]
    if abs(dz-H) < abs(dz):
        dz = dz - H
    elif abs(dz+H) < abs(dz):
        dz = dz + H
        
    dx[-1] = dz
    
    # compute incremental values of displacement vector
    dxn = [x0+ni*dx/(npoints-1) for ni in range(npoints)]
    
    # determine direction to constrain
    constrain_index = max_index(dx)

    return dxn, constrain_index
    
def index_atom_at_x(cluster, x0):
    '''Returns the index of the atom in cluster with coordinates <x0>.
    '''
    
    max_d = np.inf
    index = np.nan
    for i, atom in enumerate(cluster):
        dx = norm(atom.getCoordinates()-x0)
        if dx < max_d:
            index = i
            max_d = dx
            
    return index 
    
def make_disp_files_general(cluster, diffuse_i, basename, dxn, npoints, sysinfo,  
                       executable=None, rI_centre=np.zeros(2), do_perturb=False, 
                                            constrain_index=2, newspecies=None):
    '''Generates input files for a constrained optimization calculation of migration
    barriers along an arbitrary migration path.
    '''
    
    # set constraints
    constraint_vector = np.ones(3)
    constraint_vector[constrain_index] = 0
    cluster[diffuse_i].set_constraints(constraint_vector)
    
    # change species of diffusing atom, if requested
    if newspecies is not None:
        oldspecies = cluster[diffuse_i].getSpecies()
        cluster[diffuse_i].setSpecies(newspecies)

    # lists to hold grid spacing and energies
    grid = []
                                                                           
    for i, dxi in enumerate(dxn):
        # update dislocation structure
        if do_perturb:
                # add a small random perturbation to lift symmetry
                new_x = dxi + perturb() 
        else:
                new_x = dxi   
                
        cluster[diffuse_i].setDisplacedCoordinates(new_x)
               
        outstream = open('disp.{}.{}.gin'.format(i, basename), 'w')
        gulp.write_gulp(outstream, cluster, sysinfo, defected=True, to_cart=False,
                             rI_centre=rI_centre, relax_type='', add_constraints=True)
        outstream.close()
        
        new_z = norm(new_x-cluster[diffuse_i].getCoordinates())
        grid.append(new_z)
 
    # unset the displaced coordinates, constraints, and new species
    cluster[diffuse_i].setDisplacedCoordinates(cluster[diffuse_i].getCoordinates())
    cluster[diffuse_i].set_constraints(np.ones(3))
    if newspecies is not None:
        cluster[diffuse_i].setSpecies(oldspecies)
        
    return np.array(grid)
        
def make_disp_files_pipe(index, cluster, sysinfo, dz, npoints, basename, 
                       rI_centre=np.zeros(2), executable=None, node=0.5,
                               plane_shift=np.zeros(2),  dx=np.zeros(2)):
    '''Constructs input files for points along an atom migration path along the
    dislocation line
    '''
        
    x = cluster[index].getCoordinates()
    grid = []

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
        
        grid.append(new_z)
        
    return grid 

def get_barrier(energy_values):
    '''Calculates the maximum energy barrier that has to be surmounted getting
    between sites.
    '''
    
    eb = energy_values.max()-energy_values.min()
    
    return eb 
    
def migrate_sites_general(basename, rI, rII, bondlist, npoints, executable=None, 
                    noisy=False, plane_shift=np.zeros(3), node=0.5, threshold=1,
                    centre_on_impurity=False, do_perturb=False, newspecies=None, 
                              in_parallel=False, nprocesses=1, read_output=True):
    '''Calculates migration barriers between all pairs of atoms that are deemed
    to be bonded.
    '''
                     
    bond_dict = parse_bonds('{}.bonds.txt'.format(basename))
    
    site_pairs = []
    energy_dict = dict()
    for i in bond_dict.keys():
        start, sysinfo = gulp.cluster_from_grs('{}.{}.grs'.format(basename, i), rI, rII)
        xfinal = bond_dict[i]['site_coords']
        for j in bond_dict[i]['bonded_sites'].keys():
            # get coordinates and index of diffusing atom
            x0 = bond_dict[i]['bonded_sites'][j]
            diff_index = index_atom_at_x(start, x0)

            # check that diff_index is an integer
            if type(diff_index) is not int:
                raise AttributeError("Missing atom at site {:.0f}".format(j))
                
            dxn, constrain_index = displacement_vecs_new(start, x0, xfinal, npoints)
                                                 
            # determine centre of region I
            if centre_on_impurity:
                rI_centre=start[diff_index].getCoordinates()[:-1]
            else:
                rI_centre=np.zeros(2) 
            
            pair_name = '{}.{}.{}'.format(basename, i, j)  
            site_pairs.append(pair_name)

            grid = make_disp_files_general(start,
                                           diff_index,
                                           pair_name,
                                           dxn,
                                           npoints,
                                           sysinfo,
                                           executable=executable,
                                           rI_centre=rI_centre,
                                           do_perturb=do_perturb,
                                           constrain_index=constrain_index,
                                           newspecies=newspecies
                                          )
            
            energy_dict[pair_name] = dict()
            energy_dict[pair_name]['grid'] = np.copy(grid)
            energy_dict[pair_name]['x0'] = np.copy(x0)
            energy_dict[pair_name]['x1'] = np.copy(xfinal)
    
    # calculate energies, if requested to do so by the user
    if executable is not None:                                               
        calculate_migration_points(site_pairs, executable, npoints,
                                     in_parallel=in_parallel, np=nprocesses)
    
    if read_output:             
        heights = read_migration_energies(energy_dict, npoints, in_subdirectory=not(in_parallel)) #!
        return heights
    else:
        return None

def migrate_sites_pipe(basename, rI, rII, atom_type, npoints, executable=None, 
                               noisy=False, plane_shift=np.zeros(2), node=0.5,
                    threshold=5e-1, newspecies=None, centre_on_impurity=False,
                            in_parallel=False, nprocesses=1, read_output=True):
    '''Constructs and, if specified by user, runs input files for migration
    of vacancies along a dislocation line. <plane_shift> allows the user to 
    migrate the atom around intervening atoms (eg. oxygen ions). <adaptive> tells
    the program to perform a binary search for the maximum (with <npoints> levels)
    '''
    
    # read in list of sites
    site_info = read_sites(basename)
    
    if noisy:
        print("Preparing to construct input files for migration calculations.")
    
    site_pairs = []
    energy_dict = dict() #! 
    for site in site_info:
        sitename = '{}.{}'.format(basename, int(site[0]))
        
        #if noisy:
        #    print("Calculating migration barrier for site {}...".format(int(site[0])), end='')
        
        if not in_parallel:
            cluster, sysinfo = gulp.cluster_from_grs('{}.grs'.format(sitename), rI, rII)
        else:
            # sub-directory exists for this file
            cluster, sysinfo = gulp.cluster_from_grs('{}/{}.grs'.format(sitename, sitename), rI, rII)
        
        # height of simulation cell
        H = cluster.getHeight()
                                                
        # find atom to translate                                       
        possible_sites = adjacent_sites(site, cluster, atom_type, threshold=threshold)
        translate_index = atom_to_translate(site, possible_sites, cluster)

        for ti in translate_index:       
            #if noisy:
            #    print('...from site {}...'.format(ti), end='')
                
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
            site_pairs.append(sitepairname)
            
            grid = make_disp_files_pipe(ti,
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
            
            
            # record the grid of points along the migration path
            energy_dict[sitepairname] = dict()
            energy_dict[sitepairname]['grid'] = np.copy(grid)
            energy_dict[sitepairname]['x0'] = np.copy(x0)
            energy_dict[sitepairname]['x1'] = np.copy(site[1:4]) 
                
            # undo any changes to atom <ti>
            cluster[ti].set_constraints([1, 1, 1])
            cluster[ti].setCoordinates(x0)
            if newspecies is not None:
                # revert the species of the translated atom
                cluster[ti].setSpecies(oldspecies)
    
    # calculate energies, if requested to do so by the user
    if executable is not None:                                               
        calculate_migration_points(site_pairs, executable, npoints, noisy=noisy,
                                     in_parallel=in_parallel, np=nprocesses)
                                     
    if read_output:             
        heights = read_migration_energies(energy_dict, npoints, in_subdirectory=not(in_parallel)) #!
        return heights
    else:
        return None
    
def read_migration_energies(energy_dict, npoints, in_subdirectory=False):
    '''Reads in energies calculated for points along migration paths.
    '''
    
    heights = []
    for pair in energy_dict.keys():
        path_energies = []
        for n in range(npoints):
            prefix = 'disp.{}.{}'.format(n, pair)
            if not in_subdirectory:
                E = util.extract_energy('{}.gout'.format(prefix), 'gulp')[0]  
            else:
                E = util.extract_energy('{}/{}.gout'.format(prefix, prefix), 'gulp')[0]
                
            path_energies.append(E)
        
        path_energies = np.array(path_energies) 
        path_energies -= path_energies[0]
        Eh = get_barrier(path_energies)
      
        # produce output 
        outstream = open('disp.{}.barrier.dat'.format(pair), 'w')
        # write the components of (a) the initial site, (b) the final site, and
        # (c) the vector from one to the other
        x0 = energy_dict[pair]['x0']
        x1 = energy_dict[pair]['x1']
        dx = x1-x0
        outstream.write('# x0: {:.3f} {:.3f} {:.3f}\n'.format(x0[0], x0[1], x0[2]))
        outstream.write('# x1: {:.3f} {:.3f} {:.3f}\n'.format(x1[0], x1[1], x1[2]))
        outstream.write('# dx: {:.3f} {:.3f} {:.3f}\n'.format(dx[0], dx[1], dx[2]))
        
        # write energies at each point along the migration path
        for z, E in zip(energy_dict[pair]['grid'], path_energies):
             outstream.write('{} {:.6f}\n'.format(z, E))
             
        outstream.close()
        
        # get the indices of start and final sites
        i, j = [int(index) for index in pair.split('.')[-2:]]
        heights.append([i, j, x1[0], x1[1], Eh, path_energies[-1]]) 
        
    return np.array(heights)
    
def calculate_migration_points(site_pairs, executable, npoints, noisy=False, in_parallel=False, np=1):
    '''Optimize structures and calculate energies for all <n> points along the 
    migration paths with endpoints defined in <site_pairs>.
    '''
    
    if not in_parallel:
        for pair in site_pairs:
            for n in range(npoints):
                if noisy:
                    i, j = pair.split('.')[-2:]
                    print("Calculating barrier for migrationfrom site {} to site {}".format(i, j))
                    
                prefix = 'disp.{}.{}'.format(n, pair)
                gulp.run_gulp(executable, prefix)
    else:
        pool = Pool(processes=np)
        files_to_calc = ['disp.{}.{}'.format(ni, prefix) for ni in range(npoints) 
                                                         for prefix in site_pairs]
        for disp_file in files_to_calc:
            if noisy:
                i, j = disp_file.split('.')[-2:]
                ni = disp_file.split('.')[1]
                if int(ni) == 0:
                    message = "Calculating barrier for migration from site {} to site {}".format(i, j)
                else:
                    message = None
            else:
                message = None
                
            pool.apply_async(gulp.gulp_process, (disp_file, executable, message))
            
        pool.close()
        pool.join()
    
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
        outstream.write('{:.0f} {:.0f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(site[0],  
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
        # else
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
