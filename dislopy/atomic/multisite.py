#!/usr/bin/env python
'''Routines required to insert a defect containing multiple atoms (or vacancies)
at atomic sites in a crystal.
'''
from __future__ import print_function, absolute_import

import sys, re, os 

import numpy as np

from numpy.linalg import norm
from multiprocessing import Pool
from shutil import copyfile

from dislopy.atomic import crystal as cry
from dislopy.atomic import qe_utils as qe
from dislopy.atomic import gulpUtils as gulp
from dislopy.atomic import castep_utils as castep
from dislopy.atomic import transmutation as mutate

def periodic_distance(atom1, atom2, lattice, use_displaced=True, oned=False,
                                                to_cart=True):
    '''Calculates the smallest distance between <atom1> and any periodic image
    of <atom2> (including the one defined by the lattice vector (0, 0, 0))
    '''
    
    if type(lattice) == cry.Crystal:
        lattice = lattice.getLattice()
        
    # cell lengths
    if oned:
        # construct z-aligned vector with length equal to the cell height
        scale = np.array([0., 0., lattice.getHeight()])
    else:
        # scale is the simulation cell lengths
        scale = np.zeros(3)
        for i in range(3):
            scale[i] = norm(lattice[i])
    
    # extract appropriate coordinates
    if isinstance(atom1, np.ndarray):
        x1 = atom1
    elif use_displaced:
        x1 = atom1.getDisplacedCoordinates()
    else: 
        # use non-defected coordinates
        x1 = atom1.getCoordinates()
        
    if isinstance(atom2, np.ndarray):
        x2 = atom2
    elif use_displaced:
        x2 = atom2.getDisplacedCoordinates()
    else: 
        # use non-defected coordinates
        x2 = atom2.getCoordinates()
        
    # if <to_cart> has been specified (typically only for supercells), multiply 
    # coordinates by <scale>
    if to_cart:
        if oned:
            sc = np.array([1., 1., lattice.getHeight()])
            x1 = sc*x1
            x2 = sc*x2
        else:
            x1 = scale*x1
            x2 = scale*x2
    
    # calculate distance between atom 1 and the closest periodic image of atom 2
    mindist = np.inf
    if oned:
        # 1D-periodic cluster - look only along the axis of the cylinder
        for i in [-1, 0, 1]:
            d = np.array([0., 0., i])
            d = scale*d
            vec = x1 - (x2+d)
            dist = norm(vec)
            if dist < mindist:
                mindist = dist
                minvec = vec
    else: 
        # 3D-periodic supercell - look at periodic images in all directions
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    d = np.array([i, j, k])
                    d = scale*d
                    vec = x1-(x2 + d)
                    dist = norm(vec)
                    if dist < mindist:
                        mindist = dist
                        minvec = vec
                    
    return mindist, minvec
    
def as_unit(vector):
    '''Returns a unit vector with the same orientation as <vector>.
    '''
    
    vector = np.array(vector)
    # make sure that <vector> is not the zero vector
    if abs(norm(vector)) < 1e-6:
        raise ValueError("Direction of zero vector undefined.")
        
    return vector/norm(vector)
    
def closest_atom_in_direction(atomtype, site, supercell, direction, oned=False,
                     use_displaced=True,  to_cart=True, phitol=1e-6, dtol=1e-6):
    '''Finds the closest atom in the specified direction.
    '''
    
    # convert direction vector to unit length
    unit_dir = as_unit(direction)
    
    if isinstance(site, int):
        # assume that <site> is a site index
        atomindex = site
        site = supercell[site]
    else:
        atomindex = np.nan
    
    mindist = np.inf
    closest_index = np.nan
    for i, atom in enumerate(supercell):
        # check that <atom> is not the same as the one at <atomindex>, and that
        # it is of the correct species
        if i == atomindex or atom.getSpecies() != atomtype:
            continue
        
        # calculate the direction and magnitude of the shortest distance between
        # atom <atomindex> and any periodic repeat of <atom>
        dist, vec = periodic_distance(atom, site, supercell, to_cart=to_cart,
                                                                    oned=oned)
        unit_vec = as_unit(vec)
        
        # check to see if <atom> is the closest atom (along <direction>) so far
        # <phitol> specifies how closely the interatomic separation must align
        # with the specified direction
        if dist-dtol < mindist and abs(np.dot(unit_vec, unit_dir) - 1) < phitol:
                mindist = dist
                closest_index = i

    return closest_index 
    
def closest_atom_oftype(atom, supercell, atomtype, use_displaced=True, oned=False,
                                                       to_cart=True, skip_atoms=[]):
    '''Locates the closest atom of species <atomtype> to <atom> in the 
    provided <supercell>. Primarily useful for locating hydroxyl oxygens. 
    <skip_atoms> is a list indices for atoms which should NOT be considered.
    '''
    
    if type(supercell) == cry.Crystal:
        lattice = supercell.getLattice()
    
    mindist = np.inf
    index = -1
    
    for i, atom2 in enumerate(supercell):
        if atom2.getSpecies() != atomtype or i in skip_atoms:
            # wrong species, carry on
            continue

        dist, vec = periodic_distance(atom, atom2, supercell, use_displaced=use_displaced,
                                                                oned=oned, to_cart=to_cart)

        if dist < mindist:
            mindist = dist
            index = i

    return index
    
def hydrogens_index(coupled_defect):
    '''Checks a coupled defect to find the index of the hydrogen-containing site.
    This is relevant specifically for cases where the hydrous defect is not neutral,
    and is being charge-balanced by another chemical impurity (eg. {Ti}.._{Mg} 
    with {2H}''_{Si}, forming the titanoclinohumite defect. We assume that the 
    use has supplied a <CoupledImpurity>, and that only one site contains hydrogen
    '''
    
    # regex for Hydrogen, which may be numbered
    h_reg = re.compile(r'H\d*')
    
    # find index of H containing site    
    for i, site in enumerate(coupled_defect):
        for atom in site:
            is_h = h_reg.match(atom.getSpecies())
            if is_h:
                return i
    
    # return NaN, which can then be handled by the user            
    return np.nan
    
def hydroxyl_oxygens(hydrous_defect, supercell, hydroxyl_str, program='gulp',
                     oxy_str='O', oned=False, to_cart=True, shell_model=False):
    '''Locate the hydroxyl oxygens for molecular mechanics simulations and 
    create hydroxyl oxygens (species <hydroxyl_str>) to insert into the simulation 
    cell. We assume that the coordinates of the hydrogen atoms have been set. If
    <shell_model> is True, then the polarizability of the hydroxyl oxygens is
    represented using a Dick-Overhauser shell model.
    '''

    hydroxyl_oxys = []
    
    # if <hydrous_defect> is a <CoupledImpurity>, locate the site containing
    # hydrogen
    if mutate.is_coupled(hydrous_defect):
        hydrous_defect.site_locations(supercell)
        hydrogen_site = hydrous_defect[hydrogens_index(hydrous_defect)] 
    else: # <Impurity> object
        hydrous_defect.site_location(supercell)
    
    # find hydroxyl oxygens, making sure that no oxygen for part of more than
    # one hydroxyl group 
    already_replaced = []
    for H in hydrous_defect:
        # add an impurity of type <hydroxyl_str> to <hydroxyls>
        new_hydrox = mutate.Impurity(oxy_str, 'hydroxyl oxygen')
        if program.lower() == 'gulp':
            new_atom = gulp.GulpAtom(hydroxyl_str)
            if shell_model:
                # add a polarizable shell with the coordinates of the atom
                new_atom.addShell(np.zeros(3))
                
            new_hydrox.addAtom(new_atom)        
        else:
            new_hydrox.addAtom(cry.Atom(hydroxyl_str))
               
        # locate index of site containing nearest oxygen atom
        site_index = closest_atom_oftype(H, supercell, oxy_str, oned=oned, 
                             to_cart=to_cart, skip_atoms=already_replaced)

        new_hydrox.set_index(site_index)
        new_hydrox.site_location(supercell)
        already_replaced.append(site_index)

        hydroxyl_oxys.append(new_hydrox.copy())

    # create <CoupledImpurity> to hold hydroxyl oxygens
    full_defect = mutate.CoupledImpurity()
    
    # ...which we now add to <full_defect>
    for imp in hydroxyl_oxys:
        full_defect.add_impurity(imp)
    
    # finally, add the hydrogen atoms in the defect with the hydroxyl defects
    if mutate.is_single(hydrous_defect):
        full_defect.add_impurity(hydrous_defect)
    else: # coupled defect, eg. Ti-clinohumite defect
        full_defect = mutate.merge_coupled(hydrous_defect, full_defect)

    return full_defect
    
def locate_bonded(site, siteindex, bondatom, supercell, nbonds):
    '''Locates atoms bonded to a particular site.
    '''
    
    bondlist = []
    bonddist = []
    for i, atom in enumerate(supercell):
        if atom.getSpecies() != bondatom:
            continue
        elif i == siteindex:
            continue
        # else
        dist = distance2(site, atom)
        if len(bondlist) < nbonds:
            # list of bonded atoms not yet full
            bondlist.append(i)
            bonddist.append(dist)
            if len(bondlist) == nbonds:
                # get sorted indices
                idx = np.argsort(bonddist)
                # now sort the index and distance arrays simultaneously
                bondlist = list(np.array(bondlist)[idx])
                bonddist = list(np.array(bonddist)[idx])
        else: # len(bondlist) >= nbonds
            # check to see if <atom> is closer than any previously
            # seen by the loop
            pass
            
def which_image(x, x0, H):
    '''Determines which periodic image the of the atom at site x is closest to
    the site located at x0.
    '''
    
    # calculate distances to images
    dx_base = norm(x-x0)
    dx_up = norm(x+np.array([0, 0, H])-x0)
    dx_down = norm(x-np.array([0, 0, H])-x0)
    if dx_base < dx_up:
        if dx_base < dx_down:
            return 0
        else:
            return -1
    elif dx_down < dx_up:
        return -1
    else:
        return 1   
        
def associate_bonds(bondlist, bonded_sites, cluster, theta_thresh=0.5, 
                                                        norm_thresh=0.2):
    '''Checks to make sure that the bonds listed in <bonded_sites> correspond
    to valid bonds in the undeformed lattice. 
    '''
    
    site_pairs = dict()
    for site0 in bonded_sites.keys():
        x0 = cluster[site0].getCoordinates()
        sites_to_use = set()
        
        # check all connected sites for validity
        for site in bonded_sites[site0]:
            # calculate bond vector between site and site0
            x = cluster[site].getCoordinates()
            sgn = which_image(x, x0, cluster.getHeight())
            dx = x+sgn*np.array([0, 0, H])-x0
            
            # check for matching bonds in the base lattice
            nbonds = 0
            for bond in bondlist:
                theta = np.arccos(np.dot(dx, bond)/(norm(dx)*norm(bond)))
                if theta < theta_thresh:
                    if (norm(dx)-norm(bond))/norm(bond) < norm_thresh:
                        nbonds += 1
            
            # if a unique matching bond found -> add to list
            if nbonds == 1:            
                sites_to_use.add(site)
                
        # check that <site0> actually has valid bonds
        if len(sites_to_use) > 0:    
            site_pairs[site0] = sites_to_use
        
    return site_pairs            

def sites_to_replace(cluster, defect, radius, tol=1e-1, constraints=None,
                                                            noisy=False):
    '''For a given <cluster> of atoms, determine which sites will be replaced
    by the provided <defect>. These sites are those within <radius> of the 
    dislocation line.
    '''
    
    # find site indices           
    use_indices = []    
    for i, atom in enumerate(cluster):
        # check conditions for substitution:
        # 1. Is the atom to be replaced of the right kind?
        if atom.getSpecies() != defect.getSite():
            continue
        # 2. Is the atom within <radius> of the dislocation line?
        if norm(atom.getCoordinates()[:-1]) > (radius+tol):
            continue  
              
        # check <constraints>
        if constraints is None:
            pass
        else:
            for test in constraints:
                useAtom = test(atom) 
                if not useAtom:
                    print("Skipping atom {}.".format(i))
                    break
            if not useAtom:
                continue       
        if noisy:        
            print("Replacing atom {} (index {})...".format(str(atom), i)) 
        
        # record that atom <i> is to be replaced      
        use_indices.append(i)        
    
    # record site IDs and coordinates
    create_id_file(defect, use_indices, cluster)
    
    return use_indices
    
def sites_to_replace_bonds(cluster, defect, radius, dx_thresh, tol=1e-1, bonds=None,
                 constraints=None, noisy=False, theta_thresh=0.5, norm_thresh=0.2,
                                   has_mirror_symmetry=False, dz_min_accept=-1e-1):
    '''Calculates which sites defect energies need to be calculated for if
    diffusion calculations will be done after defect segregation energies are 
    calculated.'''
    
    # outer loop -> sites within <radius>
    h = cluster.getHeight()
    replace_at_site = defect.getSite()
    bond_pairs = dict()
    for i in range(cluster.numberOfAtoms):
        atomi = cluster[i]
        
        # check that the atom at site i is of the type that <defect> replaces
        if atomi.getSpecies() != replace_at_site:
            continue
        
        # check that atom is within the region of interest
        x0 = atomi.getCoordinates()
        if norm(x0[:-1]) > radius:
            continue
        
        # check constraints
        if constraints is None:
            pass
        else:
            for test in constraints:
                useAtom = test(atomi) 
                if not useAtom:
                    print("Skipping atom {}.".format(i))
                    break
            if not useAtom:
                continue  
                     
        if noisy:        
            print("Replacing atom {} (index {})...".format(str(atomi), i)) 
        
        bond_pairs[i] = set()
        # inner loop -> sites to which atom <i> might diffuse
        for j in range(cluster.numberOfAtoms):
            atomj = cluster[j]
            if atomj.getSpecies() != replace_at_site:
                continue
            elif i == j:
                continue
                
            ### NEED A ROUTINE HERE FOR DISLOCATIONS WITH REFLECTION SYMMETRY ###
                
            # otherwise, calculate the distance to the closest site
            x = atomj.getCoordinates()
            
            # distance vectors for atoms in the same and other images
            dx_same_im = x-x0
            dx_prev_im = x+np.array([0, 0, h])-x0
            dx_next_im = x-np.array([0, 0, h])-x0
            
            # determine which image is closest
            dx = norm(dx_same_im)
            which_image = 0
            if norm(dx_prev_im) < dx:  
                dx = norm(dx_prev_im)
                which_image = -1
            
            if norm(dx_next_im) < dx:
                dx = norm(dx_next_im)
                which_image = 1
            
            # check that jump distance is below specified threshold 
            if dx > dx_thresh:
                continue
                                                       
            # check for mirror symmetry, if the dislocation has it
            if has_mirror_symmetry:
                if which_image == 0:
                    dz = dx_same_im[-1]
                elif which_image == -1:
                    dz = dx_prev_im[-1]
                else: # which_image == 1
                    dz = dx_next_im[-1]
                    
                # if jump is back, ignore
                if dz < dz_min_accept: #! SHOULD CHANGE TO GENERAL THRESHOLD
                    continue
                else:
                    bond_pairs[i].add(j)    
            else:
                bond_pairs[i].add(j)
        
    if bonds is not None:
        # use_only atoms from jset which have matching bonds
        bond_pairs = associate_bonds(bonds, bond_pairs, cluster, theta_thresh=theta_thresh,
                                                            norm_thresh=norm_thresh)
        
    # create list of all sites for which energies must be calculated
    iset = set(bond_pairs.keys())
    jset = set()
    for j in bond_pairs.values():
        jset |= set(j)
            
    use_indices = iset | jset
    
    # record site IDs and coordinates
    create_id_file(defect, use_indices, cluster)
    
    # record list of bonds 
    create_bond_file(defect, bond_pairs, cluster)
    
    return use_indices

def create_id_file(defect, indices, cluster):
    '''Record information about sites for which defect energies should be 
    calculated.
    '''
    
    # record defect sites and IDs
    idfile = open('{}.{}.id.txt'.format(defect.getName(), defect.getSite()), 'w')
    idfile.write('# site-id x y z\n')
    # record base name of simulation files
    idfile.write('# {}.{}\n'.format(defect.getName(), defect.getSite())) 
    
    # record all indices in <idfile> for ease of restarting
    idfile.write('#')
    for i in indices:
        idfile.write(' {}'.format(i))
    idfile.write('\n')  
        
    # write atomic site coords to <idfile> for later use
    for i in indices:
        coords = cluster[i].getCoordinates()
        idfile.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i, coords[0], 
                                       coords[1], coords[2], norm(coords[:-1])))
            
    idfile.close()
    
def create_bond_file(defect, bond_pairs, cluster):
    '''Record the IDs of each site bonded to a core site to which <defect> 
    segregates in the cluster.
    '''
    
    bondfile = open('{}.{}.bonds.txt'.format(defect.getName(), defect.getSite()), 'w')
    bondfile.write('# site-id x y z\n')
    bondfile.write('# bond-id-1 x y z\n')
    bondfile.write('# ...\n')
    bondfile.write('# bond-id-n xn yn zn;\n')
    # record base name of simulation files
    bondfile.write('# {}.{}\n'.format(defect.getName(), defect.getSite()))
    
    # record bonds
    for i in bond_pairs.keys():
        # record coordinates of base site
        coords = cluster[i].getCoordinates()
        bondfile.write('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(i, coords[0], 
                                                       coords[1], coords[2], norm(coords[:-1])))
                                       
        # record coordinates of bonded sites
        for j in bond_pairs[i]:
            coords = cluster[j].getCoordinates()
            bondfile.write('\n{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(j, coords[0], 
                                                          coords[1], coords[2], norm(coords[:-1])))
        
        # add semi-colon to denote end of block for a single site
        bondfile.write(';\n')
        
    bondfile.close() 
    
def calculate_impurity(sysinfo, gulpcluster, radius, defect, gulpexec='./gulp',
         constraints=None, minimizer='bfgs', maxcyc=100, noisy=False, tol=1e-1,
                     centre_on_impurity=False, do_calc=False, dx_thresh=np.nan,
                  contains_hydroxyl=False, oh_str='Oh', o_str='O', bonds=False,
                    has_mirror_symmetry=False, in_parallel=False, nprocesses=1):
    '''Iterates through all atoms in <relaxedCluster> within distance <radius>
    of the dislocation line, and sequentially replaces one atom of type 
    <replaceType> with an impurity <newType>. dRMin is the minimum difference
    between <RI> and <radius>. Ensures that the impurity is not close to region
    II, where internal strain would not be relaxed. <constraints> contains any 
    additional tests we may perform on the atoms, eg. if the thickness is > 1||c||,
    we may wish to restrict substituted atoms to have z (x0) coordinates in the
    range [0,0.5) ( % 1). The default algorithm used to relax atomic coordinates
    is BFGS but, because of the N^2 scaling of the memory required to store the 
    Hessian, other solvers (eg. CG or numerical BFGS) should be used for large
    simulation cells.
    
    Tests to ensure that radius < (RI - dRMin) to be performed in the calling 
    routine (ie. <impurityEnergySurface> should only be called if radius < 
    (RI-dRMin) is True. 
    
    The keyword <centre_on_impurity> determines the axis of simulation region I
    (ie. the cylinder of atoms whose coordinates are to be fully relaxed). If
    this parameter is <True>, then this region is centred on each impurity in turn; 
    otherwise, region I is centred on the axis of the cluster.
    
    If <contains_hydroxyl> is True, activates functions to replace oxygen atoms 
    bonded to H atoms with their hydroxyl counterparts.
    '''
    
    # dummy variables for lattice and toCart. Due to the way the program
    # is set up, disloc is set equal to false, as the atoms are displaced 
    # and relaxed BEFORE we read them in
    lattice = np.identity(3)
    toCart = False
    disloc = False
    coordType = 'pcell'

    # test to see if <defect> is located at a single site
    if type(defect) is mutate.Impurity:
        pass
    else:
        raise TypeError('Invalid impurity type.')
    
    if bonds:
        # find all sites for which energies need to be calculated in order
        # to determine energy barriers for diffusion 
        if dx_thresh != dx_thresh:
            raise ValueError("Intersite distance must be defined.")
            
        use_indices = sites_to_replace_bonds(gulpcluster, defect, radius, dx_thresh,
                                      tol=tol, constraints=constraints, noisy=noisy, 
                                           has_mirror_symmetry=has_mirror_symmetry)
    else:     
        use_indices = sites_to_replace(gulpcluster, defect, radius, tol=tol,
                                     constraints=constraints, noisy=noisy)
                                     
    # construct input files and, if requested by the user, run calculations  
    site_list = []  
    for i in use_indices:        
        # set the coordinates and site index of the impurity
        atom = gulpcluster[i]
        defect.site_location(atom)
        defect.set_index(i)
        
        # if the defect contains hydrogen, replace normal oxygen atoms with 
        # hydroyl oxygen atoms
        if contains_hydroxyl:
            full_defect = hydroxyl_oxygens(defect, gulpcluster, oh_str, 
                                oxy_str=o_str, oned=True, to_cart=False)  
        else:
            full_defect = defect        
        
        # create .gin file for the calculation
        coords = atom.getCoordinates()
        prefix = make_outname(defect)
        outstream = open(make_outname(defect)+'.gin','w')
        
        # record prefix for later use in parallel run
        site_list.append(prefix)
       
        # write structure to output file, including the coordinates of the 
        # impurity atom(s)
        if centre_on_impurity:
            # use coordinates of Impurity in the plane
            rI_centre = coords[:-1]
        else:
            # centre on cylinder axis
            rI_centre = np.zeros(2)           
            
        gulp.write_gulp(outstream, gulpcluster, sysinfo, defected=False, to_cart=False,
                            impurities=full_defect, rI_centre=rI_centre, relax_type='',
                                                                  add_constraints=True)
                                                                  
    return 

def calculate_impurity_energies(sitelist, gulpexec, in_parallel=False, nprocesses=1):
    '''Calculates energies for dislocation clusters containing a single impurity
    previously constructed using the <calculate_impurity> function.
    '''
    
    if not in_parallel:
        print('here 1')
        for i, site in zip(use_indices, site_list):
            print('Relaxing structure with defect at site {}...'.format(i))
            gulp.run_gulp(gulpexec, site)
    else:
        print('here 2')
        # create iterable object with prefices so that map works properly
        #f = lambda prefix: gulp_process(prefix, gulpexec)
        #with Pool(processes=nprocesses) as pool:
        #    pool.map(f, (site_list)
        pool = Pool(processes=nprocesses)
        for site in site_list:
            pool.apply(gulp_process, args=(site, gulpexec))
                
        pool.close()
        pool.join()

    return
   
def parse_sitelist(dfctname, site):
    '''Reads in a list of sites for which clusters containing defects.
    '''
    
    prefix = '{}.{}'.format(dfctname, site)
    
    # read in the site IDs from the *.id.txt file
    sitefile = open('{}.id.txt'.format(prefix), 'r')
    
    # get sites and convert to int
    sitelist_str = sitefile.readlines()[2]
    sitelist_str = sitelist_str.strip('#').strip()
    
    ids = [int(x) for x in sitelist_str.strip()]
    sites = ['{}.{}'.format(prefix, i) for i in ids]
    return sites
   
def gulp_process(prefix, gulpexec):
    '''An individual GULP process to be called when running in parallel.
    '''
    
    # create the directory from which to run the GULP simulation
    if os.path.exists(prefix):
        if not os.path.isdir(prefix):
            # the name <prefix> is taken, and NOT by a directory
            raise Exception("Name {} taken by non-directory.".format(prefix))
        else:
            # assume that using directory <prefix> is fine
            pass
    else:        
        os.mkdir(prefix)
        os.chdir(prefix)
    
    # run simulation and return to the primary impurity directory    
    gulp.run_gulp(gulpexec, site)
    os.chdir('../')
    
    # copy output file to main directory 
    copyfile('{}/{}.gout'.format(prefix, prefix), '{}.gout'.format(prefix))
    return 0
    
def make_outname(defect):
    '''Returns a string containing identifying information for the defect.
    '''
    
    return '{}.{}.{}'.format(defect.getName(), defect.getSite(), defect.get_index())

def calculate_coupled_impurity(sysInfo, regionI, regionII, radius, defectCluster,
                                        gulpExec='./gulp', constraints=None):
    
    # dummy variables for lattice and toCart. Due to the way the program
    # is set up, disloc is set equal to false, as the atoms are displaced 
    # and relaxed BEFORE we read them in
    lattice = np.identity(3)
    toCart = False
    disloc = False
    coordType = 'pfractional'
    
    # test to see if <defect> is located at a single site
    if type(defectCluster) is CoupledImpurity:
        ### NOT YET IMPLEMENTED ###
        print('Coupled defects have not been implemented yet.')
        pass
    else:
        raise TypeError('Invalid impurity type.')
                   
    return
