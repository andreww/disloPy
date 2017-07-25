#!/usr/bin/env python
from __future__ import print_function, absolute_import

import numpy as np
import re

from pydis.atomic import crystal as cry
from pydis.atomic import rodSetup as rs
from pydis.atomic import aniso 
from pydis.utilities import atomistic_utils as util
from pydis.atomic import permute
from pydis.atomic.qe_utils import parse_qe
from pydis.atomic.castep_utils import parse_castep
from pydis.atomic.gulpUtils import parse_gulp
from pydis.pn import pn_1D as pn1
from pydis.pn import pn_2D as pn2

def read_unit_cell(cellname, program, shift, permutation=[0, 1, 2], path='./'):
    '''Reads in the unit from which the dislocation-bearing cluster will be 
    built.
    '''
    
    # read in cell using the program-appropriate parse function
    basestruc = cry.Crystal()
    if program == 'gulp':
        parse_fn = parse_gulp 
    elif program == 'qe':
        parse_fn = parse_qe
    elif program == 'castep':
        parse_fn = parse_castep
    else:
        raise ValueError("Program {} not supported.".format(program))
    
    sinfo = parse_fn(cellname, basestruc, path=path)
    
    if len(shift) < 3:
        # create 3-vector, populating first n elements from shift
        new_shift = np.zeros(3)
        for i in range(len(shift)):
            new_shift[i] = shift[i]
        shift = new_shift
        
    basestruc.translate_cell(np.array(shift), modulo=True)
    
    # permute the coordinates of the unit cell. Usually necessary as the glide
    # plane normal is usually aligned along z for GSF calculations, whereas here
    # it should be along y
    permute.check_permutation(permutation)
    permute.permute_cell(basestruc, program, permutation)
    
    return basestruc
    
def import_pn_pars(pnfile):
    '''Imports the parameters defining a dislocation density distribution from a
    PN simulation output file.
    '''

    pn_output = open(pnfile, 'r').read()
    
    # extract parameters
    pars = []
    for par in ['A', 'x0', 'c']:
        parmatch = re.compile(par+'\n\s*(?P<pars>-?\d+\.\d+(?:\s+-?\d+\.\d+)*)')
        for p in re.finditer(parmatch, pn_output):
            pars += [float(x) for x in p.group('pars').rstrip().split()]
    
    # get number of dimensions
    dims = int(re.search('Dimensions:\s+(?P<d>\d)', pn_output).group('d'))
    return dims, pars

def construct_xyz(unitcell, r, pn_pars, dims, spacing, b, f0, disl_type, thickness=1):
    '''Constructs a cluster/slab of atoms containing a dislocation whose
    structure is determined from the output of a PN simulation.
    '''
    
    # construct basic cluster and extend (if necessary)
    xyz = rs.TwoRegionCluster(unitCell=unitcell, R=r, regionI=r, regionII=r+1)
    if thickness > 1:
        xyz = rs.extend_cluster(xyz, thickness)
    
    # make sure that b is a vector
    if isinstance(b, int) or isinstance(b, float):
        if disl_type == 'edge':
            b = np.array([b, 0., 0.])
        elif disl_type == 'screw':
            b = np.array([0., 0., b])
    
    # generated field of lattice planes
    max_x = np.ceil(r/spacing)+5
    nplanes = 2*int(max_x)+1
    planes = np.arange(-max_x*spacing, max_x*spacing, spacing) 
    
    # calculate the magnitude of the inelastic displacement field
    if dims == 1:
        u = pn1.get_u1d(pn_pars, np.linalg.norm(b), spacing, max_x)
        rho = pn1.rho(u, planes)
        if disl_type == 'screw':
            ux = np.zeros(nplanes)
            rhox = np.zeros(nplanes-1)
            uz = u
            rhoz = rho
        elif disl_type == 'edge':
            ux = u
            rhox = rho
            uz = np.zeros(nplanes)
            rhoz = np.zeros(nplanes-1)
    elif dims == 2:
        ux, uz = pn2.get_u2d(pn_pars, np.linalg.norm(b), spacing, 
                                        max_x, disl_type)
        rhox = pn1.rho(ux, planes)
        rhoz = pn1.rho(uz, planes)

    # apply inelastic displacement field
    for atom in xyz:
        # displace atoms below the glide plane
        if atom.getCoordinates()[1] < 0.:
            xd = atom.getDisplacedCoordinates()
            # determine index of appropriate element of displacement
            # field and calculate displacement at atomic site
            i = np.where(xd[0] > planes-1e-6)[0].max()
            xn = xd[0] + ux[i]
            zn = xd[-1] + uz[i]

            atom.setDisplacedCoordinates(np.array([xn, xd[1], zn]))
            atom.setCoordinates(np.array([xn, xd[1], zn]))

    # partial dislocation located between atomic planes (for symmetry)
    newplanes = planes+spacing/2
    length = min(len(rhox), len(rhoz))
    for i in range(length):           
        # calculate displacement due to this partial disloaction
        partial = np.array([rhox[i]*b[0], 0., rhoz[i]*b[-1]])
        xyz.applyField(f0, [[newplanes[i], 0.]], [partial], use_branch=False,
                        use_dip=True, at_dip=False, centre_line=newplanes[i],
                                                            branch_thresh=0.5)
    
    return xyz
    
def symmetrise_cluster(cluster, axis=0., threshold=1., sym_thresh=0.3, use_displaced=True):
    '''Symmetrises the lower half of the dislocation cluster about the y-axis,
    with the mirror plane running through <axis>, removing atoms displaced too 
    far and merging those which overlap.
    '''
    
    n = len(cluster)
    for i in range(n):
        atom_i = cluster[i]
        if not atom_i.writeToOutput():
            continue
        else:
            if use_displaced:
                xi = atom_i.getDisplacedCoordinates()
            else:
                xi = atom_i.getCoordinates()
             
        # test to see if atom is close to axis of cluster AND in the lower half
        # of said cluster
        if xi[1] > 0:
            continue

        # bounds for the central region
        lower = axis - sym_thresh
        upper = axis + sym_thresh
        
        # delete atoms that have moved too far
        if use_displaced:
            if xi[0] > upper and atom_i.getCoordinates()[0] < lower:
                atom_i.switchOutputMode()
                continue
            elif xi[0] < lower and atom_i.getCoordinates()[0] > upper:
                atom_i.switchOutputMode()
                continue
        did_merge=False
        for j in range(n):
            atom_j = cluster[j]
            if j == i:
                continue
            elif not atom_j.writeToOutput():
                continue
            elif atom_j.getSpecies() != atom_i.getSpecies():
                continue
            else:
                if use_displaced:
                    xj = atom_j.getDisplacedCoordinates()
                else:
                    xj = atom_j.getCoordinates()
                if xj[1] > 0:
                    continue
                elif abs(xj[0] - axis) > threshold:
                    continue
                else:
                    # calculate interatomic distance
                    delta1 = np.linalg.norm(xj-xi)
                    delta2 = np.linalg.norm(xj-np.array([xi[0], xi[1], xi[2]+cluster.getHeight()]))
                    delta3 = np.linalg.norm(xj-np.array([xi[0], xi[1], xi[2]-cluster.getHeight()]))
                    delta = min(delta1, delta2, delta3)
                    if delta < threshold:
                        did_merge = True
                        # turn off atom <j> and symmetrise the coordinates
                        cluster[j].switchOutputMode()
                        new_xi = np.array([axis, xi[1], xi[2]])
                        if use_displaced:
                            cluster[i].setDisplacedCoordinates(new_xi)
                        else:
                            cluster[i].setCoordinates(new_xi)
                            
        # test to see if atom is close to the axis of symmetry
        if not did_merge:
            if abs(xi[0]-axis) < sym_thresh:
                new_xi = np.array([axis, xi[1], xi[2]]) 
                if use_displaced:
                    cluster[i].setDisplacedCoordinates(new_xi)
                else:
                    cluster[i].setCoordinates(new_xi)
                 
def centre_dislocation(xyz, par, b, spacing, dims, disl_type=None):
    '''Centres the dislocation at the origin.
    '''

    # calculate com
    if dims == 1:
        cm = pn1.com_from_pars(par, b, spacing, 100)
    else: # dims == 2
        cm = pn2.com_fom_pars2d(par, b, spacing, 100, disl_type)
    
    # translate the dislocation so that its centre is at the origin
    xyz.translate_cell(np.array([-cm, 0., 0.]), reset_disp=False)

def restrict_region(xyz, r, use_disp=True):
    '''Restricts cluster <xyz> to only those atoms within <r> of the origin.
    '''
    
    for atom in xyz:
        if not atom.writeToOutput():
            continue
        if use_disp:
            x = atom.getDisplacedCoordinates()
        else:
            x = atom.getCoordinates()
            
        if np.linalg.norm(x[:-1]) > r:
            atom.switchOutputMode()
                    
def make_xyz(unitcell, pars, dims, b, spacing, disl_type, field, r, outname, thr=1.5,
                              sym_thr=0.3, description='PN dislocation', thickness=1):
    '''Control function to read in unit cell, construct a cluster containing
    an isolated dislocation, and write it to a .xyz file.
    '''
    
    # construct cluster with dislocation and symmetrise its lower half
    xyz = construct_xyz(unitcell, r+10, pars, dims, spacing, b, field, disl_type,
                                            thickness=thickness) 
    centre_dislocation(xyz, pars, b, spacing, dims, disl_type)
    symmetrise_cluster(xyz, threshold=thr, sym_thresh=sym_thr)
    
    # make cell cylindrical
    restrict_region(xyz, r)

    util.write_xyz(xyz, outname, defected=True, description=description)
    return 
