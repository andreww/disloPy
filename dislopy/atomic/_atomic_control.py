#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np
import re
import argparse
import sys
import os
from numpy.linalg import norm

# list of atomic simulation codes currently supported by disloPy 
supported_codes = ('qe', 'gulp', 'castep', 'lammps')

#!!! need to shift the relevant functions to a simulation method-agnostic module  
from dislopy.utilities.control_functions import control_file, from_mapping, change_type,  \
                                            to_bool, change_or_map, print_control   
                            
# import modules required to set up and run a dislocation simulation
from dislopy.atomic.atomic_import import *

def array_or_int(dimension):
    '''Determine whether the user is specifying a single supercell size, or a 
    range of supercell sizes.
    '''
    
    # form of array is <startlen>, <interval>, <maxlen>
    array_string = re.match('(?P<i1>\d+)\s*,\s*(?P<di>\d+)\s*,\s*(?P<i2>\d+)', dimension)
    int_string = re.match('\d+', dimension)
    if array_string:
        i1 = int(array_string.group('i1'))
        di = int(array_string.group('di'))
        i2 = int(array_string.group('i2'))
        
        # check that input corresponds to a valid array
        if i2 <= i1:
            raise ValueError(("Final value ({}) must be greater than initial" +
                                                  " value ({})").format(i1, i2))
        elif di <= 0:
            raise ValueError("Increment ({}) must be positive.".format(di))
        elif di > (i2 - i1):
            raise ValueError(("Increment ({}) is too large. Maximum value is " +
                                                      "{}.").format(di, i2-i1))

        cell_sizes = np.arange(i1, i2+di, di)       
    elif int_string:
        # making it iterable simplifies handling supercell setup
        cell_sizes = [int(int_string.group())]
    else:
        raise AttributeError("No cell dimensions found.")
    
    return cell_sizes
    
def array_or_float(dimension):
    '''Determine whether the user is specifiying a single radius, or a range
    of radii.
    '''
    
    # form of array as above
    array_string = re.match('(?P<f1>\d+\.?\d*)\s*,\s*(?P<df>\d+\.?\d*)\s*,\s*(?P<f2>\d+\.?\d*)', 
                                                                       dimension)
    flt_string = re.match('\d+\.?\d*', dimension)
    if array_string:
        f1 = float(array_string.group('f1'))
        df = float(array_string.group('df'))
        f2 = float(array_string.group('f2'))
        
        # check that input corresponds to a valid array
        if f2 <= f1:
            raise ValueError(("Final value ({}) must be greater than initial" +
                                                  " value ({})").format(f1, f2))
        elif df <= 0:
            raise ValueError("Increment ({}) must be positive.".format(df))
        elif df > (f2 - f1):
            raise ValueError(("Increment ({}) is too large. Maximum value is " +
                                                      "{}.").format(df, f2-f1))
        radii = np.arange(f1, f2+df, df)
    elif flt_string:
        # needs to be iterable for ease of handling
        radii = [float(flt_string.group())]
    else:
        raise AttributeError("No values for the cluster radius found.")
        
    return radii
        

def vector(vec_str):
    '''Converts a string of the form "[x1,x2,...,xn]" to a numpy array.
    '''
    
    vec_regex = re.compile('(\[|\()(?P<contents>(?:-?\d+\.?\d*,\s*)*(?:-?\d+\.?\d*))(\]|\))')
    # test to make sure that the entered string conforms to vector
    # notation
    vec_match = vec_regex.match(vec_str)
    if not(vec_match):
        raise TypeError("Error: not valid vector notation")
    # else
    contents = vec_match.group('contents')
    
    # construct a numpy array from the <contents> of the vector string.
    # Since the magnitude of a burgers vector will, in general, be a floating
    # point number, we convert all elements to floats.
    vec_element = re.compile('-?\d+\.?\d*')
    elements = vec_element.findall(contents)
    base_vector = np.array([float(x) for x in elements])
        
    return base_vector

def handle_atomistic_control(param_dict):
    '''Handle each possible card for an atomistic (ie. cluster or multipole-SC)
    simulation of dislocation properties. As for Peierls-Nabarro simulations, if
    the default value of a card is <None>, the card is deemed "mission critical"
    and the program will abort if the user does not provide a value.
    '''
    
    # cards for the <&control> namelist
    #!!! Should we accommodate array-based jobs?
    control_cards = (('unit_cell', {'default': None, 'type': str}),
                     ('calc_type', {'default': None, 'type': str}),
                     ('program', {'default': None, 'type': str}),
                     ('run_sim', {'default': False, 'type': to_bool}),
                     ('make_input', {'default': False, 'type': to_bool}),
                     ('basename', {'default': 'dis', 'type': str}),
                     ('suffix', {'default': 'in', 'type': str}),
                     ('executable', {'default': '', 'type': str}),
                     ('calculate_core_energy', {'default': False, 'type': to_bool}),
                     ('maxcyc', {'default': 100, 'type': int}),
                     ('para_exec', {'default': 'mpiexec', 'type': to_bool}),
                     ('para_nproc', {'default': 1, 'type': int}),
                     ('set_omp', {'default': False, 'type': to_bool}),
                     ('omp_threads', {'default': 1, 'type': int})
                    )
    
    # cards for the <&elast> namelist. Note that if dislocations are specified 
    # in the <fields> file, they will overwrite <burgers>. Current options for
    # <field_type> are: aniso(tropic), iso_screw, iso_edge, and aniso_screw.
    # aniso_edge will be added later. <normal> is the slip plane normal.
    elast_cards = (('disl_type', {'default': None, 'type': str}),
                   ('burgers', {'default': cry.ei(3), 'type': vector}),
                   ('n', {'default': cry.ei(1), 'type': vector}),
                   ('m', {'default': cry.ei(2), 'type': vector}), 
                   ('bulk', {'default': None, 'type': float}),
                   ('shear', {'default': None, 'type': float}),
                   ('poisson', {'default': None, 'type': float}),
                   ('cij', {'default': None, 'type': aniso.readCij}),
                   ('in_gpa', {'default': True, 'type': to_bool}),
                   ('rcore', {'default': np.nan, 'type': float}),
                   ('centre', {'default': np.zeros(2), 'type': vector}),
                   ('field_type', {'default': None, 'type': str}),
                   ('randomise', {'default': False, 'type': to_bool}),
                   ('random_r', {'default': 5., 'type': float}),
                   ('amplitude', {'default': 0.01, 'type': float})
                  )
                  
    # Now move on to namelists that specify parameters for specific simulation
    # types. <xlength> and <ylength> may be integers or arrays.
    # cards for the <&multipole> namelist. Valid methods for calculating the 
    # energy are: 'compare' (screw dislocations only) and 'edge'.
    multipole_cards = (('nx', {'default': 1, 'type': array_or_int}),
                       ('ny', {'default': 1, 'type': array_or_int}),
                       ('npoles', {'default': 2, 'type': int}),
                       ('relaxtype', {'default': '', 'type': str}),
                       ('grid', {'default': True, 'type': to_bool}),
                       ('bdir', {'default': 0, 'type': int}),
                       ('alignment', {'default': 0, 'type': int}),
                       ('method', {'default': 'standard', 'type': str}),
                       ('fit_K', {'default': True, 'type': to_bool}),
                       ('e_unit_cell', {'default': np.nan, 'type': float})
                      )
                      
    # cards for the <&cluster> namelist. Remember that the Stroh sextic theory
    #  (ie. anisotropic solution) places that branch cut along the negative
    # x-axis. Method for calculating core energy is specified may be edge,
    # eregion or explicit. <rgap> is the difference between <region1> and the 
    # outer radius used for core energy fitting
    cluster_cards = (('region1', {'default': None, 'type': array_or_float}),
                     ('region2', {'default': None, 'type': array_or_float}),
                     ('scale', {'default': 1.1, 'type': float}),
                     ('use_branch', {'default': True, 'type': to_bool}),
                     ('branch_cut', {'default': [0, -1], 'type': vector}),
                     ('thickness', {'default': 1, 'type': int}),
                     ('method', {'default': '', 'type': str}),
                     ('rgap', {'default': 0, 'type': int}),
                     ('rmin', {'default': 1, 'type': int}),
                     ('dr', {'default': 1, 'type': float}),
                     ('fit_K', {'default': False, 'type': to_bool}),
                     ('cut_thresh', {'default': 0.5, 'type': float}),
                     ('centre_thresh', {'default': 1e-10, 'type': float})
                    )
                     
    namelists = ['control', 'multipole', 'cluster', 'elast', 'atoms']
    
    # that all namelists in control file
    for name in namelists:
        if name not in param_dict.keys():
            if name == 'atoms':
                # record that no atomic energies were provided
                param_dict[name] = None
            else:
                param_dict[name] = dict()
    
    # <&atoms> and <&elast> namelists need to be handled separately.        
    for i, cards in enumerate([control_cards, multipole_cards, cluster_cards]):
        # if we are running a multipole simulation, no need to look at params
        # for a cluster calculation, and vice versa
        if namelists[i] == 'multipole':
            if param_dict['control']['calc_type'] == 'cluster':
                continue
        elif namelists[i] == 'cluster':
            if param_dict['control']['calc_type'] == 'multipole':
                continue
        
        # read cards specifying simulation parameters        
        for var in cards:
            try:
                change_or_map(param_dict, namelists[i], var[0], var[1]['type'])
            except ValueError:
                default_val = var[1]['default']
                # test to see if variable is "mission-critical"
                if default_val is None:
                    raise ValueError("No value supplied for mission-critical" +
                                            " variable {}.".format(var[0]))
                else:
                    param_dict[namelists[i]][var[0]] = default_val
                    # no mapping options, but keep this line to make it easier
                    # to merge with corresponding function in <_pn_control.py>
                    if type(default_val) == str and 'map' in default_val:
                        from_mapping(param_dict, namelists[i], var[0])
                        
    # extract values from the &elast namelist
    for var in elast_cards:
        try:
            change_or_map(param_dict, 'elast', var[0], var[1]['type'])
        except ValueError:
            param_dict['elast'][var[0]] = var[1]['default']
                                         
    # test to make sure that all of the parameters required for one of isotropic
    # or anisotropic elasticity have been supplied. Should we also test to see
    # which function to use when calculating the energy coefficients?
    #!!! Need to think about this
    if not (param_dict['elast']['cij'] is None):
        # if an elastic constants tensor is given, use anisotropic elasticity
        param_dict['elast']['coefficients'] = 'aniso'
    elif not (param_dict['elast']['shear'] is None):
        # test to see if another isotropic elastic property has been provided. 
        # Preference poisson's ratio over bulk modulus, if both have been provided
        if not (param_dict['elast']['poisson'] is None):
            param_dict['elast']['coefficients'] = 'iso_nu'
        elif not (param_dict['elast']['bulk'] is None):
            param_dict['elast']['coefficients'] = 'iso_bulk'
        else:
            raise AttributeError("No elastic properties have been provided.")
    else:
        raise AttributeError("No elastic properties have been provided.")
        
    return
    
def atom_namelist(param_dict):
    '''Converts the &atoms namelist in <param_dict> into a lookup table for
    atomic energies.
    '''
    
    unformatted_atoms = param_dict['atoms']
    atom_dict = dict()
    
    for species in unformatted_atoms.keys():
        atom_dict[species] = float(unformatted_atoms[species])
        
    return atom_dict
    
def burgers_cartesian(b, lattice):
    '''Construct the cartesian representation of the Burgers vector from the
    crystal lattice and lattice representation of the Burgers vector.
    '''
    
    burgers = np.zeros(3)
    
    basis_vectors = lattice.getLattice()
    for i in range(3):
        burgers += b[i]*basis_vectors[i]
        
    return burgers
    
def poisson(K, G):
    '''Calculate isotropic poisson's ratio from the bulk modulus <K> and shear
    modulus <G>.
    '''
    
    nu = (3*K-2*G)/(2*(3*K+G))
    return nu
    
class AtomisticSim(object):
    '''handles the control file for multipole-SC and cluster based dislocation 
    simulations in <dislopy>.
    '''
    
    def __init__(self,filename):
        self.sim = control_file(filename)
        handle_atomistic_control(self.sim)
        
        if not (self.sim['atoms'] is None):
            # process provided atomic energies
            self.atomic_energies = atom_namelist(self.sim)
        else:
            # no atomic energies provided. If these are required later, program
            # will prompt the user to enter values.
            self.atomic_energies = None
        
        # functions for easy access to namelist variables
        self.control = lambda card: self.sim['control'][card]
        self.elast = lambda card: self.sim['elast'][card]
        self.multipole = lambda card: self.sim['multipole'][card]
        self.cluster = lambda card: self.sim['cluster'][card]
        
        # check that the specified atomistic simulation code is supported
        if self.control('program') in supported_codes:
            pass
        else: 
            raise ValueError("{} is not a supported atomistic simulation code.".format(self.control('program')) +
                         "Supported codes are: GULP; QE; CASTEP.")
        
        # read in the unit cell and atomistic simulation parameters
        self.read_unit_cell()
         
        # construct the displacement field function                 
        self.set_field()
        
        # elasticity stuff
        if self.elast('coefficients') == 'aniso':
            # calculate K for the SPECIFIC dislocation (ie. combination of 
            # Burgers vector and line vector) with which we are concerned
            self.K = 4*np.pi*coeff.anisotropic_K_b(self.elast('cij'),
                                                   self.elast('burgers'),
                                                   self.elast('n'),
                                                   self.elast('m')
                                                  )
        elif self.elast('coefficients').startswith('iso'):
            # extract edge and screw components
            if 'nu' in self.elast('coefficients'):
                self.K = coeff.isotropic_nu(self.elast('poisson'), 
                                              self.elast('shear'))
            elif 'bulk' in self.elast('coefficients'):
                self.K = coeff.isotropic_K(self.elast('bulk'), 
                                            self.elast('shear'))
            
            if self.elast('disl_type') == 'edge':
                self.K = 4*np.pi*self.K[0]
            elif self.elast('disl_type') == 'screw':
                self.K = 4*np.pi*self.K[1]
        
        # create input and (if specified) run simulation        
        if self.control('calc_type') == 'cluster':
            self.construct_cluster()          
        elif self.control('calc_type') == 'multipole':
            self.construct_multipole()
        else:
            raise ValueError(("{} does not correspond to a known simulation " +
                                      "type").format(self.control('calc_type')))
            
    def read_unit_cell(self):
        '''Reads the unit cell and sets a few geometrical parameters.
        '''
        
        self.base_struc = cry.Crystal()
        self.ab_initio = False
        if self.control('program') == 'gulp':
            self.parse_fn = gulp.parse_gulp
            self.write_fn = gulp.write_gulp
        elif self.control('program') == 'qe':
            self.parse_fn = qe.parse_qe
            self.write_fn = qe.write_qe
            self.ab_initio = True
        elif self.control('program') == 'castep':
            self.parse_fn = castep.parse_castep
            self.write_fn = castep.write_castep
            self.ab_initio = True
        elif self.control('program') == 'lammps':
            self.parse_fn = lammps.parse_lammps
            self.write_fn = lammps.write_lammps
        
        # read in unit cell and translate so that origin of dislocation is at
        # (0., 0.)
        self.sys_info = self.parse_fn(self.control('unit_cell'), self.base_struc)

        self.base_struc.translate_cell(np.array([self.elast('centre')[0], 
                                   self.elast('centre')[1], 0.]), modulo=True)
        
        # convert the burgers vector to cartesian coordinates
        self.burgers = burgers_cartesian(self.elast('burgers'), self.base_struc)
                                      
    def set_field(self):
        '''Sets the dislocation field type, using the values provided in <elast>.
        '''
        
        # make choice of field
        if self.elast('field_type') == 'aniso_stroh':          
            self.ufield = aniso.makeAnisoField(self.elast('cij'))
            self.sij = 0. # dummy
            
        elif self.elast('field_type') == 'iso_screw':
            self.ufield = fields.isotropicScrewField
            self.sij = 0. # dummy
            
        elif self.elast('field_type') == 'iso_edge':
            self.ufield = fields.isotropicEdgeField
            if not (self.elast('poisson') is None):
                self.sij = self.elast('poisson')
            elif self.elast('cij') is not None:
                k, g = aniso.get_isotropic(self.elast('cij'))
                self.sij = poisson(k, g)
            else:
                self.sij = poisson(self.elast('bulk'), self.elast('shear'))
                
        elif self.elast('field_type') == 'aniso_screw':
            self.ufield = fields.anisotropicScrewField
            # need to make this work for general elasticty, but currently works
            # because C_{44} = 1/S_{44}, C_{55} = 1/S_{55}
            sij = np.linalg.inv(self.elast('cij'))
            self.sij = sij[3, 3]/sij[4, 4]
            
        elif self.elast('field_type') == 'aniso_edge':
            # edge dislocation in an anisotropic medium
            self.ufield = fields.anisotropicEdgeField
            self.sij = self.elast('cij')
        
        return 
        
    def construct_cluster(self):
        '''Constructs 1D-periodic clusters containing a single dislocation.
        '''
        
        # check that equal numbers of region 1 and region 2 radii have been provided
        if len(self.cluster('region1')) != len(self.cluster('region2')):
            # if one of the regions has a single value, assume that the radius of
            # that region should be held constant while iterating over the values given for the other
            if len(self.cluster('region1')) == 1:
                self.sim['cluster']['region1'] = self.cluster('region1')[0]*np.ones(len(self.cluster('region2')))
            elif len(self.cluster('region2')) == 1:
                self.sim['cluster']['region2'] = self.cluster('region2')[0]*np.ones(len(self.cluster('region1')))
            else:
                # region radius list are incompatible => raise an error
                raise ValueError('Number of region 1 and 2 radii provided must be equal.')
        
        for r1, r2 in zip(self.cluster('region1'), self.cluster('region2')):
            self.make_cluster(r1, r2)
        
        return 
        
    def make_cluster(self, r1, r2):
        '''Construct input file for a cluster-based simulation.
        '''
        
        if self.control('program') != 'gulp': # or LAMMPS...when I implement it.
            raise Warning("Only perform cluster-based calculation with" +
                      "interatomic potentials, if you know what's good for you")
        
        print("Constructing cluster...", end='')
            
        if self.control('program') == 'gulp' and (self.control('make_input') or
                                                  self.control('run_sim')):   
            sysout = open('{}.{:.0f}.{:.0f}.sysInfo'.format(self.control('basename'),
                                                                        r1, r2), 'w')
            sysout.write('pcell\n')
            sysout.write('{:.5f} 0\n'.format(self.cluster('thickness') *
                                             norm(self.base_struc.getC())))
            for line in self.sys_info:
                sysout.write('{}\n'.format(line))
            
            sysout.close()
            
            cluster = rod.TwoRegionCluster(self.base_struc, 
                                           np.zeros(3), 
                                           r2*self.cluster('scale'),
                                           r1,
                                           r2,
                                           self.cluster('thickness')
                                          )
                                         
            # apply displacement field
            if self.elast('disl_type') == 'screw':
                # should not contain a branch cut
                cluster.applyField(self.ufield, np.array([[0., 0.]]), [self.burgers], 
                                     Sij=self.sij, branch=self.cluster('branch_cut'),
                                            branch_thresh=self.cluster('cut_thresh'),
                                         centre_thresh=self.cluster('centre_thresh'),
                                                                    use_branch=False,
                                                                    randomise=self.elast('randomise'),
                                                                    random_r=self.elast('random_r'),
                                                                    random_amp=self.elast('amplitude'))
            else:              
                # delete/merge atoms that cross the branch cut          
                cluster.applyField(self.ufield, np.array([[0., 0.]]), [self.burgers], 
                                     Sij=self.sij, branch=self.cluster('branch_cut'),
                                               use_branch=self.cluster('use_branch'),
                                            branch_thresh=self.cluster('cut_thresh'),
                                         centre_thresh=self.cluster('centre_thresh'),
                                                                    randomise=self.elast('randomise'),
                                                                    random_r=self.elast('random_r'),
                                                                    random_amp=self.elast('amplitude'))
            
            outname = '{}.{:.0f}.{:.0f}'.format(self.control('basename'), r1, r2)
                                                
            gulp.write1DCluster(cluster, self.sys_info, outname, maxiter=self.control('maxcyc'))
            
            if self.control('run_sim'):
                self.run_simulation('dis.{}'.format(outname))
                self.run_simulation('ndf.{}'.format(outname))
                
        # calculate core energy -> putting this here so that core energies
        # can be easily recalculated without needing to re-run the cluster
        # relaxation calculations. USE WITH CARE.
        if self.control('calculate_core_energy'):
            self.core_energy_cluster(r1, r2)
                
        print('done')
            
        return
        
    def construct_multipole(self):
        '''Construct 3D-periodic simulation cell containing multiple dislocations
        (whose burgers vectors must sum to zero).
        '''
        
        # find read/write functions appropriate to the atomistic simulation code
        
        
        if self.multipole('grid'):
            for nx in self.multipole('nx'):
                for ny in self.multipole('ny'):
                    self.make_multipole(nx, ny)
        else:
            for nx, ny in zip(self.multipole('nx'), self.multipole('ny')):
                self.make_multipole(nx, ny)
                
        if self.control('calculate_core_energy'):
            # calculate core energy
            self.core_energy_sc()
        
        return
        
    def make_multipole(self, nx, ny):
        '''Make a single supercell with dimensions <nx>x<ny> containing a 
        dislocation multipole.
        '''
        
        # if necessary, scale the grid of kpoints and number of valence bands
        if self.ab_initio:
            atm.scale_kpoints(self.sys_info['cards']['K_POINTS'], np.array([nx, ny, 1.]))
            if self.control('program') == 'qe':
                qe.scale_nbands(self.sys_info['namelists']['&system'], np.array([nx, ny, 1.]))
                    
        # construct supercell
        supercell = cry.superConstructor(self.base_struc, np.array([nx, ny, 1.]))           
        
        if self.elast('disl_type') == 'edge':
            if self.multipole('npoles') == 2:
                # dipole 
                mp.edge_dipole(supercell, self.burgers, bdir=self.multipole('bdir'))
            elif self.multipole('npoles') == 4:
                # quadrupole
                mp.edge_quadrupole(supercell, self.burgers, bdir=self.multipole('bdir'))
            else:
                raise ValueError("Dipoles and quadrupoles only. You monster.")
        elif self.elast('disl_type') == 'screw':
            if self.multipole('npoles') == 2:
                # dipole
                mp.screw_dipole(supercell, self.burgers, self.ufield, self.sij, 
                                           alignment=self.multipole('alignment'))
            elif self.multipole('npoles') == 4:
                # quadrupole
                mp.screw_quadrupole(supercell, self.burgers, self.ufield, self.sij)
            else:
                raise ValueError("Dipoles and quadrupoles only. You monster.")
        else:
            raise NotImplementedError("Multipole calculations currently work " +
                                      "only for pure edge and screw dislocations.")
        
        # write the dislocated structure to file
        basename = '{}.{}.{}'.format(self.control('basename'), nx, ny)
        outstream = open('{}.{}'.format(basename, self.control('suffix')), 'w')
        
        if self.multipole('relaxtype'):
            relaxtype = self.multipole('relaxtype')
        else:
            relaxtype = None
        
        if self.control('program').lower() == 'gulp':
            self.write_fn(outstream, supercell, self.sys_info, to_cart=False, defected=True,
                                                add_constraints=False, relax_type=relaxtype, 
                                                             maxiter=self.control('maxcyc'))   
        else:
            self.write_fn(outstream, supercell, self.sys_info, to_cart=False, defected=True, 
                                                add_constraints=False, relax_type=relaxtype)
        
        
        # run calculations, if requested by the user
        if self.control('run_sim'):
            self.run_simulation(basename)
              
        # write undislocated cell to file -> useful primarily if excess energy
        # of the dislocation array is being calculated using "comparison"
        if self.multipole('method') == 'compare':
            outstream = open('{}.{}.{}'.format('ndf', basename, self.control('suffix')), 'w')
            self.write_fn(outstream, supercell, self.sys_info, to_cart=False,
                      defected=False, add_constraints=False, do_relax=False, prop=False)

            # run single point calculation, if requested
            if self.control('run_sim'):
                self.run_simulation('{}.{}'.format('ndf', basename))       
        
    def run_simulation(self, basename):
        '''If specified by the user, run the simulation on the local machine.
        '''
        
        # check that a path to the atomistic program executable has been provided
        if self.control('executable') is None:
            raise ValueError('No executable provided.')
        
        if 'gulp' == self.control('program'):
            gulp.run_gulp(self.control('executable'), basename) 
        elif 'qe' == self.control('program'):
            qe.run_qe(self.control('executable'), basename)
        elif 'castep' == self.control('program'):
            castep.run_castep(self.control('executable'), basename)
        elif 'lammps' == self.control('program'):
            lammps.run_lammps(self.control('executable'), basename,
                              para_exec=self.control('para_exec'),
                              nproc=self.control('para_nproc'),
                              set_omp=self.control('set_omp'),
                              omp_threads=self.control('omp_threads')
                             )
            
        return
        
    def core_energy_cluster(self, r1, r2):
        '''Calculate the core energy of a dislocation energy in with region I
        radius <r1> and region II radius <r2>.
        '''
        
        print('Calculating core energy:')
        # calculate using cluster method
        if not (self.cluster('method') in ['explicit', 'eregion', 'edge']):
            raise ValueError(("{} does not correspond to a valid way to " +
                "calculate the core energy.").format(self.cluster('method')))
                   
        # calculate excess energy due to the presence of a dislocation       
        if self.cluster('method') == 'explicit':
            self.atomic_energies = None
        elif self.cluster('method') == 'eregion':
            self.atomic_energies = None
        if self.cluster('method') == 'edge':
            if self.atomic_energies is None: 
                # prompt user to enter atomic symbols and energies
                self.atomic_energies = ce.atom_dict_interactive() 
                    
                     
        # run single point calculations
        basename = '{}.{:.0f}.{:.0f}'.format(self.control('basename'), r1, r2)  
                
        # note that <rmax> is always taken to be the region 1 radius
        total_thick = self.cluster('thickness')*norm(self.base_struc.getC())
        par, err = ce.dis_energy(r1-self.cluster('rgap'),
                                 self.cluster('rmin'),
                                 self.cluster('dr'),
                                 basename,
                                 self.control('executable'),
                                 self.cluster('method'),
                                 norm(self.burgers),
                                 total_thick,
                                 relax_K=self.cluster('fit_K'),
                                 K=self.K,
                                 atom_dict=self.atomic_energies,
                                 rc=self.elast('rcore'),
                                 using_atomic=True
                                )
                             
        print('Finished.')              
            
    def core_energy_sc(self):
        '''Calculates the dislocation core energy from supercell calculations.
        '''
            
        # check that valid method has been supplied
        if not (self.multipole('method') in ['standard', 'edge']):
            # standard: substract energy of unit cell multiplied by nx x ny
            # edge: subtract energy of individual atoms in the perfect material 
            raise ValueError(("{} does not correspond to a valid way to " +
                   "calculate the core energy.").format(self.multipole('method')))

        # determine suffix of atomistic simulation output files
        if self.control('program') == 'gulp':
            suffix = 'gout'
        elif self.control('program') == 'qe':
            suffix = 'out'
        elif self.control('program') == 'castep':
            suffix = 'castep'
                
        # calculate excess energy of the dislocation
        # read in energy of dislocated cells
        edis = mp.gridded_energies(self.control('basename'), self.control('program'), 
                              suffix, self.multipole('nx'), j_index=self.multipole('ny'), 
                                              relax=True, gridded=self.multipole('grid')) 
                        
        if self.multipole('method') == 'standard':
            # check that a valid unit cell energy has been provided
            if self.multipole('e_unit_cell') != self.multipole('e_unit_cell'):
                raise ValueError('<e_unit_cell> must be a real number.')
                 
            dE = mp.excess_energy_standard(edis, self.multipole('e_unit_cell'))
                
        elif self.multipole('method') == 'edge':
            # calculate excess energy from energies of atoms in perfect crystal
            if self.atomic_energies is None:
                # prompt use to enter energies for all atoms
                self.atomic_energies = ce.make_atom_dict()
                    
            dE = mp.excess_energy_edge(edis, self.atomic_energies, self.parse_fn, self.control('suffix'))
                        
        # fit the core energy and other parameters
        if self.multipole('npoles') == 4:
            ndis = np.array([2, 2])
        elif self.multipole('npoles') == 2:
            if self.multipole('bdir') == 0:
                ndis = np.array([2, 1])
            else:
                ndis = np.array([1, 2])
                    
        if self.multipole('fit_K'):        
            par, err = mp.fit_core_energy_mp(dE, self.base_struc, norm(self.burgers),
                                                        self.elast('rcore'), ndis=ndis)
        else:
            par, err = mp.fit_core_energy_mp(dE, self.base_struc, norm(self.burgers),
                                                self.elast('rcore'), ndis=ndis, K=self.K)
            
        mp.write_sc_energies(self.control('basename'), dE, par, err, K=self.K, using_atomic=True)
        return
        
    def write_output(self):
        '''Writes relevant output information to <control('output')>.
        '''
            
        return
        
def main():
    '''Runs an Atomistic simulation.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, nargs='?', dest='filename', default='0')
    
    args = parser.parse_args()
    
    if args.filename != '0':
        new_simulation = AtomisticSim(args.filename)
    else:
        # read in filename from the command line
        if sys.version_info.major == 2:
            filename = raw_input('Enter name of input file: ')
        elif sys.version_info.major == 3:
            filename = input('Enter name of input file: ')
            
        new_simulation = AtomisticSim(filename)
    
if __name__ == "__main__":
    main()
