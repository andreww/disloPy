#!/usr/bin/env python
from __future__ import print_function,division

import numpy as np
import re
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')
from numpy.linalg import norm

# list of atomic simulation codes currently supported by pyDis
supported_codes = ('qe', 'gulp', 'castep')

#!!! need to shift the relevant functions to a simulation method-agnostic module  
from pyDis.pn._pn_control import control_file, from_mapping, change_type, to_bool, \
                                            change_or_map, print_control   
                            
# import modules required to set up and run a dislocation simulation
from pyDis.atomic.atomic_import import *

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

        cell_size = np.arange(i1, i2+di, di)       
    elif int_string:
        # making it iterable simplifies handling supercell setup
        cell_sizes = [int(int_string.group())]
    else:
        raise AttributeError("No cell dimensions found.")
    
    return cell_sizes

def vector(vec_str):
    '''Converts a string of the form "[x1,x2,...,xn]" to a numpy array.
    '''
    
    vec_regex = re.compile('(\[|\()(?P<contents>(?:.*,)*.*)(\]|\))')
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
    el_regex = re.compile('(\d+/\d+|\d+|\d+\.\d+)')
    elements = el_regex.finditer(contents)
    base_vector = []
    for el in elements:
        try:
            el_value = float(eval(el.group()))
        except NameError:
            print('Error: vector contains non-numerical data.')
            
        base_vector.append(el_value)
        
    return np.array(base_vector)    

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
                     ('basename', {'default': 'dis', 'type': str}),
                     ('suffix', {'default': 'in', 'type': str}),
                     ('executable', {'default': '', 'type': str}),
                     ('calculate_core_energy', {'default': True, 'type': to_bool})
                    )
    
    # cards for the <&elast> namelist. Note that if dislocations are specified 
    # in the <fields> file, they will overwrite <burgers>. Current options for
    # <field_type> are: aniso(tropic), iso_screw, iso_edge, and aniso_screw.
    # aniso_edge will be added later.           
    elast_cards = (('disl_type', {'default': None, 'type': str}),
                   ('burgers', {'default': cry.ei(3), 'type': vector}),
                   ('bulk', {'default': None, 'type': float}),
                   ('shear', {'default': None, 'type': float}),
                   ('poisson', {'default': None, 'type': float}),
                   ('cij', {'default': None, 'type': aniso.readCij}),
                   ('in_gpa', {'default': True, 'type': to_bool}),
                   ('rcore', {'default': 10., 'type': float}),
                   ('centre', {'default': np.zeros(2), 'type': vector}),
                   ('field_type', {'default': None, 'type': str})
                  )
                  
    # Now move on to namelists that specify parameters for specific simulation
    # types. <xlength> and <ylength> may be integers or arrays.
    # cards for the <&multipole> namelist
    multipole_cards = (('field_file', {'default': 'dis.cores', 'type': str}),
                       ('nx', {'default': 1, 'type': array_or_int}),
                       ('ny', {'default': 1, 'type': array_or_int}),
                       ('relaxtype', {'default': '', 'type': str})
                      )
                      
    # cards for the <&cluster> namelist. Remember that the Stroh sextic theory
    #  (ie. anisotropic solution) places that branch cut along the negative
    # x-axis. Method for calculating core energy is specified using an 
    # integer, which may take the following values: 1 == GULP method, 2 == explicit
    # calculation of region energies, 3 == edge approach (ie. atomic energies).
    cluster_cards = (('region1', {'default': None, 'type': float}),
                     ('region2', {'default': None, 'type': float}),
                     ('scale', {'default': 1.1, 'type': float}),
                     ('branch_cut', {'default': [0, -1], 'type': vector}),
                     ('thickness', {'default': 1, 'type': int}),
                     ('method', {'default': 1, 'type': int}),
                     ('rmax', {'default': 2, 'type': int}),
                     ('rmin', {'default': 1, 'type': int}),
                     ('dr', {'default': 1, 'type': int}),
                     ('fit_K', {'default': False, 'type': to_bool})
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
        for var in cards:
            try:
                change_or_map(param_dict, namelists[i], var[0], var[1]['type'])
            except ValueError:
                default_val = var[1]['default']
                # test to see if variable is "mission-critical"
                if default_val == None:
                    raise ValueError("No value supplied for mission-critical" +
                                            " variable %s.".format(var[0]))
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
    if param_dict['elast']['cij'] != None:
        # if an elastic constants tensor is given, use anisotropic elasticity
        param_dict['elast']['coefficients'] = 'aniso'
    elif param_dict['elast']['shear'] != None:
        # test to see if another isotropic elastic property has been provided. 
        # Preference poisson's ratio over bulk modulus, if both have been provided
        if param_dict['elast']['poisson']  != None:
            param_dict['elast']['coefficients'] = 'iso_nu'
        elif param_dict['elast']['bulk'] != None:
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
                
def handle_fields(field_file):
    '''Handle fields separately from the <&control> and <&geometry> namelists.
    This is necessary because each field must be defined separately, making
    it difficult to map directly into a dictionary.
    
    Each field has the following form:
    
    new_field
        <b0> <b1> <b2> <eta1> <eta2>
    end field;
    '''
    
    # list of burgers vectors and locations
    burgers_vectors = []
    dis_locations = []
    
    # regex for fields
    field_form = re.compile('new_field(?P<burgers>(?:\s+-?\d+\.?\d*){3})'+
                            '(?P<centre>(?:\s+\d\.?\d*){2})\s+end_field;')
    
    with open(field_file) as f:
        fieldstring = f.readlines()
 
    # now find all field entries and construct dislocation fields. Append entries
    # of the form [b, eta]
    dis_burgers = []
    dis_locs = []
    for field in field_form.finditer(fieldstring):
        b = field.group('burgers').split()
        b = np.array([float(bi) for bi in b])
        dis_burgers.append(b)
        
        eta = field.group('centre').split()
        eta = np.array([float(etai) for etai in eta])
        dis_locs.append(eta)
        
    if not(dis_burgers):
        # no dislocations entered
        raise Exception('No dislocations provided.')
    
    return dis_burgers, dis_locs
    
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
    
    nu = (3*K - 2*G)/(2*(3*K+G))
    return nu
    
class AtomisticSim(object):
    '''handles the control file for multipole-SC and cluster based dislocation 
    simulations in <pyDis>.
    '''
    
    def __init__(self,filename):
        self.sim = control_file(filename)
        handle_atomistic_control(self.sim)
        
        if self.sim['atoms'] != None:
            # process provided atomic energies
            self.atomic_energies = atom_namelist(self.sim)
        
        # functions for easy access to namelist variables
        self.control = lambda card: self.sim['control'][card]
        self.elast = lambda card: self.sim['elast'][card]
        self.multipole = lambda card: self.sim['multipole'][card]
        self.cluster = lambda card: self.sim['cluster'][card]
        
        # check that the specified atomistic simulation code is supported
        if self.control('program') in supported_codes:
            pass
        else: 
            raise ValueError("{} is not a supported atomistic simulation code." +
                         "Supported codes are: GULP; QE; CASTEP.")
                         
        self.set_field()
        
        # if multipole simulation, read dislocation parameters from field file
        if self.control('calc_type') == 'supercell':
            self.burgers, self.cores = handle_fields(self.multipole('field_file'))
        
        # elasticity stuff
        if self.elast('coefficients') == 'aniso':
            self.K = coeff.anisotropic_K(self.elast('cij'),
                                         self.elast('b_edge'),
                                         self.elast('b_screw'),
                                         self.elast('normal')
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
                self.K = self.K[0]
            elif self.elast('disl_type') == 'screw':
                self.K = self.K[1]
        
        # create input and (if specified) run simulation        
        if self.control('calc_type') == 'cluster':
            self.construct_cluster()          
        elif self.control('calc_type') == 'supercell':
            self.construct_multipole()
        else:
            raise ValueError(("{} does not correspond to a known simulation " +
                                      "type").format(self.control('calc_type')))
                                      
        # finally, calculate the dislocation core energy
        if self.control('calculate_core_energy'):
            self.core_energy()
                                      
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
            if self.elast('poisson') != None:
                self.sij = self.elast('poisson')
            else:
                self.sij = poisson(self.elast('bulk'), self.elast('shear'))
                
        elif self.elast('field_type') == 'aniso_screw':
            self.ufield = fields.anisotropicScrewField
            # need to make this work for general elasticty, but currently works
            # because C_{44} = 1/S_{44}, C_{55} = 1/S_{55}
            self.sij = self.elast('cij')[4, 4]/self.elast('cij')[3, 3]
        
        return 
        
    def construct_cluster(self):
        '''Construct input file for a cluster-based simulation.
        '''
        
        if self.control('program') != 'gulp': # or LAMMPS...when I implement it.
            raise Warning("Only perform cluster-based calculation with" +
                      "interatomic potentials if you know what's good for you")
        
        print("Constructing cluster...", end='')
            
        self.base_struc = cry.Crystal()
        if self.control('program') == 'gulp': # add option for LAMMPS later
            sys_info = gulp.parse_gulp(self.control('unit_cell'), self.base_struc)
            
            self.burgers = burgers_cartesian(self.elast('burgers'), self.base_struc)
            
            #!!! need to work out why I included this bit
            sysout = open('{}.{:.0f}.{:.0f}.sysInfo'.format(self.control('basename'),
                                                    self.cluster('region1'),
                                                    self.cluster('region2')), 'w')
            sysout.write('pcell\n')
            sysout.write('{:.5f} 0\n'.format(self.cluster('thickness') *
                                             norm(self.base_struc.getC())))
            for line in sys_info:
                sysout.write('{}\n'.format(line))
            
            sysout.close()
            
            cluster = rod.TwoRegionCluster(self.base_struc, 
                                          self.elast('centre'), 
                                          self.cluster('region2')*self.cluster('scale'),
                                          self.cluster('region1'),
                                          self.cluster('region2'),
                                          self.cluster('thickness')
                                         )
                                         
            # apply displacement field
            cluster.applyField(self.ufield, np.array([[0., 0.]]), [self.burgers], 
                                    Sij=self.sij, branch=self.cluster('branch_cut'))
            
            outname = '{}.{:.0f}.{:.0f}'.format(self.control('basename'), 
                                                self.cluster('region1'),
                                                self.cluster('region2'))
                                                
            gulp.write1DCluster(cluster, sys_info, outname)
            
            if self.control('run_sim'):
                self.run_simulation('dis.{}'.format(outname))
                self.run_simulation('ndf.{}'.format(outname))
                
        print('done')
            
        return
        
    def construct_multipole(self):
        '''Construct 3D-periodic simulation cell containing multiple dislocations
        (whose burgers vectors must sum to zero).
        '''
        
        self.base_struc = cry.Crystal()
        
        # find read/write functions appropriate to the atomistic simulation code
        ab_initio = False
        if self.control('program') == 'gulp':
            parse_fn = gulp.parse_gulp
            write_fn = gulp.write_gulp
        elif self.control('program') == 'qe':
            parse_fn = qe.parse_qe
            write_fn = qe.write_qe
            ab_initio = True
        elif self.control('program') == 'castep':
            parse_fn = castep.parse_castep
            write_fn = castep.write_castep
            ab_initio = True
        
        sys_info = parse_fn(self.control('unit_cell'), self.base_struc)
        
        for nx in self.multipole('nx'):
            for ny in self.multipole('ny'):
                # if calculation uses DFT, need to scale k-point grid
                if ab_initio:
                    atm.scale_kpoints(sys_info['cards']['K_POINTS'], np.array([nx, ny, 1.]))
                
                # construct supercell
                supercell = cry.superConstructor(base_struc, np.array([nx, ny, 1.]))
                                                                  
                supercell.applyField(self.field, self.cores, self.burgers, Sij=self.sij)
                
                basename = '{}.{}.{}'.format(self.control('basename'), nx, ny)
                outstream = open('{}.{}.{}.{}'.format(basename, self.control('suffix')), 'w')
                write_fn(outstream, supercell, sys_info, to_cart=False, defected=True, 
                          add_constraints=False, relax_type=self.multipole('relaxtype'))
                          
                # run calculations, if requested by the user
                self.run_simulation(basename)
        
        return
        
    def run_simulation(self, basename):
        '''If specified by the user, run the simulation on the local machine.
        '''
        
        # check that a path to the atomistic program executable has been provided
        if self.control('executable') == None:
            raise ValueError('No executable provided.')
        
        if 'gulp' == self.control('program'):
            gulp.run_gulp(self.control('executable'), basename) 
        elif 'qe' == self.control('program'):
            qe.run_qe(self.control('executable'), basename)
        elif 'castep' == self.control('program'):
            castep.run_castep(self.control('executable'), basename)
            
        return
        
    def core_energy(self):
        '''Calculate the core energy of the dislocation.
        
        Probably need to modify <cluster_energy> and <edge_energy> so that burgers
        magnitude can be input.
        '''
        
        if not self.control('run_sim'):
            # must have relaxed the dislocation structure to calculate it E_core
            pass
        elif not self.control('calculate_core_energy'):
            pass      
        elif self.control('calc_type') == 'cluster':
            print('Calculating core energy:')
            #!!! Need to completely rewrite this to accommodate changes to the 
            #!!! energy modules -> will be shorter.
            # calculate using cluster method
            print(self.cluster('method') == 'edge')
            if (self.cluster('method') != 'eregion' 
                and self.cluster('method') != 'explicit'
                and self.cluster('method') != 'edge'):
                raise ValueError(("{} does not correspond to a valid way to " +
                   "calculate the core energy.").format(self.cluster('method')))
            if self.cluster('method') == 1:
                method = 'eregion'
                self.atomic_energies = None
            elif self.cluster('method') == 2:
                method = 'explicit'
                self.atomic_energies = None
            if self.cluster('method') == 3:
                if self.atomic_energies == None: 
                    # prompt user to enter atomic symbols & energies
                    self.atomic_energies = ce.make_atom_dict() 
                          
            # run single point calculation
            basename = '{}.{:.0f}.{:.0f}'.format(self.control('basename'),
                                                 self.cluster('region1'),
                                                 self.cluster('region2'))    
            ce.dis_energy(self.cluster('rmax'),
                          self.cluster('rmin'),
                          self.cluster('dr'),
                          basename,
                          self.control('executable'),
                          self.cluster('method'),
                          self.burgers[0],
                          self.cluster('thickness')*norm(self.base_struc.getC()),
                          relax_K=self.cluster('fit_K'),
                          K=self.K,
                          atom_dict=self.atomic_energies,
                          rc=self.elast('rcore')
                         )
            print('Finished.')              
            
        else: # supercell calculation
            #!!! Need to think about this
            if self.multipole('method') == 1:
                pass
            elif self.multipole('method') == 2:
                pass
            
        return
        
def main(filename):
    new_simulation = AtomisticSim(filename)
    
if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except IndexError:
        main(raw_input('Enter name of control file: '))
