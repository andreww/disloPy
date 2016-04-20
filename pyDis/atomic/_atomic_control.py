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

        cell_sizes = np.arange(i1, i2+di, di)       
    elif int_string:
        # making it iterable simplifies handling supercell setup
        cell_sizes = [int(int_string.group())]
    else:
        raise AttributeError("No cell dimensions found.")
    
    return cell_sizes

def vector(vec_str):
    '''Converts a string of the form "[x1,x2,...,xn]" to a numpy array.
    '''
    
    vec_regex = re.compile('(\[|\()(?P<contents>(?:\d+\.?\d*,\s*)*(?:\d+\.?\d*))(\]|\))')
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
    vec_element = re.compile('\d+\.?\d*')
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
                   ('rcore', {'default': np.nan, 'type': float}),
                   ('centre', {'default': np.zeros(2), 'type': vector}),
                   ('field_type', {'default': None, 'type': str})
                  )
                  
    # Now move on to namelists that specify parameters for specific simulation
    # types. <xlength> and <ylength> may be integers or arrays.
    # cards for the <&multipole> namelist. Valid methods for calculating the 
    # energy are: 'comparison' (screw dislocations only) and 'edge'.
    multipole_cards = (('nx', {'default': 1, 'type': array_or_int}),
                       ('ny', {'default': 1, 'type': array_or_int}),
                       ('npoles', {'default': 2, 'type': int}),
                       ('relaxtype', {'default': '', 'type': str}),
                       ('grid', {'default': True, 'type': to_bool}),
                       ('bdir', {'default': 0, 'type': int}),
                       ('method', {'default': '', 'type': int})
                      )
                      
    # cards for the <&cluster> namelist. Remember that the Stroh sextic theory
    #  (ie. anisotropic solution) places that branch cut along the negative
    # x-axis. Method for calculating core energy is specified may be edge,
    # eregion or explicit.
    cluster_cards = (('region1', {'default': None, 'type': float}),
                     ('region2', {'default': None, 'type': float}),
                     ('scale', {'default': 1.1, 'type': float}),
                     ('branch_cut', {'default': [0, -1], 'type': vector}),
                     ('thickness', {'default': 1, 'type': int}),
                     ('method', {'default': '', 'type': str}),
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
                if default_val == None:
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
        
        # read in the unit cell and atomistic simulation parameters
        self.read_unit_cell()
         
        # construct the displacement field function                 
        self.set_field()
        
        # elasticity stuff
        if self.elast('coefficients') == 'aniso':
            self.K = 4*np.pi*coeff.anisotropic_K(self.elast('cij'),
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
                                      
        # finally, calculate the dislocation core energy
        if self.control('calculate_core_energy'):
            self.core_energy()
            
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
            
        if self.control('program') == 'gulp': # add option for LAMMPS later            
            #!!! need to work out why I included this bit
            sysout = open('{}.{:.0f}.{:.0f}.sysInfo'.format(self.control('basename'),
                                                    self.cluster('region1'),
                                                    self.cluster('region2')), 'w')
            sysout.write('pcell\n')
            sysout.write('{:.5f} 0\n'.format(self.cluster('thickness') *
                                             norm(self.base_struc.getC())))
            for line in self.sys_info:
                sysout.write('{}\n'.format(line))
            
            sysout.close()
            
            cluster = rod.TwoRegionCluster(self.base_struc, 
                                           np.zeros(3), 
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
                                                
            gulp.write1DCluster(cluster, self.sys_info, outname)
            
            if self.control('run_sim'):
                self.run_simulation('dis.{}'.format(outname))
                self.run_simulation('ndf.{}'.format(outname))
                
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
        
        return
        
    def make_multipole(self, nx, ny):
        '''Make a single supercell with dimensions <nx>x<ny> containing a 
        dislocation multipole.
        '''
        
        if self.ab_initio:
            atm.scale_kpoints(self.sys_info['cards']['K_POINTS'], np.array([nx, ny, 1.]))
                    
        # construct supercell
        supercell = cry.superConstructor(self.base_struc, np.array([nx, ny, 1.]))
        
        ''' #!!! Changing this to use functions from <super_edge>
        supercell.applyField(self.ufield, self.cores, self.burgers, Sij=self.sij)
        '''            
        
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
                mp.screw_dipole(supercell, self.burgers, self.ufield, self.sij)
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
        self.write_fn(outstream, supercell, self.sys_info, to_cart=False, defected=True, 
              add_constraints=False, relax_type=self.multipole('relaxtype'))
        
        # run calculations, if requested by the user
        self.run_simulation(basename)
              
        # if the excess energy of the dislocation will be calculated using the 
        # comparison method, write an undislocated cell to file
        if self.multipole('method') == 'comparison':
            outstream = open('{}.{}.{}'.format('ndf', basename, 
                                    self.control('suffix')), 'w')
            self.write_fn(outstream, supercell, self.sys_info, to_cart=False,
                         defected=False, add_constraints=False, do_relax=False)

            # run single point calculation, of requested
            self.run_simulation('{}.{}'.format('ndf', basename))       
        
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
                if self.atomic_energies == None: 
                    # prompt user to enter atomic symbols and energies
                    self.atomic_energies = ce.make_atom_dict() 
                          
            # run single point calculation
            basename = '{}.{:.0f}.{:.0f}'.format(self.control('basename'),
                                                 self.cluster('region1'),
                                                 self.cluster('region2'))  
            print(self.K)  
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
                          rc=self.elast('rcore'),
                          using_atomic=True
                         )
            print('Finished.')              
            
        else: # supercell calculation
            # check that valid method has been supplied
            if not (self.multipole('method') in ['comparison', 'edge']):
                raise ValueError(("{} does not correspond to a valid way to " +
                   "calculate the core energy.").format(self.cluster('method')))

            # calculate excess energy of the dislocation
            if self.multipole('method') == 'comparison':
                # compare energies of dislocated cells with defect-free cells
                # begin by reading in the energies of the cells with dislocations...
                pass 
                # ...and without
                
                # compute the excess energy of the dislocation(s)
                
            elif self.multipole('method') == 'edge':
                # calculate excess energy from energies of atoms in perfect crystal
                if self.atomic_energies == None:
                    # prompt use to enter energies for all atoms
                    self.atomic_energies = ce.make_atom_dict()
            
        return
        
def main(filename):
    new_simulation = AtomisticSim(filename)
    
if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except IndexError:
        main(raw_input('Enter name of control file: '))
