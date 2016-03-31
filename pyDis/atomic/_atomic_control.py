#!/usr/bin/env python
from __future__ import print_function,division

import numpy as np
import re
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')
import numpy.linalg as L

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
        cell_sizes = int(int_string.group())
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

def handle_atomistic_control(test_dict):
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
                     ('simulation_name', {'default': 'dis', 'type': str}),
                     ('suffix', {'default': 'in', 'type': str}),
                     ('executable', {'default': '', 'type': str}),
                    )
    
    # cards for the <&elast> namelist. Note that if dislocations are specified 
    # in the <fields> file, they will overwrite <burgers>               
    elast_cards = (('disl_type', {'default': None, 'type': str}),
                   ('burgers', {'default': cry.ei(3), 'type': vector}),
                   ('bulk', {'default': None, 'type': float}),
                   ('shear', {'default': None, 'type': float}),
                   ('possion', {'default': None, 'type': float}),
                   ('in_gpa', {'default': True, 'type': to_bool}),
                   ('rcore', {'default': 10., 'type': float}),
                   ('centre', {'default': np.zeros(2), 'type': vector})
                  )
                  
    # Now move on to namelists that specify parameters for specific simulation
    # types. <xlength> and <ylength> may be integers or arrays.
    # cards for the <&multipole> namelist
    multipole_cards = (('field_file', {'default': 'dis.cores', 'type': str}),
                       ('xlength', {'default': 1, 'type': array_or_int}),
                       ('ylength', {'default': 1, 'type': array_or_int})
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
                     
    namelists = ['control','elast', 'multipole', 'cluster']
    
    # that all namelists in control file
    for name in namelists:
        if name not in param_dict.keys():
            param_dict[name] = dict()
            
    for i,cards in enumerate([control_cards, elast_cards, multipole_cards,
                                                           cluster_cards]):
        for var in cards:
            try:
                change_or_map(param_dict,namelist[i],var[0],var[1]['type'])
            except ValueError:
                default_var = var[1]['default']
                # test to see if variable is "mission-critical"
                if (default_vale == None) and var[0] != 'cij_file':
                    raise ValueError("No value supplied for mission-critical" +
                                            " variable %s." % var[0])
                else:
                    param_dict[namelists[i]][var[0]] = default_val
                    # no mapping options, but keep this line to make it easier
                    # to merge with corresponding function in <_pn_control.py>
                    if type(default_val) == str and 'map' in default_val:
                        pass
                        
    return
                
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
    
    #with open(field_file) as f:
    #    fieldstring = f.readlines()
    fieldstring = field_file    
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
    
class AtomisticSim(object):
    '''handles the control file for multipole-SC and cluster based dislocation 
    simulations in <pyDis>.
    '''
    
    def __init__(self,filename):
        self.sim = control_file(filename)
        handle_atomistic_control(self.sim)
        
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
        
        # if multipole simulation, read dislocation parameters from field file
        if self.control('calc_type') == 'supercell':
            self.burgers, self.cores = handle_fields(self.multipole('field_file'))
        
        # calculate the energy coefficients -> could be turned into own function
        # need to rethink for analytic solutions with anisotropic elasticity
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
            
            if self.control('disl_type') == 'edge':
                self.K = self.K[0]
            elif self.control('disl_type') == 'screw':
                self.K = self.K[1]
        
    def construct_cluster(self):
        return
        
    def construct_multipole(self):
        return
        
    def run_simulation(self):
        '''If specified by the user, run the simulation on the local machine.
        '''
        
        # check that a path to the atomistic program executable has been provided
        if self.control('executable') == None:
            raise ValueError('No executable provided.')
        
        if self.control('run_sim'):
            # run simulation
            if 'gulp' == args.prog.lower():
                gulp.run_gulp(args.progexec, basename) 
            elif 'qe' == args.prog.lower():
                qe.run_qe(args.progexec, basename)
            elif 'castep' == args.prog.lower():
                castep.run_castep(args.progexec, basename)
        else:
            pass
            
        return
        
    def core_energy(self):
        '''Calculate the core energy of the dislocation.
        '''
        
        if not self.control('calculate_core_energy'):
            pass
        
        elif self.control('calc_type') == 'cluster':
            # calculate using cluster method
            if self.cluster('method') == 1: # Normal GULP method
                cluster_energy.main([self.cluster('rma
                pass
            elif self.cluster('method') == 2: # direct calculation on regions
                pass
            elif self.cluster('method') == 3: # edge method
                pass
            else: 
                raise ValueError(("{} does not correspond to a valid way to " +
                   "calculate the core energy.").format(self.cluster('method')))
        else: # supercell calculation
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
