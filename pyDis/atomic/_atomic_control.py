#!/usr/bin/env python
from __future__ import print_function,division

import numpy as np
import re
import sys
import numpy.linalg as L

## temporary path definition -> will generalise later
#sys.path.append('/home/richard/code_bases/pyDis_pn/') 

# need to shift the relevant functions to a simulation method-agnostic module                        
from ../pn._pn_control import control_file,from_mapping,change_type,change_or_map, \ 
                            print_control   
                            
# import modules required to set up and run a dislocation simulation
import gulpUtils as gulp
import crystal as cry
import rodSetup as rod
import disField as fields
import aniso

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
    control_cards = (('unit_cell',{'default':None,'type':str}),
                     ('path',{'default':'./','type':str}),
                     ('calc_type',{'default':None,'type':str}),
                     ('gulp_exec',{'default':'gulp','type':str),
                     ('sim_name',{'default':'dislocation','type':str),
                     ('max_cyc',{'default':1000,'type':int})
                    )
                    
    # cards for the <&geometry> namelist
    geometry_cards = (('eta',{'default':[0.,0.],'type':vector}),
                      ('RI',{'default':25.,'type':float}),
                      ('RII',{'default':15.,'type':float}),
                      ('branch_cut',{'default':[0,-1],'type':vector})
                     )
                     
    # cards for the <&fields> namelist. Elasticity is the parameter (eg. 
    # poisson's ratio, S44/S55 ratio, etc.) required to set up simulation
    # cij_file is the file containing the elastic constants matrix. Note that
    # pyDis only checks if <cij_file> has been defined if the Stroh theory
    # is being used to calculate the displacement fields.
    field_cards = (('field_type',{'default':None,'type':str}),
                    ('nfields',{'default':1,'type':int}),
                    ('elasticity',{'default':0.5,'type':float}),
                    ('cij_file',{'default':None,'type':str})
                   )
                     
    namelists = ['control','geometry','fields']
    # that all namelists in control file
    for name in namelists:
        try:
            x = test_dict[name]
        except KeyError:
            test_dict[name] = dict()
            
    for i,cards in enumerate([control_cards,geometry_cards,field_cards]):
        for var in cards:
            try:
                change_or_map(test_dict,namelist[i],var[0],var[1]['type'])
            except ValueError:
                default_var = var[1]['default']
                # test to see if variable is "mission-critical"
                if (default_vale == None) and var[0] != 'cij_file':
                    raise ValueError("No value supplied for mission-critical" +
                                            " variable %s." % var[0])
                else:
                    test_dict[namelists[i]][var[0]] = default_val
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
    
    new <type of field> :
        <important values (eg. burgers vectors, Sij)>
    end field;
    '''
    
    # list of burgers vectors and locations
    burgers_vectors = []
    dis_locations = []
    
    # regex for fields
    field_entry = re.compile('\s*new\s+field\s*:\s*\n(?P<dis>(?:\s*.*=.*;\s*\n))' +
                                            '\s*end field;')
    
    # begin by reading in field file and checking syntax
    field_options = {'burgers_length':{'default':None,'type':float},
                     'burgers_direction':{'default':[0,0,1],'type':vector},
                     'core':{'default':[0.,0.],'type':vector}
                    }
    
    with open(field_file) as f:
        lines = f.readlines()
        
        # find all field entries
        fields = re.findall(field_entry,lines)
        
        if not(fields):
            # no dislocations entered
            raise Exception('No dislocations provided.')
        else:
            # extract stuff
                   
    
    return burgers_vectors,dis_locations
    
class AtomisticSim(object):
    '''handles the control file for multipole-SC and cluster based dislocation 
    simulations in <pyDis>.
    '''
    
    def __init__(self,filename):
        self.sim = control_file(filename)
        handle_atomistic_control(self.sim)
        
        # functions for easy access to namelist variables
        self.control = lambda card: self.sim['control'][card]
        self.geometry = lambda card: self.sim['geometry'][card]
        self.fields = lambda card: self.sim['fields'][card]
        
        # construct displacement fields
        self.burgers_vectors,self.core_locations = handle_fields(filename,
                                                        self.fields('nfields'))
        
        # If generating displacement fields for an anisotropic medium using the
        # Stroh theory, read in .cij file and construct the appropriate 
        # field function
        
    def make_common_input(self):
        '''Performs setup tasks common to both the multipole-SC and cluster 
        methods.
        '''

        self.gulp_struc = cry.Crystal()
        self.sys_info = gulp.parse_gulp(self.control('unit_cell'), gulp_struc,
                                                     path=self.control('path'))
        
    def make_cluster_input(self):
        '''Perform setup tasks specific to the cluster-based calculation.
        '''
        
        # write file containing system parameters
        sys_out = open(self.control('sim_name') + '%d.%d.sysInfo' % 
                             (self.geometry('RI'),self.geometry('RII')),'w')
                             
        sys_out.write('pcell\n')
        sys_out.write('%.5f 0\n' % (self.thickness*L.norm(gulp_struc.getC())))
        for line in self.sys_info:
            sys_out.write(line + '\n')
        sys_out.close()
        
        self.b = self.make_burgers()
        
        return
        
    def make_burgers(self):
        self.
        
def main(filename):
    new_simulation = AtomisticSim(filename)
    
if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except IndexError:
        main(raw_input('Enter name of control file: '))
    

