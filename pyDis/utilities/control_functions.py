#!/usr/bin/env python
'''Contains all generic functions used by <pydis> control programs, such as
<_atomic_control> and <_pn_control>.
'''
from __future__ import absolute_import, print_function

import re

def control_file(filename):
    '''Opens the control file <filename> for a PN simulation and constructs a 
    dictionary containing the (unformatted) values for all necessary simulation
    parameters.
    '''
    
    sim_parameters = dict()
    input_style = re.compile('\s*(?P<par>.+)\s*=\s*[\'"]?(?P<value>[^\'"]*)[\'"]?\s*;')
    in_namelist = False
    with open(filename) as f:
        for line in f:
            temp = line.strip()
            # handle the line
            if temp.startswith('#'): # comment
                continue
            if temp.startswith('&'): # check to see if <line> starts a namelist
                card_name = re.search('&(?P<name>\w+)\s+{', temp).group('name')
                sim_parameters[card_name] = dict()
                in_namelist = True
                continue
            if in_namelist:
                if '};;' in temp: # end of namelist
                    in_namelist = False
                    continue
                elif not(temp.strip()):
                    # empty line
                    continue
                # else
                inp = re.search(input_style, temp)
                sim_parameters[card_name][inp.group('par').strip()] = \
                                                    inp.group('value')

    return sim_parameters
    
def from_mapping(top_dict, namelist, card):
    '''Generates a value for <top_dict[namelist][card]> from a provided 
    functional form.
    '''
    
    map_form = re.compile('\s*map\s+(?P<namelist>[^\s]+)\s*>\s*(?P<val>.+):' +
               '\s*(?P<var>\w+)\s*-\>\s*(?P<expr>.*)')
               
    in_str = top_dict[namelist][card]
    found_map = re.search(map_form, in_str)
    
    # construct the mapping from <val> -> <var>
    new_func = 'lambda {}: {}'.format(found_map.group('var'), found_map.group('expr'))
    input_value = top_dict[found_map.group('namelist')][found_map.group('val')]
    
    # evaluate function using the provided value
    val = eval('({})({})'.format(new_func, input_value))
    top_dict[namelist][card] = val
    return

    
def change_type(top_dict, namelist, card, new_type):
    '''Converts the string in the specified <card> in <namelist> to the required 
    type.
    '''
    
    try:
        top_dict[namelist][card] = new_type(top_dict[namelist][card])
    except ValueError:
        # test to see if np.e or np.pi in input
        if 'np.pi' in top_dict[namelist][card] or 'np.e' in top_dict[namelist][card]:
            top_dict[namelist][card] = eval(top_dict[namelist][card])

    return
    
def change_or_map(param_dict, namelist, card, new_type):
    '''Decides whether the value in <param_dict[namelist][card]> can be converted
    directly to a bool or numerical type (ie. int, float, or complex) or whether the
    provided value is a mapping involving some other parameter.
    '''

    # check that value is defined
    try:
        x = param_dict[namelist][card]
    except KeyError:
        # not defined -> go to default value
        raise ValueError

    if 'map' in  param_dict[namelist][card]:
        # provided value is a mapping
        from_mapping(param_dict, namelist, card)
    else:
        # convert to <new_type>
        change_type(param_dict, namelist, card, new_type)
        
    return
    
def to_bool(in_str):
    '''Routine to convert <in_str>, which may take the values "True" or "False", 
    a boolean (needed because bool("False") == True)
    '''
    
    bool_vals = {"True":True, "False":False}
    
    try:
        new_value = bool_vals[in_str]
    except KeyError:
        raise ValueError("{} is not a boolean value.".format(in_str))
        
    return new_value
    
def print_control(control_dict, print_types=True):
    '''Print the values of each card in the namelists in <control_dict>
    including, optionally, their types. Useful mainly as a debugging tool
    if adding new cards (or namelists).
    '''
    
    for key in control_dict:
        print("### Namelist: {} ###\n".format(key.upper()))
        for key1 in control_dict[key]:
            print("CARD: {}".format(key1))
            print("VALUE: {}".format(control_dict[key][key1]))
            
            if print_types:
                print("TYPE: {}".format(type(control_dict[key][key1])))
            else:
                print()
    
    return 
