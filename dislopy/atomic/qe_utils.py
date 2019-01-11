#!/usr/bin/env python
'''Utilities required for interfacing with the pw-DFT code Quantum Espresso.
'''
from __future__ import print_function, absolute_import

import re
import numpy as np

import sys

from dislopy.atomic import crystal as cry
from dislopy.utilities import atomistic_utils as util
from dislopy.atomic import transmutation as mutate

namelists= ['&control','&system','&electrons','&ions','&cell']
cards = ['CELL_PARAMETERS','ATOMIC_SPECIES','ATOMIC_POSITIONS','CONSTRAINTS',
                                            'OCCUPATIONS','FORCES','K_POINTS']

def parse_qe(filename, qe_struc, path='./'):
    '''Parses qe file <filename> and extracts structure to
    <qe_struc>.
    '''
    
    qe_lines = util.read_file(filename, path)
    # extract cards and namelists
    # begin by extracting indices of card and namelist block entries
    name_i = [i for i, line in enumerate(qe_lines) if any([name in
                                      line for name in namelists])]
    card_i = [i for i, line in enumerate(qe_lines) if any([card in
                                          line for card in cards])]
    
    # extract namelist blocks
    name_dict = dict()
    for i in range(len(name_i)):
        # extract the namelist key
        for key in namelists:
            if key in qe_lines[name_i[i]]:
                use_key = key
                break
        # extract elements of namelist to dictionary
        try:
            name_dict[use_key] = qe_lines[name_i[i]:name_i[i+1]]
        except IndexError:
            name_dict[use_key] = qe_lines[name_i[i]:card_i[0]]
            
    # extract card blocks
    card_dict = dict()
    for i in range(len(card_i)):
        # extract the card key
        for key in cards:
            if key in qe_lines[card_i[i]]:
                use_key = key
                break
        # extract elements of card block to the card dictionary
        try:
            card_dict[use_key] = qe_lines[card_i[i]:card_i[i+1]]
        except IndexError:
            card_dict[use_key] = qe_lines[card_i[i]:]
            
    # lattice parameters and atomic coordinates
    for i, cell_param in enumerate(card_dict['CELL_PARAMETERS'][1:]):
        cell_param = cell_param.split()
        new_vec = np.array([float(x) for x in cell_param])
        qe_struc.setVector(new_vec, i)
        
    # extract atoms
    for atom_line in card_dict['ATOMIC_POSITIONS'][1:]:
        temp = atom_line.split()
        name = temp[0]
        coords = np.array([float(x) for x in temp[1:]])
        qe_struc.addAtom(cry.Atom(name, coords))
    
        sys_info = extract_parameters(name_dict, card_dict)                   
    return sys_info

def extract_parameters(name_dict, card_dict):
    '''Extract the simulation parameters to sys_info.
    '''
    
    sys_info = dict()
    
    # regex to capture variable definition
    var_form = re.compile('(?P<name>\w[\w\d_\(\)]*(?:\([\w\d_,\s]+\))?)\s*=\s*(?P<value>[^,]+)')
    
    # extract namelist guff to <sys_info>
    sys_info['namelists'] = dict()
    for key in name_dict:
        sys_info['namelists'][key] = dict()
        for el in name_dict[key]:
            found = var_form.finditer(el)
            if found:
                for entry in found:
                    sys_info['namelists'][key][entry.group('name')] = entry.group('value') 
    
    # extract the card guff to <sys_info>. Because entries in the card blocks do
    # not share a common format, each possible card must be parsed separately.
    sys_info['cards'] = dict()            
    for key in card_dict:
        if key == 'CELL_PARAMETERS':
            # extract cell parameter units
            units = re.search(r'{\s*(?P<units>(alat|bohr|angstrom))\s*}',
                                          card_dict['CELL_PARAMETERS'][0])
            if units:
                sys_info['cards']['CELL_PARAMETERS'] = units.group('units')
            else:
                print('No units supplied. Assuming bohr.')
                sys_info['cards']['CELL_PARAMETERS'] = None
        if key == 'K_POINTS':
            header = re.compile('K_POINTS\s+{\s*(?P<type>(automatic|gamma))\s*}')
            preamble = header.search(card_dict['K_POINTS'][0])
            if not preamble:
                raise ValueError("Specified grid generation scheme not supported.")
            else:
                if preamble.group('type') == 'gamma':
                    sys_info['cards']['K_POINTS'] = None
                else:
                    grid_form = re.compile('(?P<grid>(?:\d+\s+){3})(?P<shift>\d+\s+\d+\s+\d+)')
                    grid = re.search(grid_form, card_dict['K_POINTS'][1])
                    sys_info['cards']['K_POINTS'] = dict()
                    sys_info['cards']['K_POINTS']['spacing'] = np.array([float(x)
                                           for x in grid.group('grid').split()])
                    sys_info['cards']['K_POINTS']['shift'] = grid.group('shift')                                 
        elif key == 'ATOMIC_SPECIES':
            # handle pseudopotentials
            sys_info['cards'][key] = card_dict[key][1:]
        elif key == 'OCCUPATIONS':
            # handle occupations -> not yet implemented
            print("Occupations not implemented. Skipping...")
        else:
            # information stored elsewhere
            pass
            
    return sys_info
    
def add_psps(sim_info, new_psps):
    '''Add additional pseudopotentials to <sim_info>. Useful for 
    impurity calculations where impurity atoms are not present in the
    bulk material. <new_psps> is a list of objects of class <Pseudopotential>.
    '''
    
    for psp in new_psps:
        sim_info['cards']['ATOMIC_SPECIES'].append(str(psp))

class Pseudopotential(object):
    '''Holds information for a QE pseudopotential.
    '''
    
    def __init__(self, species, atomic_weight, psp):
        self.species = species
        self.weight = atomic_weight
        self.psp = psp

    def __str__(self):
        return '  {} {:.4f} {}'.format(self.species, self.weight, 
                                                     self.psp)
                                                     
def scale_nbands(system_nmlst, sc_dims):
    '''If the <nbands> variable (number of valence bands) is specified in the 
    system information for the base cell, scale by the size of the new supercell
    defined by <sc_dims> (i.e. the multiples of unit cell parameters in x,y,z).
    '''
    
    if 'nbnd' in system_nmlst.keys():
        # increase number of bands to reflect new sc size
        old_nbands = int(system_nmlst['nbnd'])
        new_nbands = old_nbands*sc_dims[0]*sc_dims[1]*sc_dims[2]
        system_nmlst['nbnd'] = new_nbands
    
    return 
        
def write_qe(outstream, qe_struc, sys_info, defected=True, to_cart=False,
       add_constraints=False, relax_type='scf', impurities=None, do_relax=None, prop=None):
    '''Writes crystal structure contained in <qe_struc> to <outstream>.<prop> is
    a dummy variable to make input consistent with <write_gulp>.
    '''
            
    # if isolated/coupled defects have been supplied, add these to the structure
    if not (impurities is None):
        if mutate.is_single(impurities):
            mutate.cell_defect(qe_struc, impurities, use_displaced=True)
        elif mutate.is_coupled(impurities):
            mutate.cell_defect_cluster(qe_struc, impurities, use_displaced=True)
        else:
            raise TypeError("Supplied defect not of type <Impurity>/<CoupledImpurity>")
    
    # write namelists
    for block in namelists:
        # test that the namelist <block> is not empty
        if not (block in sys_info['namelists'].keys()):
            continue
        
        # else, write the block    
        outstream.write(' {}\n'.format(block))
        for variable in sys_info['namelists'][block]:
            if variable == 'calculation':
                if not (relax_type is None):
                    outstream.write('    calculation = \'{}\'\n'.format(relax_type))
                else:
                    print("No calculation type specified; defaulting to scf")
                    outstream.write('    calculation = \'scf\'\n')
            elif variable == 'nat':
                outstream.write('    nat = {}\n'.format(len(qe_struc)))
            elif variable == 'ntyp':
                outstream.write('    ntyp = {}\n'.format(qe_struc.number_of_elements()))
            else:
                outstream.write('    {} = {}\n'.format(variable, 
                                        sys_info['namelists'][block][variable]))
        outstream.write(' /\n')
        
    # write pseudopotentials
    outstream.write(" ATOMIC_SPECIES\n")
    for psp in sys_info['cards']['ATOMIC_SPECIES']:
        outstream.write(psp + '\n')
        
    # write atomic coordinates
    outstream.write(' ATOMIC_POSITIONS { crystal }\n')
    qe_struc.write(outstream, defected=defected, add_constraints=add_constraints)
    
    # write lattice 
    outstream.write(' CELL_PARAMETERS')
    outstream.write(' {{ {} }}\n'.format(sys_info['cards']['CELL_PARAMETERS']))
    qe_struc.writeLattice(outstream.write)
        
    # write k-point grid
    if not sys_info['cards']['K_POINTS']:
        # use the gamma point
        outstream.write(' K_POINTS { gamma }\n')
    else:
        # use automatically generated Monkhorst-Pack grid
        outstream.write(' K_POINTS { automatic }\n')
        grid = sys_info['cards']['K_POINTS']['spacing']
        outstream.write('  {} {} {}'.format(grid[0], grid[1], grid[2]))
        outstream.write(' {}\n'.format(sys_info['cards']['K_POINTS']['shift']))
        
    outstream.close()
    
    # finally, remove any impurity atoms that have been appended to <qe_struc>
    if not (impurities is None):
        mutate.undo_defect(qe_struc, impurities)
    
    return
