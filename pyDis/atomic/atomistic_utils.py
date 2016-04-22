#!/usr/bin/env python
'''A module to hold miscellaneous functions/classes/etc. that are generally

useful (eg. scaling k-point grids and reading input files), but have no obvious
home in any other module and are not substantial enough to form the basis of

their own modules. If you want to implement a minor helper function, this is the

module to do it in.
'''

from __future__ import print_function, division

import sys
sys.path.append('/home/richard/code_bases/dislocator2/')
from numpy.linalg import norm
import numpy as np
import re

from pyDis.atomic import crystal as cry

# currently supported atomistic simulation codes
supported_codes = ('gulp', 'castep', 'qe')

def read_file(filename,path='./', return_str=False):
    '''Reads a file and prepares it for parsing. Has the option to return the

    output as a single string (with newline characters included), which can be
    useful if the structure of the input file makes regex easy (eg. CASTEP, QE)
    '''

    lines = []
    with open('%s%s' % (path, filename)) as input_file:
        for line in input_file:
            temp = line.rstrip()
            if temp:
                lines.append(temp)

    if return_str:
        all_lines = ''
        # stitch all elements of lines together
        for line in lines:
            all_lines += line + '\n'
        lines = all_lines

    return lines

def ceiling(x):
    '''Returns the smallest integer >= x.
    '''

    if abs(int(x) - x) < 1e-12:
        # if x has integer value (note: NOT type), return x
        return float(int(x))
    else:
        return float(int(x + 1.0))

def scale_kpoints(kgrid, sc_dimensions):
    '''Scales the k-point grid to reflect new supercell dimensions. 
    '''

    new_grid = []
    try:
        use_grid = kgrid['old_spacing']
    except KeyError:
        kgrid['old_spacing'] = kgrid['spacing']
        use_grid = kgrid['spacing']       
        
    for k, dim in zip(use_grid, sc_dimensions):
        new_grid.append(max(int(ceiling(k / dim)), 1))
    
    # store the original grid and 
    kgrid['spacing'] = new_grid
    return

def write_kgrid(write_fn, kgrid):
    '''Writes k-point grid.
    '''

    write_fn('%s ' % kgrid['preamble'])
    for k in kgrid['spacing']:
        write_fn(' %d' % k)
    write_fn('\n')
    return

def isiter(x):
    '''Tests to see if x is an iterable object whose class is NOT 
    <Basis> or any class derived from <Basis> (eg. <Crystal>, 
    <TwoRegionCluster>, etc.).
    '''
    
    if isinstance(x, (cry.Basis, cry.Atom)):
        # NOTE: need to implement __getitem__ for <cry.Atom>
        return False
    # else
    try:
        a = iter(x)
        return True
    except TypeError:
        return False

def write_xyz(input_crystal, filename, defected=False, description='xyz file',
                                                       to_cart=False, r=np.inf):
    '''Writes the atoms in <input_basis> to the specified .xyz file. 
    '''
    

    xyz_lines = ''
    natoms = 0
    
    for atom in input_crystal:
        # check that atom is to be written to output
        if norm(atom.getDisplacedCoordinates()[:-1]) > r:
            continue
        
        #else
        natoms += 1
            
        # write coordinates in deformed crystal if <defected> is True
        if defected:
            x = atom.getDisplacedCoordinates()
        else:
            x = atom.getCoordinates()
            
        # convert to Cartesian coordinates if <to_cart> is True. This is of 
        # particular relevance when the <input_crystal> is 3D periodic, in which
        # case the atomic coordinates are likely expressed in fractional coordinates
        if to_cart:
            x = cry.fracToCart(x, input_crystal.getLattice())
        
        xyz_lines += '{} {:.6f} {:.6f} {:.6f}\n'.format(atom.getSpecies(), x[0],
                                                                     x[1], x[2])
                                                                     
    xyz_file = open(filename, 'w')
    xyz_file.write('{}\n'.format(natoms))
    xyz_file.write('{}\n'.format(description))
    xyz_file.write(xyz_lines)
        
    xyz_file.close()
    
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
    
def extract_energy(cellname, program, relax=True):
    '''Reads in the final energy (or enthalpy) from an atomistic calculation.
    If the calculation involved relaxation of the atomic coordinates or the 
    cell shape/size, <relax> == True, otherwise it should be False. 
    '''
    
    
    
    # regex to match the final energy output from relaxation calculations 
    energy_relax = {"gulp": re.compile(r"\n\s*Final energy\s+=\s+" +
                                  "(?P<E>-?\d+\.\d+)\s*(?P<units>eV)\s*\n"),
                    "castep": re.compile(r"\n\s*BFGS:\s*Final\s+Enthalpy\s+=\s+" +
                              "(?P<E>-?\d+\.\d+E\+\d+)\s*(?P<units>\w+)\s*\n"),
                    "qe": re.compile(r"\n\s*Final (?:energy|enthalpy)\s+=\s+" +
                                  "(?P<E>-?\d+\.\d+)\s+(?P<units>\w+)\s*\n")
                   }
    
    # regex to match energy from a single-point calculation               
    energy_fixed = {'gulp': re.compile(r'\n\s*Total lattice energy\s+=\s+'+
                                    '(?P<E>-?\d+\.\d+)\s*(?P<units>eV)\s*\n'),
                    'castep': re.compile(r'\n\s*Final energy\s+=\s+' +
                                     '(?P<E>-?\d+\.\d+)\s*(?P<units>\w+)\s*\n'),
                    'qe': re.compile(r'\n\s*!\s+total energy\s+=\s+'+
                                    '(?P<E>-?\d+\.\d+)\s*(?P<units>\w+)\s*')
                   }
    
    # stuff to match total force in GULP output
    acceptable_gnorm = 0.2               
    get_gnorm = re.compile(r"Final Gnorm\s*=\s*(?P<gnorm>\d+\.\d+)")
    
    if not (program in supported_codes):
        raise ValueError("{} is not a supported atomistic code.".format(program))
    elif relax:
        energy_regex = energy_relax[program]
    else: # SP calc.
        energy_regex = energy_fixed[program]
        
    # read in the output file from the atomistic code
    outfile = open(cellname)
    output_lines = outfile.read()
    matched_energies = re.findall(energy_regex, output_lines)
    
    E = np.nan
    units = ''
    if not matched_energies:
        print("Warning: No energy block found.")
    else:
        # check that structure is, in fact, relaxed
        if program.lower() == 'gulp':
            # triggers for failed energy convergence
            gulp_flag_failure = ["Conditions for a minimum have not been satisfied",
                                                "Too many failed attempts to optimise"]
            if ((gulp_flag_failure[0] in output_lines) or 
                (gulp_flag_failure[1] in output_lines)):
                gnorm = float(get_gnorm.search(output_lines).group("gnorm"))
            else:
                gnorm = 0.
            
            # check to see if the total force is below reasonable threshold
            if gnorm < acceptable_gnorm:       
                E = float(matched_energies[-1][0])
                units = matched_energies[-1][1]
            else:
                E = np.nan
                units = None

        else: 
            # Other codes - we don't check for convergence, perhaps we should
            # use the last matched energy (ie. optimised structure)
            E = float(matched_energies[-1][0])
            units = matched_energies[-1][1]
        
    return E, units    
