#!/usr/bin/env python
'''Contains utilities required for interfacing with the molecular-mechanics
massively parallel MD code LAMMPS.
'''
from __future__ import print_function

import re
import numpy as np
import sys
sys.path.append('/home/richitensor/programs/pyDis/')

from pyDis.atomic import crystal as cry
from pyDis.atomic import atomistic_utils as atm

class LammpsAtom(cry.Atom):
    '''Same as <cry.Atom>, but with charge.
    '''

    def __init__(self, atomic_symbol, coordinates=np.zeros(3), q=0.0, 
                                                        index=0):
        '''Creates an <Atom> object with additional properties relating to
        the realization of atoms in LAMMPS. 
        '''

        super(LammpsAtom, self).__init__(atomic_symbol, coordinates)
        self._charge = q
        self._index = int(index)

    def write(self, outstream, lattice=cry.Lattice(), defected=True, to_cart=True,
                                                          add_constraints=False):
        '''Writes the atom to <outstream>, including geometric constraints if
        <add_constraints> is true. LAMMPS requires every atom in the input data
        file to have a unique <index>. If an index has not already been assigned
        to the atom, one MUST be given here. If <index> is specified, this value
        supercedes the one given in <self._number>.
        '''

        # test to see if atom should be output
        if not self.writeToOutput():
            return # otherwise write atom to output

        # check that an index has been provided
        if self._index < 1:
                raise AttributeError("Cannot find value of atom index.")

        atom_format = '{} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'

        # write coordinates and/or constraints of atom
        if defected:
            coords = self.getDisplacedCoordinates()
        else:
            coords = self.getCoordinates()

        outstream.write(atom_format.format(self._index, self.getSpecies(), self.q,
                                            coords[0], coords[1], coords[2]))

        if add_constraints:
            # add constraints if non-trivial 
            pass
        else:
            outstream.write('\n')

        return
    
    def set_index(self, i):
        '''Set index to <i>.
        '''

        # index must be int
        self._index = int(i)

    def copy(self):
        '''Creates a copy of the atom.
        '''

        new_atom = LammpsAtom(self.getSpecies(), self.getCoordinates(),
                                                    index=self._index)
        
        # make sure new atom is writable to output
        if self.writeToOutput():
            pass
        else:
            new_atom.switchOutputMode()

        return new_atom

def assign_indices(lmp_struc):
    '''Numbers atoms in a LAMMPS crystal structure.
    '''

    for i, atom in enumerate(lmp_struc):
        atom.set_index(i+1)

    return

def parse_lammps(basename, unit_cell, use_data=False, datafile=None, path='./'):
    '''Parses LAMMPS file (FILES?) contained in <basename>, extracting geometrical
    parameters to <unit_cell> and simulation parameters to <sys_info>.

    vital system info includes: number of atom types, atomic masses
    '''

    # regex to find cell lengths, cell angles, and atoms
    # atom line has the format atom-ID atom-type q (charge) x y z
    lattice_reg = re.compile('^\s*0+\.0+(?:e\+0+)?\s+(?P<x>\d+\.\d+)(?:e\+0+)?' +
                                                '\s+(?P<vec>\w)lo\s+\whi')
    
    atom_reg = re.compile('^\s*\d+\s+(?P<i>\d+)\s+(?P<q>-?\d+\.\d+)' +
                            '(?P<coords>(?:\s+-?\d+\.\d+){3})')

    angle_reg = re.compile('\s*(?P<angles>-?\d+\.?\d*(?:\s+-?\d+\.?\d*){2})\s+xy\s+xz\s+yz')

    
    # check that the user has passed a data.* file if <use_data> is True
    if use_data and datafile == None:
        raise NameError("No file containing simulation data specified.")

    struc_file = atm.read_file(basename, path=path)

    after_lattice = False
    after_atoms = False
    cell_lengths = np.zeros(3)

    for line in struc_file:
        # try to match lattice
        lattmatch = lattice_reg.match(line)
        if lattmatch:
            if lattmatch.group('vec') == 'x':
                index = 0
            elif lattmatch.group('vec') == 'y':
                index = 1
            else: # z
                index = 2

            cell_lengths[index] = float(lattmatch.group('x'))
        else:
            # look for atoms
            atommatch = atom_reg.match(line)
            if atommatch:
                # parse coordinates
                coords = np.array([float(x) for atommatch.group('coords').split()])
                new_atom = LammpsAtom(atommatch.group('i'), coords, 
                                    q=float(atommatch.group('q')))
                
                unit_cell.addAtom(new_atom)
            else:
                # look for angles
                angmatch = angle_reg.match(line)

    sys_info = None
    return sys_info

def write_lammps(outstream, lmp_struc, sys_info, atom_ids, defected=True, 
                   to_cart=False, add_constraints=False, impurities=None,  
                          relax_type=None, use_data=False, datafile=None):
    '''Writes structure contained in <lmp_struc> to <outstream>. If <use_data>
    is True, write the atomic coordinates to a separate data.* file
    '''
    
    if use_data:
        # check that a data.* filename has been provided
        if datafile == None:
            raise NameError("No data.* file defined.")
            
    datastream = open(datafile, 'w')
            
    for i, atom in enumerate(lmp_struc):

    pass
