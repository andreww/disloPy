#!/usr/bin/env python
'''Contains utilities required for interfacing with the molecular-mechanics
massively parallel MD code LAMMPS.
'''
from __future__ import print_function, absolute_import

import re
import numpy as np
import subprocess
import os

from numpy.linalg import norm

from dislopy.atomic import crystal as cry
from dislopy.utilities import atomistic_utils as atm
from dislopy.atomic import transmutation as mutate

class LammpsAtom(cry.Atom):
    '''Same as <cry.Atom>, but with charge.
    '''

    def __init__(self, atomic_symbol, coordinates=np.zeros(3), q=None, index=0):
        '''Creates an <Atom> object with additional properties relating to
        the realization of atoms in LAMMPS. Note that, if the specified atom_style
        is <atomic>, then <charge> will not be printed
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

        # scales for the x, y, and z components of the atomic coordinates
        scalex = norm(lattice[0])
        scaley = norm(lattice[1])
        scalez = norm(lattice[2])

        # test to see if atom should be output
        if not self.writeToOutput():
            return # otherwise write atom to output

        # check that an index has been provided
        if self._index < 1:
                raise AttributeError("Cannot find value of atom index.")

        # write coordinates and/or constraints of atom
        if defected:
            coords = self.getDisplacedCoordinates()
        else:
            coords = self.getCoordinates()

        if not (self._charge is None):
            # using charges -> need to include q in atom line
            atom_format = '{} {} {:.6f} {:.6f} {:.6f} {:.6f}'
            outstream.write(atom_format.format(self._index, 
                                               self.getSpecies(), 
                                               self._charge, 
                                               coords[0]*scalex, 
                                               coords[1]*scaley, 
                                               coords[2]*scalez))
        else:
            atom_format = '{} {} {:.6f} {:.6f} {:.6f}'
            outstream.write(atom_format.format(self._index,
                                               self.getSpecies(),
                                               coords[0]*scalex, 
                                               coords[1]*scaley, 
                                               coords[2]*scalez))

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
                                      index=self._index, q=self._charge)

        new_atom.set_constraints(self.get_constraints())

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

def parse_lammps(datafile, unit_cell, input_script, path='./'):
    '''Parses LAMMPS file (FILES?) contained in <basename>, extracting geometrical
    parameters to <unit_cell> and simulation parameters to <sys_info>.

    vital system info includes: number of atom types, atomic masses
    '''

    # record atomic masses and the contents of the input script as system info
    sysinfo = dict()
    sysinfo['masses'] = []
    input_script_file = open(input_script, 'r')
    input_script_lines = input_script_file.read()
    input_script_file.close()
    sysinfo['input_script'] = input_script_lines

    # regex to find cell lengths, cell angles, masses, and atomic coordinates
    # atom line has the format atom-ID atom-type q (charge) x y z
    anynum = '-?\d+\.?\d*(?:e(?:\+|-)\d+)?'

    # determine which style is used for the Atoms input
    style_reg = re.search('atom_style\s+(?P<atom_style>\w+)', sysinfo['input_script'])
    atom_style = style_reg.group('atom_style')
    if atom_style == 'charge':
        atom_line = '^\s*\d+\s+(?P<i>\d+)\s+(?P<q>{})(?P<coords>(?:\s+{}){{3}})'.format(anynum, anynum)
    elif atom_style == 'atomic':
        atom_line = '^\s*\d+\s+(?P<i>\d+)(?P<coords>(?:\s+{}){{3}})'.format(anynum)
    else:
        raise ValueError("Atom input style {} not currently supported.".format(atom_style))
    
    tilt_line = '^\s*(?P<proj>{}(?:\s+{}){{2}})\s+xy\s+xz\s+yz'.format(anynum, anynum)
    lattice_line = '^\s*0+\.0+(?:e\+0+)?\s+(?P<x>{})\s+(?P<vec>\w)lo\s+\whi'.format(anynum)
    mass_line = '^\s*(?P<i>\d+)\s+(?P<m>{})\s*'.format(anynum)

    lattice_reg = re.compile(lattice_line)
    atom_reg = re.compile(atom_line)  
    tilt_reg = re.compile(tilt_line)
    mass_reg = re.compile(mass_line)

    # check that the user has passed a data.* file if <use_data> is True
    #if use_data and datafile is None:
    #    raise NameError("No file containing simulation data specified.")

    struc_file = atm.read_file(datafile, path=path)
    cell_lengths = np.zeros(3) 

    in_atoms = False
    in_masses = False 
    for line in struc_file:
        # try to match lattice
        lattmatch = lattice_reg.search(line)
        # look for skews
        tiltmatch = tilt_reg.search(line)
        if lattmatch:
            if lattmatch.group('vec') == 'x':
                index = 0
            elif lattmatch.group('vec') == 'y':
                index = 1
            else: # z
                index = 2

            cell_lengths[index] = float(lattmatch.group('x'))
            continue
        if tiltmatch:
            projections = [float(x) for x in tiltmatch.group('proj').split()]
            continue
        if line.strip().split()[0].lower() == 'atoms':
            in_atoms = True
            continue
        if in_atoms:
            # look for atoms
            atommatch = atom_reg.search(line)
            if atommatch:
                # parse coordinates and add atom to <unit_cell>
                coords = np.array([float(x) for x in atommatch.group('coords').split()])
                if atom_style == 'charge':
                    new_atom = LammpsAtom(atommatch.group('i'), coords, q=float(atommatch.group('q')))
                elif atom_style == 'atomic':
                    new_atom = LammpsAtom(atommatch.group('i'), coords)

                unit_cell.addAtom(new_atom)
                continue
            elif (not line.strip()) or line.strip().startswith('#'):
                # empty line or comment -> may still be in the Atoms section
                continue
            else:
                in_atoms = False
                # still want to check some other stuff
        if line.strip().split()[0].lower() == 'masses':
            in_masses = True
            continue
        if in_masses:
            mass_match = mass_reg.search(line)
            if mass_match:
                sysinfo['masses'].append([int(mass_match.group('i')), 
                                         float(mass_match.group('m'))])
                continue
            elif (not line.strip()) or line.strip().startswith('#'):
                # as above
                continue
            else:
                in_masses = False
                pass
        
    # construct lattice vectors and set latt vecs of <unit_cell> 
    x = np.array([cell_lengths[0], 0., 0.])
    unit_cell.setVector(x, 0)
    y = np.array([projections[0], cell_lengths[1], 0.])
    unit_cell.setVector(y, 1)
    z = np.array([projections[1], projections[2], cell_lengths[2]])
    unit_cell.setVector(z, 2)

    # scale the atomic coordinates (this is right...maybe?)
    for atom in unit_cell:
        atom.to_cell(unit_cell.getLattice())

    return sysinfo

def write_lammps(outstream, struc, sys_info, defected=True, do_relax=True, to_cart=True,
                    add_constraints=False, relax_type='conv', impurities=None):
    '''Writes structure contained in <lmp_struc> to <outstream>. Note that,
    because atomic coordinates in LAMMPS are, by default, given in cartesian
    coordinates, the write function here sets the default value of <to_cart>
    to be <True>.

    #!!!At present, only crystals with orthogonal lattice vectors are supported.
    '''


    outstream.write('This is the first line of a LAMMPS file\n\n')

    # insert any impurities
    if not (impurities is None):
        if mutate.is_single(impurities):
            mutate.cell_defect(struc, impurities, use_displaced=defected)
        elif mutate.is_coupled(impurities):
            mutate.cell_defect_cluster(struc, impurities, use_displaced=defected)
        else:
            raise TypeError("Supplied defect not of type <Impurity>/<CoupledImpurity>")   

    # calculate total number of atoms in <lmp_struc> plus any impurities
    outstream.write('  {} atoms\n\n'.format(len(struc)))

    # number of distinct elements
    outstream.write('  {} atom types\n\n'.format(struc.number_of_elements()))

    # simulation cell dimensions 
    lattice = struc.getLattice()
    outstream.write(' 0. {:.6f} xlo xhi\n'.format(lattice[0, 0]))
    outstream.write(' 0. {:.6f} ylo yhi\n'.format(lattice[1, 1]))
    outstream.write(' 0. {:.6f} zlo zhi\n'.format(lattice[2, 2]))
    outstream.write(' {:.3f} {:.3f} {:.3f} xy xz yz\n\n'.format(lattice[1, 0],
                                                lattice[2, 0], lattice[2, 1]))

    # write atoms to file
    outstream.write('Atoms\n\n')
    assign_indices(struc)
    for i, atom in enumerate(struc):
        #atom.set_index(i+1)
        atom.write(outstream, lattice=lattice, defected=defected, to_cart=to_cart)

    # atomic masses
    outstream.write('\nMasses\n\n')
    for species in sys_info['masses']:
        outstream.write('{} {:.2f}\n'.format(species[0], species[1]))

    # change the input script
    read_and_write_data(outstream, sysinfo)

    iscript = open('script.{}'.format(outstream.name), 'w')
    iscript.write(sysinfo['input_script'])
    iscript.close() 

    outstream.close()    
    return

def read_and_write_data(outstream, sysinfo):
    '''Changes the values of the <read_data> and <write_data> variables in 
    the lammps input script to match the datafile containing the dislocation(s),
    <outstream>.
    '''

    read_data = re.search('read_data\s+\S+', sysinfo['input_script']).group()
    write_data = re.search('write_data\s+\S+', sysinfo['input_script']).group()

    sysinfo['input_script'].replace(read_data, 'read_data {}'.format(outstream.name))
    sysinfo['input_script'].replace(write_data, 'write_data new.{}'.format(outstream.name))
    return
    
def run_lammps(lammps_exec, basename, nproc=1, para_exec='mpiexec', 
                                                  set_omp=False, omp_threads=1):
    '''Runs a lammps simulation. Since lammps is a parallel code, there is the 
    option to run it in parallel, with the number of processors given by <nproc>.    
    '''

    # check the number of processors used -> if > 1, use parallel executable
    # (default MPI)
    if nproc < 1:
        raise ValueError("Number of processors must be positive.")
    elif nproc > 1:
        use_para = True
    else:
        use_para = False

    # set number of openMPI threads per process (if > 1)
    if use_para and set_omp and omp_threads >= 1:        
        os.putenv('OMP_NUM_THREADS', str(int(omp_threads)))

    # input and output files
    lin = open('script.{}'.format(basename), 'r')
    lout = open('out.{}'.format(basename), 'w')

    # run lammps    
    if use_para:
        subprocess.call([para_exec, '-np', str(nproc), lammps_exec], stdin=lin,
                                                                    stdout=lout)
    else:
        subprocess.call(lammps_exec, stdin=lin, stdout=lout)

    lin.close()
    lout.close()
    return
