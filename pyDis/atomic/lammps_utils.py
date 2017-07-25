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

from pyDis.atomic import crystal as cry
from pyDis.utilities import atomistic_utils as atm
from pyDis.atomic import transmutation as mutate

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
        
        if not (q is None):
            # using charges -> need to include q
            atom_format = '{} {} {:.6f} {:.6f} {:.6f} {:.6f}'
        else:
            atom_format = '{} {} {:.6f} {:.6f} {:.6f}'

        # write coordinates and/or constraints of atom
        if defected:
            coords = self.getDisplacedCoordinates()
        else:
            coords = self.getCoordinates()

        if not (q is None):
            outstream.write(atom_format.format(self._index, self.getSpecies(), 
                                   self.q, coords[0]*scalex, coords[1]*scaley, 
                                                              coords[2]*scalez))
        else:
            outstream.write(atom_format.format(self._index, self.getSpecies(),
                          coords[0]*scalex, coords[1]*scaley, coords[2]*scalez))

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

def parse_lammps(basename, unit_cell, path='./'):
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

    # regex to find the projection of the b and c lattice vectors onto the x (b)
    # and x and y (c) axes, because LAMMPS is weird. What's wrong with a 3x3 array?
    tilt_reg = re.compile('\s*(?P<proj>-?\d+\.?\d*(?:e\+\d+)?(?:\s+-?\d+\.?\d*'+
                                '(?:e\+\d+)?){2})\s+xy\s+xz\s+yz')

    
    # check that the user has passed a data.* file if <use_data> is True
    if use_data and datafile is None:
        raise NameError("No file containing simulation data specified.")

    struc_file = atm.read_file(basename, path=path)

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
                coords = np.array([float(x) for x in atommatch.group('coords').split()])
                new_atom = LammpsAtom(atommatch.group('i'), coords, 
                                    q=float(atommatch.group('q')))
                
                unit_cell.addAtom(new_atom)
            else:
                # look for skews
                tiltmatch = tilt_reg.match(line)
                if tiltmatch:
                    projections = [float(x) for x in tiltmatch.group('proj').split()]

    # construct lattice vectors and set latt vecs of <unit_cell> 
    x = np.array([cell_lengths[0], 0., 0.])
    y = np.array([projections[0], cell_lengths[1], 0.])
    z = np.array([projections[1], projections[2], cell_lengths[2]])

    unit_cell.setVector(x, 0)
    unit_cell.setVector(y, 1)
    unit_cell.setVector(z, 2)

    sys_info = None
    return sys_info

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
        elif mutate.is_coupled(impurities);
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
                                                lattice[2, 0], lattice[2, 1])
    
    # write atoms to file
    outstream.write('Atoms\n')
    for i, atom in enumerate(struc):
        atom.set_index(i)
        atom.write(outstream, lattice=lattice, defected=defected, to_cart=to_cart)
        
    # atomic masses
    outstream.write('\nMasses\n\n')
    for species in sys_info['masses']:
        outstream.write('{} {:.2f}\n'.format(species[0], species[1]))

    pass
    
def run_lammps(lammps_exec, basename, nproc=1, para_exec='mpiexec', set_omp=False,
                                                                  omp_threads=1):
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
    lin = open('in.{}'.format(basename), 'r')
    lout = open('out.{}'.format(basename), 'w')
    
    # run lammps    
    if use_para:
        subprocess.call([para_exec, '-np', str(nproc), lammps_exec], stdin=lin,
                                                                    stdout=lout)
    else:
        subprocess.call(lammps_exec, stdin=lin, stdout=lout)
        
    lin.close()
    lout.close()
