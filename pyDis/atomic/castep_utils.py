#!/usr/bin/env python
from __future__ import print_function

import re
import numpy as np
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic import crystal as cry
from pyDis.utilities import atomistic_utils as util
from pyDis.atomic import transmutation as mutate

class CastepBasis(cry.Basis):
    '''Subclass of <cry.Basis> with additional functionality to accommodate 
    CASTEP's absolutely nonsensical lack simple syntax for constraining atomic
    coordinates during relaxation.
    '''

    def __init__(self):
        # set up simulation cell
        super(CastepBasis,self).__init__()
        # add dictionary holding number of each atomic species present
        self.atom_dict = dict()

    def addAtom(self,new_atom):
        '''Adds a <CastepAtom> to the <CastepBasis>, assigning it both an atom
        index (ie. how many atoms come before it + 1) and a species index (ie.
        how many atoms of its TYPE come before it + 1).
        '''

        # add the atom as usual
        super(CastepBasis,self).addAtom(new_atom)

        # now update <atom_dict> to reflect the insertion of this atom, and work
        # out its species index.
        if new_atom.getSpecies() in self.atom_dict:
            self.atom_dict[new_atom.getSpecies()] += 1
        else:
            self.atom_dict[new_atom.getSpecies()] = 1
        
        self[-1].index = self.atom_dict[new_atom.getSpecies()]

    def write_constraints(self, write_fn):
        '''Write constraints to <write_fn> (usually a file I/O stream) in the 
        CASTEP format, listing the number of the atom in the list of all atoms
        contained in the basis (<n_atom>), as well as which number of atom of 
        this particular species we are writing to output.
        '''

        if type(write_fn) is file:
            write_fn = write_fn.write
            end = '\n'
        else:
            end = ''

        # test to see if any atom actually has constraints
        use_constraints = False
        for atom in self:
            if atom.has_constraints:
                use_constraints = True
                self._currentindex = 0
                break

        if not use_constraints:
            # no need to write an ionic constraints block
            return

        # otherwise, write the ionic constraints block to <write_fn>
        write_fn('%BLOCK IONIC_CONSTRAINTS' + end)
        n_constraints = 0
        for atom in self:
            # test for constraints
            for i,const in enumerate(atom.get_constraints()):
                if int(const) == 0: # fix atom in place
                    # note that, in castep, 1 implies that motion is constrained
                    fix = cry.ei(i+1)
                    n_constraints += 1
                    write_fn('%d %s %d %.2f %.2f %.2f%s' % (n_constraints,
                         atom.getSpecies(),atom.index,fix[0],fix[1],fix[2],end))

        write_fn("%ENDBLOCK IONIC_CONSTRAINTS" + end + end)
   
        return

class CastepCrystal(cry.Lattice,CastepBasis):
    '''New <Crystal> class to give access to features in <CastepBasis>. Identical
    to <cry.Crystal> in all other respects.
    '''

    def __init__(self,a=cry.ei(1),b=cry.ei(2),c=cry.ei(3)):
        '''Identical to <__init__> for <cry.Crystal>, but with the <__init__>
        call for the basis done using class <CastepBasis>.
        '''

        cry.Lattice.__init__(self,a,b,c)
        CastepBasis.__init__(self)

def parse_castep(basename, unit_cell, path='./'):
    '''Parses the CASTEP .cell and .param files with root name <basename> and
    extracts the atoms and lattice vectors to <unit_cell> (which should be a 
    <CastepCrystal> object). Returns a list containing essential, system-specific
    input information (pseudopotentials, etc.)
    '''

    # check if the user has provided a .cell file or a root name
    if basename.endswith('.cell'):
        # need to extract root so that .param file can be parsed
        cell_name = basename
        root_name = re.compile('(?P<root>.+)\.cell')
        param_name = root_name.match(basename).group('root') + '.param'
    else:
        cell_name = basename + '.cell'
        param_name = basename + '.param'

    # regular expressions to find blocks containing (in order) atoms, cell
    # parameters, and pseudopotentials.
    atoms_block = re.compile("%BLOCK\s+POSITIONS_[A-Za-z]+\s*\n\s*" +
                           "(?:\s*[A-Z][a-z]?\d*(?:\s+-?\d+\.\d+){3}\s*\n)+"
                             "\s*%ENDBLOCK\s+POSITIONS_[A-Za-z]+",re.IGNORECASE)   
    lattice_block = re.compile("%BLOCK LATTICE\w+\s*\n(?:(?:\s*-?\d+\.\d+){3}" +
                                          "\s*\n){3}\s*%ENDBLOCK",re.IGNORECASE)
    psp_block = re.compile('%BLOCK\s+SPECIES_POT.+%ENDBLOCK\s+SPECIES_POT',
                                                  re.DOTALL | re.IGNORECASE)
    kgrid_key = re.compile('KPOINTS?_MP_GRID(?:\s+\d){3}',re.IGNORECASE)

    # import .cell input file and extract atoms and cell vectors
    cas_lines = util.read_file(cell_name, path=path, return_str=True)
    atoms = atoms_block.search(cas_lines)
    vecs = lattice_block.search(cas_lines)

    # extract import system information (currently just psps and k-point grid).
    # Unlike with GULP, where the system information in simple, independent of
    # the geometry, and occurs only at the end of an input file, we use a dict
    # to store system information (in particular, so that the k-grid can be 
    # manipulated later.
    sys_info = dict()
    psps = psp_block.search(cas_lines)
    if not psps: # if no pseudopotentials are given, CASTEP will generate its own
        sys_info['psps'] = None
    else:
        sys_info['psps'] = psps.group()

    # extract Monkhorst Pack k-point grid
    kgrid = kgrid_key.search(cas_lines)
    if not kgrid:
        raise RuntimeError("No k-point grid found.")
    else:
        grid_parse = re.compile('(?P<preamble>.+)(?P<points>(?:\s+\d+){3})',    
                                                                   re.DOTALL)
        grid = grid_parse.search(kgrid.group())
        kgrid = {'preamble': grid.group('preamble'),'spacing': np.array([
                              int(i) for i in grid.group('points').split()])}
        sys_info['mp_kgrid'] = kgrid

    # finally, read in the .param file
    with open('%s%s' % (path,param_name)) as f:
        sys_info['param'] = f.read()

    # regular expressions to find a line containing a single atom, as well as
    # a line containing a single unit cell vector.
    atom_line = re.compile('(?P<sym>[A-Z][a-z]?\d*)(?P<coords>(?:\s+-?\d+\.\d+){3})',
                                                                      re.IGNORECASE)
    vector_line = re.compile('(?:-?\d+\.\d+.*)',re.IGNORECASE)

    # extract cell parameters and atoms in the associated %BLOCKS found earlier
    # to <unit_cell>
    cell_index = 0
    for cell_param in vector_line.finditer(vecs.group()):
        # convert cell_param to vector
        cell_param = cell_param.group().split()
        new_vec = np.array([float(x) for x in cell_param])
        unit_cell.setVector(new_vec,cell_index)
        cell_index += 1

    for atom in atom_line.finditer(atoms.group()):
        coords = atom.group('coords').split()
        coords = np.array([float(x) for x in coords])
        unit_cell.addAtom(cry.Atom(atom.group('sym'),coords))

    return sys_info

def write_castep(outstream, cas_struc, sys_info, defected=True, to_cart=False,
         add_constraints=False, relax_type=None, impurities=None, do_relax=None, prop=None):
    '''Writes the information in <cas_struc> and <sys_info> to the specified 
    output stream. <prop> is a dummy variable to make input consistent with 
    <write_gulp>.
    
    #!!! Do we need some facility to change the calculation type (eg. from 
    #!!! fixed-cell relaxation to a SCF calculation) without the user having
    #!!! to edit the .param (.cell?) file directly?
    '''
    
    # insert defect(s), if supplied
    if not (impurities is None):
        if mutate.is_single(impurities):
            mutate.cell_defect(cas_struc, impurities, use_displaced=True)
        elif mutate.is_coupled(impurities):
            mutate.cell_defect_cluster(cas_struc, impurities, use_displaced=True)
        else:
            raise TypeError("Supplied defect not of type <Impurity>/<CoupledImpurity>")

    # begin by writing the cell block
    outstream.write('%BLOCK LATTICE_CART\n')
    cas_struc.writeLattice(outstream.write)
    outstream.write('%ENDBLOCK LATTICE_CART\n')
    outstream.write('FIX_ALL_CELL true\n\n')

    # write the atom block to file
    outstream.write('%BLOCK POSITIONS_FRAC\n')
    # can this be done using <cas_struc.write(outstream, defected=defected)> ?
    for atom in cas_struc:
        atom.write(outstream.write, defected=defected)
        
    outstream.write('%ENDBLOCK POSITIONS_FRAC\n\n')

    # write constraints
    if add_constraints:
        cas_struc.write_constraints(outstream)

    # write pseupotential filenames and Monkhorst-Pack k-point grid
    if sys_info['psps']:
        outstream.write('%s\n\n' % sys_info['psps'])

    util.write_kgrid(outstream.write, sys_info['mp_kgrid'])

    outstream.close()

    # now we write the .param file
    # first, extract the root name from <outstream>
    root_name = re.compile('(?P<root>.+)\.cell')
    param_name = root_name.match(outstream.name).group('root') + '.param'
    
    # write simulation parameters to the .param file
    param_file = open(param_name, 'w')
    param_file.write(sys_info['param'])
    param_file.close()
    
    # remove any defects that have been inserted
    if not (impurities is None):
        mutate.undo_defect(cas_struc, impurities)
        
    return
