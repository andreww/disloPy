#!/usr/bin/env python
'''Contains utility functions to extract converged cell parameters and internal
coordinates from a PWSCF output file and update the corresponding PWscf input
file.

Currently needs to be rewritten depending on whether you use <celldm> or 
<CELL_PARAMETERS> to give the lattice parameters.
'''
from __future__ import print_function

import re
import numpy as np
import sys

# regular expressions to match important parts of qe input file
namelist = re.compile(r'(?P<nmlst>&.*?\n\s*/\s*)\n', re.DOTALL)
psps = re.compile('ATOMIC_SPECIES\s*\n(?:\s*[A-Z][a-z]?\s+\d+\.\d+\s+.+UPF\s*\n)+')
kpoints = re.compile('K_POINTS.+?\n\s*\d+(?:\s+\d+){2,5}', re.DOTALL)

# regular expressions to match important parts of qe output file
cell_block = re.compile('CELL_PARAMETERS\s*?(?:\(|{)?(?P<units>[^\(\{]*?)(?:\)|})?\s*?\n' +
                            '(?P<pars>(?:(?:\s*-?\d+\.\d+)(?:\s+-?\d+\.\d+){2}\s*?\n){3})')
atoms_block = re.compile('(?P<block>ATOMIC_POSITIONS\s+\(\w+\)\s*\n' +
                  '(?:[A-Z][a-z]?(?:\s+-?\d+\.\d+){3}(?:(?:\s+\d){3})?\s*?\n)+)')

def write_system(sys_namelist, outstream):
    ''' Modify the &system namelist to make sure that celldm(1) = 0.0, all other
    celldm(i) are absent and ibrav = 0. This ensures that we can use the
    CELL_PARAMETERS card to set the cell shape, allowing us to treat all pairs
    of input/output files on the same footing.
    '''
    
    outstream.write('&system\n')
    outstream.write('   ibrav = 0,\n')
    outstream.write('   celldm(1) = 0.0,\n'),    
    for match in re.finditer(r'(?P<item>[^&\n]+?)(?:,|\n|\n\s*/\s*\n)', 
                                              sys_namelist, re.DOTALL):
        if ('celldm' in match.group('item')  or 'ibrav' in match.group('item')
                                            or 'system' in match.group('item')):
            pass
        else:
            outstream.write('   {},\n'.format(match.group('item').strip()))
    outstream.write(' /\n')
    
    return
    
def format_cell(cell):
    '''Modifies the cell block in a qe output file to conform with the format 
    used in the qe input files.
    '''
    
    if 'alat' in cell.group('units'):
        # extract basic length scale
        alat = float(re.search(r'\d+\.\d+', cell.group('units')).group())
        
        # extract and scale vectors
        new_cell = cell.group('pars').split('\n')
        cell = ' CELL_PARAMETERS { bohr }\n'
        for vector in new_cell:
            if not vector: # blank line
                continue
            else:
                # scale components of lattice parameter
                temp = [alat*float(x) for x in vector.split()]
                cell += '  {:.8f} {:.8f} {:.8f}\n'.format(temp[0], temp[1], temp[2]) 
    else:
        cell = re.sub('\(', '{', re.sub('\)', '}', cell.group()))
    
    return cell

def main(argv):
    '''Update QE input file with results from corresponding output file.
    '''
    
    # extract all lines from qe input and output files
    qe_input = open(argv[0]).read()
    qe_output = open(argv[1]).read()
    
    # reopen qe input file, but as an output stream
    outstream = open(argv[0], 'w')
    
    # test to see if the calculation is a variable-cell relaxation calculation
    if 'vc-relax' in qe_input:
        is_vc = True
    else:
        is_vc = False

    # extract namelist and pseudopotentials and write to new input. 
    for match in namelist.finditer(qe_input):
        namelist_block = match.group('nmlst')
        if ('&system' in namelist_block) and is_vc:
            # cell parameters need to be extracted from output. This is most 
            # easily done if we set ibrav = 0 and write the CELL_PARAMETERS card
            # out explicitly
            write_system(namelist_block, outstream)
        else:
            outstream.write(namelist_block+'\n')

    outstream.write(psps.search(qe_input).group())
    
    # extract optimized atomic positions and cell parameters from qe output
    # file, starting with atoms block
    atoms_opt = atoms_block.findall(qe_output)[-1]
    atoms_opt = re.sub('\(', '{', re.sub('\)', '}', atoms_opt))
    
    outstream.write(atoms_opt)
    
    # extract the cell parameters. As the calculation may be done with
    # either a fixed or variable cell size, may need to read cell parameters from
    # either the input or output file.
    
    if is_vc:
        # extract cell parameters from output file and format for qe input
        cell = None
        for cell in cell_block.finditer(qe_output):
            pass
        cell = format_cell(cell)
        outstream.write(cell)
    else:
        # extract cell parameters from input file. This may not be necessary
        # if the cell dimensions have already been specified in the &system 
        # namelist.
        cell = cell_block.search(qe_input)
        cell = format_cell(cell)
        if cell:
            outstream.write(cell)
        else:
            pass
    
    # finally, write k-point grid
    outstream.write(kpoints.search(qe_input).group())
    outstream.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])
