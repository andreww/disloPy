#!/usr/bin/env python
'''A bunch of functions to convert a gulp .grs file into a regular .gin file
'''
from __future__ import print_function, division

import sys
import re
import numpy as np

from mix import read_file
                       
def cell2cart(a, b, c, alpha, beta, gamma):
    if abs(alpha - 90.0) < 1e-12:
        sina = 1.0
        cosa = 0.0
    else:
        sina = np.sin(np.radians(alpha))
        cosa = np.cos(np.radians(alpha))
    if abs(beta-90.0) < 1e-12:
        sinb = 1.0
        cosb = 0.0
    else:
        sinb = np.sin(np.radians(beta))
        cosb = np.cos(np.radians(beta))
    if abs(gamma-90.0) < 1e-12:
        sing = 1.0
        cosg = 0.0
    else:
        sing = np.sin(np.radians(gamma))
        cosg = np.cos(np.radians(gamma))
        
    c_x = 0.0
    c_y = 0.0
    c_z = c
        
    b_x = 0.0
    b_y = b*sina
    b_z = b*cosa
        
    a_z = a*cosb
    a_y = a*(cosg - cosa*cosb)/sina
    trm1 = a_y/a
    a_x = a*np.sqrt(1.0 - cosb**2 - trm1**2)
        
    return np.array([[a_x, a_y, a_z],    
                     [b_x, b_y, b_z],
                     [c_x, c_y, c_z]])
                     
def makegin(outstream, grs_lines, needs_restart=False):
    '''Given the content of a GULP .grs restart file, produces a .gin file (with
    the lattice given in cartesian coordinates) that can be read by disloPy. If 
    <needs_restart> is True, then the 
    '''
    
    atom_line = re.compile('^\s*(?P<lab>[A-Z][a-z]?\d*\s+\w+)(?P<coords>' +
                        '(?:\s+(-?\d+\.\d+|\d+/\d+)){3})(?:(?:\s+-?\d+\.\d+){3})?')
    
    skip = False
    for i, line in enumerate(grs_lines):
        if line.startswith('#'):
            continue
        elif 'totalenergy' in line:
            continue
        elif ('dump' in line) or ('output' in line):
            continue
        elif skip:
            skip = False
            continue
        #else
        if 'frac' in line.split()[0] or 'pfrac' in line.split()[0]:
            outstream.write('fractional\n')
            continue
        elif 'cell' in line:
            outstream.write('vectors\n')
            par = grs_lines[i+1]
            par = [float(x) for x in par.split()]
            vec = cell2cart(*par)
            for i in range(3):
                for j in range(3):
                    outstream.write(' {:.6f}'.format(vec[i, j]))
                outstream.write('\n')
            skip = True
            continue
        found_atom = atom_line.search(line)
        if found_atom:
            coords = [eval('float({})'.format(x)) for x in found_atom.group('coords').split()]
            outstream.write('{} {:.6f} {:.6f} {:.6f}\n'.format(found_atom.group('lab'), 
                                                       coords[0], coords[1], coords[2]))
        else:
            outstream.write('{}\n'.format(line))
            
    if needs_restart:
        outstream.write('dump {}.grs'.format(re.search(r'(\w+)\.gin', 
                                                outstream.name).group(1)))
            
    outstream.close()

def main(argv):
    '''Routine to convert GULP .grs file into a .gin file.
    '''
    
    grslines = read_file(argv[0])
    ginfile = open('{}.opt.gin'.format(re.match(r'([/\w\.\d]+)\.grs', 
                                              argv[0]).group(1)), 'w')
    
    if len(argv) > 1:
        if argv[1].lower() == 'True' or argv[1] == '1':
            makegin(ginfile, grslines, needs_restart=True)
        else:
            makegin(ginfile, grslines)
    else:
        makegin(ginfile, grslines)
    
    print("Converted {} to .gin format.".format(argv[0]))

if __name__ == "__main__":
    main(sys.argv[1:])
