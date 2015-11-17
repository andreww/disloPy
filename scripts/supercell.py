#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append('/home/richard/code_bases/dislocator2/')
import numpy as np

from pyDis.atomic import qe_utils as qe
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import atomistic_utils as util
from pyDis.atomic import crystal as cry

def main(argv):
    '''Make supercell. This is just a utility script for personal use, so we'll
    hard code all of the parameters.
    
    CURRENT_VERSION: GULP
    '''
    
    basename = argv[0]
    outname = argv[1]
    
    dimensions = np.array([float(argv[2]), float(argv[3]), float(argv[4])])
    
    base_struc = cry.Crystal()
    sys_info = gulp.parse_gulp(basename, base_struc)
    
    supercell = cry.superConstructor(base_struc, dimensions)
    
    outstream = open(outname, 'w')
    gulp.write_gulp(outstream, supercell, sys_info, defected=False, relax=False)

if __name__ == "__main__":
    main(sys.argv[1:])
