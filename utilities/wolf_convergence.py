#!/usr/bin/env python

import sys
sys.path.append('/home/richard/code_bases/')

from pyDis.atomic import atomistic_utils as util
from pyDis.atomic import wolf

def main(argv):

    base_lines = util.read_file(argv[0])
    base_name = argv[1]
    
    gulp_exec = '/home/richard/programs/atomistic/gulp/Src/./gulp'
    wolf.gulp_wolf(base_name, base_lines, gulp_exec)
    
if __name__ == "__main__":
    main(sys.argv[1:])
