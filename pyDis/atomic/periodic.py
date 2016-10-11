#!/usr/bin/env python
'''Used to subtract the contribution of interactions between dislocations in the 
simulation cell and their periodic images from the total cell energy. This is
done using the method outlined in Cai et al. (2003).
'''
from __future__ import print_function

import numpy as np
import sys
import os
sys.path.append(os.environ['PYDISPATH'])

from pyDis.atomic.aniso import readCij

def 

