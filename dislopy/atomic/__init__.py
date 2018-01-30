#!/usr/bin/env python
''''
atomic
======
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
======
<atomic> is a subpackage that contains modules to facilitate setting up, 
running, and analysing atomistic simulations of dislocations in crystals. At 
present, the two methods implemented are:

    1. Cluster-based modelling, in which a single dislocation is embedded in a
    1D periodic cylinder of atoms containing a fixed outer region.
    2. Multipole simulations, with multiple dislocations (whose Burgers vectors
    sum to zero) embedded in a 3D periodic supercell. Unlike cluster-based
    modelling this can use DFT to determine the cell energy.
    
At present, supported atomistic codes are:
GULP
CASTEP
QE

Partial support is provided for:
LAMMPS
'''
