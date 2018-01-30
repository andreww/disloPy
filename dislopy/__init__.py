#!/usr/bin/env python
'''
PyDis
=====
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
=====

PyDis is a suite of modules designed to facilitate the atomic-scale simulation
of dislocations in crystalline materials. Three distinct methods are available:

1. Cluster-based simulations
2. Multipole simulation
3. The Peierls-Nabarro (PN) method

The first two of these methods, both of which are implemented in modules 
contained in the subpackage <atomic>, allow the user to carry out fully atomistic
simulations of dislocation core structures and energies. The PN approach, by
contrast, is a sequential multiscale method, in which inelastic interactions in
a classical model are parameterised using the output of atomistic calculations of
generalised stacking fault energies. The modules required to set up and run PN
calculations are contained in the subpackage <pn>.

In addition, we have provided facilities for modelling point defect-dislocation
interactions.

subpackages:
pn
    Peierls-Nabarro modelling of planar dislocations
atomic
    Fully atomistic modelling of dislocation core structures using the multipole
    or cluster-based approaches
'''
