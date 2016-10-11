#!/usr/bin/env python
'''Simple tool to create queue manager files for HPC system with TORQUE job
queuing. Be aware that this script was created with dislocation simulations
in mind (especially gamma line/surface calculations) and can only be used 
unmodified for single jobs and regular 1- and 2-dimensional grids.

The file used as a template for the job submission scripts must have the form

_SCRIPT = """#PBS -N castep
#PBS -l select=5
#PBS -l walltime=09:00:00
#PBS -A wouldntyouliketoknow
#PBS -j oe
#PBS -m n

# Go to the submit dir - and set environment vars for
# fortran temp files, OMP etc.
cd $(readlink -f $PBS_O_WORKDIR)
export TMPDIR=$(readlink -f $PBS_O_WORKDIR)
export GFORTRAN_TMPDIR=$(readlink -f $PBS_O_WORKDIR)
export OMP_NUM_THREADS=1

# Load castep module
export EXECNAME={execname}

# Run with 120 MPI processes (n), 24 per 24 core node (N),
# split 12 per NUMA region (S), one thread per MPI process (d)
# and one hyperthread per core (j). This is 5 nodes
echo "Starting castep"
aprun -n $n $EXECNAME {additional_arguments} {basename}
'''
from __future__ import print_function

import sys
import argparse

def specify_options():

    options = argparse.ArgumentParser()
    options.add_argument('-s', '--script-template', type=str, dest='template',
                             default=None, help='Template for the job script.')
    

def make_script():

    return script
    
def job_grid():
   
    return

def main():
    
    return
    
if __name__ == "__main__":
    main(sys.argv[1:])
