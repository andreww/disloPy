#!/usr/bin/env python

_SCRIPT = """#PBS -N castep
#PBS -l select=5
#PBS -l walltime=09:00:00
#PBS -A n03-walk
#PBS -j oe
#PBS -m n

# Go to the submit dir - and set environment vars for
# fortran temp files, OMP etc.
cd $(readlink -f $PBS_O_WORKDIR)
export TMPDIR=$(readlink -f $PBS_O_WORKDIR)
export GFORTRAN_TMPDIR=$(readlink -f $PBS_O_WORKDIR)
export OMP_NUM_THREADS=1

# Load castep module
export CASTEP=/usr/local/packages/castep/8.0.0-phase1/bin/castep.mpi

# Run with 120 MPI processes (n), 24 per 24 core node (N),
# split 12 per NUMA region (S), one thread per MPI process (d)
# and one hyperthread per core (j). This is 5 nodes
echo "Starting castep"
aprun -n 120 -N 24 -S 12 -d 1 -j 1 $CASTEP {basename}
echo "Castep done"
"""

def create_runcastep(basename, filename='runcastep.sh'):
    
    f = open(filename, 'w')
    f.write(_SCRIPT.format(basename=basename))
    f.close()

if __name__ == "__main__":
    import sys
    import os

    for dir in sys.argv[1:]:
        file = os.path.join(dir, 'runcastep.sh')
        create_runcastep(dir, file)

