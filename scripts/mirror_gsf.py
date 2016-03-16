#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import pyDis.pn.fit_gsf

def main(file_in, file_out):
    gsf, units = pyDis.pn.fit_gsf.read_numerical_gsf(file_in)

    # Now turn into 3D array
    gsf_mesh = np.zeros([9,9,3])
    point = 0
    for i in range(np.shape(gsf_mesh)[0]):
        for j in range(np.shape(gsf_mesh)[1]):
            gsf_mesh[i,j,0] = gsf[point, 0]
            gsf_mesh[i,j,1] = gsf[point, 1]
            gsf_mesh[i,j,2] = gsf[point, 2]
            point = point + 1

    # Do the mirror op
    gsf_mesh = pyDis.pn.fit_gsf.mirror2d(gsf_mesh)

    # return to 2D array
    #gsf = np.zeros([17*17,3])
    #point = 0
    #for i in range(np.shape(gsf_mesh)[0]):
    #    for j in range(np.shape(gsf_mesh)[1]):
    #        gsf[point, 0] = gsf_mesh[i, j, 0]
    #        gsf[point, 1] = gsf_mesh[i, j, 1]
    #        gsf[point, 2] = gsf_mesh[i, j, 2]
    #        point = point + 1
    #
    #np.savetxt(file_out, gsf)

    # Write out the data
    with open(file_out, 'w') as outstream:
        outstream.write("# %s\n" % (units))
        for i in range(np.shape(gsf_mesh)[0]):
            for j in range(np.shape(gsf_mesh)[1]):
                outstream.write("%d %d %.4f\n" % (i,j,gsf_mesh[i,j,2]))
            # include a space between slices of constant x (for gnuplot)
            outstream.write("\n")

if __name__ == "__main__":
    import sys
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    main(file_in, file_out)


