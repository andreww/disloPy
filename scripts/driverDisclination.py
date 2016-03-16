#!/usr/bin/env python
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

import numpy as np
import numpy.linalg as L

from pyDis.atomic import crystal as cry
from pyDis.atomic import rodSetup as rs
from pyDis.atomic import fields as df
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import aniso

def dipole(x,Omega,*args):
    uPlus = df.isotropicWedgeField(x,Omega,np.array([1.,0.,0.]))
    uMinus = df.isotropicWedgeField(x,-Omega,np.array([-1.,0.,0.]))
    return uPlus + uMinus

if (__name__ == "__main__"):
    
    # scale typically needs to be larger than for a dislocation
    SCALE = 1.20
    
    print 'Input name of base gulp file: ',
    gulpName = raw_input()
    print 'Input path to base file: ',
    path = raw_input()
    print 'Specify RI: ',
    RI = float(raw_input())
    print 'Specify RII: ',
    RII = float(raw_input())
    # maximum R to use when constructing the cluster
    Rmax = SCALE*RII
    print 'Enter name of output file: ',
    outName = raw_input()
    print 'Enter x component of eta: ',
    etax = float(raw_input())
    print 'Enter y component of eta: ',
    etay = float(raw_input())

    gulpStruc = cry.Crystal()
    sysInfo = gulp.parse_gulp(gulpName,gulpStruc)
    
    ### TEMPORARY for anisotropic disclinations
    #Cij = aniso.readCij('diamond100_base')
    #uField = aniso.anisoWedgeDisclination(Cij)
    ### \TEMPORARY
    
    # write a file containing system parameters
    sysOut = open(outName + '.%d.%d.sysInfo' % (RI,RII),'w')
    sysOut.write('pcell\n')
    sysOut.write('%.5f 0\n' % L.norm(gulpStruc.getC()))
            
    for line in sysInfo:
        sysOut.write(line + '\n')
               
    sysOut.close()
    
    eta = np.array([etax,etay])
    Frank = np.array([0.,0.,0.24498])
    disFrank = np.array([Frank])
    disCores = np.array([[0.,0.]])
    
    gulpCluster = rs.TwoRegionCluster(gulpStruc,eta,Rmax,RI,RII)    
    gulpCluster.applyField(df.isotropicWedgeField,disCores,disFrank,0.08654,
                                                    branch=[-1.,0.]) 
    #gulpCluster.applyField(uField,disCores,disFrank,0.,branch=[-1,0])  
    
    outStream = outName+'.%d.%d' % (RI,RII)   
    gulp.write1DCluster(gulpCluster,sysInfo,outStream)
    
