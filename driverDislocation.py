#!/usr/bin/env python

import numpy as np
import numpy.linalg as L

import crystal as cry
import rodSetup as rs
import disField as df
import gulpUtils as gulp

if (__name__ == "__main__"):

    SCALE = 1.10
    
    print 'Input name of base gulp file: ',
    gulpName = raw_input()
    print 'Input path to base file: ',
    path = raw_input()
    print 'Specify RI: ',
    RI = float(raw_input())
    print 'Specify RII: ',
    RII = float(raw_input())
    # set max R to RII*SCALE -> can be pruned later
    Rmax = RII*SCALE
    print 'Enter name of output file: ',
    outName = raw_input()
    print 'Enter x component of eta: ',
    etax = float(raw_input())
    print 'Enter y component of eta: ',
    etay = float(raw_input())
    print 'Enter thickness: ',
    thickness = int(raw_input())
    #thickness = 2
    
    gulpLines = gulp.readGulpFile(gulpName,path)
    gulpStruc = cry.Crystal()
    sysInfo = gulp.parseGulpFile(gulpLines,gulpStruc)
    
    # write a file containing system parameters
    sysOut = open(outName + '.%d.%d.sysInfo' % (RI,RII),'w')
    sysOut.write('pcell\n')
    sysOut.write('%.5f 0\n' % L.norm(gulpStruc.getC()))
            
    for line in sysInfo:
        sysOut.write(line + '\n')
               
    sysOut.close()
    
    eta = np.array([etax,etay])
    b = np.array([0.,0.,L.norm(gulpStruc.getC())])
    disBurgers = np.array([b])
    disCores = np.array([[0.,0.]])
    
    gulpCluster = rs.GulpCluster(gulpStruc,eta,Rmax,RI,RII,thickness)    
    gulpCluster.applyField(df.isotropicScrewField,disCores,disBurgers,0.34672)   
    
    outStream2 = outName+'.%d.%d' % (RI,RII)   
    gulp.write1DCluster(gulpCluster,sysInfo,outStream2)
    
