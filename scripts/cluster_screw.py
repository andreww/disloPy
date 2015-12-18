#!/usr/bin/env python
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

import numpy as np
import numpy.linalg as L

from pyDis.atomic import crystal as cry
from pyDis.atomic import rodSetup as rs
from pyDis.atomic import disField as df
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import aniso

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
    print 'Enter x component of eta (<enter> for 0.): ',
    try:
        etax = float(raw_input())
    except ValueError:
        etax = 0.0
    print 'Enter y component of eta (<enter> for 0.): ',
    try:
        etay = float(raw_input())
    except ValueError:
        etay = 0.0
    print 'Enter thickness (<enter> for 1): ',
    thickness = raw_input()
    try:
        thickness = int(thickness)
    except ValueError:
        thickness = 1
        
    #print 'Use force minimization (y/(n))?',
    #force_min=raw_input()
    #if 'y' in force_min.lower():
    #    useforce = True
    #else:
    #    useforce = False
        
    #print 'Enter Sij: ',
    #Sij = float(raw_input())

    gulpStruc = cry.Crystal()
    sysInfo = gulp.parse_gulp(gulpName,gulpStruc)
    
    ### TEMPORARY
    #Cij = aniso.readCij('uo2')
    #uField = aniso.makeAnisoField(Cij)
    ### \TEMPORARY
    
    # write a file containing system parameters
    sysOut = open(outName + '.%d.%d.sysInfo' % (int(RI),int(RII)),'w')
    sysOut.write('pcell\n')
    sysOut.write('%.5f 0\n' % (thickness*L.norm(gulpStruc.getC())))
            
    for line in sysInfo:
        sysOut.write(line + '\n')
               
    sysOut.close()
    
    eta = np.array([etax,etay])
    #b = np.array([0.,L.norm(gulpStruc.getB()),0.])
    b = np.array([0.0,0.0,L.norm(gulpStruc.getC())])
    disBurgers = np.array([b])
    disCores = np.array([[0.,0.]]) 
    
    gulpCluster = rs.TwoRegionCluster(gulpStruc,eta,Rmax,RI,RII,thickness)    
    #gulpCluster.applyField(df.isotropicEdgeField,disCores,disBurgers,Sij=0.30444,
    #                                        branch=[0.,-1])  
    gulpCluster.applyField(df.isotropicScrewField, disCores, disBurgers, Sij=1.684)
    ### TEMPORARY
    #gulpCluster.applyField(uField,disCores,disBurgers,0.08658,branch=[-1,0],THRESH=1.0)
    ### \TEMPORARY
    
    outStream2 = outName+'.%d.%d' % (RI,RII)   
    gulp.write1DCluster(gulpCluster,sysInfo,outStream2)
    
