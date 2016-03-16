#!/usr/bin/env python
'''Driver program for supercell construction in gulp. For the moment, 
assume that the simulation cell has orthogonal unit cell vectors. This
means that a quadrupole arrangement of dislocations must be used.
Currently, set up for screw dislocation in isotropic material only.'''
import sys
sys.path.append('/home/richard/code_bases/dislocator2/')

from pyDis.atomic import crystal as cry
from pyDis.atomic import fields as df
from pyDis.atomic import gulpUtils as gulp

import numpy as np
import numpy.linalg as L
import sys

def moduloCoords(crystalFile):
    '''Takes atomic coordinates modulo 1.
    '''
    for atom in crystalFile:
        x = atom.getDisplacedCoordinates()
        x = x % 1.
        atom.setDisplacedCoordinates(x)
        
    return

if __name__ == "__main__":
    
    print 'Enter name of input gulp file: ',
    gulpName = raw_input()
    print 'Enter path to %s: ' % gulpName,
    gulpPath = raw_input()
    print 'Enter supercell x-width: ',
    xWidth = int(raw_input())
    print 'Enter supercell y-width: ',
    yWidth = int(raw_input())
    print 'What file do you wish to write the output to? ',
    outputName = raw_input()
    print 'Enter Sij: ',
    try:
        sij = float(raw_input())
    except:
        sij = 1.
   
    
    #gulpName = 'mgo100.gin'
    #gulpPath = 'tests/'
    #outputName = 'SCMgO100Simp'

    gulpStruc = cry.Crystal()
    sysInfo = gulp.parse_gulp(gulpName,gulpStruc)
            
    superGulp = cry.superConstructor(gulpStruc,np.array([xWidth,yWidth,0]))
            
    core1 = np.array([0.25/xWidth,0.5/yWidth])
    b = L.norm(gulpStruc.getC())*cry.ei(3)
    # QUADRUPOLE
    disBurgers = np.array([b,-b,b,-b])
    #disCores = np.array([core1+np.array([0.25,0.25]),core1+np.array([0.75,0.25]),
    #                     core1+np.array([0.75,0.75]),core1+np.array([0.25,0.75])])
    disCores = np.array([core1,core1+np.array([0.5,0.]),core1+np.array([0.5,0.5]),
                                                    core1+np.array([0.,0.5])])
    ######
    # DIPOLE
    #disBurgers = np.array([b,-b])
    #disCores = np.array([core1+np.array([0.5,0.]),core1+np.array([0.5,0.5])])
    ######
                
    superGulp.applyField(df.anisotropicScrewField, disCores, disBurgers, Sij=sij)
    moduloCoords(superGulp)
                
    relaxType = 'conv'
    
    outputFile2 = '%s.%d.%d' % (outputName,xWidth,yWidth)
    gulp.writeSuper(superGulp,sysInfo,outputFile2,relaxType)
        
        
        
