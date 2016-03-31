#!/usr/bin/env python
from __future__ import print_function

import re
import numpy as np
import os
import sys

import gulpUtils as gulp
import cluster_energy as ce
import crystal as cry

# regex guff
formula_parser = re.compile('\s+Formula\s+=\s+' +
    '([A-Z][a-z]?\d+)+')
atom_format = re.compile('(?P<name>[A-Z][a-z]?)(?P<nat>\d+)')
latticeEnergy = re.compile(r'\n\s*Total lattice energy\s+=\s+' +
                                  '(?P<energy>-?\d+\.\d+)\s+eV\s*\n')

def iterateEdgeRI(startRI, finalRI, dRI, base_name, gulpexec, E_atom):
    '''Routine to calculate single-point energies for an edge dislocation -> 
    will merge with the routines for screw dislocations.
    
    Currently only works for single species crystals.
    '''
    
    sysInfo = ce.readSystemInfo(base_name)

    # extract relaxed dislocation
    rIAtoms = cry.Basis()
    rIIAtoms = cry.Basis()
    gulp.extractRegions('dis.{}.grs'.format(base_name), rIAtoms, rIIAtoms)

    # need to extract undislocated cluster for <clusterEnergy> routines
    # -> should fix this
    rIperf = cry.Basis()
    rIIperf = cry.Basis()
    gulp.extractRegions('ndf.{}.grs'.format(base_name), rIperf, rIIperf)
    
    currentRI = startRI 
    
    excess_energies = []
  
    # iterate over radii
    while currentRI >= finalRI:
    
        print('Setting RI = {:.1f}...'.format(currentRI))
        print('Calculating energy', end=' ')
         
        [newRI, newRII, tempRI, tempRII] = ce.newRegions(rIperf, rIAtoms,
                    rIIperf, rIIAtoms, currentRI)    
                    
        ce.writeSPSections(newRI, newRII, sysInfo, 'sp.{}.{}'.format(base_name,
                                                               currentRI), True)
                                        
        for reg in ['RI', 'RII', 'BOTH']:
            gulp.run_gulp(gulpexec, 'sp.{}.{}.{}'.format(base_name, currentRI, reg))

        # calculate excess energies
        rIfile = ce.readOutputFile('sp.{}.{}.RI.gout'.format(base_name, currentRI))
        rIIfile = ce.readOutputFile('sp.{}.{}.RII.gout'.format(base_name, currentRI))
        bothFile = ce.readOutputFile('sp.{}.{}.BOTH.gout'.format(base_name, currentRI))
        ERI = float(re.search(latticeEnergy, rIfile).group('energy'))
        ERII = float(re.search(latticeEnergy, rIIfile).group('energy'))
        EBoth = float(re.search(latticeEnergy, bothFile).group('energy'))
        
        total_E_RI = ERI + 0.5*(EBoth - (ERI+ERII))
        E_excess = total_E_RI - newRI.numberOfAtoms*E_atom
        excess_energies.append([currentRI, E_excess])
        currentRI -= dRI
        print('Energy is {:.6f} eV'.format(E_excess)) 
        
    outStream = open(base_name + '.energies', 'w')
    for r, E in excess_energies:
        outStream.write('{} {:.4f}\n'.format(r, E))
        
    outStream.close()
    return
    
def iterateIonicEdge(startRI, finalRI, dRI, base_name, gulpexec, E_atoms):
    '''Iterate over values of RI in the cluster cell to find the energy function
    E(r) for an edge dislocation.
    '''

    
    # read system specific information required to run simulation
    sysInfo = ce.readSystemInfo(base_name)

    # extract relaxed atomic positions in dislocated cell
    rIAtoms = cry.Basis()
    rIIAtoms = cry.Basis()
    gulp.extractRegions('dis.{}.grs'.format(base_name), rIAtoms, rIIAtoms)

    # extract atoms in perfect cell (need to get rid of this)
    rIperf = cry.Basis()
    rIIperf = cry.Basis()
    gulp.extractRegions('ndf.{}.grs'.format(base_name), rIperf, rIIperf)
    currentRI = startRI
    excess_energies = []
    # iterate over radii
    while currentRI >= finalRI:

        print('Setting RI = {:.1f}...'.format(currentRI))
        print('Calculating energy', end=' ')
        [newRI, newRII, tempRI, tempRII] = ce.newRegions(rIperf, rIAtoms,
                                           rIIperf, rIIAtoms, currentRI)
        ce.writeSPSections(newRI, newRII, sysInfo, 'sp.{}.{}'.format(base_name, currentRI),
                                                                                  True)
        for reg in ['RI', 'RII', 'BOTH']:
            gulp.run_gulp(gulpexec, 'sp.{}.{}.{}'.format(base_name, currentRI, reg))
            
        rIfile = ce.readOutputFile('sp.{}.{}.RI.gout'.format((base_name, currentRI))
        rIIfile = ce.readOutputFile('sp.{}.{}.RII.gout'.format(base_name, currentRI))
        bothFile = ce.readOutputFile('sp.{}.{}.BOTH.gout'.format(base_name, currentRI))
        
        ERI = float(re.search(latticeEnergy, rIfile).group('energy'))
        ERII = float(re.search(latticeEnergy, rIIfile).group('energy'))
        EBoth = float(re.search(latticeEnergy, bothFile).group('energy'))
        total_E_RI = ERI + 0.5*(EBoth - (ERI+ERII))
        E_perfect = 0.0
        
        for atom in newRI:
            E_perfect += E_atoms[atom.getSpecies()]
            
        E_excess = total_E_RI - E_perfect
        excess_energies.append([currentRI, E_excess])
        currentRI -= dRI
        print('Energy is {:.6f} eV'.format(E_excess))
        
    outStream = open(base_name + '.energies', 'w')
    for r, E in excess_energies:
        outStream.write('{} {:.4f}\n'.format(r, E))
        
    outStream.close()
    return
    
def make_atom_dict():
    new_dict = dict()
    while True:
        new_key = raw_input("Enter species (<enter> to exit): ")
        if not new_key:
            break
        value = float(raw_input("Enter energy: "))
        new_dict[new_key] = value
    return new_dict

def main(argv):
    '''Driver function for energy fitting.
    '''
    
    # read in simulation parameters
    startRI = eval(argv[1])
    finalRI = eval(argv[2])
    dRI = eval(argv[3])
    basename = str(argv[4])
    relax_K = eval(argv[5]) # True or False, do we fit K?
    E_atom = eval(argv[6])
    
    try:
        gulpExec = str(argv[7])
    except IndexError:
        # if no path to the GULP executable is given, assume that it is in the
        # present working directory. 
        gulpExec = './gulp'
        
    # calculate energy as a function of radius
    print('####CALCULATING ENERGY OF EDGE DISLOCATION AS A FUNCTION OF R####')
    iterateEdgeRI(startRI, finalRI, dRI, basename, gulpExec, E_atom)
    
    # calculate the core energy and energy coefficient
    # begin by prompting user to enter burgers vector and length of simulation
    # cell
    notBurgers = True
    b = raw_input('\n\nEnter Burgers vector: ')
    while notBurgers:
        try:
            b = float(b)
        except ValueError:
            b = raw_input('Invalid Burgers vector entered. Please try again: ')
        else:
            notBurgers = False
            
    not_thickness = True   
    thickness = raw_input('Enter cell thickness (return to use b): ')
    
    # if the user has pressed return, set <thickness> = burger vector length
    if not(thickness):
        thickness = b
        not_thickness = False
    else:
        thickness = float(thickness)
        
    # Fit core energy and energy coefficient, and write to output
    Ecore, K, cov = ce.fitCoreEnergy(basename, b, thickness, 2*b, fit_K=relax_K)
    KGPa = K*ce.CONV_EV_TO_GPA
    if len(cov) > 1:  
        errKGPa = np.sqrt(cov[1, 1])*ce.CONV_EV_TO_GPA
    else:
        errKGPa = 0. # did not fit K
        
    EString = "Core energy: {:.4f} +- {:.4f} eV/angstrom".format(Ecore, np.sqrt(cov[0, 0]))
    KString = "Energy coefficient: {:.2f} +- {:.2f} GPa".format(KGPa, errKGPa)
                          
    print('\n\n' + EString)
    print(KString)
                          
    # generate fitted energies and output to file
    print('\n\nWriting fitted energies to file...\n\n')
    fittedEnergies = open('{}.fitted.energies'.format(basename), 'w')
    r, E = ce.numericalEnergyCurve(startRI, Ecore, K, b, 2*b)
    fittedEnergies.write('# {}; {}\n'.format(EString, KString))
    for i, energy in enumerate(E):
        fittedEnergies.write('{:.3f} {:.6f}\n'.format(r[i], energy))
    fittedEnergies.close()
    
    delete = raw_input('Delete single point calculation input/output files: ')
    if delete.lower() in 'yes' or not delete:
        os.system('rm -f sp.*')
    
    print('                     ####FINISHED####')
    
    
if __name__ == "__main__":
    main(sys.argv)                                           
