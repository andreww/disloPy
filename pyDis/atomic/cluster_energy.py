#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import numpy.linalg as L
import re
import os
import sys
from scipy.optimize import curve_fit

import crystal as cry
import gulpUtils as gulp

# convert elastic modulus in eV/ang**3 to GPa
CONV_EV_TO_GPA = 160.2176487 
    
def R(atom):
    '''Radial distance from dislocation line.
    '''
    
    # use first two elements of coordinates
    x = atom.getCoordinates()
    distance = np.sqrt(x[0]**2+x[1]**2)
    return distance
 
def newRegions(rIPerfect, rIDislocated, rIIPerfect, rIIDislocated, Rnew):
    '''Creates new <cry.Basis> objects to hold the lists of 
    region I and region II atoms with a smaller radius <Rnew> 
    '''   
    
    newRI = cry.Basis()
    newRII = rIIDislocated.copy()
    newRIPerfect = cry.Basis()
    newRIIPerfect = rIIPerfect.copy()
    
    for i, atom in enumerate(rIDislocated):
        # calculate radial distance of the i-th atom from dislocation line
        # in the relaxed structure. 
        Rxy = R(atom)
        if Rxy < Rnew:
            # atom is sufficiently close to dislocation to include in region I
            newRI.addAtom(atom)
            newRIPerfect.addAtom(rIPerfect[i])
        else:
            # fix atom in region II if Rxy greater than new radius
            newRII.addAtom(atom)
            newRIIPerfect.addAtom(rIPerfect[i])
            
    return newRI, newRII, newRIPerfect, newRIIPerfect
    
def readSystemInfo(filename):
    '''Reads the file containing all relevant system information (interatomic 
    potentials etc.), plus the value of pcell.
    '''
    
    sysFile = open(filename + '.sysInfo', 'r')
    sysInfo = sysFile.readlines()
    sysFile.close()
    
    return sysInfo
    
def writeSinglePoint(rIBasis, rIIBasis, sysInfo, outName, disloc):
    '''Creates a gulp input file for a single point cluster calculation
    using the atoms in <rIBasis> and <rIIBasis>, with the pcell value and 
    interatomic potentials in <sysInfo>. <disloc> is Boolean variable
    whose value is <True> if the system contains a dislocation.
    '''
    
    dummyLattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    outStream = open(outName + '.gin', 'w')
    
    # permit charged cells
    outStream.write('qok eregion\n')
    
    outStream.write('pcell\n')
    outStream.write(sysInfo[1])
    
    # Before writing regions to output, test to see if they are empty.
    if rIBasis.getAtoms():  
        gulp.writeRegion(rIBasis, dummyLattice, outStream, 1, disloc, use_cart=False,
                                                       coordType='pfractional')
    if rIIBasis.getAtoms():
        gulp.writeRegion(rIIBasis, dummyLattice, outStream, 2, disloc, use_cart=False,
                                                        coordType='pfractional')
    
    # finally, write interatomic potentials, etc. to file
    for line in sysInfo[2:]:
        outStream.write(line)
    return
    
def writeSPSections(regionI, regionII, sysInfo, outName, disloc):
    '''Takes in region I and region II atoms, and performs single-point energy
    calculations on 1D periodic simulation cells containing only atoms from
    region I (RI), only atoms from region II (RII), and all atoms (BOTH).
    '''
    
    # initialise an empty basis object. We will pass this to <writeSinglePoint>
    # in place of any region that we do not wish to output
    emptyBasis = cry.Basis()
    
    # region I only
    writeSinglePoint(regionI, emptyBasis, sysInfo, outName+'.RI', disloc)
    # region II only
    writeSinglePoint(emptyBasis, regionII, sysInfo, outName+'.RII', disloc)
    # both regions 
    writeSinglePoint(regionI, regionII, sysInfo, outName+'.BOTH', disloc)
    
    return
    
def nameBits(baseName):
    '''Extracts basic filename and value of RII from <baseName>. Useful for 
    naming single-point calculation input files.
    '''
    
    baseFormat = re.match(r'(\w+)\.\d+\.(\d+)', baseName)
    systemName = baseFormat.group(1)
    RII = baseFormat.group(2) 
    
    return systemName, RII
    
def readOutputFile(filename):
    '''Reads in a gulp single point energy calculation output file and formats.
    Returns a string containing every line in filename, including \\n characters
    '''
    
    outFile = open(filename, 'r')
    outLines = outFile.readlines()
    outFile.close()
    
    # concatenate all lines in file together
    singleString = ''
    for line in outLines:
        singleString += line
        
    return singleString
    
def dislocationEnergyExplicit(gulpName, outStream, currentRI):
    '''Constructs energy curve for dislocation using output files of single
    point calculations. Outputs results to <outName>
    '''
    
    # regular expression which matches the section of the gulp output file that
    # breaks the energies down into regions
    regionEnergies = re.compile(r'\n\s+1\s+(?P<RI>-?\d+\.\d+)\s*\n\s+2\s+' 
                           + '(?P<RIRII>-?\d+\.\d+)\s+(?P<RII>-?\d+\.\d+)\s*\n')
                                    
    # open single point calc. output files
    dislocated = readOutputFile(gulpName + '.dislocated.gout')
    perfect = readOutputFile(gulpName + '.perfect.gout')
    
    # calculated total Region I energies for both clusters                                
    energiesDislocated = regionEnergies.search(dislocated)
    disERI = float(energiesDislocated.group('RI'))
    disERIRII = float(energiesDislocated.group('RIRII'))
    totalEnergyDis = disERI + 0.5*disERIRII
    
    energiesPerfect = regionEnergies.search(perfect)
    perfERI = float(energiesPerfect.group('RI'))
    perfERIRII = float(energiesPerfect.group('RIRII'))
    totalEnergyPerf = perfERI + 0.5*perfERIRII
    
    # calculate energy of introducing a dislocation
    eDis = totalEnergyDis - totalEnergyPerf
    
    outStream.write('{} {:.6f}\n'.format(currentRI, eDis))
    
    return eDis
    
def dislocationEnergySections(gulpName, outStream, currentRI):
    '''Constructs energy curve for dislocation using RI, RII, and BOTH
    output files from single point calculations. Outputs results to <outName>.
    '''
    
    # regex to find the total energy of the simulation cell
    latticeEnergy = re.compile(r'\n\s*Total lattice energy\s+=\s+' +
                                  '(?P<energy>-?\d+\.\d+)\s+eV\s*\n')
                                  
    # list of simulation regions
    simulations = ('RI', 'RII', 'BOTH')
    
    energies = dict()

    # extract energies of each individual single point calculation
    # energy of dislocated cell...
    for cellType in ('dislocated', 'perfect'):
        # create a subdictionary to hold energies of individual calculations
        energies[cellType] = dict()
        for region in simulations:
            outputFile = readOutputFile(gulpName + '.{}.{}.gout'.format(cellType, region))
            E = float(re.search(latticeEnergy, outputFile).group('energy'))
            energies[cellType][region] = E
         
        # compute region I - region II interaction energy for polymer <cellType>    
        energies[cellType]['RIRII'] = (energies[cellType]['BOTH'] -
                    energies[cellType]['RI'] - energies[cellType]['RII'])
        # compute total energy of region I (ERI + 0.5*Einteraction)
        energies[cellType]['Etot'] = (energies[cellType]['RI'] + 
                                    0.5*energies[cellType]['RIRII'])
    
    # calculate energy of dislocation                
    eDis = energies['dislocated']['Etot'] - energies['perfect']['Etot']
    
    outStream.write('{} {:.6f}\n'.format(currentRI, eDis))
        
    return eDis

def iterateOverRI(startRI, finalRI, dRI, baseName, gulpExec,
                                                explicitRegions=True):
    '''Decreases the radius of region I from <startRI> to <finalRI> by
    decrementing by dRI (> 0). <baseName> gives the root name for both .grs
    files and the .sysInfo file. <gulpExec> is the path to the GULP 
    executable. If <explicitRegions>, gulp can calculate the total energy of
    the dislocation explicitly (E(RI)+0.5*E(RI-RII)). 
    '''
    
    # open the output file to stream energy data to
    outStream = open(baseName + '.energies', 'w')
    sysInfo = readSystemInfo(baseName)
   
    # get bits for output names
    systemName, RIIval = nameBits(baseName)
    
    # initialise <Basis> objects to hold input regions
    rIPerfect = cry.Basis()
    rIIPerfect = cry.Basis()
    rIDislocated = cry.Basis()
    rIIDislocated = cry.Basis()
    
    # populate these bases from the input files
    gulp.extractRegions('ndf.{}.grs'.format(baseName), rIPerfect, rIIPerfect)
    gulp.extractRegions('dis.{}.grs'.format(baseName), rIDislocated, rIIDislocated)

    # ensure that dRI is > 0
    dRI = abs(dRI)
    if (dRI - 0.) < 1e-10:
        print('Error: Decrement interval must be nonzero.')
        sys.exit(1)
    elif startRI < finalRI:
        print('Error: Initial radius is lower than final radius.')
        sys.exit(1)
    elif finalRI <= 0.:
        print('Error: Final radius must be >= 0.')
        sys.exit(1)

    currentRI = startRI
        
    while currentRI >= finalRI:
    
        print('Setting RI = {:.1f}...'.format(currentRI))
        print('Calculating energy', end=' ') 
        
        [newRI, newRII, newRIPerfect, newRIIPerfect] = newRegions(rIPerfect,
                             rIDislocated, rIIPerfect, rIIDislocated, currentRI)

        derivedName = 'sp.{}.{:.1f}.{}'.format(systemName, currentRI, RIIval)
        
        # works to here
        
        if explicitRegions: # use GULP's eregion functionality     
            print('using eregion.')              
            writeSinglePoint(newRI, newRII, sysInfo, derivedName+'.dislocated',
                                                                        True)
            writeSinglePoint(newRIPerfect, newRIIPerfect, sysInfo, derivedName 
                                                         + '.perfect', False)
                                                         
            # run single point calculations and extract dislocation energy
            for cellType in ('dislocated', 'perfect'):
                gulp.run_gulp(gulpExec, '{}.{}'.format(derivedName, cellType))
                                                
            eDis = dislocationEnergyExplicit(derivedName, outStream, currentRI) 
            
        else: # calculate energy of region I by calculating energies of regions
            print ('using sections.')
            writeSPSections(newRI, newRII, sysInfo, derivedName + '.dislocated',
                                                                        True)
            writeSPSections(newRIPerfect, newRIIPerfect, sysInfo, derivedName 
                                                         + '.perfect', False)
                                                         
            # run single point calculations and extract dislocation energy                                             
            for cellType in ('dislocated', 'perfect'):
                for regionsUsed in ('RI', 'RII', 'BOTH'):
                    gulp.run_gulp(gulpExec, '%s.%s.%s' % (derivedName, cellType,
                                                            regionsUsed))
                                                          
            eDis = dislocationEnergySections(derivedName, outStream, currentRI) 
            
        print('Energy is {:.6f} eV'.format(eDis))    
                                                         
        currentRI = currentRI - dRI        
        
    return
    
# routines to fit the core energy of the dislocation

def readEnergies(basename):
    '''Read in the file containing the radius-region I energy data.
    '''
    
    E = []
    with open(basename+'.energies', 'r') as EFile:
        for line in EFile:
            line = line.rstrip().split()
            E.append([float(line[0]), float(line[1])])
            
    return np.array(E)
    
def EDis(r, Ecore, K, b, rcore=10.):
    '''Strain energy between rcore and r of a dislocation with core energy 
    <Ecore>, Burgers vector <b>, and energy coefficient K. Returns in units of
    eV/b.
    '''   
    
    return Ecore + K*b**2/(4*np.pi)*np.log(r/rcore)
    
def fitCoreEnergy(basename, b, thickness, rcore=10, fit_K=False):
    '''Fit the core energy and energy coefficient of the dislocation whose 
    radius-energy data is stored in <basename>.energies. Returns K in eV/ang**3
    and Ecore in eV/ang. <thickness> is the length of the simulation cell.
    '''
    
    E = readEnergies(basename)/np.array([1., thickness])
   
    # define dislocation energy function -> contains specified core radius. If
    # fit_K is true, fit the value of K, otherwise, use the material specific
    # value (which we prompt the user to enter
    if fit_K:
        def specific_energy(r, Ecore, K):
            return Ecore + K*b**2/(4*np.pi)*np.log(r/rcore)
    else:
        K = raw_input("Enter the energy coefficient K (in GPa): ")
        # convert to eV/ang**3
        K = float(K)/CONV_EV_TO_GPA
        def specific_energy(r, Ecore):
            return Ecore + K*b**2/(4*np.pi)*np.log(r/rcore)
        
    # fit the core energy and energy coefficient
    par, cov = curve_fit(specific_energy, E[:, 0], E[:, 1])
    Ecore = par[0] 
    if fit_K:
        K = par[1]
    
    return Ecore, K, cov
    
def numericalEnergyCurve(rmax, Ecore, K, b, rcore=10, rmin=1, dr=0.1):
    '''Computes energy as a function of r between <rmin> and <rmax> (at
    intervals of <dr>) using the fitted dislocation energy function.
    '''
    
    # number of samples
    n = int((rmax-rmin)/dr)
    r = np.linspace(rmin, rmax, n)
    
    # generate energy as a function of r
    E = EDis(r, Ecore, K, b, rcore)
    
    return r, E    
    
def main(argv):
    '''Driver function for energy fitting.
    '''
    
    # read in simulation parameters
    startRI = eval(argv[1])
    finalRI = eval(argv[2])
    dRI = eval(argv[3])
    basename = str(argv[4])
    relax_K = eval(argv[5])
    
    try:
        gulpExec = str(argv[6])
    except IndexError:
        # if no path to the GULP executable is given, assume that it is in the
        # present working directory. 
        gulpExec = './gulp'
        # tell program whether or not to use <eregion> to partition energies 
        # between regions I and II
        # <explicitRegions> == False => Do three single point calculations: one
        # with the whole simulation cell, one with the region I atoms only, and
        # one with the region II atoms only. 
        try:
            explicitRegions = eval(argv[6])
        except IndexError:
            explicitRegions = True
    else:
        # as above    
        try:
            explicitRegions = eval(argv[7])
        except IndexError:
            explicitRegions = True
        
    # calculate energy as a function of radius
    print('####CALCULATING ENERGY OF DISLOCATION AS A FUNCTION OF R####')
    iterateOverRI(startRI, finalRI, dRI, basename, gulpExec, explicitRegions)
    
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
    thickness = raw_input('Enter cell thickness (<return> to use b): ')
    
    # if the user has pressed return, set <thickness> = burger vector length
    if not(thickness):
        thickness = b
        not_thickness = False
        
    while not_thickness:
        try:
            thickness = float(thickness)
        except ValueError:
            thickness = raw_input('Invalid thickness entered. Please try again: ')
        else:
            not_thickness = False
    
    # Fit core energy and energy coefficient, and write to output
    Ecore, K, cov = fitCoreEnergy(basename, b, thickness, 2*b, fit_K=relax_K)
    KGPa = K*CONV_EV_TO_GPA
    if len(cov) > 1:  
        errKGPa = np.sqrt(cov[1, 1])*CONV_EV_TO_GPA
    else:
        errKGPa = 0. # did not fit K
        
    EString = "Core energy: {:.4f} +- {:.4f} eV/angstrom".format(Ecore, np.sqrt(cov[0, 0]))
    KString = "Energy coefficient: {:.2f} +- {:.2f} GPa".format(KGPa, errKGPa)
                          
    print('\n\n' + EString)
    print(KString)
                          
    # generate fitted energies and output to file
    print('\n\nWriting fitted energies to file...\n\n')
    fittedEnergies = open('{}.fitted.energies'.format(basename), 'w')
    r, E = numericalEnergyCurve(startRI, Ecore, K, b, 2*b)
    fittedEnergies.write('# {}; {}\n'.format(EString, KString))
    for i, energy in enumerate(E):
        fittedEnergies.write('{:.3f} {:.6f}\n'.format(r[i], energy))
    fittedEnergies.close()
    
    delete = raw_input('Delete single point calculation input/output files' +
                                    ' (press <return> for yes): ')
    if delete.lower() in 'yes' or not(delete):
        os.system('rm -f sp.*')
    
    print('                     ####FINISHED####')
    
    
if __name__ == "__main__":
    main(sys.argv)    
    
        
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
