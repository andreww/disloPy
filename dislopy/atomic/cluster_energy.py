#!/usr/bin/env python
'''Routines to calculate the core energy of a dislocation embedded in a 1D 
periodic simulation cell.
'''
from __future__ import print_function, absolute_import

import sys

import numpy as np
import numpy.linalg as L
import re
import os
import argparse
from scipy.optimize import curve_fit

from dislopy.atomic import crystal as cry
from dislopy.atomic import gulpUtils as gulp
from dislopy.utilities import atomistic_utils as atm

# convert elastic modulus in eV/ang**3 to GPa
CONV_EV_TO_GPA = 160.2176487 

# regex guff -> find new home
atom_format = re.compile('(?P<name>[A-Z][a-z]?)(?P<nat>\d+)')
#latticeEnergy = re.compile(r'\n\s*Total lattice energy\s+=\s+' +
#                                  '(?P<energy>-?\d+\.\d+)\s+eV\s*\n')
    
def R(atom):
    '''Radial distance from dislocation line.
    '''
    
    # use first two elements of coordinates
    x = atom.getCoordinates()
    distance = np.sqrt(x[0]**2+x[1]**2)
    return distance
    
### HANDLE ATOMS ###

def handle_atoms(atomlist):
    '''Convert a list of atomic symbols and energies into a lookup table.
    '''
    
    # check that two values (symbol + energy) have been provided for every 
    # atom.
    if atomlist is None:
        return None
    if len(atomlist) % 2 == 1:
        raise ValueError("")
        
    atom_dict = dict()
    for i in range(len(atomlist)):
        if i % 2 == 1:
            continue
        else:
            atom_dict[atomlist[i]] = float(atomlist[i+1])
        
    return atom_dict
    
def atom_dict_interactive():
    '''Prompts the user to create (interactively) a dictionary containing all
    atomic species present and their energies in the perfect (ie. dislocation
    free) crystal. Note: not the only way that energies may be provided.
    '''
    
    new_dict = dict()
    while True:
        new_key = raw_input("Enter species (<enter> to exit): ")
        if not new_key:
            break
        else:
            value = float(raw_input("Enter energy: "))
            new_dict[new_key] = value
        
    return new_dict
    
### READ AND WRITE ATOMISTIC SIMULATION INPUT/OUTPUT ###
 
def newRegions(rIPerfect, rIDislocated, rIIPerfect, rIIDislocated, Rnew, edge=False):
    '''Creates new <cry.Basis> objects to hold the lists of region I and region 
    II atoms with a smaller radius <Rnew>. If <edge> is True, do this only for
    the dislocated cell, and return emptu bases for the perfect cell.
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
            if not edge:
                newRIPerfect.addAtom(rIPerfect[i])
        else:
            # fix atom in region II if Rxy greater than new radius
            newRII.addAtom(atom)
            if not edge:
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
    
    baseFormat = re.match(r'([\w.]+)\.\d+\.(\d+)', baseName)
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
    
### ENERGY CALCULATION FUNCTIONS ###
    
def dislocationEnergyEregion(gulpName, outStream, currentRI):
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
    
    return eDis
    
def dislocationEnergySections(gulpName, outStream, currentRI):
    '''Constructs energy curve for dislocation using RI, RII, and BOTH
    output files from single point calculations. Outputs results to <outName>.
    '''
                                  
    # list of simulation regions
    simulations = ('RI', 'RII', 'BOTH')
    
    energies = dict()

    # extract energies of each individual single point calculation
    # energy of dislocated cell...
    for cellType in ('dislocated', 'perfect'):
        # create a subdictionary to hold energies of individual calculations
        energies[cellType] = dict()
        for region in simulations:
            E, units = atm.extract_energy('{}.{}.{}.gout'.format(gulpName, 
                                   cellType, region), 'gulp', relax=False)
            energies[cellType][region] = E
         
        # compute region I - region II interaction energy for polymer <cellType>    
        energies[cellType]['RIRII'] = (energies[cellType]['BOTH'] -
                    energies[cellType]['RI'] - energies[cellType]['RII'])
        # compute total energy of region I (ERI + 0.5*Einteraction)
        energies[cellType]['Etot'] = (energies[cellType]['RI'] + 
                                    0.5*energies[cellType]['RIRII'])
    
    # calculate energy of dislocation                
    eDis = energies['dislocated']['Etot'] - energies['perfect']['Etot']
        
    return eDis
    
    
def dislocation_energy_edge(base_name, outStream, currentRI, newRI, E_atoms):
    '''Calculate energy using atomic energies. As the edge dislocation setup code
    is non-conservative, the use of this method is mandatory for edge dislocations.
    '''
                                  
    # list of simulation regions
    simulations = ('RI', 'RII', 'BOTH')
    
    energies = dict()
    
    # extract energies from single point calculations
    ERI, units = atm.extract_energy('{}.RI.gout'.format(base_name), 'gulp', relax=False)
    ERII, units = atm.extract_energy('{}.RII.gout'.format(base_name), 'gulp', relax=False)
    EBoth, units = atm.extract_energy('{}.BOTH.gout'.format(base_name), 'gulp', relax=False)
    total_E_RI = ERI + 0.5*(EBoth - (ERI+ERII))
    
    # calculate energy of equivalent number of atoms in perfect crystal
    E_perfect = 0.0        
    for atom in newRI:
        E_perfect += E_atoms[atom.getSpecies()]
            
    eDis = total_E_RI - E_perfect
    
    return eDis

def iterateOverRI(startRI, finalRI, dRI, baseName, gulpExec, thick, 
                                     use_eregion=True, E_atoms=None):
    '''Decreases the radius of region I from <startRI> to <finalRI> by
    decrementing by dRI (> 0). <baseName> gives the root name for both .grs
    files and the .sysInfo file. <gulpExec> is the path to the GULP 
    executable. If <explicitRegions>, gulp can calculate the total energy of
    the dislocation explicitly (E(RI)+0.5*E(RI-RII)). 
    '''
    
    # if atomic energies have been supplied, record that we are using the 
    # edge method
    if not (E_atoms is None):
        using_edge = True
    else:
        using_edge = False
    
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
    
    # populate the bases for regions I and II of the dislocated cell
    gulp.extractRegions('dis.{}.grs'.format(baseName), rIDislocated, rIIDislocated)
    
    # unless atomic energies have been supplied (in which case the energy will
    # be calculated using those instead of a reference, perfect cluster), read
    # in the associated undislocated cluster.
    if not using_edge:
        gulp.extractRegions('ndf.{}.grs'.format(baseName), rIPerfect, rIIPerfect)

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
                             rIDislocated, rIIPerfect, rIIDislocated, currentRI,
                                                            edge=using_edge)

        derivedName = 'sp.{}.{:.1f}.{}'.format(systemName, currentRI, RIIval)
        
        if using_edge:
            print('using edge.')
            writeSPSections(newRI, newRII, sysInfo, derivedName, True)
            
            # run single-point calculations
            for regionsUsed in ('RI', 'RII', 'BOTH'):
                gulp.run_gulp(gulpExec, '{}.{}'.format(derivedName, regionsUsed))
            
            eDis = dislocation_energy_edge(derivedName, outStream, currentRI, 
                                                        newRI, E_atoms) 
        elif use_eregion: # use GULP's eregion functionality     
            print('using eregion.')              
            writeSinglePoint(newRI, newRII, sysInfo, derivedName+'.dislocated',
                                                                        True)
            writeSinglePoint(newRIPerfect, newRIIPerfect, sysInfo, derivedName 
                                                         + '.perfect', False)
                                                         
            # run single point calculations and extract dislocation energy
            for cellType in ('dislocated', 'perfect'):
                gulp.run_gulp(gulpExec, '{}.{}'.format(derivedName, cellType))
                                                
            eDis = dislocationEnergyEregion(derivedName, outStream, currentRI) 
            
        else: # calculate energy of region I by calculating energies of regions
            print ('using sections.')
            writeSPSections(newRI, newRII, sysInfo, derivedName + '.dislocated',
                                                                        True)
            writeSPSections(newRIPerfect, newRIIPerfect, sysInfo, derivedName 
                                                         + '.perfect', False)
                                                         
            # run single point calculations and extract dislocation energy
            for cellType in ('dislocated', 'perfect'):
                for regionsUsed in ('RI', 'RII', 'BOTH'):
                    gulp.run_gulp(gulpExec, '{}.{}.{}'.format(derivedName, cellType,
                                                            regionsUsed))
                                                          
            eDis = dislocationEnergySections(derivedName, outStream, currentRI) 
         
        # convert <eDis> to energy_units/angstrom and write to output streams
        eDis /= thick    
        print('Energy is {:.6f} eV/ang.'.format(eDis))  
        outStream.write('{} {:.6f}\n'.format(currentRI, eDis))  
                                                         
        currentRI = currentRI - dRI        
        
    return
  
### ENERGY FITTING ROUTINES ###

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
    
def fitCoreEnergy(basename, b, rcore=10, fit_K=False, in_K=-1,
                                            using_atomic=False):
    '''Fit the core energy and energy coefficient of the dislocation whose 
    radius-energy data is stored in <basename>.energies. Returns K in eV/ang**3
    and Ecore in eV/ang. <thickness> is the length of the simulation cell.
    '''

    E = readEnergies(basename)
   
    # define dislocation energy function -> contains specified core radius. If
    # fit_K is true, fit the value of K, otherwise, use the material specific
    # value (which we prompt the user to enter
    if fit_K:
        def specific_energy(r, Ecore, K):
            return Ecore + K*b**2/(4*np.pi)*np.log(r/rcore)
    else:
        if in_K == -1: # prompt user to provide energy coefficient
            K = raw_input("Enter the energy coefficient K (in GPa): ")
        else:
            K = in_K
            
        # convert to eV/ang**3
        if not using_atomic:
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
    
def dis_energy(rmax, rmin, dr, basename, executable, method, b, thick, relax_K=True,
                            K=None, atom_dict=None, rc=np.nan, using_atomic=False):
    '''Calculate the energy of a dislocated cluster as a function of radius. This
    is the control function.
    '''
    
    # calculate the energy function of the dislocation using the method appropriate
    # to the type of dislocation and interatomic potentials being used
    print('####CALCULATING ENERGY OF DISLOCATION AS A FUNCTION OF R####')
    if method == 'explicit':
        iterateOverRI(rmax, rmin, dr, basename, executable, thick, use_eregion=False)
    elif method == 'eregion':
        iterateOverRI(rmax, rmin, dr, basename, executable, thick, use_eregion=True)
    elif method == 'edge' and not (atom_dict is None):
        iterateOverRI(rmax, rmin, dr, basename, executable, thick, E_atoms=atom_dict)
    
    # if rc == None, use the normal value of twice the burgers vector length
    if rc != rc:  
        rc = 2*b
  
    # Fit core energy and energy coefficient, and write to output
    Ecore, K, cov = fitCoreEnergy(basename, b, rc, fit_K=relax_K, in_K=K,
                                                    using_atomic=using_atomic)
    KGPa = K*CONV_EV_TO_GPA
    
    # calculate 1-sigma uncertainty in the value of the energy coefficient
    if len(cov) > 1:  
        errKGPa = np.sqrt(cov[1, 1])*CONV_EV_TO_GPA
    else:
        errKGPa = 0. # did not fit K
    
    # generate fitted energy function...
    r, E = numericalEnergyCurve(rmax, Ecore, K, b, rc)

    # and output to file
    fittedEnergies = open('{}.fitted.energies'.format(basename), 'w') 
    
    # write core energy and energy coefficient   
    EString = "Core energy: {:.4f} +- {:.4f} eV/angstrom".format(Ecore, np.sqrt(cov[0, 0]))
    KString = "Energy coefficient: {:.2f} +- {:.2f} GPa".format(KGPa, errKGPa)
    fittedEnergies.write('# {}; {}\n'.format(EString, KString))
    
    for i, energy in enumerate(E):
        fittedEnergies.write('{:.3f} {:.6f}\n'.format(r[i], energy))
        
    fittedEnergies.close()
    
    # remove single point calculation files
    os.system('rm -f sp.*')
    
    return (Ecore, KGPa), (np.sqrt(cov[0, 0]), errKGPa)
        
### COMMAND LINE/INTERACTIVE USE FUNCTIONS ###   
    
def energy_command_line():
    '''Command line options for a cluster energy calculation. Note that this 
    is not necessary if using <_atomic_control.py> (which accesses <dis_energy>
    directly.
    '''
    
    options = argparse.ArgumentParser()
    options.add_argument('-rmax', type=int, dest='rmax', default=10, 
                            help='Maximum radius of cluster')
    options.add_argument('-rmin', type=int, dest='rmin', default=1,
                            help='Minimum radius of cluster')
    options.add_argument('-dr', type=int, dest='dr', default=1, 
                            help='Radial increment.')
    options.add_argument('-name', type=str, dest='basename', default='dis',
                            help='Base name for single point calculations.')
    options.add_argument('-exec', type=str, dest='executable', default='',
                            help='Location of the GULP executable.')
    options.add_argument('-method', choices=['explicit', 'eregion', 'edge'],
                            default='explicit')
    options.add_argument('-b', type=float, dest='b', default=np.nan, 
                            help='Burgers vector magnitude.')
    options.add_argument('-thick', type=float, dest='thick', default=np.nan, 
                            help='Thickness of simulation cell (along line).')
    options.add_argument('-rc', type=float, dest='rc', default=np.nan,
                            help='Radius of dislocation core.')
    options.add_argument('-k', type=float, default=np.nan, dest='K', 
                            help='Energy coefficient')
    options.add_argument('-fit', choices=[0, 1], type=int, dest='fitk', default=1,
                                help='Fit the energy coefficient (0==F, 1==T)')
    options.add_argument('-atoms', nargs='*', dest='atoms', help="List of atom" +
                            " energies in the perfect crystal.")
    
    return options   
    
def main():
    '''Run the energy program from the command line.
    '''
    
    options = energy_command_line()
    args = options.parse_args()
    
    args.fitk = bool(args.fitk)
    
    # check that all supplied simulation parameters are legal
    if args.executable == '':
        raise AttributeError("No program executable found.")
    if args.b != args.b: # no burgers vector supplied
        raise ValueError("Burgers vector magnitude cannot be NaN.")
    if args.thick != args.thick: # no cell thickness provided
        raise ValueError("Cell thickness cannot be NaN.")
    if (not args.fitk) and (args.K != args.K):
        raise AttributeError("Must provide energy coefficient if <fitk> == False")
        
    atom_dict = handle_atoms(args.atoms)
        
    Ec, K = dis_energy(args.rmax, args.rmin, args.dr, args.basename, args.executable, 
                        args.method, args.b, args.thick, relax_K=args.fitk, K=args.K,
                                                      atom_dict=atom_dict, rc=args.rc)
                
    print(Ec, K)
    
if __name__ == "__main__":
    main()
