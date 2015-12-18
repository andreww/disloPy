#!/usr/bin/env python
from __future__ import print_function

import re
import argparse
import sys

import numpy as np

# dictionary containing regex to match final energies for a variety of codes
energy_lines = {"gulp": re.compile(r"\n\s*Final energy\s+=\s+" +
                                  "(?P<E>-?\d+\.\d+)\s*(?P<units>\w+)\s*\n"),
                "castep":re.compile(r"\n\s*BFGS:\s*Final\s+Enthalpy\s+=\s+" +
                              "(?P<E>-?\d+\.\d+E\+\d+)\s*(?P<units>\w+)\s*\n")
               }
               
get_gnorm = re.compile(r"Final Gnorm\s*=\s*(?P<gnorm>\d+\.\d+)")
               
def command_line_options():
    '''Parse command line options to control extraction of gamma surface.
    '''
    
    options = argparse.ArgumentParser()
    
    options.add_argument("-b","--base-name",type=str,dest="base_name",
                         default="gsf",help="Base name for GSF calculations")
    options.add_argument("-x","--x_max",type=int,dest="x_max",default=0,
                                            help="Number of steps along x")
    options.add_argument("-y","--y_max",type=int,dest="y_max",default=0,
                                            help="Number of steps along y")
    options.add_argument("-p","--program",type=str,dest="program",default="gulp",
                                     help="Program used to calculate GSF energies")
    options.add_argument("-o","--output",type=str,dest="out_name",default="gsf.dat",
                                  help="Output file for the gamma surface energies")
    options.add_argument("-s","--suffix",type=str,dest="suffix",default="out",
                                help="Suffix for ab initio/MM-FF output files")
    options.add_argument("--indir", action="store_true", 
                         help="Each GSF point is in its own directory")
                                  
    return options
               
def get_gsf_energy(energy_regex,base_name,suffix,i,j=None,indir=False):
    '''Extracts calculated energy from a generalized stacking fault calculation
    using the regular expression <energy_regex> corresponding to the code used
    to calculate the GSF energy.

    Set argument indir to True if mkdir was True in gsf_setup.
    '''
    
    if indir:
        # files are in directories named like the files.
        if j is None: # gamma-line
            filename = "%s.%d/%s.%d.%s" % (base_name,i,base_name,i,suffix)
        else: # gamma surface
            filename = "%s.%d.%d/%s.%d.%d.%s" % (base_name,i,j,base_name,i,j,suffix)
    else:
        # Files are in the current directory
        if j is None: # gamma-line
            filename = "%s.%d.%s" % (base_name,i,suffix)
        else: # gamma surface
            filename = "%s.%d.%d.%s" % (base_name,i,j,suffix)
    
    outfile = open(filename)
    output_lines = outfile.read()
    outfile.close()
    matched_energies = re.findall(energy_regex, output_lines)
    
    # flags that we use to see if convergence has failed without resulting in
    # a divergent energy (which would stop GULP).
    flag_failure = ["Conditions for a minimum have not been satisfied",
                    "Too many failed attempts to optimise"]
    
                    
    if not(matched_energies):
        #raise AttributeError("Calculation does not have an energy.")
        E = np.nan
        units = None       
    else:
        if (flag_failure[0] in output_lines) or (flag_failure[1] in output_lines):
            gnorm = float(get_gnorm.search(output_lines).group("gnorm"))
        else:
            gnorm = 0.
            
        if gnorm < 0.2:       
            for match in matched_energies:
                E = float(match[0])
                units = match[1]
        else:
            E = np.nan
            units = None
        
    return E, units
        
def main():
    
    if not sys.argv[1:]:
        raise ValueError("Input arguments must be supplied.")
    else:
        options = command_line_options()
        args = options.parse_args()
        
    # test to make sure that the number of increments along at least one axis
    # has been specified
    gamma_line = False
    if (not args.x_max) and (not args.y_max):
        # number of increments in both directions == 0
        raise ValueError("Number of increments in at least one direction must" +
                                                                      "be >= 1.")    
    elif args.x_max and (not args.y_max):
        gamma_line = True  
        i_max = args.x_max
    elif (not args.x_max) and args.y_max:
        gamma_line = True
        i_max = args.y_max
    else: # gamma surface
        pass
        
    # determine which regular expression to use to match output energies
    try:
        regex = energy_lines[(args.program).lower()]
    except KeyError:
        print("Invalid program name supplied. Implemented programs are: ")
        for prog in energy_lines.keys():
            print("***%s***" % prog)
        
    outstream = open("%s" % args.out_name,"w")    
    
    if gamma_line:
        # handle gamma line
        for x in xrange(i_max+1):
            E,units = get_gsf_energy(regex,args.base_name,args.suffix,i,indir=args.indir)
    else:
        energies = np.zeros((args.x_max+1,args.y_max+1))
        # handle gamma surface
        for i in xrange(args.x_max+1):
            for j in xrange(args.y_max+1):
                E,units = get_gsf_energy(regex,args.base_name,args.suffix,i,j,indir=args.indir)
                energies[i, j] = E

        # get rid of nan values -> Should implement this as a separate function
        E_perfect = energies[0, 0]
        for i in xrange(args.x_max+1):
            for j in xrange(args.y_max+1):
                if energies[i, j] != energies[i, j] or (energies[i, j] < E_perfect - 1.):
                    # average neighbouring energies, excluding nan values        
                    approx_energy = 0.
                    num_real = 0 # tracks number of adjacent non-NaNs
                    for signature in [((i+1) % args.x_max, j),
                                      ((i-1) % args.x_max, j),
                                      (i, (j+1) % args.y_max),
                                      (i, (j-1) % args.y_max)]:
                        if energies[signature] !=  energies[signature]:
                            pass
                        elif energies[signature] < E_perfect - 1.:
                            # energy is nonsense -> probably a sign of weird 
                            # interatomic potentials.
                            pass
                        else:
                            approx_energy += energies[signature]
                            num_real += 1

                    if num_real == 0:
                        #raise Exception("Entry (%d, %d) has no real neighbours" 
                        #                                               % (i, j))
                        print("Warning: no real neighbours.")
                        energies[i, j] = -10000
                    else:
                        energies[i, j] = approx_energy/num_real
                    print("({}, {}): {:.2f}".format(i, j, energies[i, j]))

                outstream.write("%d %d %.4f\n" % (i,j,energies[i,j]))
                
            # include a space between slices of constant x (for gnuplot)
            outstream.write("\n")
    
    outstream.close()
        
if __name__ == "__main__":
    main()
        
                                                                       
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      

