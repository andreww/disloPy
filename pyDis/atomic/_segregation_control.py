#!/usr/bin/env python
from __future__ import print_function, division

import re
import os
import sys
sys.path.append(os.environ['PYDISPATH'])

import numpy as np

supported_codes = ('gulp')

# note that this uses a slightly modified version of <control_file>, to allow
# the user to supply multiple impurity atoms of the same species, as is the case,
# for instance, in protonated vacancies in NAMs.
from pyDis.pn._pn_control import from_mapping, change_type, to_bool, change_or_map, print_control   

# pyDis specific stuff
from pyDis.atomic import crystal as cry
from pyDis.atomic import gulpUtils as gulp
from pyDis.atomic import multisite as ms
from pyDis.atomic import rodSetup as rs
from pyDis.atomic import transmutation as mut
from pyDis.atomic import segregation as seg
from pyDis.atomic import migration_cluster as mig

from pyDis.atomic._atomic_control import vector

def control_file_seg(filename):
    '''Opens the control file <filename> for a segregation calculation and makes
    the dictionary containing the (unformatted) values for all necessary 
    parameters, in addition to a list of impurity atoms. 
    '''
    
    sim_parameters = dict()
    impurities = []
    
    # regex for simulation parameters, and the different types of atom
    input_style = re.compile('\s*(?P<par>.+)\s*=\s*[\'"]?(?P<value>[^\'"]*)[\'"]?\s*;')
    core_regex = re.compile('^\s*(?P<symbol>\w+)(?P<coords>(?:\s+-?\d+\.\d*){3});')
    shel_regex = re.compile('^\s*(?P<symbol>\w+)\s+(?P<sheltype>(bshe|shel))'+
                                          '(?P<coords>(?:\s+-?\d+\.\d*){3});')
    in_namelist = False
    in_defects = False
    with open(filename) as f:
        for line in f:
            temp = line.strip()
            # handle the line
            if temp.startswith('#'): # comment
                pass
            elif temp.startswith('&'): # check to see if <line> starts a namelist
                card_name = re.search('&(?P<name>\w+)\s+{', temp).group('name')
                if card_name == 'defects':
                    in_defects = True
                else: # regular namelist 
                    sim_parameters[card_name] = dict()
                    in_namelist = True
            elif in_namelist:
                if '};;' in temp: # end of namelist
                    in_namelist = False
                elif not(temp.strip()):
                    # empty line
                    pass
                else: # regular namelist
                    inp = re.search(input_style, temp)
                    sim_parameters[card_name][inp.group('par').strip()] = \
                                                        inp.group('value')
            elif in_defects:
                if '};;' in temp: # end of <defects>
                    in_defects = False
                elif not temp:
                    # empty line
                    pass
                else:
                    # extract atoms and construct
                    new_atom = dict()
                    is_core = core_regex.search(temp)
                    is_shel = shel_regex.search(temp)
                    if is_core:
                        new_atom['symbol'] = is_core.group('symbol')
                        new_atom['coords'] = is_core.group('coords')
                        new_atom['has_shel'] = False
                        new_atom['is_bshe'] = False
                    elif is_shel:
                        new_atom['symbol'] = is_shel.group('symbol')
                        new_atom['coords'] = is_shel.group('coords')
                        # add appropriate polarizable shell stuff -> solely
                        # for GULP and (eventually) DL_POLY
                        new_atom['has_shel'] = True
                        if new_atom.group('sheltype') == 'bshe':
                            new_atom['is_bshe'] = True
                        else:
                            new_atom['is_bshe'] = False
                            
                    impurities.append(new_atom)
                    
    return sim_parameters, impurities

def handle_segregation_control(param_dict):
    '''Handle all possible cards for a segregation energy calculation. As in
    other control files used in <pyDis>, the use of <None> as the default value
    for a parameter denotes that parameter as mission critical, and the program
    will abort if no specific value has been provided. <program> presently 
    defaults to GULP, but this will likely change as support for other atomistic
    simulation codes is added.
    ''' 
    
    # cards for the <&control> namelist
    control_cards = (('dislocation_file', {'default': '', 'type': str}),
                     ('program', {'default': 'gulp', 'type': str}),
                     ('do_calc', {'default': True, 'type': to_bool}),
                     ('noisy', {'default': False, 'type': to_bool}),
                     ('executable', {'default': '', 'type': str}),
                     ('region_r', {'default': 10, 'type': int}),
                     ('new_r1', {'default': 10, 'type': int}),
                     ('centre_on_impurity', {'default': True, 'type': to_bool}),
                     ('site', {'default': None, 'type': str}),
                     ('label', {'default': 'dfct', 'type': str}),
                     ('uses_hydroxyl', {'default': False, 'type': to_bool}),
                     ('o_str', {'default': 'O', 'type': str}),
                     ('oh_str', {'default': 'Oh', 'type': str}),
                     ('n', {'default': 1, 'type': int}),
                     ('analyse', {'default': False, 'type': to_bool}),
                     ('no_setup', {'default': False, 'type': to_bool}),
                     ('migration', {'default': False, 'type': to_bool})
                    )
                     
    # cards for the <&migration> namelist
    migration_cards = (('do_calc', {'default': True, 'type': to_bool}),
                       ('no_setup', {'default': False, 'type': to_bool}),
                       ('npoints', {'default': 3, 'type': int}),
                       ('nlevels', {'default': 1, 'type': int}),
                       ('adaptive', {'default': False, 'type': to_bool}),
                       ('node', {'default': 0.5, 'type': float}),
                       ('plane_shift', {'default': np.zeros(2), 'type': vector}),
                       ('threshold', {'default': 0.5, 'type': float}),
                       ('plot_migration', {'default': False, 'type': to_bool}),
                       ('new_species', {'default': '', 'type': str})
                      )
                    
    # cards for the <&constraints> namelist
    constraints_cards = (('height_min', {'default':np.nan, 'type':float}),
                         ('height_max', {'default':np.nan, 'type':float}),
                         ('phi_min', {'default':np.nan, 'type':float}),
                         ('phi_max', {'default':np.nan, 'type':float})
                        )
                        
    # cards for the <&analysis> namelist
    analysis_cards = (('E0', {'default': np.nan, 'type': float}),
                      ('dE0', {'default': np.nan, 'type': float}),
                      ('mirror', {'default': False, 'type': to_bool}),
                      ('axis', {'default': 1, 'type': int}),
                      ('mirror_both', {'default': False, 'type': to_bool}),
                      ('inversion', {'default': False, 'type': to_bool}),
                      ('plot_scatter', {'default': True, 'type': to_bool}),
                      ('plot_contour', {'default': True, 'type': to_bool}),
                      ('plot_name', {'default': '', 'type': int}),
                      ('figformat', {'default': 'tif', 'type': str}),
                      ('do_fit', {'default': True, 'type': to_bool}),
                      ('tolerance', {'default': 1.0, 'type': float})
                     )
                        
    # read in specific namelists
    namelists = ['control', 'migration', 'constraints', 'analysis']
    
    # initialise namelists
    for name in namelists:
        if name not in param_dict.keys():
            param_dict[name] = dict()
                
    # populate namelists 
    for i, cards in enumerate([control_cards, migration_cards, constraints_cards, 
                                                                analysis_cards]):
        # read in cards specifying sim. parameters
        for var in cards:
            try:
                change_or_map(param_dict, namelists[i], var[0], var[1]['type'])
            except ValueError:
                default_val = var[1]['default']
                # test to see if variable is "mission-critical"
                if default_val is None:
                    raise ValueError("No value supplied for mission-critical" +
                                            " variable {}.".format(var[0]))
                else:
                    param_dict[namelists[i]][var[0]] = default_val
                    # no mapping options, but keep this line to make it easier
                    # to merge with corresponding function in <_pn_control.py>
                    if type(default_val) == str and 'map' in default_val:
                        from_mapping(param_dict, namelists[i], var[0])
                        
    return

class SegregationSim(object):
    '''Handles the control file used to set up the calculations required to 
    calculate the segregation energies of defects occupying a range of sites
    near a dislocation core. 
    '''

    def __init__(self, filename):
        '''Creates dictionaries of simulation parameters and guides the 
        simulation from setup to analysis of results.
        '''
        
        self.sim, self.atoms = control_file_seg(filename)
        handle_segregation_control(self.sim)
        
        # make it easy to access namelists
        self.control = lambda card: self.sim['control'][card]
        self.constraints = lambda card: self.sim['constraints'][card]
        self.analysis = lambda card: self.sim['analysis'][card]
        self.migration = lambda card: self.sim['migration'][card]
        
        if self.control('program') in supported_codes:
            pass
        else:
            raise ValueError('{} is not a supported atomistic simulation code.'.format(self.control('program')) +
                                                     'Supported codes are: GULP.')
                                                     
        # check that the user does not wish to simultaneously invert and mirror
        # the sites
        if self.analysis('mirror') and self.analysis('inversion'):
            raise ValueError('Mirror and inversion symmetry cannot both be True.')
        
        # make cluster
        self.make_cluster()
        
        if not self.control('no_setup'):
            # check that the user has supplied a .grs file containing a dislocation
            if not self.control('dislocation_file'):
                raise ValueError('Must supply a dislocation.')
            
            # create constraint functions
            self.form_constraints()
            
            # make and segregate the defect
            self.make_defect()
            self.segregate_defect()   
        else:
            # user may simply not want to create input files, eg. if simulations
            # have already been run on another machine and they just want to 
            # parse the results
            pass 
        
        # if the user wishes the output to be analysed, do so now -> check that
        # the calculation has been run
        if self.control('analyse') and self.control('executable'):
            self.analyse_results()  
            
        if self.control('migration'):
            self.migration_barriers() 
        
    def make_cluster(self):
        '''Constructs the base cluster that will be used in all subsequent
        calculations.
        '''
        
        # extract region I and II radii from the .grs filename
        rmatch = re.search(r'.+.(?P<r1>\d+)\.(?P<r2>\d+).(?:grs|gin)',
                                    self.control('dislocation_file'))
        self.r1 = int(rmatch.group('r1'))
        self.r2 = int(rmatch.group('r2'))
        
        base_clus, self.sysinfo = gulp.cluster_from_grs(self.control('dislocation_file'),
                                          self.r1, self.r2, new_rI=self.control('new_r1'))
                                                       
        self.cluster = rs.extend_cluster(base_clus, self.control('n'))
        
    def make_defect(self):
        '''Constructs the Impurity object whose segregation energy surface
        around the dislocation is to be calculated.
        '''
        
        # create empty impurity
        self.dfct = mut.Impurity(self.control('site'), self.control('label'))
        
        # add any impurity atoms supplied by the user
        for atom in self.atoms:
            if self.control('program') == 'gulp': # support other programs later
                coords = np.array([float(x) for x in atom['coords'].rstrip().split()])
                new_atom = gulp.GulpAtom(atom['symbol'], coords)
                
                # check to see if atom has a polarizable shell
                if atom['has_shel']:
                    if atom['is_bshe']:
                        shelltype = 'bshe'
                    else:
                        shelltype = 'shel'
                    
                    new_atom.addShell(np.zeros(3), shellType=shelltype)
                    
                # add to impurity
                self.dfct.addAtom(new_atom)                
        
    def form_constraints(self):
        '''Creates functions for any applicable constraints.
        '''
        
        self.cons_funcs = []
        
        # height of simulation cell
        H = self.cluster.getHeight()
        
        # create height constraint. The supplied values of <height_min> and
        # <height_max> are in fractional units.
        if self.constraints('height_min') is not np.nan:
            hmin = self.constraints('height_min')*H
            if self.constraints('height_max') is not np.nan:
                hmax = float(self.constraints('height_max'))*H
            else:
                # assume max height is the top of the unitcell
                hmax = hmin+H/float(self.control('n'))
                
            # create constraint function
            self.cons_funcs.append(lambda atom: mut.heightConstraint(hmin, hmax, 
                                                                atom, period=H))
        elif self.constraints('height_max') is not np.nan:
            # note that height_min is NaN -> assume base of cell
            hmin = 0.
            hmax = self.constraints('height_max')*H
            self.cons_funcs.append(lambda atom: mut.heightConstraint(hmin, hmax, 
                                                                atom, period=H))
            
        # create azimuthal constraints, if supplied
        if self.constraints('phi_min') is not np.nan:
            fmin = float(self.constraints('phi_min'))
            if self.constraints('phi_max') is not np.nan:
                fmax = float(self.constraints('phi_max'))
            else:
                # assume phi_max is upper bound of range of possible azimuthal
                # values
                fmax = np.pi
            self.cons_funcs.append(lambda atom: mut.azimuthConstraint(fmin, fmax, atom))
        elif self.constraints('phi_max') is not np.nan:
            # phi_min guaranteed to be NaN, set to lower bound of range of 
            # possible values
            fmin = -np.pi
            fmax = float(self.constraints('phi_max'))
            self.cons_funcs.append(lambda atom: mut.azimuthConstraint(fmin, fmax, atom))
        
    def segregate_defect(self):
        '''Creates the input files for the segregation energy calculation.
        '''
        
        # if no executable has been provided, assume that the user wishes only
        # to create the input files, not run them
        if self.control('executable') and self.control('do_calc'):
            do_calc = True
        else:
            do_calc = False
        
        if self.control('uses_hydroxyl'):
            ms.calculate_hydroxyl(self.sysinfo, 
                                  self.cluster, 
                                  self.control('region_r'),
                                  self.dfct, 
                                  do_calc=do_calc, 
                                  gulpexec=self.control('executable'),
                                  centre_on_impurity=self.control('centre_on_impurity'),
                                  constraints=self.cons_funcs,
                                  o_str=self.control('o_str'),
                                  oh_str=self.control('oh_str'),
                                  noisy=self.control('noisy')
                                 )
        else:
            # no need to test for hydroxyl-bonded oxygen ions
            gulp.calculateImpurity(self.sysinfo, 
                                   self.cluster, 
                                   self.control('region_r'),
                                   self.dfct, 
                                   do_calc=do_calc, 
                                   gulpexec=self.control('executable'),
                                   centre_on_impurity=self.control('centre_on_impurity'),
                                   constraints=self.cons_funcs,
                                   noisy=self.control('noisy')
                                  )
                                  
    def analyse_results(self):
        '''Analyses the output of the segregation calculation, getting segregation
        energies, fitting the size effect and inhomogeneity terms of the elastic
        energy, and plotting the segregation energy surface.
        '''
        
        seg.analyse_segregation_results('{}.{}'.format(self.control('label'), 
                                        self.control('site')), 
                                        self.analysis('E0'), 
                                        self.analysis('dE0'),
                                        self.control('n'), 
                                        self.control('region_r'),
                                        mirror=self.analysis('mirror'),
                                        mirror_axis=self.analysis('axis'),
                                        mirror_both=self.analysis('mirror_both'), 
                                        inversion=self.analysis('inversion'),
                                        plot_scatter=self.analysis('plot_scatter'),
                                        plot_contour=self.analysis('plot_contour'),
                                        plotname=self.analysis('plot_name'),
                                        figformat=self.analysis('figformat'),
                                        fit=self.analysis('do_fit'),
                                        tolerance=self.analysis('tolerance')
                                       )
                                       
    def migration_barriers(self):
        '''Calculates migration barriers for pipe diffusion at each defect site.
        '''

        # basename to use
        basename = '{}.{}'.format(self.control('label'), self.control('site'))
        
        if self.migration('adaptive'):
            npar = self.migration('nlevels')
            
        else:
            npar = self.migration('npoints')
        
        if self.migration('do_calc'):                            
            executable = self.control('executable')
        else:
            # user may want just to read in results
            executable = None
            
        # check to see if the user wants to migrate an impurity
        if self.migration('new_species'):
            newspecies = self.migration('new_species')
        else:
            newspecies = None
        
        heights= []
        if not self.migration('no_setup'):    
            heights = mig.migrate_sites(basename, 
                                        self.control('n'), 
                                        self.control('new_r1'),  
                                        self.r2, 
                                        self.control('site'), 
                                        npar, 
                                        executable=executable, 
                                        node=self.migration('node'),
                                        plane_shift=self.migration('plane_shift'),
                                        adaptive=self.migration('adaptive'),
                                        threshold=self.migration('threshold'),
                                        newspecies=newspecies,
                                        noisy=self.control('noisy')
                                       )
                                   
        # read in energies output by non-adaptive calculations
        try:
            if not self.migration('adaptive') and not heights:
                heights = mig.extract_barriers_even(basename, npar)  
        except IOError:
            return
            
        # write barrier heights to file
        mig.write_heights(basename, heights)  
        
        if self.migration('plot_migration'):  
            mig.plot_barriers(heights, 
                              'barrier.{}'.format(self.analysis('plot_name')),
                              self.control('region_r'), 
                              mirror_both=self.analysis('mirror_both'),
                              mirror=self.analysis('mirror'), 
                              mirror_axis=self.analysis('axis'),
                              inversion=self.analysis('inversion'),
                              tolerance=self.analysis('tolerance')
                             )
                                                                                                   
def main(filename):
    new_simulation = SegregationSim(filename)

if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except IndexError:
        pass
