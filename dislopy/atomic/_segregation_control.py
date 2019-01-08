#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import re
import os
import sys
import argparse

import numpy as np

from shutil import copyfile
from numpy.linalg import norm

supported_codes = ('gulp')

# note that this uses a slightly modified version of <control_file>, to allow
# the user to supply multiple impurity atoms of the same species, as is the case,
# for instance, in protonated vacancies in NAMs.
from dislopy.utilities.control_functions import from_mapping, change_type, to_bool, \
                                                    change_or_map, print_control   

# dislopy specific stuff
from dislopy.atomic import crystal as cry
from dislopy.atomic import gulpUtils as gulp
from dislopy.atomic import multisite as ms
from dislopy.atomic import rodSetup as rs
from dislopy.atomic import transmutation as mut
from dislopy.atomic import segregation as seg
from dislopy.atomic import migration_cluster as mig

from dislopy.atomic._atomic_control import vector

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
                    new_impurity = []
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
                    impurities.append(new_impurity)
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
                            
                    new_impurity.append(new_atom)
                    
    return sim_parameters, impurities

def handle_segregation_control(param_dict):
    '''Handle all possible cards for a segregation energy calculation. As in
    other control files used in <dislopy>, the use of <None> as the default value
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
                     ('calc_type', {'default': None, 'type': str}), # NEW
                     ('region_r', {'default': 10, 'type': float}),
                     ('new_r1', {'default': 10, 'type': int}),
                     ('centre_on_impurity', {'default': True, 'type': to_bool}),
                     ('n', {'default': 1, 'type': int}),
                     ('analyse', {'default': False, 'type': to_bool}),
                     ('no_setup', {'default': False, 'type': to_bool}),
                     ('migration', {'default': False, 'type': to_bool}),
                     ('parallel', {'default': False, 'type': to_bool}),
                     ('np', {'default': 1, 'type': int}),                    
                     ('collate', {'default': True, 'type': to_bool}),
                     ('site', {'default': None, 'type': str}),
                     ('label', {'default': 'dfct', 'type': str}),
                     ('suffix', {'default': 'in', 'type': str}),
                     ('uses_hydroxyl', {'default': False, 'type': to_bool}),
                     ('o_str', {'default': 'O', 'type': str}),
                     ('oh_str', {'default': 'Oh', 'type': str})
                    )
                     
    # cards for the <&migration> namelist
    migration_cards = (('do_calc', {'default': True, 'type': to_bool}),
                       ('no_setup', {'default': False, 'type': to_bool}),
                       ('npoints', {'default': 3, 'type': int}),
                       ('node', {'default': 0.5, 'type': float}),
                       ('plane_shift', {'default': np.zeros(2), 'type': vector}),
                       ('threshold', {'default': 0.5, 'type': float}),
                       ('plot_migration', {'default': False, 'type': to_bool}),
                       ('new_species', {'default': '', 'type': str}),
                       ('find_bonded', {'default': False, 'type': to_bool}),
                       ('dx_thresh', {'default': np.nan, 'type': float}),
                       ('pipe_only', {'default': True, 'type': to_bool}),
                       ('has_mirror_symmetry', {'default': False, 'type': to_bool})
                      )
                    
    # cards for the <&constraints> namelist
    constraints_cards = (('height_min', {'default':np.nan, 'type':float}),
                         ('height_max', {'default':np.nan, 'type':float}),
                         ('phi_min', {'default':np.nan, 'type':float}),
                         ('phi_max', {'default':np.nan, 'type':float}),
                         ('x0', {'default': 0., 'type': float}), #NEW
                         ('y0', {'default': 0., 'type': float}) #NEW
                        )
                        
    # cards for the <&analysis> namelist
    analysis_cards = (('E0', {'default': np.nan, 'type': float}),
                      ('dE0', {'default': np.nan, 'type': float}),
                      ('mirror', {'default': False, 'type': to_bool}),
                      ('axis', {'default': 1, 'type': int}),
                      ('mirror_both', {'default': False, 'type': to_bool}),
                      ('inversion', {'default': False, 'type': to_bool}),
                      ('do_fit', {'default': True, 'type': to_bool}),
                      ('fit_r', {'default': 2.0, 'type': float}),
                      ('tolerance', {'default': 1.0, 'type': float}),
                      ('plot_scatter', {'default': True, 'type': to_bool}),
                      ('plot_contour', {'default': True, 'type': to_bool}),
                      ('plot_name', {'default': '', 'type': int}),
                      ('figformat', {'default': 'tif', 'type': str}),
                      ('vmin', {'default': np.nan, 'type': float}),
                      ('vmax', {'default': np.nan, 'type': float}),
                      ('nticks', {'default': 4, 'type': int})
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
        
        self.sim, self.impurities = control_file_seg(filename)
        handle_segregation_control(self.sim)
        
        # if <self.impurities> is empty, the user will want to run a vacancy 
        # segregation calculation. Add an empty list to <self.impurities> to 
        # represent this vacancy    
        if len(self.impurities) == 0:
            self.impurities.append([])
        
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
        
        # index to track for which defect in <self.impurities> we are 
        # calculating energies.
        self.current_index = 0
        
        # if no executable has been provided, assume that the user wishes only
        # to create the input files, not run them
        if self.control('executable') and self.control('do_calc'):
            do_calc = True
        elif self.control('do_calc') and not self.control('executable'):
            raise Warning('Atomistic calculation requested but no executable ' +
                          'provided. Setting <do_calc> to False.')
            do_calc = False
        else:
            do_calc = False
        
        if not self.control('no_setup'):
            # check that the user has supplied a .grs file containing a dislocation
            # user may simply not want to create input files, eg. if simulations
            # have already been run on another machine and they just want to 
            # parse the results
            if not self.control('dislocation_file'):
                raise ValueError('Must supply a dislocation.')
            
            # construct point defect-free simulation cell, together with the 
            # appropriate set of contraints
            
            if self.control('calc_type').lower() == 'cluster':     #NEW
                self.make_cluster()
            elif self.control('calc_type').lower() == 'multipole': #NEW
                self.make_supercell() #NEW
            else: #NEW
                raise ValueError('{} is an invalid simulaion type'.format(self.control('calc_type'))) #NEW 
                
            self.form_constraints()

        # run separately for 1 vs. multiple defect configurations
        if len(self.impurities) == 1:    
            if not self.control('no_setup'):
                # create defect and input files
                self.make_defect()
                self.segregate_defect()
            if do_calc:
                self.calculate_defect()
        else:    
            for i in range(len(self.impurities)): 
                # make and segregate the defect
                self.current_index = i
                dirname = '{}{:.0f}'.format(self.control('label'), self.current_index)
                if not self.control('no_setup'):
                    # make subdirectories for each configuration
                    if not os.path.exists(dirname):
                        os.mkdir(dirname)             
                    elif not os.path.isdir(dirname):
                        # make sure that <dirname> is a directory
                        raise TypeError('{} exists and is not a directory.'.format(dirname))
                    
                    os.chdir(dirname)
                    
                    # create defect and generate input files
                    self.make_defect()
                    self.segregate_defect()  
                else:
                    # change to <dirname> for calculations, but first check that it exists
                    if os.path.exists(dirname):
                        os.chdir(dirname)
                    else:
                        raise NameError('{} does not exist.'.format(dirname))
                         
                if do_calc:
                    self.calculate_defect()
              
                # move to the next defect configuration
                os.chdir('../')
        
        # if the user wishes the output to be analysed, do so now -> check that
        # the calculation has been run
        if self.control('analyse') and self.control('executable'):
            if len(self.impurities) == 1:
                self.analyse_results()  
            else:
                # start by analyzing the results for each configuration 
                # individually
                for i in range(len(self.impurities)):
                    # analyze results for configuration <i>
                    self.current_index = i
                    os.chdir('{}{:.0f}'.format(self.control('label'), self.current_index))
                    self.analyse_results()
                    os.chdir('../')
                
                if self.control('collate'):    
                    # collect lowest energy configurations for each atomic site in a
                    # separate directory
                    energies = self.all_config_energies()
                    emin, min_i, degen = self.minimum_energies(energies)
                    self.collate_energies(min_i)
                    self.save_optimum_configs(min_i, emin)
                    
                    # go down into collection directory and analyse PES
                    os.chdir(self.control('label'))
                    self.analyse_results()
                    os.chdir('../')
         
        # calculate migration barriers; at present, migration barriers can be
        # calculated only if the number of defect configurations provided is 1   
        if self.control('migration'):
            if len(self.impurities) == 1:
                self.migration_barriers() 
            else:
                raise ValueError("<self.control('migration')> cannot be True " +
                                 "if more than one defect configuration has " +
                                 "been provided.")
        
    def make_cluster(self):
        '''Constructs the base cluster that will be used in all subsequent
        calculations.
        '''
        
        # extract region I and II radii from the .grs filename
        rmatch = re.search(r'.+.\.(?P<r1>\d+)\.(?P<r2>\d+).(?:grs|gin)',
                                    self.control('dislocation_file'))
        self.r1 = int(rmatch.group('r1'))
        self.r2 = int(rmatch.group('r2'))
        
        base_clus, self.sysinfo = gulp.cluster_from_grs(self.control('dislocation_file'),
                                          self.r1, self.r2, new_rI=self.control('new_r1'))
                                                       
        self.cluster = rs.extend_cluster(base_clus, self.control('n'))
        
    def make_supercell(self): #NEW
        '''Constructs the supercell that will be used in subsequent simulations.
        '''
        
        base_sc = cry.Crystal() #NEW
        self.sysinfo = gulp.parse_gulp(self.control('dislocation_file'), base_sc) #NEW
        
        # extend normal to dislocation line
        self.supercell = cry.superConstructor(base_sc, np.array([1., 1., self.control('n')])) #NEW       
        
    def make_defect(self):
        '''Constructs the Impurity object whose segregation energy surface
        around the dislocation is to be calculated.
        '''
        
        # create empty impurity
        self.dfct = mut.Impurity(self.control('site'), self.control('label'))
        
        # add any impurity atoms supplied by the user
        for atom in self.impurities[self.current_index]:
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
        if self.control('calc_type').lower() == 'cluster':
            H = self.cluster.getHeight()
        else: # supercell 
            H = 1.
            
        # location of the dislocation line
        self.x0 = [self.constraints('x0'), self.constraints('y0')]
        
        # create height constraint. The supplied values of <height_min> and
        # <height_max> are in fractional units.
        if (self.constraints('height_min') is not np.nan) or (self.constraints('height_max') is not np.nan):
            if self.constraints('height_min') is not np.nan:
                hmin = self.constraints('height_min')*H                
            else:
                # note that height_min is NaN -> assume base of cell
                hmin = 0.
                    
            if self.constraints('height_max') is not np.nan:  
                hmax = self.constraints('height_max')*H
            else:
                # assume max height is the top of the unitcell
                hmax = hmin+H/float(self.control('n'))
        else:
            # neither defined -> assume that all sites to be replaced will be 
            # in the lowermost replicate
            hmin = 0.
            hmax = H/float(self.control('n'))
            
        # create constraint function
        self.cons_funcs.append(lambda atom: mut.heightConstraint(hmin, hmax, atom, period=H))
            
        # create azimuthal constraints, if supplied
        if (self.constraints('phi_min') is not np.nan) or (self.constraints('phi_max') is not np.nan):
            if self.constraints('phi_min') is not np.nan:
                fmin = self.constraints('phi_min')
            else:
                # set to lower bound of range of possible values
                fmin = -np.pi
                    
            if self.constraints('phi_max') is not np.nan:
                fmax = self.constraints('phi_max')
            else:
                # assume phi_max is upper bound of range of possible azimuthal values
                fmax = np.pi
                            
            self.cons_funcs.append(lambda atom: mut.azimuthConstraint(fmin, fmax, atom, x0=self.x0))  
        
    def segregate_defect(self):
        '''Creates the input files for the segregation energy calculation.
        '''
        
        # calculate impurity energies
        if self.control('calc_type').lower() == 'cluster':
            use_cell = self.cluster
        else: 
            use_cell = self.supercell
            
        ms.insert_defect(self.sysinfo, 
                         use_cell, 
                         self.control('region_r'),
                         self.dfct, 
                         self.control('new_r1'),
                         calc_type=self.control('calc_type'),
                         suffix=self.control('suffix'),
                         centre_on_impurity=self.control('centre_on_impurity'),
                         constraints=self.cons_funcs,              
                         noisy=self.control('noisy'),
                         contains_hydroxyl=self.control('uses_hydroxyl'),
                         o_str=self.control('o_str'),
                         oh_str=self.control('oh_str'),
                         bonds=self.migration('find_bonded'), 
                         has_mirror_symmetry=self.migration('has_mirror_symmetry'),
                         dx_thresh=self.migration('dx_thresh'),
                         x0 = self.x0
                        )
                                 
    def calculate_defect(self):
        '''Calculates energies of defect-bearing clusters.
        '''
        
        # read in the site IDs
        sitelist = ms.parse_sitelist(self.control('label'), self.control('site'))
        
        # run calculation
        ms.calculate_impurity_energies(sitelist, self.control('executable'), 
                                       in_parallel=self.control('parallel'), 
                                       nprocesses=self.control('np'),
                                       suffix=self.control('suffix'))
                                  
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
                                        fit=self.analysis('do_fit'),
                                        fit_r=self.analysis('fit_r'),
                                        tolerance=self.analysis('tolerance'),
                                        plot_scatter=self.analysis('plot_scatter'),
                                        plot_contour=self.analysis('plot_contour'),
                                        plotname=self.analysis('plot_name'),
                                        figformat=self.analysis('figformat'),
                                        vmin=self.analysis('vmin'),
                                        vmax=self.analysis('vmax'),
                                        nticks=self.analysis('nticks')
                                       )

    ###!!!EXPERIMENTAL!!!### 

    def single_config_energy(self):
        '''Gets a list of defect segregation energies for the <confignum>-th 
        configuration of the defect.
        '''
        
        os.chdir('{}{:.0f}'.format(self.control('label'), self.current_index)) 
        sim_name = '{}.{}'.format(self.control('label'), self.control('site'))
        self.site_info = seg.parse_control(sim_name)
        E = seg.get_energies(sim_name, self.site_info)
        dE = seg.defect_excess_energy(E, self.analysis('E0'), self.control('n'))
        E_seg = seg.segregation_energy(dE, self.analysis('dE0'))
        os.chdir('../')
        return E_seg

    def all_config_energies(self):
        '''Get defect segregation energies for all configurations of the
        defect.'''
        
        # extract energies
        energies = []
        for i in range(len(self.impurities)):
            self.current_index = i
            Eseg_i = self.single_config_energy()
            if i == 0:
                # record length of <i>-th energy list
                e_length = len(Eseg_i)
            else:
                if len(Eseg_i) != e_length:
                    raise ValueError("Number of energies for configuration {:0.f} is incorrect".format(i))
            energies.append(Eseg_i)
        
        # take transpose of energies matrix so that all energies for the same
        # atomic site are on the same row
        energies = np.array(energies)
        energies = energies.T
        return energies

    def minimum_energies(self, energies):
        '''Create a list containing the minimum segregation energy
        for each site among those calculated for different configurations
        of a particular defect.
        '''
        
        if len(energies[0]) == 1:
            # there is only a single configuration - this shouldn't happen
            return energies, np.zeros(len(energies)), None
            
        # otherwise, find the lowest energy for each site
        min_energies = []
        min_indices = []
        degenerates = None
        for i, site in enumerate(energies):
            min_E = min(site)
            min_energies.append(min_E)
            min_j = np.where(site == min_E)[0]
            if len(min_j) == 1:
                min_indices.append(min_j[0])
            else:
                print("Degenerate energies for site {:.0f}".format(i))
                # save the first config to copy to the master directory, and
                # then record which configurations are energy degenerate
                min_indices.append(min_j[0])
                if degenerates is None:
                    degenerates = dict()
                degenerates[i] = min_j
                
        return min_energies, min_indices, degenerates

    def collate_energies(self, min_i, program='gulp'):
        '''Moves the data files associated with the minimum energy configurations <min_i> at each 
        site in <site_info> to a single directory for post-processing.
        '''

        if os.path.exists(self.control('label')):
            if os.path.isdir(self.control('label')):
                pass
            else:
                raise UserWarning('Name for aggregation directory already in use by non-directory.')
        else:
            os.mkdir(self.control('label'))

        if program.lower() == 'gulp':
            suffices = ['gin', 'gout', 'grs', 'xyz']
        else:
            raise ValueError('Currently, programs other than GULP are unsupported for this simulation type.')
        
        sim_base = '{}.{}'.format(self.control('label'), self.control('site'))
        for i, j in zip(min_i, self.site_info):
            site_j = int(j[0])
            for suffix in suffices:
                copyfile('{}{:.0f}/{}.{:.0f}.{}'.format(self.control('label'), i, sim_base, site_j, suffix),
                         '{}/{}.{:.0f}.{}'.format(self.control('label'), sim_base, site_j, suffix))

        # move one of the *.id.txt files into the collation directory
        copyfile('{}0/{}.id.txt'.format(self.control('label'), sim_base), '{}/{}.id.txt'.format(self.control('label'), sim_base))
        
        return
        
    def save_optimum_configs(self, min_i, energies):
        '''Save the optimum configuration (and its energy) of the defect at each
        site in <self.site_info>.
        '''
        
        ostream = open('{}.{}.opt.txt'.format(self.control('label'), self.control('site')), 'w')
        
        # write header
        ostream.write('# site-index x y config-number E\n')
        
        for site, i, E in zip(self.site_info, min_i, energies):
            ostream.write('{:.0f} {:.6f} {:.6f} {:.0f} {:.6f}\n'.format(site[0],
                                                     site[1], site[2], i, E))
                                                     
        ostream.close()
        return

    ###!!!END EXPERIMENTAL!!!###    
                                       
    def migration_barriers(self):
        '''Calculates migration barriers for pipe diffusion at each defect site.
        '''
        
        # basename to use
        basename = '{}.{}'.format(self.control('label'), self.control('site'))
        
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
            if self.migration('pipe_only'):   
                heights = mig.migrate_sites_pipe(basename, 
                                                 self.control('new_r1'),  
                                                 self.r2, 
                                                 self.control('site'), 
                                                 self.migration('npoints'), 
                                                 executable=executable, 
                                                 node=self.migration('node'),
                                                 plane_shift=self.migration('plane_shift'),
                                                 threshold=self.migration('threshold'),
                                                 newspecies=newspecies,
                                                 noisy=self.control('noisy'),
                                                 centre_on_impurity=self.control('centre_on_impurity'),
                                                 in_parallel=self.control('parallel'),
                                                 nprocesses=self.control('np')
                                                )
            else:
                heights = mig.migrate_sites_general(basename, 
                                                    self.control('new_r1'),  
                                                    self.r2, 
                                                    self.control('site'), 
                                                    self.migration('npoints'), 
                                                    executable=executable, 
                                                    node=self.migration('node'),
                                                    plane_shift=self.migration('plane_shift'),
                                                    threshold=self.migration('threshold'),
                                                    newspecies=newspecies,
                                                    noisy=self.control('noisy'),
                                                    centre_on_impurity=self.control('centre_on_impurity'),
                                                    in_parallel=self.control('parallel'),
                                                    nprocesses=self.control('np')
                                                   )
                                   
        # read in energies output by non-adaptive calculations
        try:
            if heights is None:
                heights = mig.extract_barriers_even(basename, self.migration('npoints'))
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
                                                                                                   
def main():
    '''Runs a segregation simulation.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, nargs='?', dest='filename', default='0')
    
    args = parser.parse_args()
    if args.filename != 0:
        new_sim = SegregationSim(args.filename)
    else:
        # read in filename from the command line
        if sys.version_info.major == 2:
            filename = raw_input('Enter name of input file: ')
        elif sys.version_info.major == 3:
            filename = input('Enter name of input file: ')
            
        new_simulation = SegregationSim(filename)

if __name__ == "__main__":
    main()
