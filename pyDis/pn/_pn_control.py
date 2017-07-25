#!/usr/bin/env python
from __future__ import absolute_import, print_function

import sys

import numpy as np
import re
import matplotlib.pyplot as plt
from numpy.linalg import norm

# PN modules from pyDis package
from pyDis.pn import pn_1D as pn1
from pyDis.pn import pn_2D as pn2
from pyDis.pn import fit_gsf as fg
from pyDis.pn import peierls_barrier as pb
from pyDis.pn import energy_coeff as coeff
from pyDis.pn import visualise_pn as vp
from pyDis.atomic import aniso
from pyDis.utilities.control_functions import control_file, from_mapping, change_type,  \
                                            to_bool, change_or_map, print_control  

atomic_to_GPa =  160.2176487
    
def to_vector(in_str, datatype=float):
    '''Converts <in_str>, which is a list of numbers, in an array with elements
    of type float.
    '''
    
    new_vec = []
    for x in re.finditer('-?\d+\.?\d*', in_str):
        new_vec.append(datatype(x.group()))
    
    return np.array(new_vec, dtype=datatype)
    
def to_int_vector(in_str):
    '''Calls <to_vector> with datatype=int. Used for parsing the PN input file.
    '''
    
    arr = to_vector(in_str, datatype=int)
    return arr

def handle_pn_control(param_dict):
    '''handle each possible card in turn. If default value is <None>, the card is 
    considered "mission critical" and the program will abort if a value is not 
    provided.
    
    TO DO: List defining control parameters
    '''

    # cards for the <&control> namelist. If <run_sim> is False, we won't actually
    # run the PN simulation. Do this only if you want to visualise the results
    # without re-running the simulation
    control_cards = (('run_sim', {'default':True, 'type':to_bool}),
                     ('gsf_file', {'default':None, 'type':str}),
                     ('output', {'default':'pyDisPN.out', 'type':str}),
                     ('title_line', {'default':'Peierls-Nabarro model', 'type':str}),
                     ('n_iter', {'default':1, 'type':int}),
                     ('dimensions', {'default':2, 'type':int}),
                     ('max_x', {'default':100, 'type':int}),
                     ('n_funcs', {'default':np.nan, 'type':int}),
                     ('disl_type', {'default':'edge', 'type':str}),
                     ('use_sym', {'default':True, 'type':to_bool}),
                     ('plot', {'default':False, 'type':to_bool}),
                     ('plot_name', {'default':'pn_plot.tif', 'type':str}),
                     ('plot_both', {'default':False, 'type':to_bool}),
                     ('visualize', {'default':False, 'type':to_bool}),
                     ('record', {'default':True, 'type':to_bool}),
                     ('nplanes', {'default':20, 'type':int}),
                     ('noisy', {'default': False, 'type': to_bool}),
                   )

    # cards for the <&struc> namelist. Need to find some way to incorporate 
    # anisotropic elasticity
    struc_cards = (('a', {'default':None, 'type':float}),
                   ('b', {'default':'map struc > a: x -> x', 'type':float}),
                   ('burgers', {'default':'map struc > a: x -> x', 'type':float}),
                   ('spacing', {'default':'map struc > a: x -> x', 'type':float}),
                   ('parameters', {'default':np.array([]), 'type':to_vector})
                  )
                  
    # cards for the &elast namelist. <k_para> and <k_norm> allow the user to 
    # supply predetermined elastic coefficients for the systems parallel and 
    # perpendicular to the Burgers vector
    elast_cards = (('bulk', {'default':None, 'type':float}),
                   ('shear', {'default':None, 'type':float}),
                   ('poisson', {'default':None, 'type':float}),
                   ('k_parallel', {'default':None, 'type':float}),
                   ('k_normal', {'default':np.nan, 'type':float}),
                   ('cij', {'default':None, 'type':aniso.readCij}),
                   ('b_edge', {'default':np.array([1., 0., 0.]), 'type':to_vector}),
                   ('b_screw', {'default':np.array([0., 0., 1.]), 'type':to_vector}),
                   ('normal', {'default':np.array([0., 1., 0.]), 'type':to_vector}),
                   ('in_gpa', {'default':True, 'type':to_bool})
                  )

    # cards for the <&surface> namelist. Where dimensions == 1 but the gsf 
    # file is two-dimensional (ie. a gamma surface), <map_ux> corresponds to the 
    # gamma line direction. If <dimensions> == 2, <map_ux> (<max_uy>) is the
    # direction of the edge (screw) component of the displacement. 
    surf_cards = (('x_length', {'default':'map struc > a: x -> x', 'type':float}),
                  ('y_length', {'default':'map struc > b: x -> x', 'type':float}),
                  ('angle', {'default':np.pi/2, 'type':float}),
                  ('map_ux', {'default':'remap: (ux, uy) -> ux', 'type':str}),
                  ('map_uy', {'default':'remap: (ux, uy) -> uy', 'type':str}),
                  ('gamma_shift', {'default':0., 'type':float}),
                  ('has_vacuum', {'default':True, 'type':to_bool}),
                  ('do_fourier', {'default':False, 'type':to_bool}),
                  ('fourier_N', {'default':3, 'type': int}),
                  ('fourier_M', {'default':-1, 'type':int})
                 )

    # cards for the <&properties> namelist
    prop_cards = (('max_rho', {'default':True, 'type':to_bool}),
                  ('width', {'default':True, 'type':to_bool}),
                  ('center', {'default':True, 'type':to_bool})
                 )
    
    # cards for the <&stress> namelist
    stress_cards = (('calculate_stress', {'default':True, 'type':to_bool}),
                    ('calculate_barrier', {'default':False, 'type':to_bool}),
                    ('dtau', {'default':0.0001, 'type':float}),
                    ('use_GPa', {'default':True, 'type':to_bool}),
                    ('threshold', {'default':0.5, 'type':float})
                   )
                   
    # cards for the <&visualize> namelist
    vis_cards = (('xyz_name', {'default':'pn.xyz', 'type':str}),
                 ('unitcell', {'default':'', 'type':str}),
                 ('path_to_cell', {'default':'./', 'type':str}),
                 ('threshold', {'default':1., 'type':float}),
                 ('sym_thresh', {'default':0.3, 'type':float}),
                 ('radius', {'default':10., 'type':float}),
                 ('program', {'default':'', 'type':str}),
                 ('shift', {'default':np.zeros(3), 'type':to_vector}),
                 ('permutation', {'default':np.array([0, 1, 2], dtype=int), 'type':to_int_vector}),
                 ('thickness', {'default': 1, 'type':int})
                )
    
    # list of valid namelists 
    namelists = ['control', 'struc', 'elast', 'surface', 'properties', 'stress',
                                                        'visualize']
    # check that all namelists were in the control file and add them as keys
    # (with empty dicts as values) in the control dictionary
    for name in namelists:
        if name not in param_dict.keys():
            param_dict[name] = dict()

    # handle the input namelists, except for the elasticity namelist, which
    # requires special handling.
    for i, cards in enumerate([control_cards, struc_cards, elast_cards, surf_cards, 
                                               prop_cards, stress_cards, vis_cards]):
        if namelists[i] == 'elast':
            # handle the elastic parameters later
            continue
        # else    
        for var in cards:
            try:
                change_or_map(param_dict, namelists[i], var[0], var[1]['type'])
            except ValueError:
                default_val = var[1]['default']
                # test to see if variable is "mission-critical" using None != None
                if default_val is None:
                    raise ValueError("No value supplied for mission-critical" +
                                                 " variable {}.".format(var[0]))
                else:
                    param_dict[namelists[i]][var[0]] = default_val
                    # if <default_val> is a mapping, needs to be evaluated
                    if type(default_val) == str and 'map' in default_val:
                        from_mapping(param_dict, namelists[i], var[0])
                        
    # extract values from the &elast namelist
    for var in elast_cards:
        try:
            change_or_map(param_dict, 'elast', var[0], var[1]['type'])
        except ValueError:
            param_dict['elast'][var[0]] = var[1]['default']
                                         
    # test to make sure that all of the parameters required for one of isotropic
    # or anisotropic elasticity have been supplied. Should we also test to see
    # which function to use when calculating the energy coefficients?
    if not (param_dict['elast']['k_parallel'] is None):
        param_dict['elast']['coefficients'] = 'specific'
    elif not (param_dict['elast']['cij'] is None):
        # if an elastic constants tensor is given, use anisotropic elasticity
        param_dict['elast']['coefficients'] = 'aniso'
    elif not (param_dict['elast']['shear'] is None):
        # test to see if another isotropic elastic property has been provided. 
        # Preference poisson's ratio over bulk modulus, if both have been provided
        if not (param_dict['elast']['poisson']  is None):
            param_dict['elast']['coefficients'] = 'iso_nu'
        elif not (param_dict['elast']['bulk'] is None):
            param_dict['elast']['coefficients'] = 'iso_bulk'
        else:
            raise AttributeError("No elastic properties have been provided.")
    else:
        raise AttributeError("No elastic properties have been provided.")
            
    return
    
class PNSim(object):
    # handles the control file for a PN simulation in <pyDis>
    
    def __init__(self, filename):
        self.sim = control_file(filename)
        handle_pn_control(self.sim)
        
        # functions to access values in namelists easily
        self.control = lambda card: self.sim['control'][card]
        self.struc = lambda card: self.sim['struc'][card]
        self.elast = lambda card: self.sim['elast'][card]
        self.surf = lambda card: self.sim['surface'][card]
        self.prop = lambda card: self.sim['properties'][card]
        self.stress = lambda card: self.sim['stress'][card]
        self.vis = lambda card: self.sim['visualize'][card]
        
        if self.control('run_sim'):
            # calculate the energy coefficients
            if self.elast('coefficients') == 'specific':          
                self.K = coeff.predefined(self.elast('k_parallel'), self.elast('k_normal'),
                                                    using_atomic=not(self.elast('in_gpa')))
            elif self.elast('coefficients') == 'aniso':
                self.K = coeff.anisotropic_K(self.elast('cij'),
                                             self.elast('b_edge'),
                                             self.elast('b_screw'),
                                             self.elast('normal'),
                                             using_atomic=not(self.elast('in_gpa'))
                                            )
            elif self.elast('coefficients') == 'iso_nu':
                self.K = coeff.isotropic_nu(self.elast('poisson'), self.elast('shear'),
                                                 using_atomic=not(self.elast('in_gpa')))
            elif self.elast('coefficients') == 'iso_bulk':
                self.K = coeff.isotropic_K(self.elast('bulk'), self.elast('shear'),
                                               using_atomic=not(self.elast('in_gpa')))
                                               

            # if restricting calculation to one component of displacement, extract
            # the appropriate energy coefficient.    
            if self.control('dimensions') == 1:
                # store a copy of the edge and screw energy coefficients
                self.K_store = self.K
                if self.control('disl_type').lower() == 'edge':
                    self.K = self.K[0]
                else:
                    # for the moment, default to shear
                    self.K = self.K[1]
            
            # construct the gamma surface/gamma line
            if self.control('noisy'):
                print("Constructing gamma surface/line")
            self.construct_gsf()
            
            # calculate fit parameters and energy of the dislocation
            if self.control('noisy'):
                print("Relaxing dislocation core structure.")
                 
            self.run_sim()
            
            if self.control('noisy'):
                print("Core relaxation complete.")
                
            # calculate Peierls stress and Peierls barrier, if requested
            if self.control('noisy'):
                print('Calculating Peierls stress.')
                
            self.peierls() 
                   
            if self.control('noisy'):
                print('Peierls stress calculated.')
            
            # do post-processing and/or plotting, if requested by the user
            if self.control('noisy'):
                print('Doing post-processing.')       
            self.post_processing()

            # write results to ouput
            self.write_output()
        
        if self.control('visualize'):
            if self.control('noisy'):
                print('Visualising core structure.')
            self.visualise()
        
    def run_sim(self):
        '''Calculates an optimum(ish) dislocation structure.
        '''
        
        # check to see if the user has provided parameters specifying a trial
        # disregistry profile (in the order A x0 c).
        if len(self.struc('parameters')) != 0:
            # use supplied parameters and perform only a single energy minimization
            inpar = self.struc('parameters')
            niter = 1
        else:
            # no parameters -> tell program that it should generate a trial 
            # configuration
            inpar = None
            niter = self.control('n_iter')
            
        if self.control('dimensions') == 1:
            # calculate dislocation energy and disregistry profile
            self.E, self.par = pn1.run_monte1d(
                                               niter,
                                               self.control('n_funcs'),
                                               self.K,
                                               max_x=self.control('max_x'),
                                               energy_function=self.gsf,
                                               use_sym=self.control('use_sym'),
                                               b=self.struc('burgers'),
                                               spacing=self.struc('spacing'),
                                               noisy=self.control('noisy'),
                                               params=inpar
                                              ) # Done Monte Carlo 1D
            
        elif self.control('dimensions') == 2:
            self.E, self.par = pn2.run_monte2d(
                                               niter,
                                               self.control('n_funcs'),
                                               self.control('disl_type'),
                                               self.K, 
                                               max_x=self.control('max_x'),
                                               energy_function=self.gsf,
                                               use_sym=self.control('use_sym'),
                                               b=self.struc('burgers'),
                                               spacing=self.struc('spacing'),
                                               noisy=self.control('noisy'),
                                               params=inpar
                                              ) # Done Monte Carlo 2D              
                                                            
        return
                                
    def construct_gsf(self):
        '''Makes a function mapping disregistry values to misfit energies.
        '''
    
        gsf_grid, self.units = fg.read_numerical_gsf(self.control('gsf_file'))

        # determine which fitting function to use from the shape of <gsf_grid>
        # original test was self.control("dimensions")
        n_columns = np.shape(gsf_grid)[-1]
        if n_columns == 2:
            self.gsf = fg.spline_fit1d(gsf_grid, self.surf('x_length'), 
                                                 self.surf('y_length'),
                                                 angle=self.surf('angle'),
                                                 units=self.units,
                                                 hasvac=self.surf('has_vacuum'),
                                                 do_fourier_fit=self.surf('do_fourier'),
                                                 n_order=self.surf('fourier_N'))
                                                 
        elif n_columns == 3: # 2-dimensional misfit function
            # fit the gamma surface
            if self.control('dimensions') == 2:
                if self.surf('fourier_M') < 0:
                    m_order = self.surf('fourier_N')
                else:
                    m_order = self.surf('fourier_M')
                    
                base_func = fg.spline_fit2d(gsf_grid, self.surf('x_length'), 
                                                      self.surf('y_length'),
                                                   angle=self.surf('angle'),
                                                           units=self.units,
                                             hasvac=self.surf('has_vacuum'),
                                     do_fourier_fit=self.surf('do_fourier'),
                                             n_order=self.surf('fourier_N'),
                                             m_order=m_order)
                                                  
                self.gsf = fg.new_gsf(base_func, self.surf('map_ux'), self.surf('map_uy'))
                            
            elif self.control('dimensions') == 1: 
                # begin by performing a spline fit to the surface 
                base_func = fg.spline_fit2d(gsf_grid, self.surf('x_length'), 
                                                      self.surf('y_length'),
                                                   angle=self.surf('angle'),
                                                           units=self.units,
                                             hasvac=self.surf('has_vacuum'),
                                                       do_fourier_fit=False)
                
                # remap input so that x and y correspond to ux and uy                                       
                temp_gsf = fg.new_gsf(base_func, self.surf('map_ux'), self.surf('map_uy'))
                
                # project out the dislocation parallel component
                # determine which axis to use (0/x if edge, 1/y if screw)
                if self.control('disl_type') == 'edge':
                    use_axis = 0
                elif self.control('disl_type') == 'screw':
                    use_axis = 1
                    
                temp_1d = fg.projection(temp_gsf, self.surf('gamma_shift'), use_axis)
                
                # set the correct periodicity
                temp_1d2 = lambda x: temp_1d(x % self.struc('burgers'))
                
                if self.surf('do_fourier'):
                    # fit a fourier series to <temp_1d2>.
                    self.gsf = fg.gline_fourier(temp_1d, self.surf('fourier_N'),
                                                           self.struc('burgers'))
                else:
                    self.gsf = temp_1d2
                     
        else:
            raise ValueError("GSF grid has invalid number of dimensions")

        return
                            
    def post_processing(self):
        '''Calculate the displacement field and dislocation density, record these
        values in appropriate output files, plot them (if requested to by the
        user), and then calculate the dislocation height and width.
        '''
        
        if (self.control('plot') or self.prop('max_rho') or self.prop('width') or
                                                        self.control('record')):
                                                        
            if self.control('record'):
                # open output files for the disregistry and dislocation density
                outstreamu = open('disreg.{}'.format(self.control('output')), 'w')
                outstreamrho = open('rho.{}'.format(self.control('output')), 'w')
                
            # lattice points
            r = self.struc('spacing')*np.arange(-self.control('max_x'), self.control('max_x'))
            
            # calculate misfit profile and dislocation density and write to output
            # files
            if self.control('dimensions') == 1:
                self.ux = pn1.get_u1d(self.par, self.struc('burgers'), 
                                   self.struc('spacing'), self.control('max_x')) 
                
                # record disregistry and dislocation density
                if self.control('record'):
                    # header for output files
                    outstreamu.write('# r (\AA) u (\AA)\n')
                    outstreamrho.write('# r (\AA) rho\n')
                    
                    # record values of rho and u at each lattice point
                    rho = pn1.rho(self.ux, r) 
                    for i in range(len(r)):
                        outstreamu.write('{:.3f} {:.6f}\n'.format(r[i], self.ux[i]))
                        if i != 0:
                            # rho not defined for first lattice plane
                            # Record at intervals between lattice planes
                            ri = 0.5*(r[i]+r[i-1])
                            outstreamrho.write('{:.3f} {:.6f}\n'.format(ri, rho[i-1]))
                        
                    outstreamu.close()
                    outstreamrho.close()
                    
            elif self.control('dimensions') == 2:
                self.ux, self.uy = pn2.get_u2d(self.par, self.struc('burgers'),
                                    self.struc('spacing'), self.control('max_x'),
                                                      self.control('disl_type'))
                
                # record both components of disregistry and dislocation density
                if self.control('record'):
                    # header lines
                    outstreamu.write('# r (\AA) ux (\AA) uy (\AA)\n')
                    outstreamrho.write('# r (\AA) rhox  rhoy\n')
                    
                    # record values of u and rho at each lattice point
                    rhox = pn1.rho(self.ux, r)
                    rhoy = pn1.rho(self.uy, r)
                    for i in range(len(r)):
                        outstreamu.write('{:.3f} {:.6f} {:.6f}\n'.format(r[i], self.ux[i],
                                                                              self.uy[i]))
                        if i != 0:
                            # as above, rho not defined for first lattice plane.
                            # Record at intervals between lattice planes
                            ri = 0.5*(r[i]+r[i-1])
                            outstreamrho.write('{:.3f} {:.6f} {:.6f}\n'.format(ri, 
                                                                rhox[i-1], rhoy[i-1]))
                        
                    outstreamu.close()
                    outstreamrho.close()
                             
            # create plot of dislocation density and misfit profile
            if self.control('plot'):   
                if self.control('dimensions') == 1:
                    self.fig, self.ax = pn1.plot_both(self.ux, r, self.struc('burgers'),
                                                        self.struc('spacing'), 
                                                        nplanes=self.control('nplanes'))
                    plt.savefig(self.control('plot_name'))
                    plt.close()             
                elif self.control('plot_both'): # dimension == 2
                    self.fig, self.ax = pn1.plot_both(self.ux, r, self.struc('burgers'),
                                                               self.struc('spacing'), 
                                                        nplanes=self.control('nplanes'))
                    plt.savefig('edge.' + self.control('plot_name'))
                    plt.close()
                    self.fig, self.ax = pn1.plot_both(self.uy, r, self.struc('burgers'),
                                                               self.struc('spacing'), 
                                                        nplanes=self.control('nplanes'))
                    plt.savefig('screw.' + self.control('plot_name'))
                    plt.close()
                else: # dimensions == 2
                    if self.control('disl_type') in 'edge':
                        self.fig, self.ax = pn1.plot_both(self.ux, r, self.struc('burgers'),
                                                               self.struc('spacing'), 
                                                        nplanes=self.control('nplanes'))
                    else: # screw
                        self.fig, self.ax = pn1.plot_both(self.uy, r, self.struc('burgers'),
                                                               self.struc('spacing'), 
                                                        nplanes=self.control('nplanes'))
                    plt.savefig(self.control('plot_name'))
                    plt.close()
                
            # calculate dislocation width and height
            if (self.control('disl_type') in 'edge' 
                or self.control('dimensions') == 1):
                self.rhos = pn1.rho(self.ux, r)
            else: # screw
                self.rhos = pn1.rho(self.uy, r)
            
            if self.control('disl_type') in 'edge' or self.control('dimensions') == 1:
                self.dis_width = pn1.dislocation_width(self.ux, r, self.struc('burgers'))
            else: # screw dislocation with edge component of displacement 
                self.dis_width = pn1.dislocation_width(self.uy, r, self.struc('burgers'))
                
            self.max_density = pn1.max_rho(self.rhos, self.struc('spacing'))
            
            # calculate the centre of mass
            if self.control('dimensions') == 1:
                self.com = pn1.com_from_pars(self.par, self.struc('burgers'),
                               self.struc('spacing'), self.control('max_x'))
            else:
                self.com = pn2.com_from_pars2d(self.par, self.struc('burgers'),
                                  self.struc('spacing'), self.control('max_x'),
                                                      self.control('disl_type'))

        return
            
    def peierls(self):
        '''Calculate the Peierls stress for the lowest-energy dislocation.
        '''
        
        if self.stress('calculate_barrier'):
            # calculate change in dislocation energy with position       
            e_shifted, pars = pb.shift_energies(self.par, 
                                          self.control('max_x'),
                                          self.gsf, 
                                          self.K, 
                                          self.struc('burgers'), 
                                          self.struc('spacing'), 
                                          dims=self.control('dimensions'),
                                          disl_type=self.control('disl_type'))
            
            # Check to see if any of the shifted energies is lower than the 
            # original energy calculated for the dislocation. If so, use the
            # parameters for the shifted dislocation
            En = e_shifted[np.argmin(e_shifted[:, 1]), 1]
            if En < self.E:
                self.E = En
                self.par = pars[np.argmin(e_shifted[:, 1])]
             
            self.wp_shift = e_shifted[:, 1].max()-e_shifted[:, 1].min()
            
            # calculate the Peierls stress
            self.taup_shift = pb.sigmap_from_wp(self.struc('spacing'), e_shifted[:, 1], 
                                                        self.struc('burgers'))
            if self.stress('use_GPa'):
                self.taup_shift *= atomic_to_GPa
                
            if self.control('record'):
                # retain a copy of the shifted dislocation energies
                outstream_e = open('e_peierls.{}'.format(self.control('output')), 'w')
                
                for x, E in e_shifted:
                    outstream_e.write('{:.3f} {:.6f}\n'.format(x, E))
                    
                outstream_e.close()
             
        if self.stress('calculate_stress'):
            self.taup, self.taup_av = pb.taup(
                                              self.par, 
                                              self.control('max_x'), 
                                              self.gsf,
                                              self.K, 
                                              self.struc('burgers'), 
                                              self.struc('spacing'),
                                              self.control('dimensions'), 
                                              self.control('disl_type'),
                                              dtau=self.stress('dtau'), 
                                              in_GPa=self.stress('use_GPa'),
                                              thr=self.stress('threshold')
                                             )
                                                           
            # calculate average and direction-dependent Peierls barriers
            self.wp_av = pb.peierls_barrier(self.taup_av, self.struc('burgers'),
                                                  in_GPa=self.stress('use_GPa'))
                                                     
        return
        
    def write_output(self):
        '''Write the results of the PN calculation to the specified output file 
        (<control('output')>)
        '''
        
        outstream = open(self.control('output'), 'w')
        
        # write front matter
        boiler_plate = 'Peierls-Nabarro calculation with pyDis - a Python' + \
                       ' interface for atomistic modelling of dislocations\n'                       
        outstream.write(boiler_plate)
        outstream.write('{}\n\n'.format(self.control('title_line')))
        
        # write parameters used in the simulation
        outstream.write('SIMULATION PARAMETERS:\n')
        outstream.write('Dimensions: {}\n'.format(self.control('dimensions')))
        outstream.write('Dislocation type: {}\n'.format(self.control('disl_type')))
        outstream.write('Burgers vector: {:.3f} ang.\n'.format(self.struc('burgers')))
        outstream.write('Interlayer spacing: {:.3f} ang.\n'.format(self.struc('spacing')))
        outstream.write('Number of iterations: {}\n'.format(self.control('n_iter')))
        outstream.write('Number of atomic planes used: {}\n'.format(2*self.control('max_x')+1))
        
        # write stacking fault energy at midpoint of gamma line
        if self.control('dimensions') == 1:
            sfe = self.gsf(self.struc('burgers')/2.)
        else:
            if self.control('disl_type') == 'edge':
                sfe = self.gsf(self.struc('burgers')/2., 0.)
            else:
                sfe = self.gsf(0., self.struc('burgers')/2.)
                
        if not self.surf('do_fourier'):
            sfe = float(sfe)
                
        outstream.write('Stacking fault energy: {:.4f} eV/ang.^2\n'.format(sfe))
        
        # energy coefficients -> only write relevant coefficient if 1D
        if self.control('dimensions') == 2:
            outstream.write('Edge energy coefficient: {:.4f} eV/ang.^3\n'.format(self.K[0]))
            outstream.write('Screw energy coefficient: {:.4f} eV/ang.^3\n'.format(self.K[1]))
        else: # dimensions == 1
            outstream.write('Edge energy coefficient: {:.4f} eV/ang.^3\n'.format(self.K_store[0]))
            outstream.write('Screw energy coefficient: {:.4f} eV/ang.^3\n'.format(self.K_store[1]))
            
        outstream.write('\n\n')
        
        # write fit parameters
        outstream.write('Fit parameters: \n')
        
        # needed in case user set input <parameters> manually
        N = len(self.par)/3
        
        if self.control('dimensions') == 2:
            # write parameters for the edge component of displacement
            outstream.write('----X1----\n')
            
            outstream.write('A\n')
            for A in self.par[:N/2]:
                outstream.write('{:.3f} '.format(A))
            outstream.write('\n')
            
            outstream.write('x0\n')
            for x0 in self.par[N:N+N/2]:
                outstream.write('{:.3f} '.format(x0))
            outstream.write('\n')
            
            outstream.write('c\n')
            for c in self.par[2*N:2*N+N/2]:
                outstream.write('{:.3f} '.format(c))
            outstream.write('\n')

            # write parameters for screw component of displacement
            outstream.write('----X2----\n')
            
            outstream.write('A\n')
            for A in self.par[N/2:N]:
                outstream.write('{:.3f} '.format(A))
            outstream.write('\n')
            
            outstream.write('x0\n')
            for x0 in self.par[N+N/2:2*N]:
                outstream.write('{:.3f} '.format(x0))
            outstream.write('\n')
            
            outstream.write('c\n')
            for c in self.par[2*N+N/2:]:
                outstream.write('{:.3f} '.format(c))
            outstream.write('\n')

        elif self.control('dimensions') == 1:
            outstream.write('----X----\n')

            outstream.write('A\n')
            for A in self.par[:N]:
                outstream.write('{:.3f} '.format(A))
            outstream.write('\n')
            
            outstream.write('x0\n')
            for x0 in self.par[N:2*N]:
                outstream.write('{:.3f} '.format(x0))
            outstream.write('\n')
            
            outstream.write('c\n')
            for c in self.par[2*N:]:
                outstream.write('{:.3f} '.format(c))
            outstream.write('\n')
        
        outstream.write('\n\n')
        
        # write results
        outstream.write('Results:\n')
        outstream.write('E = {:.3f} eV\n'.format(self.E))
        
        # write Peierls stress related stuff
        if self.stress('use_GPa'):
            units = 'GPa'
        else:
            units = 'eV/ang.^3'
            
        if self.stress('calculate_stress'):
            outstream.write('Using applied stress method:\n')
                
            outstream.write('Left Peierls stress: {:.3f} {}\n'.format(
                                                   self.taup[0], units))
            outstream.write('Right Peierls stress: {:.3f} {}\n'.format(
                                                    self.taup[1], units))
                                                    
            # average Peierls stress with APPROXIMATE uncertainty
            e_taup = abs(self.taup[0] - self.taup_av)
            outstream.write('Average Peierls stress: {:.3f} +/- {:.3f}{}\n'.format(
                                                     self.taup_av, e_taup, units))
                                                    
            outstream.write('Peierls barrier: {:.3f} eV/ang\n\n'.format(self.wp_av))
            
        if self.stress('calculate_barrier'):
            outstream.write('Using dislocation translation method:\n')
            outstream.write('Peierls stress: {:.3f} {}\n'.format(self.taup_shift, units))
            outstream.write('Peierls barrier: {:.3f} eV/ang.\n\n'.format(self.wp_shift))
            
        if self.prop('max_rho'):
            outstream.write('Maximum density: {:.3f}\n'.format(self.max_density))
        if self.prop('width'):
            outstream.write('Dislocation width: {:.3f} ang.\n'.format(abs(self.dis_width)))
        if self.prop('center'):
            outstream.write('Dislocation centre (/spacing): {:.3f}\n'.format(
                                            self.com/self.struc('spacing')))
            outstream.write('Dislocation centre (ang.): {:.3f}\n'.format(self.com))
            
        outstream.write('\n\n**Finished**')
        outstream.close()
        return
        
    def visualise(self):
        '''Constructs an atomistic representation of the dislocation core.
        '''
        
        if not self.control('run_sim'):
            # read in dislocation parameters
            dims, self.par = vp.import_pn_pars(self.control('output'))
            
        if self.vis('program') == '':
            raise AttributeError("Atomistic simulation code not specified")
        else:
            basestruc = vp.read_unit_cell(self.vis('unitcell'), 
                                          self.vis('program'),
                                          self.vis('shift'), 
                                          permutation=self.vis('permutation'),
                                          path=self.vis('path_to_cell'))
        
        # construct elastic displacement field function
        field = aniso.makeAnisoField(self.elast('cij'), n=self.elast('b_edge'),
                                        m=self.elast('normal')) 
                                
        vp.make_xyz(basestruc, self.par, self.control('dimensions'), self.struc('burgers'),
                                   self.struc('spacing'), self.control('disl_type'), field,
                       self.vis('radius'), self.vis('xyz_name'), thr=self.vis('threshold'), 
                      sym_thr=self.vis('sym_thresh'), description=self.control('title_line'),
                                                        thickness=self.vis('thickness'))
        
def main(filename):
    '''Runs an Peierls-Nabarro simulation.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, nargs='?', dest='filename', default='0')
    
    args = parser.parse_args()
    new_sim = PNSim(filename)
    
if __name__ == "__main__":
    main()
