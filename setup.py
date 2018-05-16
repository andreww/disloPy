#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='dislopy',
      version=0.99,
      description='Python-based dislocation modelling utilities',
      author='Richard Skelton, Andrew Walker',
      author_email='richard.skelton@anu.edu.au',
      packages=find_packages(exclude="docs"),
      install_requires=['numpy>1.7',
                        'scipy'],
      
      entry_points = {
            'console_scripts': [
                'atomistic = bin.command_line_scripts:main_atomistic',
                'peierls = bin.command_line_scripts:main_pn',
                'segregation = bin.command_line_scripts:main_segregation',
            ]
      }
     )
