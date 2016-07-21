#!/usr/bin/env python
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

setup(name='slim_python',
      version='1.0',
      description='learn optimized scoring systems from data',
      long_description = '''
    slim-python is a free software package to train SLIM scoring systems
    using IBM ILOG CPLEX API and the Python programming language. ''',
      author='Berk Ustun',
      author_email='ustunb@mit.edu',
      url='https://www.berkustun.com/',
      packages=['slim_python'],
      )