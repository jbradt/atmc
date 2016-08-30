from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize
from sys import platform


extra_args = ['-Wall', '-Wno-unused-function', '-std=c++11', '-g']
if platform == 'darwin':
    extra_args.append('-mmacosx-version-min=10.9')

include_path = [np.get_include()]

ext_kwargs = dict(include_dirs=[np.get_include()],
                  libraries=['mcopt'],
                  language='c++',
                  extra_compile_args=extra_args,
                  extra_link_args=extra_args)

exts = [Extension('atmc.mcopt_wrapper', ['atmc/mcopt_wrapper.pyx'], **ext_kwargs),
        Extension('atmc.armadillo', ['atmc/armadillo.pyx'], **ext_kwargs)]

setup(name='atmc',
      version='2.1.0',
      description='Particle tracking and MC optimizer module',
      packages=['atmc'],
      ext_modules=cythonize(exts),
      )
