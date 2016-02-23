from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

# mcoptmodule = Extension(
#     'atmc.mcopt_wrapper',
#     include_dirs=[np.get_include(), '/usr/local/include', '/usr/include'],
#     libraries=['armadillo', 'mcopt'],
#     library_dirs=['/usr/local/lib'],
#     sources=['atmc/mcopt_wrapper.cpp'],
#     language='c++',
#     extra_compile_args=['-Wall', '-std=c++11', '-mmacosx-version-min=10.9'],
#     )

include_path = [np.get_include()]

ext_kwargs = dict(include_dirs=[np.get_include(), '/usr/local/include'],
                  libraries=['mcopt'],
                  library_dirs=['/usr/local/lib'],
                  language='c++',
                  extra_compile_args=['-Wall', '-std=c++11', '-mmacosx-version-min=10.9', '-g'],)

exts = [Extension('atmc.mcopt_wrapper', ['atmc/mcopt_wrapper.pyx'], **ext_kwargs),
        Extension('atmc.armadillo', ['atmc/armadillo.pyx'], **ext_kwargs)]

setup(name='atmc',
      version='2.1.0',
      description='Particle tracking and MC optimizer module',
      packages=['atmc'],
      ext_modules=cythonize(exts),
      )
