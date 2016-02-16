from setuptools import setup, Extension
import numpy as np

mcoptmodule = Extension(
    'atmc.mcopt_wrapper',
    include_dirs=[np.get_include(), '/usr/local/include', '/usr/include'],
    libraries=['armadillo', 'mcopt'],
    library_dirs=['/usr/local/lib'],
    sources=['atmc/mcopt_wrapper.cpp'],
    language='c++',
    extra_compile_args=['-Wall', '-std=c++11', '-mmacosx-version-min=10.9'],
    )

setup(name='atmc',
      version='2.1.0',
      description='Particle tracking and MC optimizer module',
      packages=['atmc'],
      ext_modules=[mcoptmodule],
      )
