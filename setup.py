from setuptools import setup
from setuptools import find_packages
# from distutils.core import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

name = "UCRL"

#from distutils.extension import Extension
from setuptools.extension import Extension
from Cython.Build import cythonize
import os

import numpy

libraries = []
if os.name == 'posix':
    libraries.append('m')

extra_compile_args = []
# extra_compile_args = ["-O3"]

extensions = [
    Extension("UCRL.evi.evi",
              ["UCRL/evi/evi.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args),
    Extension("UCRL.evi._utils",
              ["UCRL/evi/_utils.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args),
    Extension("UCRL.cython.ExtendedValueIteration",
              ["UCRL/cython/ExtendedValueIteration.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args),
    Extension("UCRL.cython.max_proba",
              ["UCRL/cython/max_proba.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args),
]

setup(
    name=name,
    version='0.1.dev0',
    packages=[package for package in find_packages()
              if package.startswith(name)],
    license='BOOOOO',
    install_requires=requires_list,
    ext_modules=cythonize(extensions, gdb_debug=True),
)
