from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

name = "UCRL"

from distutils.extension import Extension
from Cython.Build import cythonize
import os

import numpy

libraries = []
if os.name == 'posix':
    libraries.append('m')

extensions = [
    Extension("UCRL.evi.evi", ["UCRL/evi/evi.pyx"],
        include_dirs = [numpy.get_include()],
        libraries = libraries,
                         extra_compile_args=["-O3"]),
    # Everything but primes.pyx is included here.
    Extension("UCRL.evi._utils", ["UCRL/evi/_utils.pyx"],
        include_dirs = [numpy.get_include()],
        libraries = libraries,
                         extra_compile_args=["-O3"]),
]

setup(
    name=name,
    version='0.1.dev0',
    packages=[package for package in find_packages()
              if package.startswith(name)],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires=requires_list,
    ext_modules = cythonize(extensions),
)