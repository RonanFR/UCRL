from setuptools import setup
from setuptools import find_packages
import numpy as np
# from distutils.core import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

name = "rlexplorer"

#from distutils.extension import Extension
from setuptools.extension import Extension
from Cython.Build import cythonize
from distutils.command.clean import clean as Clean
import os

import numpy
import shutil

import rlexplorer
VERSION = rlexplorer.__version__

# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('rlexplorer'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

cmdclass = {'clean': CleanCommand}

libraries = []
if os.name == 'posix':
    libraries.append('m')

extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"]
extra_link_args = ['-fopenmp']

extensions = [
    Extension("rlexplorer.evi.evi",
              ["rlexplorer/evi/evi.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("rlexplorer.evi._lpproba",
              ["rlexplorer/evi/_lpproba.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("rlexplorer.evi._utils",
              ["rlexplorer/evi/_utils.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("rlexplorer.evi.vi",
              ["rlexplorer/evi/vi.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("rlexplorer.evi.scopt",
              ["rlexplorer/evi/scopt.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("rlexplorer.evi.tevi",
              ["rlexplorer/evi/tevi.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
]

setup(
    name=name,
    version=VERSION,
    packages=[package for package in find_packages()
              if package.startswith(name)],
    license='BOOOOO',
    install_requires=requires_list,
    ext_modules=cythonize(extensions, gdb_debug=False, include_path=[np.get_include()]),
    cmdclass=cmdclass,
)
