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
from distutils.command.clean import clean as Clean
import os

import numpy
import shutil

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
        for dirpath, dirnames, filenames in os.walk('UCRL'):
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
    Extension("UCRL.evi.evi",
              ["UCRL/evi/evi.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi._max_proba",
              ["UCRL/evi/_max_proba.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi._utils",
              ["UCRL/evi/_utils.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi.free_evi",
              ["UCRL/evi/free_evi.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.cython.ExtendedValueIteration",
              ["UCRL/cython/ExtendedValueIteration.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.cython.max_proba",
              ["UCRL/cython/max_proba.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi.prova",
              ["UCRL/evi/prova.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi._free_utils",
              ["UCRL/evi/_free_utils.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args)
]

setup(
    name=name,
    version='0.1.dev0',
    packages=[package for package in find_packages()
              if package.startswith(name)],
    license='BOOOOO',
    install_requires=requires_list,
    ext_modules=cythonize(extensions, gdb_debug=False),
    cmdclass=cmdclass,
)
