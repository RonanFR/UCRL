from distutils.core import setup
from Cython.Build import cythonize


# Setup file used for Cython (transform python code in C code and compile it in .so)

setup(
  # ext_modules = cythonize("max_proba.pyx"),
  ext_modules=cythonize("ExtendedValueIteration.pyx"),
)
