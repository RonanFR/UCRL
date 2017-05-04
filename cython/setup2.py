from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "ExtendedValueIteration2",
        ["ExtendedValueIteration2.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
  # ext_modules = cythonize("hello.pyx"),
  # ext_modules = cythonize("max_proba.pyx"),
  ext_modules = cythonize(ext_modules),
)
