# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See _lpproba.pyx for details.

from ._utils cimport DTYPE_t, SIZE_t

cdef DTYPE_t LPPROBA_bernstein(DTYPE_t* v, DTYPE_t[:] p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t[:] beta,
                          SIZE_t reverse) nogil

