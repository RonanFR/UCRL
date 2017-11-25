# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See _max_proba.pyx for details.

from ._utils cimport DTYPE_t, SIZE_t

cdef void max_proba_purec(DTYPE_t[:] p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t beta, DTYPE_t[:] new_p,
                          SIZE_t reverse) nogil

cdef void max_proba_purec2(DTYPE_t* p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t beta, DTYPE_t[:] new_p,
                          SIZE_t reverse) nogil

cdef void max_proba_bernstein(DTYPE_t[:] p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t[:] beta, DTYPE_t[:] new_p,
                          SIZE_t reverse) nogil

cdef void max_proba_bernstein_cin(DTYPE_t* p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t* beta, DTYPE_t[:] new_p,
                          SIZE_t reverse) nogil
