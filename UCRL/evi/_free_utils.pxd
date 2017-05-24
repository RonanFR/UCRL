# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See free_evi.pyx for details.

import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t, SIZE_t

#  /* Auxiliary routine: printing eigenvalues */
cdef void print_eigenvalues( char* desc, SIZE_t n, DTYPE_t* wr, DTYPE_t* wi ) nogil

# /* Auxiliary routine: printing eigenvectors */
cdef void print_eigenvectors( char* desc, SIZE_t n, DTYPE_t* wi, DTYPE_t* v, SIZE_t ldv ) nogil

# /* Auxiliary routine: printing a matrix */
cdef void print_matrix( SIZE_t ordering, char* desc, SIZE_t m, SIZE_t n, DTYPE_t* a, SIZE_t lda ) nogil

# /* Auxiliary routine: printing a vectors */
cdef void print_int_vector( char* desc, SIZE_t n, SIZE_t* a ) nogil
cdef void print_double_vector( char* desc, SIZE_t n, DTYPE_t* a ) nogil

cdef SIZE_t get_mu_and_ci_c(DTYPE_t* sq_mtx, SIZE_t N,
                          DTYPE_t* condition_number, DTYPE_t* mu) nogil