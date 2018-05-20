# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See _utils.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cimport cython

cdef struct IntVectorStruct:
    SIZE_t* values
    SIZE_t dim

cdef struct DoubleVectorStruct:
    DTYPE_t* values
    SIZE_t dim

@cython.boundscheck(False)
cdef inline DTYPE_t dot_prod(DTYPE_t[:] x, DTYPE_t* y, SIZE_t dim) nogil:
    cdef SIZE_t i
    cdef DTYPE_t total = 0.
    for i in range(dim):
        total += x[i] * y[i]
    return total

@cython.boundscheck(False)
cdef inline DTYPE_t sparse_dot_prod(DTYPE_t[:] x, DTYPE_t* y, SIZE_t* indices, SIZE_t dim) nogil:
    cdef SIZE_t i
    cdef DTYPE_t total = 0.
    for i in range(dim):
        total += x[indices[i]] * y[indices[i]]
    return total

cdef inline SIZE_t pos2index_2d(SIZE_t n_row, SIZE_t n_col, SIZE_t row, SIZE_t col) nogil:
    # return col * n_row + row
    return row * n_col + col

cdef DTYPE_t sign(DTYPE_t a, DTYPE_t tol=*) nogil

cdef int isclose_c(DTYPE_t a, DTYPE_t b, DTYPE_t rel_tol=*, DTYPE_t abs_tol=*) nogil

cdef DTYPE_t check_end(DTYPE_t* x, DTYPE_t* y, SIZE_t dim,
                      DTYPE_t* min_y, DTYPE_t* max_y) nogil

cdef DTYPE_t check_end_sparse(DTYPE_t* x, DTYPE_t* y, SIZE_t[:] states, SIZE_t dim,
                      DTYPE_t* min_y, DTYPE_t* max_y) nogil

cdef SIZE_t isinsortedvector(SIZE_t value, SIZE_t[:] vector, SIZE_t dim) nogil

# =============================================================================
# Sorting Algorithms
# =============================================================================
cdef void get_sorted_indices(DTYPE_t *x, SIZE_t dim, SIZE_t *sort_idx) nogil

cdef void quicksort_indices(DTYPE_t *x, SIZE_t l, SIZE_t h, SIZE_t *sort_idx) nogil

cdef SIZE_t hoare_partition(DTYPE_t *x, SIZE_t l, SIZE_t h, SIZE_t *sort_idx) nogil
