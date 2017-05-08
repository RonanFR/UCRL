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


cdef DTYPE_t sign(DTYPE_t a, DTYPE_t tol=*) nogil

cdef int isclose_c(DTYPE_t a, DTYPE_t b, DTYPE_t rel_tol=*, DTYPE_t abs_tol=*) nogil

cdef DTYPE_t check_end(DTYPE_t* x, DTYPE_t* y, SIZE_t dim,
                      DTYPE_t* min_y, DTYPE_t* max_y) nogil


# =============================================================================
# Sorting Algorithms
# =============================================================================
cdef void get_sorted_indices(DTYPE_t *x, SIZE_t dim, SIZE_t *sort_idx) nogil

# Merge sort (see Wikipedia [https://en.wikipedia.org/wiki/Merge_sort#Top-down_implementation])
cdef void TopDownMergeSort(SIZE_t* A, SIZE_t* B, SIZE_t n, DTYPE_t* X) nogil
cdef void TopDownSplitMerge(SIZE_t* B, SIZE_t iBegin, SIZE_t iEnd, SIZE_t* A, DTYPE_t* X) nogil
cdef void TopDownMerge(SIZE_t* A, SIZE_t iBegin, SIZE_t iMiddle, SIZE_t iEnd, SIZE_t* B, DTYPE_t* X) nogil