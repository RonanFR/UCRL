# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

from libc.string cimport memcpy
from libc.math cimport fabs

cdef inline DTYPE_t sign(DTYPE_t a, DTYPE_t tol=1e-09) nogil:
    if (a < -tol):
        return -1.
    elif (a > tol):
        return 1.
    else:
        return 0.

cdef inline int isclose_c(DTYPE_t a, DTYPE_t b, DTYPE_t rel_tol=1e-09, DTYPE_t abs_tol=0.0) nogil:
    return fabs(a-b) <= max(rel_tol * max(fabs(a), fabs(b)), abs_tol)

cdef DTYPE_t check_end(DTYPE_t* x, DTYPE_t* y, SIZE_t dim,
                      DTYPE_t* min_y, DTYPE_t* max_y) nogil:
    cdef SIZE_t i, j
    cdef DTYPE_t diff
    cdef DTYPE_t diff_min = x[0]-y[0], diff_max = x[0] - y[0]
    min_y[0] = y[0]
    max_y[0] = y[0]
    for i in range(0, dim):
        diff = x[i] - y[i]
        if diff_min > diff:
            diff_min = diff
        if diff_max < diff:
            diff_max = diff
        if min_y[0] > y[i]:
            min_y[0] = y[i]
        if max_y[0] < y[i]:
            max_y[0] = y[i]
    return diff_max - diff_min

# =============================================================================
# Sorting Algorithms
# =============================================================================

cdef void get_sorted_indices(DTYPE_t *x, SIZE_t dim, SIZE_t *sort_idx) nogil:
    cdef SIZE_t i, j, tmp
    for i in range(dim-1):
        for j in range(i+1, dim):
            if x[sort_idx[j]] < x[sort_idx[i]]:
                tmp = sort_idx[i]
                sort_idx[i] = sort_idx[j]
                sort_idx[j] = tmp


#Array A[] has the items to sort; array B[] is a work array.
cdef void TopDownMergeSort(SIZE_t* A, SIZE_t* B, SIZE_t n, DTYPE_t* X) nogil:
    memcpy(B, A, n * sizeof(SIZE_t)) # duplicate array A[] into B[]
    # CopyArray(A, 0, n, B)             # duplicate array A[] into B[]
    TopDownSplitMerge(B, 0, n, A, X)     # sort data from B[] into A[]

# Sort the given run of array A[] using array B[] as a source.
# iBegin is inclusive; iEnd is exclusive (A[iEnd] is not in the set).
cdef void TopDownSplitMerge(SIZE_t* B, SIZE_t iBegin, SIZE_t iEnd, SIZE_t* A, DTYPE_t* X) nogil:
    cdef SIZE_t iMiddle
    if (iEnd - iBegin < 2):                     # if run size == 1
        return                                  #   consider it sorted
    # split the run longer than 1 item into halves
    iMiddle = (iEnd + iBegin) / 2               # iMiddle = mid point
    # recursively sort both runs from array A[] into B[]
    TopDownSplitMerge(A, iBegin,  iMiddle, B, X)  # sort the left  run
    TopDownSplitMerge(A, iMiddle,    iEnd, B, X)  # sort the right run
    # merge the resulting runs from array B[] into A[]
    TopDownMerge(B, iBegin, iMiddle, iEnd, A, X)

# Left source half is A[ iBegin:iMiddle-1].
# Right source half is A[iMiddle:iEnd-1   ].
# Result is            B[ iBegin:iEnd-1   ].
cdef void TopDownMerge(SIZE_t* A, SIZE_t iBegin, SIZE_t iMiddle, SIZE_t iEnd, SIZE_t* B, DTYPE_t* X) nogil:
    cdef SIZE_t i, j, k
    i = iBegin
    j = iMiddle

    # While there are elements in the left or right runs...
    for k in range(iBegin, iEnd):
        # If left run head exists and is <= existing right run head.
        if (i < iMiddle) and (j >= iEnd or X[A[i]] <= X[A[j]]):
            B[k] = A[i]
            i = i + 1
        else:
            B[k] = A[j]
            j = j + 1
