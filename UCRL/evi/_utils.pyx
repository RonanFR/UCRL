# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

from libc.string cimport memcpy
from libc.math cimport fabs
import numpy as np
cimport numpy as np


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

cdef DTYPE_t check_end_sparse(DTYPE_t* x, DTYPE_t* y, SIZE_t[:] states, SIZE_t dim,
                      DTYPE_t* min_y, DTYPE_t* max_y) nogil:
    cdef SIZE_t i, k
    cdef DTYPE_t diff
    cdef DTYPE_t diff_min = x[0]-y[0], diff_max = x[0] - y[0]
    min_y[0] = y[0]
    max_y[0] = y[0]
    for k in range(0, dim):
        i = states[k]
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

cdef SIZE_t isinsortedvector(SIZE_t value, SIZE_t[:] vector, SIZE_t dim) nogil:
    cdef SIZE_t l=0, r=dim-1, m

    while 1:
        if l > r:
            return -1

        m = (l + r) / 2
        if vector[m] < value:
            l = m + 1
        elif vector[m] > value:
            r = m - 1
        else:
            return m

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


cdef void quicksort_indices(DTYPE_t *x, SIZE_t l, SIZE_t h, SIZE_t *sort_idx) nogil:
    cdef SIZE_t p
    if l < h:
        p = hoare_partition(x, l, h, sort_idx)
        quicksort_indices(x, l, p, sort_idx)
        quicksort_indices(x, p+1, h, sort_idx)

cdef SIZE_t hoare_partition(DTYPE_t *x, SIZE_t l, SIZE_t h, SIZE_t *sort_idx) nogil:
    cdef SIZE_t i, j, temp
    cdef DTYPE_t pivot

    pivot = x[sort_idx[l]]
    i = l
    j = h
    while 1:
        while x[sort_idx[i]] < pivot:
            i += 1

        while x[sort_idx[j]] > pivot:
            j -= 1

        if i >= j:
            return j

        temp = sort_idx[j]
        sort_idx[j] = sort_idx[i]
        sort_idx[i] = temp

# =============================================================================
# PYTHON interfaces
# =============================================================================

def py_quicksort_indices(np.ndarray[DTYPE_t, ndim=1, mode="c"] v):
    cdef DTYPE_t* vpt
    cdef SIZE_t* sorted_idxpt
    sorted_idx = np.arange(len(v))

    vpt = <DTYPE_t*> np.PyArray_GETPTR1(v, 0)
    sorted_idxpt = <SIZE_t*> np.PyArray_GETPTR1(sorted_idx, 0)
    quicksort_indices(vpt, 0, len(v)-1, sorted_idxpt)
    return sorted_idx

def py_sorted_indices(np.ndarray[DTYPE_t, ndim=1, mode="c"] v):
    cdef DTYPE_t* vpt
    cdef SIZE_t* sorted_idxpt
    sorted_idx = np.arange(len(v))

    vpt = <DTYPE_t*> np.PyArray_GETPTR1(v, 0)
    sorted_idxpt = <SIZE_t*> np.PyArray_GETPTR1(sorted_idx, 0)
    get_sorted_indices(vpt, len(v), sorted_idxpt)
    return sorted_idx

def py_isinsortedvector(SIZE_t c, np.ndarray[SIZE_t, ndim=1] v):
    return isinsortedvector(c, v, len(v))
