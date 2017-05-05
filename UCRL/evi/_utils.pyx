# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

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

cdef void get_sorted_indices(DTYPE_t *x, SIZE_t dim, SIZE_t *sort_idx) nogil:
    cdef SIZE_t i, j
    cdef DTYPE_t tmp
    for i in range(dim-1):
        for j in range(i+1, dim):
            if x[sort_idx[i]] < x[sort_idx[j]]:
                tmp = x[sort_idx[i]]
                x[sort_idx[i]] = x[sort_idx[j]]
                x[sort_idx[j]] = tmp

cdef DTYPE_t check_end(DTYPE_t* x, DTYPE_t* y, SIZE_t dim,
                      DTYPE_t* min_y, DTYPE_t* max_y) nogil:
    cdef SIZE_t i, j
    cdef DTYPE_t diff
    cdef DTYPE_t diff_min = x[0]-y[0], diff_max = x[0] - y[0]
    min_y[0] = y[0]
    max_y[0] = y[0]
    for i in range(1, dim):
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

