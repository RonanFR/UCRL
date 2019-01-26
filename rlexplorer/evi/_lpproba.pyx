# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdio cimport printf
import numpy as np
cimport numpy as np

cdef DTYPE_t LPPROBA_bernstein(DTYPE_t* v, DTYPE_t[:] p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t[:] beta, DTYPE_t bplus, SIZE_t reverse) nogil:
    cdef SIZE_t i, idx
    cdef DTYPE_t delta, new_delta, fval, a, b

    delta = 1.
    fval = 0.
    for i in range(n):
        idx = asc_sorted_indices[i]
        a = max(0, p[idx] - beta[idx])
        delta -= a
        fval += a * v[idx]
    i = n - 1
    while delta > 0 and i >= 0:
        idx = n - 1 - i if reverse else i
        idx = asc_sorted_indices[idx]
        b = p[idx] + beta[idx] + bplus
        a = max(0, p[idx] - beta[idx])
        new_delta = min(delta, b - a)
        fval += new_delta * v[idx]
        delta -= new_delta
        i -= 1
    return fval
