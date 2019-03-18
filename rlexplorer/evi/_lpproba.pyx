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


cdef DTYPE_t LPPROBA_hoeffding(DTYPE_t* v, DTYPE_t[:] p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t beta, SIZE_t reverse) nogil:
    cdef SIZE_t i, idx, pos
    cdef DTYPE_t temp, fval
    cdef DTYPE_t sum_p = 0.0

    idx = asc_sorted_indices[(n-1)*(1-reverse)]
    temp = min(1., p[idx] + beta/2.0)
    sum_p = temp
    fval = temp * v[idx]
    if temp < 1.:
        for i in range(0+reverse, n-1+reverse):
            pos = asc_sorted_indices[i]
            temp = p[pos]
            sum_p += temp
            fval += temp * v[pos]
        i = 0
        while sum_p > 1.0 and i < n:
            idx = n-1-i if reverse else i
            pos = asc_sorted_indices[idx]
            sum_p -= p[pos]
            fval -= p[pos]*v[pos]
            temp = max(0.0, 1. - sum_p)
            fval += temp*v[pos]
            sum_p += temp
            i += 1
    return fval


# =============================================================================
# Python interface
# =============================================================================
def py_LPPROBA_hoeffding(
            np.ndarray[DTYPE_t, ndim=1] v,
            np.ndarray[DTYPE_t, ndim=1] p,
            np.ndarray[SIZE_t, ndim=1] asc_sorted_idx,
            DTYPE_t beta,
            reverse):
    cdef SIZE_t* asc_pt
    cdef DTYPE_t* v_pt
    cdef SIZE_t n = len(p)
    cdef SIZE_t rev = 1 if reverse else 0

    v_pt = <DTYPE_t*> np.PyArray_GETPTR1(v, 0)
    asc_pt = <SIZE_t*> np.PyArray_GETPTR1(asc_sorted_idx, 0)
    return LPPROBA_hoeffding(v_pt, p, n, asc_pt, beta, rev)

def py_LPPROBA_bernstein(
            np.ndarray[DTYPE_t, ndim=1] v,
            np.ndarray[DTYPE_t, ndim=1] p,
            np.ndarray[SIZE_t, ndim=1] asc_sorted_idx,
            np.ndarray[DTYPE_t, ndim=1] beta,
            DTYPE_t bplus,
            reverse):
    cdef SIZE_t* asc_pt
    cdef DTYPE_t* v_pt
    cdef SIZE_t n = len(p)
    cdef SIZE_t rev = 1 if reverse else 0

    v_pt = <DTYPE_t*> np.PyArray_GETPTR1(v, 0)
    asc_pt = <SIZE_t*> np.PyArray_GETPTR1(asc_sorted_idx, 0)
    return LPPROBA_bernstein(v_pt, p, n, asc_pt, beta, bplus, rev)
