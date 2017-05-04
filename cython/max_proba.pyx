import numpy as np
cimport numpy as np

ctypedef np.int64_t np_int_t
ctypedef np.float64_t np_float_t

def maxProba(np.ndarray[np_float_t, ndim=1] p, np.ndarray[np_int_t, ndim=1]  sorted_indices, np_float_t beta):
    cdef np_int_t i, n
    cdef np_float_t min1, max1, s, s2
    cdef np.ndarray[np_float_t, ndim=1] p2

    n = np.size(sorted_indices)
    p2 = np.copy(p)

    min1 = min(1, p2[sorted_indices[n-1]] + beta)
    s = 1-p2[sorted_indices[n-1]]+min1
    p2[sorted_indices[n-1]] = min1

    s2 = s
    for i in sorted_indices:
        max1 = max(0, 1-s + p2[i])
        s2 += (max1 -p2[i])
        p2[i] = max1
        s = s2
        if s <= 1: break

    return p2
