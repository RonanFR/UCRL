from cython.parallel import prange

from libc.stdio cimport printf
# =============================================================================
# Max Probabilities give CI [Near-optimal Regret Bounds for RL]
# =============================================================================

cdef void max_proba_purec(DTYPE_t[:] p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t beta, DTYPE_t[:] new_p) nogil:
    cdef SIZE_t i
    cdef DTYPE_t temp
    cdef DTYPE_t sum_p = 0.0

    temp = min(1., p[asc_sorted_indices[n-1]] + beta/2.0)
    new_p[asc_sorted_indices[n-1]] = temp
    sum_p = temp
    if temp < 1.:
        for i in range(0, n-1):
            temp = p[asc_sorted_indices[i]]
            new_p[asc_sorted_indices[i]] = temp
            sum_p += temp
        i = 0
        while sum_p > 1.0 and i < n:
            sum_p -= p[asc_sorted_indices[i]]
            new_p[asc_sorted_indices[i]] = max(0.0, 1. - sum_p)
            sum_p += new_p[asc_sorted_indices[i]]
            i += 1
    else:
        for i in range(0, n):
            new_p[i] = 0
        new_p[asc_sorted_indices[n-1]] = temp

cdef void max_proba_purec2(DTYPE_t* p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t beta, DTYPE_t[:] new_p) nogil:
    cdef SIZE_t i
    cdef DTYPE_t temp
    cdef DTYPE_t sum_p = 0.0

    temp = min(1., p[asc_sorted_indices[n-1]] + beta/2.0)
    new_p[asc_sorted_indices[n-1]] = temp
    sum_p = temp
    if temp < 1.:
        for i in range(0, n-1):
            temp = p[asc_sorted_indices[i]]
            new_p[asc_sorted_indices[i]] = temp
            sum_p += temp
        i = 0
        while sum_p > 1.0 and i < n:
            sum_p -= p[asc_sorted_indices[i]]
            new_p[asc_sorted_indices[i]] = max(0.0, 1. - sum_p)
            sum_p += new_p[asc_sorted_indices[i]]
            i += 1
    else:
        for i in range(0, n):
            new_p[i] = 0
        new_p[asc_sorted_indices[n-1]] = temp

cdef void max_proba_reduced(DTYPE_t[:] p,
                            SIZE_t n,
                            SIZE_t* asc_sorted_indices,
                            DTYPE_t beta, DTYPE_t[:] new_p) nogil:
    cdef SIZE_t i
    cdef DTYPE_t temp, thr = 0.
    cdef DTYPE_t sum_p = 0.0

    temp = min(1., p[asc_sorted_indices[n-1]] + beta/2.0)
    sum_p = 1.0 + temp - p[asc_sorted_indices[n-1]]
    new_p[asc_sorted_indices[n-1]] = temp
    if temp - 1. < thr:
        for i in range(0, n-1):
            if sum_p > 1.0:
                sum_p -= p[asc_sorted_indices[i]]
                new_p[asc_sorted_indices[i]] = max(0.0, 1. - sum_p)
                sum_p += new_p[asc_sorted_indices[i]]
            else:
                new_p[asc_sorted_indices[i]] = p[asc_sorted_indices[i]]
    else:
        for i in prange(0, n):
            new_p[i] = 0
        new_p[asc_sorted_indices[n-1]] = temp

# =============================================================================
# Max Probabilities give Bernstein-CI
# [Sample complexity of episodic fixed-horizon reinforcement learning]
# =============================================================================
cdef void max_proba_bernstein(DTYPE_t[:] p,
                          SIZE_t n,
                          SIZE_t* asc_sorted_indices,
                          DTYPE_t[:] beta, DTYPE_t[:] new_p) nogil:
    cdef SIZE_t i, idx
    cdef DTYPE_t delta, new_delta

    delta = 1.
    for i in range(n):
        new_p[i] = max(0, p[i] - beta[i])
        delta -= new_p[i]
    i = n - 1
    while delta > 0 and i >= 0:
        idx = asc_sorted_indices[i]
        new_delta = min(delta, p[idx] + beta[idx] - new_p[idx])
        new_p[idx] += new_delta
        delta -= new_delta
        i -= 1

