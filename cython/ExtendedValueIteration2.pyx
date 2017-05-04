import numpy as np
cimport numpy as np
from cpython cimport array
import array
from cython.parallel import prange
from libcpp cimport bool

ctypedef np.int64_t np_int_t
ctypedef np.float64_t np_float_t


def extended_value_iteration2(double[:] policy_indices, double[:] policy,
                              np_int_t nb_states, list state_actions,
                              np.ndarray[np_float_t, ndim=3] estimated_probabilities,
                              np.ndarray[np_float_t, ndim=2] estimated_rewards,
                              np.ndarray[np_float_t, ndim=2] estimated_holding_times,
                              np.ndarray[np_float_t, ndim=2] beta_r,
                              np.ndarray[np_float_t, ndim=2] beta_p,
                              np.ndarray[np_float_t, ndim=2] beta_tau,
                              np_float_t tau_max,
                              np_float_t r_max,
                              np_float_t tau,
                              np_float_t tau_min,
                              np_float_t epsilon):
    cdef np_int_t s, c
    cdef np.ndarray[np_int_t, ndim=1] sorted_indices
    cdef np.ndarray[np_float_t, ndim=1] u1, u2, vec
    cdef np.ndarray[np_float_t, ndim=3] p
    cdef bool first_action
    cdef np_float_t r_optimal, v, tau_optimal

    u1 = np.zeros(nb_states)
    sorted_indices = np.arange(nb_states)
    u2 = np.zeros(nb_states)
    p = np.copy(estimated_probabilities)
    while True:
        for s in prange(nb_states, nogil=True):
            first_action = True
            for c, a in enumerate(state_actions[s]):
                vec = maxProba(p[s][c], sorted_indices, beta_p[s][c])
                vec[s] -= 1
                r_optimal = min(tau_max*r_max,
                                estimated_rewards[s][c] + beta_r[s][c])
                v = r_optimal + np.dot(vec, u1) * tau
                tau_optimal = min(tau_max, max(tau_min, estimated_holding_times[s][c] - np.sign(v) * beta_tau[s][c]))
                if first_action or v/tau_optimal + u1[s] > u2[s]:
                    u2[s] = v/tau_optimal + u1[s]
                    policy_indices[s] = c
                    policy[s] = a
                first_action = False
        if max(u2-u1)-min(u2-u1) < epsilon:  # stopping condition
            break
        else:
            u1 = np.copy(u2)
            sorted_indices = np.argsort(u1)

cdef maxProba(np.ndarray[np_float_t, ndim=1] p, np.ndarray[np_int_t, ndim=1]  sorted_indices, np_float_t beta) nogil:
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
