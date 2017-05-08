import numpy as np
cimport numpy as np
from cpython cimport array
import array
import math as m

ctypedef np.int64_t np_int_t
ctypedef np.float64_t np_float_t

from libc.stdio cimport printf


def extended_value_iteration(np.ndarray[np_int_t, ndim=1] policy_indices,
                             np.ndarray[np_int_t, ndim=1] policy,
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
    cdef np_int_t s, c, counter = 0
    cdef np.ndarray[np_int_t, ndim=1] sorted_indices
    cdef np.ndarray[np_float_t, ndim=1] u1, u2
    cdef np.ndarray[np_float_t, ndim=3] p

    u1 = np.zeros(nb_states)
    sorted_indices = np.arange(nb_states)
    u2 = np.zeros(nb_states)
    p = estimated_probabilities
    while True:
        for s in range(0, nb_states):
            first_action = True
            for c, a in enumerate(state_actions[s]):
                vec = maxProba(p[s][c], sorted_indices, beta_p[s][c], nb_states)
                vec[s] -= 1
                r_optimal = min(tau_max*r_max,
                                estimated_rewards[s][c] + beta_r[s][c])
                v = r_optimal + np.dot(vec, u1) * tau
                tau_optimal = min(tau_max, max(max(tau_min, r_optimal/r_max), estimated_holding_times[s][c] - np.sign(v) * beta_tau[s][c]))
                if first_action or v/tau_optimal + u1[s] > u2[s] or m.isclose(v/tau_optimal + u1[s], u2[s]):
                    u2[s] = v/tau_optimal + u1[s]
                    policy_indices[s] = c
                    policy[s] = a
                first_action = False
        counter = counter + 1
        # printf("**%d\n", counter)
        # for i in range(nb_states):
        #     printf("%.2f[%.2f] ", u1[i], u2[i])
        # printf("\n")

        if max(u2-u1)-min(u2-u1) < epsilon:  # stopping condition
            # print("%d\n", counter)
            return max(u1) - min(u1), u1, u2
        else:
            u1 = u2
            u2 = np.empty(nb_states)
            sorted_indices = np.argsort(u1)
            # for i in range(nb_states):
            #     printf("%d , ", sorted_indices[i])
            # printf("\n")
            

cdef maxProba(np.ndarray[np_float_t, ndim=1] p, np.ndarray[np_int_t, ndim=1]  sorted_indices, np_float_t beta, np_int_t nb_states):
    cdef np_int_t i, n
    cdef np_float_t min1, max1, s, s2, proba
    cdef np.ndarray[np_float_t, ndim=1] p2
    
    n = np.size(sorted_indices)
    min1 = min(1, p[sorted_indices[n-1]] + beta/2)
    if min1 == 1:
        p2 = np.zeros(nb_states)
        p2[sorted_indices[n-1]] = 1
    else:
        sorted_p = p[sorted_indices]
        support_sorted_p = np.nonzero(sorted_p)[0]
        restricted_sorted_p = sorted_p[support_sorted_p]
        support_p = sorted_indices[support_sorted_p]
        p2 = np.zeros(nb_states)
        p2[support_p] = restricted_sorted_p
        s = 1 - p[sorted_indices[n-1]] + min1
        p2[sorted_indices[n-1]] = min1
        s2 = s
        for i, proba in enumerate(restricted_sorted_p):
            max1 = max(0, 1-s + proba)
            s2 += (max1 - proba)
            p2[support_p[i]] = max1
            s = s2
            if s <= 1: break
    return p2

