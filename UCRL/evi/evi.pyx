# cython: boundscheck=False


# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset

from cython.parallel import prange

import numpy as np
cimport numpy as np

from ._utils cimport sign
from ._utils cimport isclose_c
from ._utils cimport get_sorted_indices
from ._utils cimport check_end

# =============================================================================
# Max Probabilities give CI [Near-optimal Regret Bounds for RL]
# =============================================================================

cdef void max_proba_purec(DTYPE_t[:] p, SIZE_t* asc_sorted_indices,
                          DTYPE_t beta, DTYPE_t[:] new_p) nogil:
    cdef SIZE_t i, n
    cdef DTYPE_t temp
    cdef DTYPE_t sum_p = 0.0
    n = p.shape[0]

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

cdef inline DTYPE_t dot_prod(DTYPE_t[:] x, DTYPE_t* y, SIZE_t dim) nogil:
    cdef SIZE_t i
    cdef DTYPE_t total = 0.
    for i in range(dim):
        total += x[i] * y[i]
    return total

# =============================================================================
# Extended Value Iteration Class
# =============================================================================

cdef class EVI:

    def __init__(self, nb_states):
        self.nb_states = nb_states
        self.u1 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u2 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))

        # allocate indices and memoryview (may slow down)
        self.sorted_indices = <SIZE_t *> malloc(nb_states * sizeof(SIZE_t))
        for i in range(nb_states):
            self.u1[i] = 0
            self.u2[i] = 0
            self.sorted_indices[i] = i

        # allocate space for matrix of max probabilities and the associated memory view
        self.mtx_maxprob = <DTYPE_t *>malloc(nb_states * nb_states * sizeof(DTYPE_t))
        self.mtx_maxprob_memview = <DTYPE_t[:nb_states, :nb_states]> self.mtx_maxprob

    def __dealloc__(self):
        free(self.u1)
        free(self.u2)
        free(self.mtx_maxprob)
        free(self.sorted_indices)

    cpdef DTYPE_t evi(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     SIZE_t[:, :] state_actions,
                     SIZE_t[:] num_actions_per_state,
                     DTYPE_t[:,:,:] estimated_probabilities,
                     DTYPE_t[:,:] estimated_rewards,
                     DTYPE_t[:,:] estimated_holding_times,
                     DTYPE_t[:,:] beta_r,
                     DTYPE_t[:,:] beta_p,
                     DTYPE_t[:,:] beta_tau,
                     DTYPE_t tau_max,
                     DTYPE_t r_max,
                     DTYPE_t tau,
                     DTYPE_t tau_min,
                     DTYPE_t epsilon):

        cdef SIZE_t s, i, a_idx
        cdef SIZE_t first_action
        cdef DTYPE_t c1
        cdef DTYPE_t min_u1, max_u1, r_optimal, v, tau_optimal
        cdef SIZE_t nb_states = self.nb_states

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef SIZE_t* sorted_indices = self.sorted_indices

        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview

        with nogil:
            for i in prange(nb_states):
                u1[i] = 0.0
                u2[i] = 0.0
                sorted_indices[i] = i

            while True:
                for s in prange(nb_states):
                    first_action = 1
                    for a_idx in range(num_actions_per_state[s]):
                        max_proba_purec(estimated_probabilities[s][a_idx],
                                    sorted_indices, beta_p[s][a_idx],
                                    mtx_maxprob_memview[s])
                        mtx_maxprob_memview[s][s] = mtx_maxprob_memview[s][s] - 1. #????
                        r_optimal = min(tau_max*r_max,
                                        estimated_rewards[s][a_idx] + beta_r[s][a_idx])
                        v = r_optimal + dot_prod(mtx_maxprob_memview[s], u1, nb_states) * tau
                        tau_optimal = min(tau_max, max(
                            max(tau_min, r_optimal/r_max),
                            estimated_holding_times[s][a_idx] - sign(v) * beta_tau[s][a_idx]
                        ))
                        c1 = v / tau_optimal + u1[s]
                        if first_action or c1 > u2[s] or isclose_c(c1, u2[s]):
                            u2[s] = c1
                            policy_indices[s] = a_idx
                            policy[s] = state_actions[s][a_idx]

                        first_action = 0

                # stopping condition
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    return max_u1 - min_u1
                else:
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)
