# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset

from libc.stdio cimport printf

from cython.parallel import prange

import numpy as np
cimport numpy as np

from ._utils cimport sign
from ._utils cimport isclose_c
from ._utils cimport get_sorted_indices
from ._utils cimport check_end
from ._utils cimport dot_prod

from ._max_proba cimport max_proba_purec
from ._max_proba cimport max_proba_bernstein


# =============================================================================
# Extended Value Iteration Class
# =============================================================================

cdef class EVI:

    def __init__(self, nb_states, list actions_per_state, use_bernstein=0, random_state=0):
        cdef SIZE_t n, m, i, j
        self.nb_states = nb_states
        self.u1 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u2 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))

        self.bernstein_bound = use_bernstein
        self.random_state = random_state

        # allocate indices and memoryview (may slow down)
        self.sorted_indices = <SIZE_t *> malloc(nb_states * sizeof(SIZE_t))
        for i in range(nb_states):
            self.u1[i] = 0.0
            self.u2[i] = 0.0
            self.sorted_indices[i] = i

        # allocate space for matrix of max probabilities and the associated memory view
        self.mtx_maxprob = <DTYPE_t *>malloc(nb_states * nb_states * sizeof(DTYPE_t))
        self.mtx_maxprob_memview = <DTYPE_t[:nb_states, :nb_states]> self.mtx_maxprob

        n = len(actions_per_state)
        assert n == nb_states
        self.actions_per_state = <IntVectorStruct *> malloc(n * sizeof(IntVectorStruct))
        for i in range(n):
            m = len(actions_per_state[i])
            self.actions_per_state[i].dim = m
            self.actions_per_state[i].values = <SIZE_t *> malloc(m * sizeof(SIZE_t))
            for j in range(m):
                self.actions_per_state[i].values[j] = actions_per_state[i][j]

    def __dealloc__(self):
        cdef SIZE_t i
        free(self.u1)
        free(self.u2)
        free(self.mtx_maxprob)
        free(self.sorted_indices)
        for i in range(self.nb_states):
            free(self.actions_per_state[i].values)
        free(self.actions_per_state)

    cpdef DTYPE_t evi(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] estimated_probabilities,
                     DTYPE_t[:,:] estimated_rewards,
                     DTYPE_t[:,:] estimated_holding_times,
                     DTYPE_t[:,:] beta_r,
                     DTYPE_t[:,:,:] beta_p,
                     DTYPE_t[:,:] beta_tau,
                     DTYPE_t tau_max,
                     DTYPE_t r_max,
                     DTYPE_t tau,
                     DTYPE_t tau_min,
                     DTYPE_t epsilon):

        cdef SIZE_t s, i, a_idx, counter = 0
        cdef SIZE_t first_action
        cdef DTYPE_t c1
        cdef DTYPE_t min_u1, max_u1, r_optimal, v, tau_optimal
        cdef SIZE_t nb_states = self.nb_states

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef SIZE_t* sorted_indices = self.sorted_indices

        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview

        with nogil:
            c1 = u1[0]
            for i in range(nb_states):
                u1[i] = u1[i] - c1 # 0.0
                u2[i] = 0.0
                sorted_indices[i] = i
            get_sorted_indices(u1, nb_states, sorted_indices)


            while True: #counter < 5:
                for s in prange(nb_states):
                    first_action = 1
                    for a_idx in range(self.actions_per_state[s].dim):
                        if self.bernstein_bound == 0:
                            # max_proba_purec
                            # max_proba_reduced
                            max_proba_purec(estimated_probabilities[s][a_idx], nb_states,
                                        sorted_indices, beta_p[s][a_idx][0],
                                        mtx_maxprob_memview[s])
                        else:
                            max_proba_bernstein(estimated_probabilities[s][a_idx], nb_states,
                                        sorted_indices, beta_p[s][a_idx],
                                        mtx_maxprob_memview[s])
                        mtx_maxprob_memview[s][s] = mtx_maxprob_memview[s][s] - 1.
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
                            policy[s] = self.actions_per_state[s].values[a_idx]

                        first_action = 0
                counter = counter + 1
                # printf("**%d\n", counter)
                # for i in range(nb_states):
                #     printf("%.2f[%.2f] ", u1[i], u2[i])
                # printf("\n")

                # stopping condition
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    # printf("%d\n", counter)
                    return max_u1 - min_u1
                else:
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)
                    # for i in range(nb_states):
                    #     printf("%d , ", sorted_indices[i])
                    # printf("\n")


    cpdef get_uvectors(self):
#        cdef np.npy_intp shape[1]
#        shape[0] = <np.npy_intp> self.nb_states
#        npu1 = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, self.u1)
#        print(npu1)
#        npu2 = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, self.u2)
#        return npu1, npu2
        u1n = -99*np.ones((self.nb_states,))
        u2n = -99*np.ones((self.nb_states,))
        for i in range(self.nb_states):
            u1n[i] = self.u1[i]
            u2n[i] = self.u2[i]
        return u1n, u2n