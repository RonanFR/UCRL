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
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stdio cimport printf

from cython.parallel import prange

import numpy as np
cimport numpy as np

from ._utils cimport sign
from ._utils cimport isclose_c
from ._utils cimport get_sorted_indices, quicksort_indices
from ._utils cimport isinsortedvector
from ._utils cimport check_end_sparse
from ._utils cimport dot_prod
from ._utils cimport sparse_dot_prod
from ._utils cimport pos2index_2d

from ._max_proba cimport max_proba_purec
from ._max_proba cimport max_proba_bernstein


# =============================================================================
# TRUNCATED Extended Value Iteration Class
# =============================================================================

cdef class TEVI:

    def __init__(self, nb_states, list actions_per_state, bound_type, int random_state, DTYPE_t gamma=1.):
        cdef SIZE_t n, m, i, j
        self.nb_states = nb_states
        self.u1 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u2 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))

        if bound_type == "chernoff":
            self.bound_type = CHERNOFF
#        elif bound_type == "bernstein":
#            self.bound_type = BERNSTEIN
        else:
            raise ValueError("Unknown bound type")

        # allocate indices and memoryview (may slow down)
        self.sorted_indices = <SIZE_t *> malloc(nb_states * sizeof(SIZE_t))
        self.sorted_indices_reachable = <SIZE_t *> malloc(nb_states * sizeof(SIZE_t))
        for i in range(nb_states):
            self.u1[i] = 0.0
            self.u2[i] = 0.0
            self.sorted_indices[i] = i
            self.sorted_indices_reachable[i] = i

        # allocate space for matrix of max probabilities and the associated memory view
        self.mtx_maxprob = <DTYPE_t *>malloc(nb_states * nb_states * sizeof(DTYPE_t))
        self.mtx_maxprob_memview = <DTYPE_t[:nb_states, :nb_states]> self.mtx_maxprob

        n = len(actions_per_state)
        assert n == nb_states
        self.actions_per_state = <IntVectorStruct *> malloc(n * sizeof(IntVectorStruct))
        self.max_macroactions_per_state = 0
        for i in range(n):
            m = len(actions_per_state[i])
            if m > self.max_macroactions_per_state:
                self.max_macroactions_per_state = m
            self.actions_per_state[i].dim = m
            self.actions_per_state[i].values = <SIZE_t *> malloc(m * sizeof(SIZE_t))
            for j in range(m):
                self.actions_per_state[i].values[j] = actions_per_state[i][j]
        self.random_state = random_state

        self.gamma = gamma

    def __dealloc__(self):
        cdef SIZE_t i
        free(self.u1)
        free(self.u2)
        free(self.mtx_maxprob)
        free(self.sorted_indices)
        free(self.sorted_indices_reachable)
        for i in range(self.nb_states):
            free(self.actions_per_state[i].values)
        free(self.actions_per_state)

    def reset_u(self, u):
        for i in range(self.nb_states):
            self.u1[i] = u[i]
            self.u2[i] = u[i]

    cpdef DTYPE_t run(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] estimated_probabilities,
                     DTYPE_t[:,:] estimated_rewards,
                     DTYPE_t[:,:] estimated_holding_times,
                     DTYPE_t[:,:] beta_r,
                     DTYPE_t[:,:,:] beta_p,
                     DTYPE_t[:,:] beta_tau,
                     SIZE_t[:,:] is_truncated_sa,
                     SIZE_t[:] unreachable_states,
                     SIZE_t[:] states_terminalcond,
                     DTYPE_t tau_max,
                     DTYPE_t r_max,
                     DTYPE_t tau,
                     DTYPE_t tau_min,
                     DTYPE_t epsilon,
                     DTYPE_t span_constraint=-1,
                     SIZE_t initial_recenter = 1,
                     SIZE_t relative_vi = 0
                     ):

        cdef SIZE_t ITERATIONS_LIMIT = 1000000

        cdef SIZE_t s, i, a_idx, counter = 0, j
        cdef SIZE_t first_action
        cdef DTYPE_t c1, dotp, new_span, old_span
        cdef DTYPE_t gamma = self.gamma
        cdef DTYPE_t min_u1, max_u1, r_optimal, v, tau_optimal
        cdef SIZE_t nb_states = self.nb_states
        cdef SIZE_t max_nb_actions = self.max_macroactions_per_state
        cdef SIZE_t action

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef SIZE_t* sorted_indices = self.sorted_indices
        cdef SIZE_t* sorted_indices_reachable = self.sorted_indices_reachable
        cdef SIZE_t num_reachable_state = nb_states - len(unreachable_states)
        cdef SIZE_t nb_states_tc = len(states_terminalcond)

        cdef BoundType bound_type = CHERNOFF

        if beta_p.shape[2] > 1 and num_reachable_state < nb_states:
            return -32
        if beta_p.shape[2] == nb_states and num_reachable_state == nb_states:
            bound_type = BERNSTEIN

        if num_reachable_state == 0:
            return -14

        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview

        cdef DTYPE_t* action_noise

        action_noise = <DTYPE_t*> malloc(max_nb_actions * sizeof(DTYPE_t))

        epsilon = max(1e-12, epsilon)

        local_random = np.random.RandomState(self.random_state)
        # printf('action_noise (EVI): ')
        cdef DTYPE_t noise_factor = 0.1 * min(1e-6, epsilon)
        for a in range(max_nb_actions):
            action_noise[a] = noise_factor * local_random.random_sample()
        #     printf('%f ', action_noise[a])
        # printf('\n')

        ITERATIONS_LIMIT = min(ITERATIONS_LIMIT, nb_states * max_nb_actions * 200)

        old_span = np.finfo('f').max

        with nogil:
            c1 = u2[0] if initial_recenter else 0.
            for i in range(nb_states):
                u1[i] = u2[i] - c1 # 0.0
                u2[i] = 0.0
                sorted_indices[i] = i
            get_sorted_indices(u1, nb_states, sorted_indices)
            # quicksort_indices(u1, 0, nb_states-1, sorted_indices)
            # get sorted index containing only reachable states
            if num_reachable_state < nb_states:
                j = 0
                for i in range(nb_states):
                    if isinsortedvector(sorted_indices[i], unreachable_states, nb_states - num_reachable_state) < 0:
                        sorted_indices_reachable[j] = sorted_indices[i]
                        j += 1

                if j != num_reachable_state:
                    return -12

            while True:
                for s in prange(nb_states):
                    first_action = 1

                    for a_idx in range(self.actions_per_state[s].dim):
                        action = self.actions_per_state[s].values[a_idx]
                        if bound_type == CHERNOFF:
                            if is_truncated_sa[s, a_idx]:
                                max_proba_purec(estimated_probabilities[s][a_idx], num_reachable_state,
                                                            sorted_indices_reachable,
                                                            beta_p[s][a_idx][0],
                                                            mtx_maxprob_memview[s], 0)
                                dotp = sparse_dot_prod(mtx_maxprob_memview[s], u1, sorted_indices_reachable, num_reachable_state)
                                dotp -= u1[s]
                            else:
                                max_proba_purec(estimated_probabilities[s][a_idx], nb_states,
                                            sorted_indices, beta_p[s][a_idx][0],
                                            mtx_maxprob_memview[s], 0)
                                mtx_maxprob_memview[s][s] = mtx_maxprob_memview[s][s] - 1.
                                dotp = dot_prod(mtx_maxprob_memview[s], u1, nb_states)
                        else:
                            max_proba_bernstein(estimated_probabilities[s][a_idx], nb_states,
                                        sorted_indices, beta_p[s][a_idx],
                                        mtx_maxprob_memview[s], 0)
                            mtx_maxprob_memview[s][s] = mtx_maxprob_memview[s][s] - 1.
                            dotp = dot_prod(mtx_maxprob_memview[s], u1, nb_states)

                        r_optimal = min(tau_max*r_max,
                                        estimated_rewards[s][a_idx] + beta_r[s][a_idx])
                        v = r_optimal + gamma * dotp * tau
                        tau_optimal = min(tau_max, max(
                            max(tau_min, r_optimal/r_max),
                            estimated_holding_times[s][a_idx] - sign(v) * beta_tau[s][a_idx]
                        ))
                        c1 = v / tau_optimal + gamma * u1[s]

                        # printf('%d, %d: %f [%f]\n', s, a_idx, c1, v)

                        if first_action:
                            u2[s] = c1
                            policy[s] = action
                            policy_indices[s] = a_idx
                        elif c1 + action_noise[a_idx] > u2[s] + action_noise[policy_indices[s]]:
                            u2[s] = c1
                            policy[s] = action
                            policy_indices[s] = a_idx

                        first_action = 0

                counter = counter + 1
                # printf("**%d\n", counter)
                # for i in range(nb_states):
                #     printf("%.7f[%.7f] \n", u1[i], u2[i])
                # printf("\n")

                # stopping condition
                new_span = check_end_sparse(u2, u1, states_terminalcond, nb_states_tc, &min_u1, &max_u1)
                if old_span - new_span < epsilon:
                    # printf("%d\n", counter)
                    free(action_noise)
                    return max_u1 - min_u1
                else:
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)
                    # quicksort_indices(u1, 0, nb_states-1, sorted_indices)
                    # get sorted index containing only reachable states
                    if num_reachable_state < nb_states:
                        j = 0
                        for i in range(nb_states):
                            if isinsortedvector(sorted_indices[i], unreachable_states, nb_states - num_reachable_state) < 0:
                                sorted_indices_reachable[j] = sorted_indices[i]
                                j += 1

                        if j != num_reachable_state:
                            return -14

                    if relative_vi:
                        c1 = u1[0]
                        for i in range(nb_states):
                            u1[i] = u1[i] - c1
                    #     printf("%d , ", sorted_indices[i])
                    # printf("\n")

                old_span = new_span

                if counter > ITERATIONS_LIMIT:
                    free(action_noise)
                    return -54


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
