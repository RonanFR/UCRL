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
from ._utils cimport get_sorted_indices
from ._utils cimport check_end
from ._utils cimport dot_prod
from ._utils cimport pos2index_2d

from ._max_proba cimport max_proba_purec
from ._max_proba cimport max_proba_bernstein


# =============================================================================
# Bias Span Constrained Extended Value Iteration Class
# =============================================================================

cdef class SpanConstrainedEVI:

    def __init__(self, nb_states, list actions_per_state, str bound_type,
                    DTYPE_t span_constraint, SIZE_t relative_vi,
                    int random_state, DTYPE_t gamma=1.):
        cdef SIZE_t n, m, i, j
        self.nb_states = nb_states
        self.u1 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u2 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))

        if bound_type == "chernoff":
            self.bound_type = CHERNOFF
        elif bound_type == "bernstein":
            self.bound_type = BERNSTEIN
        else:
            raise ValueError("Unknown bound type")

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
        self.span_constraint = span_constraint
        self.relative_vi = relative_vi

    def __dealloc__(self):
        cdef SIZE_t i
        free(self.u1)
        free(self.u2)
        free(self.mtx_maxprob)
        free(self.sorted_indices)
        for i in range(self.nb_states):
            free(self.actions_per_state[i].values)
        free(self.actions_per_state)

    def reset_u(self, u):
        for i in range(self.nb_states):
            self.u1[i] = u[i]
            self.u2[i] = 0.0

    cpdef DTYPE_t run(self, SIZE_t[:,:] policy_indices, DTYPE_t[:,:] policy,
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
                     DTYPE_t epsilon,
                     DTYPE_t span_constraint = -1,
                     SIZE_t initial_recenter = 1,
                     SIZE_t relative_vi = -1,
                     str operator_type = "T"):

        span_constraint = self.span_constraint if span_constraint < 0 else span_constraint
        relative_vi = self.relative_vi if relative_vi < 0 else relative_vi

        cdef SIZE_t isT = 1 if operator_type == "T" else 0

        cdef SIZE_t s, i, a_idx, counter = 0
        cdef SIZE_t first_action
        cdef DTYPE_t c1
        cdef DTYPE_t gamma = self.gamma
        cdef DTYPE_t min_u1, max_u1, r_optimal, v, tau_optimal
        cdef SIZE_t nb_states = self.nb_states
        cdef SIZE_t max_nb_actions = self.max_macroactions_per_state
        cdef SIZE_t action
        cdef SIZE_t exists_policy = 0

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef SIZE_t* sorted_indices = self.sorted_indices

        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview

        cdef SIZE_t* action_max_min
        cdef SIZE_t* action_max_min_indices
        cdef SIZE_t pos_mina, pos_maxa
        cdef DTYPE_t truncated_value, min_v, max_v, w1, w2
        cdef DTYPE_t* action_noise
        cdef DTYPE_t* u2_mina


        action_max_min = <SIZE_t*> malloc(nb_states * 2 * sizeof(SIZE_t))
        action_max_min_indices = <SIZE_t*> malloc(nb_states * 2 * sizeof(SIZE_t))

        action_noise = <DTYPE_t*> malloc(max_nb_actions * sizeof(DTYPE_t))
        u2_mina = <DTYPE_t*> malloc(nb_states * sizeof(DTYPE_t))

        # break ties in a random way. For each action select a random (small) noise
        local_random = np.random.RandomState(self.random_state)
        # printf('action_noise (SCEVI): ')
        for a in range(max_nb_actions):
            action_noise[a] = 1e-4 * local_random.random_sample()
        #     printf('%f ', action_noise[a])
        # printf('\n')

        with nogil:
            # let's start from the value of the previous episode
            # but recenter the value u1 = u1 -u1[0]

            c1 = u2[0] if initial_recenter else 0.
            for i in range(nb_states):
                u1[i] = u2[i] - c1
                u2[i] = 0.0
                # printf('%.7f ', u1[i])
                sorted_indices[i] = i # reinitialize indices
            get_sorted_indices(u1, nb_states, sorted_indices) # get sorted indices
            # printf('\n')

            while True:
                exists_policy = 1 #let's hope for a policy

                # each state is independently updated
                for s in prange(nb_states):
                    first_action = 1

                    # for each action available in the state
                    for a_idx in range(self.actions_per_state[s].dim):
                        # get the real action (used to update the policy)
                        action = self.actions_per_state[s].values[a_idx]

                        ########################################################
                        # compute optimistic value
                        ########################################################
                        if self.bound_type == CHERNOFF:
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
                        v = r_optimal + gamma * dot_prod(mtx_maxprob_memview[s], u1, nb_states) * tau
                        tau_optimal = min(tau_max, max(
                            max(tau_min, r_optimal/r_max),
                            estimated_holding_times[s][a_idx] - sign(v) * beta_tau[s][a_idx]
                        ))
                        c1 = v / tau_optimal + gamma * u1[s]

                        # printf('%d, %d: %f [%f]\n', s, a_idx, c1, v)

                        ########################################################
                        # Update u2[s] (taking the maximum)
                        # Update u_min[s] (taking the minimum)
                        # Update max and min action in this state
                        ########################################################
                        pos_mina = pos2index_2d(nb_states, max_nb_actions, s, 0)
                        pos_maxa = pos2index_2d(nb_states, max_nb_actions, s, 1)
                        # printf("%.3f, %.3f (%d, %d) -> ", u2_mina[s], u2[s], action_max_min_indices[pos_mina], action_max_min_indices[pos_maxa])
                        if first_action:
                            u2[s] = c1
                            u2_mina[s] = c1
                            # update max action (column 1)
                            action_max_min[pos_maxa] = action
                            action_max_min_indices[pos_maxa] = a_idx
                            # update min action (column 0)
                            action_max_min[pos_mina] = action
                            action_max_min_indices[pos_mina] = a_idx
                        else:
                            if c1 + action_noise[a_idx] > u2[s] + action_noise[action_max_min_indices[pos_maxa]]:
                                u2[s] = c1

                                # update max action (column 1)
                                action_max_min[pos_maxa] = action
                                action_max_min_indices[pos_maxa] = a_idx

                            if c1 + action_noise[a_idx] < u2_mina[s] + action_noise[action_max_min_indices[pos_mina]]:
                                u2_mina[s] = c1

                                # update min action (column 0)
                                action_max_min[pos_mina] = action
                                action_max_min_indices[pos_mina] = a_idx

                        # printf("%.3f, %.3f (%d, %d) \n\n", u2_mina[s], u2[s], action_max_min_indices[pos_mina], action_max_min_indices[pos_maxa])

                        first_action = 0

                counter = counter + 1
                # printf("**%d\n", counter)
                # for i in range(nb_states):
                #     printf("%.7f[%.7f] \n", u1[i], u2[i])
                # printf("\n")

                ########################################################
                # Truncate value function based on span
                # Compute "greedy" policy
                ########################################################
                min_v = u2[0]
                max_v = u2[0]
                for s in range(1, nb_states):
                    if u2[s] < min_v:
                        min_v = u2[s]
                    elif u2[s] > max_v:
                        max_v = u2[s]

                # printf('mm_v: %f, %f\n', min_v, max_v)
                for s in range(nb_states):
                    truncated_value = min(u2[s], min_v + span_constraint)

                    if u2_mina[s] > truncated_value:
                        pos_mina = pos2index_2d(nb_states, max_nb_actions, s, 0)
                        pos_maxa = pos2index_2d(nb_states, max_nb_actions, s, 1)

                        if isT:
                            # at this iteration does not exist a policy fot T
                            exists_policy = 0

                            # reset policy
                            policy_indices[s, 0] = -1
                            policy_indices[s, 1] = -1
                            policy[s, 0] = -1
                            policy[s, 1] = -1
                        else:
                            # printf('state: %d\n', s)
                            # the new value is the one of the min action
                            truncated_value = u2_mina[s]

                            # the new policy is just the min action
                            policy_indices[s, 0] = action_max_min_indices[pos_mina]
                            policy_indices[s, 1] = action_max_min_indices[pos_maxa]
                            policy[s, 0] = 1.
                            policy[s, 1] = 0

                    else:

                        pos_mina = pos2index_2d(nb_states, max_nb_actions, s, 0)
                        pos_maxa = pos2index_2d(nb_states, max_nb_actions, s, 1)
                        w1 = w2 = 0.

                        if isclose_c(u2_mina[s], u2[s]):
                            # if the values are equal pick the action according to the noise
                            if u2_mina[s] + action_noise[action_max_min_indices[pos_mina]] > u2[s] + action_noise[action_max_min_indices[pos_maxa]]:
                                w1 = 1
                            else:
                                w2 = 1.
                        elif isclose_c(u2_mina[s], truncated_value):
                            w1 = 1.
                        elif isclose_c(u2[s], truncated_value):
                            w2 = 1.
                        else:
                            w1 = (u2[s] - truncated_value) / (u2[s] - u2_mina[s])
                            w2 = (truncated_value - u2_mina[s]) / (u2[s] - u2_mina[s])

                        if not isclose_c(truncated_value, w1*u2_mina[s] + w2 * u2[s])\
                            or w1 < 0 or w2 < 0 or not isclose_c(w1+w2, 1.):
                            printf('(%d) %.3f %.3f %.3f [%.3f, %.3f]\n',s,u2_mina[s], u2[s], truncated_value, w1, w2)
                            return -22

                        ##########################
                        # Compute "greedy" policy
                        ##########################
                        policy_indices[s, 0] = action_max_min_indices[pos_mina]
                        policy_indices[s, 1] = action_max_min_indices[pos_maxa]
                        policy[s, 0] = w1
                        policy[s, 1] = w2

                    # update with truncated value
                    u2[s] = truncated_value

                # stopping condition
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    if exists_policy == 0 and isT:
                        return -21

                    free(action_max_min)
                    free(action_max_min_indices)
                    free(u2_mina)
                    free(action_noise)
                    return max_u1 - min_u1
                else:
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)

                    if relative_vi:
                        c1 = u1[0]
                        for i in range(nb_states):
                            u1[i] = u1[i] - c1
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

    cpdef get_span_constraint(self):
        return self.span_constraint
