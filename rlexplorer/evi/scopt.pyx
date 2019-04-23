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
from cpython cimport bool

import numpy as np
cimport numpy as np

from ._utils cimport sign
from ._utils cimport isclose_c
from ._utils cimport get_sorted_indices
from ._utils cimport check_end
from ._utils cimport pos2index_2d

from ._lpproba cimport LPPROBA_bernstein, LPPROBA_hoeffding


# =============================================================================
# Bias Span Constrained Extended Value Iteration Class
# =============================================================================

cdef class SCOPT:

    def __init__(self, nb_states, list actions_per_state, str bound_type,
                    DTYPE_t span_constraint, SIZE_t relative_vi,
                    int random_state, int augmented_reward, DTYPE_t gamma=1.):
        cdef SIZE_t n, m, i, j
        self.nb_states = nb_states
        self.u1 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u2 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u3_mina = <DTYPE_t*> malloc(nb_states * sizeof(DTYPE_t))
        self.u3 = <DTYPE_t*> malloc(nb_states * sizeof(DTYPE_t))

        if bound_type == "hoeffding":
            self.bound_type = CHERNOFF
        elif bound_type == "bernstein":
            self.bound_type = BERNSTEIN
        else:
            raise ValueError("Unknown bound type")

        self.operator_type = TOP
        self.augment_reward = augmented_reward

        # allocate indices and memoryview (may slow down)
        self.sorted_indices = <SIZE_t *> malloc(nb_states * sizeof(SIZE_t))
        for i in range(nb_states):
            self.u1[i] = 0.0
            self.u2[i] = 0.0
            self.u3_mina[i] = -1.0
            self.u3[i] = -1.0
            self.sorted_indices[i] = i

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
        free(self.sorted_indices)
        free(self.u3_mina)
        free(self.u3)
        for i in range(self.nb_states):
            free(self.actions_per_state[i].values)
        free(self.actions_per_state)

    def reset_u(self, u):
        for i in range(self.nb_states):
            self.u1[i] = u[i]
            self.u2[i] = u[i]
            self.u3_mina[i] = -1.0
            self.u3[i] = -1.0

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
                     SIZE_t relative_vi = -1):

        cdef SIZE_t ITERATIONS_LIMIT = 1000000
        span_constraint = self.span_constraint / tau if span_constraint < 0 else span_constraint / tau
        relative_vi = self.relative_vi if relative_vi < 0 else relative_vi
        cdef OperatorType opt_type = self.operator_type

        cdef SIZE_t s, i, a_idx, counter = 0
        cdef SIZE_t first_action
        cdef DTYPE_t c1, dotp
        cdef DTYPE_t gamma = self.gamma
        cdef DTYPE_t min_u1, max_u1, r_optimal, v, tau_optimal
        cdef SIZE_t nb_states = self.nb_states
        cdef SIZE_t max_nb_actions = self.max_macroactions_per_state
        cdef SIZE_t action
        cdef SIZE_t exists_policy = 0

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef SIZE_t* sorted_indices = self.sorted_indices

        cdef SIZE_t* action_max_min
        cdef SIZE_t* action_max_min_indices
        cdef SIZE_t pos_mina, pos_maxa
        cdef DTYPE_t th, min_v, max_v, w1, w2
        cdef DTYPE_t* action_noise
        cdef DTYPE_t* u3_mina = self.u3_mina
        cdef DTYPE_t* u3 = self.u3


        # note that 2 stands for the number of actions (bernoulli policy)
        action_max_min = <SIZE_t*> malloc(nb_states * 2 * sizeof(SIZE_t))
        action_max_min_indices = <SIZE_t*> malloc(nb_states * 2 * sizeof(SIZE_t))

        action_noise = <DTYPE_t*> malloc(max_nb_actions * sizeof(DTYPE_t))
        # u3_mina = <DTYPE_t*> malloc(nb_states * sizeof(DTYPE_t))
        # u3 = <DTYPE_t*> malloc(nb_states * sizeof(DTYPE_t))

        epsilon = max(1e-12, epsilon)

        # break ties in a random way. For each action select a random (small) noise
        local_random = np.random.RandomState(self.random_state)
        # printf('action_noise (SCEVI): ')
        cdef DTYPE_t noise_factor = 0.1 * min(1e-6, epsilon)
        for a in range(max_nb_actions):
            action_noise[a] = noise_factor * local_random.random_sample()
        #     printf('%f ', action_noise[a])
        # printf('\n')


        # ITERATIONS_LIMIT = min(ITERATIONS_LIMIT, max(nb_states * max_nb_actions * 200, 30000))

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
                            dotp = LPPROBA_hoeffding(u1, estimated_probabilities[s][a_idx], nb_states,
                                            sorted_indices, beta_p[s][a_idx][0], 0)
                        else:
                            dotp = LPPROBA_bernstein(u1, estimated_probabilities[s][a_idx], nb_states,
                                        sorted_indices, beta_p[s][a_idx], 0, 0)

                        r_optimal = min(tau_max*r_max,
                                        estimated_rewards[s][a_idx] + beta_r[s][a_idx])
                        v = r_optimal + gamma * dotp * tau
                        tau_optimal = min(tau_max, max(
                            max(tau_min, r_optimal/r_max),
                            estimated_holding_times[s][a_idx] - sign(v) * beta_tau[s][a_idx]
                        ))
                        c1 = v / tau_optimal + gamma * (1. - tau) * u1[s]

                        # printf('%d, %d: %f [%f]\n', s, a_idx, c1, v)

                        ########################################################
                        # Update u3[s] (taking the maximum)
                        # Update max action in this state
                        ########################################################
                        # we have just to actions (pick action 1 [max] in state s)
                        pos_maxa = pos2index_2d(nb_states, 2, s, 1)
                        # printf("%.3f, %.3f (%d, %d) -> ", u3_mina[s], u3[s], action_max_min_indices[pos_mina], action_max_min_indices[pos_maxa])
                        if first_action:
                            u3[s] = c1 # initialize the maximum

                            # update max action (column 1)
                            action_max_min[pos_maxa] = action
                            action_max_min_indices[pos_maxa] = a_idx

                        else:
                            if c1 + action_noise[a_idx] > u3[s] + action_noise[action_max_min_indices[pos_maxa]]:
                                u3[s] = c1

                                # update max action (column 1)
                                action_max_min[pos_maxa] = action
                                action_max_min_indices[pos_maxa] = a_idx


                        # printf("%.3f, %.3f (%d, %d) \n\n", u3_mina[s], u3[s], action_max_min_indices[pos_mina], action_max_min_indices[pos_maxa])

                        first_action = 0

                counter = counter + 1
                # printf("**%d\n", counter)
                # for i in range(nb_states):
                #     printf("%.7f[%.7f] \n", u1[i], u3[i])
                # printf("\n")

                ########################################################
                # Truncate value function based on span
                ########################################################
                min_v = u3[0]
                max_v = u3[0]
                for s in range(1, nb_states):
                    if u3[s] < min_v:
                        min_v = u3[s]
                    elif u3[s] > max_v:
                        max_v = u3[s]

                th = min_v + span_constraint
                for s in range(nb_states):
                    if opt_type == TOP:
                        u2[s] = min(u3[s], th)
                ################################################################
                # stopping condition
                ################################################################
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    exists_policy = 1
                    ###############################
                    # Compute the associated policy
                    ###############################
                    for s in range(nb_states):
                        # we have just two actions (pick action 0 [min] and 1 [max] in state s)
                        pos_mina = pos2index_2d(nb_states, 2, s, 0)
                        pos_maxa = pos2index_2d(nb_states, 2, s, 1)
                        if u3[s] < th:
                            # the policy is just the greedy action
                            # we put the same index
                            # this is true both for T and N (for example i have not computed the min action for T, and I don't need it)
                            policy_indices[s, 0] = action_max_min_indices[pos_maxa]
                            policy_indices[s, 1] = action_max_min_indices[pos_maxa]
                            policy[s, 0] = 0
                            policy[s, 1] = 1.
                        else:
                            if opt_type == TOP:
                                ############################################################
                                # Compute pessimistic action
                                ############################################################
                                # for each action available in the state
                                first_action = 1
                                for a_idx in range(self.actions_per_state[s].dim):
                                    # get the real action (used to update the policy)
                                    action = self.actions_per_state[s].values[a_idx]
                                    ########################################################
                                    # compute pessimistic value
                                    ########################################################
                                    if self.bound_type == CHERNOFF:
                                        dotp = LPPROBA_hoeffding(u1, estimated_probabilities[s][a_idx], nb_states,
                                                        sorted_indices, beta_p[s][a_idx][0], 1)
                                    else:
                                        dotp = LPPROBA_bernstein(u1, estimated_probabilities[s][a_idx], nb_states,
                                                    sorted_indices, beta_p[s][a_idx], 0, 1)
                                    if self.augment_reward == 0:
                                        r_optimal = max(0., min(r_max, estimated_rewards[s][a_idx]) - beta_r[s][a_idx])
                                    else:
                                        r_optimal = 0.
                                    v = r_optimal + gamma * dotp * tau
                                    tau_optimal = min(tau_max, max(
                                        max(tau_min, r_optimal/r_max),
                                        estimated_holding_times[s][a_idx] - sign(v) * beta_tau[s][a_idx]
                                    ))
                                    c1 = v / tau_optimal + gamma * (1. - tau) * u1[s]

                                    ########################################################
                                    # Update u3_mina[s] (taking the minimum)
                                    # Update min action in this state
                                    ########################################################
                                    # we have just to actions (pick action 0 [min] in state s)
                                    pos_mina = pos2index_2d(nb_states, 2, s, 0)
                                    if first_action:
                                        u3_mina[s] = c1 # initialize the maximum

                                        # update min action (column 0)
                                        action_max_min[pos_mina] = action
                                        action_max_min_indices[pos_mina] = a_idx

                                        first_action = 0
                                    else:
                                        if c1 + action_noise[a_idx] < u3_mina[s] + action_noise[action_max_min_indices[pos_mina]]:
                                            u3_mina[s] = c1

                                            # update min action (column 0)
                                            action_max_min[pos_mina] = action
                                            action_max_min_indices[pos_mina] = a_idx

                            ###### finish to update the policy

                            if u3_mina[s] > th and opt_type == TOP:
                                exists_policy = 0

                                # reset policy
                                policy_indices[s, 0] = -1
                                policy_indices[s, 1] = -1
                                policy[s, 0] = -1
                                policy[s, 1] = -1
                            else:
                                w1 = w2 = 0.

                                if isclose_c(u3_mina[s], u3[s]):
                                    # if the values are equal pick the action according to the noise
                                    if u3_mina[s] + action_noise[action_max_min_indices[pos_mina]] > u3[s] + action_noise[action_max_min_indices[pos_maxa]]:
                                        w1 = 1
                                    else:
                                        w2 = 1.
                                elif isclose_c(u3_mina[s], th):
                                    w1 = 1.
                                elif isclose_c(u3[s], th):
                                    w2 = 1.
                                else:
                                    w1 = (u3[s] - th) / (u3[s] - u3_mina[s])
                                    w2 = (th - u3_mina[s]) / (u3[s] - u3_mina[s])

                                if not isclose_c(th, w1 * u3_mina[s] + w2 * u3[s])\
                                    or w1 < 0 or w2 < 0 or not isclose_c(w1 + w2, 1.):
                                    printf('(%d) %.3f %.3f %.3f [%.3f, %.3f]\n',s,u3_mina[s], u3[s], th, w1, w2)
                                    free(action_max_min)
                                    free(action_max_min_indices)
                                    # free(u3_mina)
                                    # free(u3)
                                    free(action_noise)
                                    return -22

                                ##########################
                                # Compute "greedy" policy
                                ##########################
                                policy_indices[s, 0] = action_max_min_indices[pos_mina]
                                policy_indices[s, 1] = action_max_min_indices[pos_maxa]
                                policy[s, 0] = w1
                                policy[s, 1] = w2


                    if exists_policy == 0 and opt_type == TOP:
                        return -21
                    free(action_max_min)
                    free(action_max_min_indices)
                    # free(u3_mina)
                    # free(u3)
                    free(action_noise)
                    return max_u1 - min_u1
                else:
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)

                    if relative_vi:
                        c1 = u1[0]
                    if relative_vi:
                        c1 = u1[0]
                        for i in range(nb_states):
                            u1[i] = u1[i] - c1
                    #     printf("%d , ", sorted_indices[i])
                    # printf("\n")

                if counter > ITERATIONS_LIMIT:
                    free(action_max_min)
                    free(action_max_min_indices)
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

    cpdef get_u3vectors(self):
        u3_mina = -99*np.ones((self.nb_states,))
        u3_maxa = -99*np.ones((self.nb_states,))
        for i in range(self.nb_states):
            u3_mina[i] = self.u3_mina[i]
            u3_maxa[i] = self.u3[i]
        return u3_mina, u3_maxa

    cpdef get_span_constraint(self):
        return self.span_constraint

    cpdef get_operator_type(self):
        return 'T' if self.operator_type == TOP else 'N'

    cpdef use_relative_vi(self):
        return self.relative_vi
