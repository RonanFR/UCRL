


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
from libc.math cimport log

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
from ._max_proba cimport max_proba_purec2

from libc.stdio cimport printf

# =============================================================================
# Extended Value Iteration Class
# =============================================================================

cdef class FreeEVIAlg1:

    def __init__(self,
                 int nb_states,
                 int nb_options,
                 list macro_actions_per_state,
                 list mdp_actions_per_state,
                 int threshold,
                 list option_policies,
                 list reachable_states_per_option,
                 list options_terminating_conditions):
        cdef SIZE_t n, m, i, j, idx
        cdef SIZE_t max_states_per_option = 0
        self.nb_states = nb_states
        self.nb_options = nb_options
        self.threshold = threshold
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

        n = len(macro_actions_per_state)
        assert n == nb_states
        self.actions_per_state = <IntVectorStruct *> malloc(n * sizeof(IntVectorStruct))
        for i in range(n):
            m = len(macro_actions_per_state[i])
            self.actions_per_state[i].dim = m
            self.actions_per_state[i].values = <SIZE_t *> malloc(m * sizeof(SIZE_t))
            for j in range(m):
                self.actions_per_state[i].values[j] = macro_actions_per_state[i][j]

        n = len(reachable_states_per_option)
        assert n == nb_options
        self.reachable_states_per_option = <IntVectorStruct *> malloc(n * sizeof(IntVectorStruct))
        # self.x = <DoubleVectorStruct *> malloc(n * sizeof(DoubleVectorStruct))
        self.r_tilde_opt = <DoubleVectorStruct *> malloc(nb_options * sizeof(DoubleVectorStruct))
        self.mu_opt = <DoubleVectorStruct *> malloc(nb_options * sizeof(DoubleVectorStruct))
        self.cn_opt = <DTYPE_t *> malloc(nb_options * sizeof(DTYPE_t))

        self.options_policies = <SIZE_t *>malloc(nb_options * nb_states * sizeof(SIZE_t)) # (O x S)
        self.options_policies_indices_mdp = <SIZE_t *>malloc(nb_options * nb_states * sizeof(SIZE_t)) # (O x S)
        self.options_terminating_conditions = <DTYPE_t *>malloc(nb_options * nb_states * sizeof(SIZE_t)) # (O x S)

        for i in range(n):
            m = len(reachable_states_per_option[i])
            if m > max_states_per_option:
                max_states_per_option = m
            self.reachable_states_per_option[i].dim = m
            self.reachable_states_per_option[i].values = <SIZE_t *> malloc(m * sizeof(SIZE_t))

            # self.x[i].dim = m
            # self.x[i].values = <DTYPE_t *> malloc(m * sizeof(DTYPE_t))

            self.r_tilde_opt[i].dim = m
            self.r_tilde_opt[i].values = <DTYPE_t *> malloc(m * sizeof(DTYPE_t))

            self.mu_opt[i].dim = m
            self.mu_opt[i].values = <DTYPE_t *> malloc(m * sizeof(DTYPE_t))

            for j in range(m):
                self.reachable_states_per_option[i].values[j] = reachable_states_per_option[i][j]

            for j in range(nb_states): # index of the state to be considered
                idx = pos2index_2d(nb_options, nb_states, i, j)
                self.options_policies[idx] = option_policies[i][j]
                self.options_terminating_conditions[idx] = options_terminating_conditions[i][j]
                for kk, kk_v in enumerate(mdp_actions_per_state[j]): # save the index of the primitive action
                    if kk_v == self.options_policies[idx]:
                        self.options_policies_indices_mdp[idx] = kk
                        break

        self.sorted_indices_mu = <SIZE_t *> malloc(nb_states * max_states_per_option * sizeof(SIZE_t))
        self.xx = <DTYPE_t *> malloc(nb_states * max_states_per_option * sizeof(DTYPE_t))
        self.max_reachable_states_per_opt = max_states_per_option
        self.beta_mu_p = <DTYPE_t *> malloc(nb_options * sizeof(DTYPE_t))

    def __dealloc__(self):
        cdef SIZE_t i
        free(self.u1)
        free(self.u2)
        free(self.mtx_maxprob)
        free(self.sorted_indices)
        for i in range(self.nb_states):
            free(self.actions_per_state[i].values)
        free(self.actions_per_state)
        for i in range(self.nb_options):
            free(self.reachable_states_per_option[i].values)
            #free(self.x[i].values)
            free(self.mu_opt[i].values)
            free(self.r_tilde_opt[i].values)
        free(self.mu_opt)
        free(self.r_tilde_opt)
        free(self.reachable_states_per_option)
        #free(self.x)
        free(self.cn_opt)
        free(self.options_terminating_conditions)
        free(self.options_policies)
        free(self.options_policies_indices_mdp)
        free(self.sorted_indices_mu)
        free(self.xx)
        free(self.beta_mu_p)

    cpdef compute_mu_info(self,
                          DTYPE_t[:,:,:] estimated_probabilities_mdp,
                          DTYPE_t[:,:] estimated_rewards_mdp,
                          DTYPE_t[:,:] beta_r,
                          SIZE_t[:,:] nb_observations_mdp,
                          DTYPE_t range_mu_p,
                          DTYPE_t total_time,
                          DTYPE_t delta,
                          DTYPE_t max_nb_actions):
        cdef SIZE_t nb_options = self.nb_options
        cdef SIZE_t o, opt_nb_states, i, j, s, nexts, idx, mdp_index_a
        cdef DTYPE_t prob, condition_nb
        cdef SIZE_t visits = 0



        for o in range(nb_options):
            option_reach_states = self.reachable_states_per_option[o]

            opt_nb_states = option_reach_states.dim

            Q_o = np.zeros((opt_nb_states, opt_nb_states))

            visits = 0
            for i in range(option_reach_states.dim):
                s = option_reach_states.values[i]
                # a = self.options_policies[pos2index_2d(nb_options, self.nb_states, o, s)]
                mdp_index_a = self.options_policies_indices_mdp[pos2index_2d(nb_options, self.nb_states, o, s)]

                self.r_tilde_opt[o].values[i] = estimated_rewards_mdp[s, mdp_index_a] + beta_r[s,mdp_index_a]

                visits = nb_observations_mdp[s, mdp_index_a]

                for j in range(option_reach_states.dim):
                    nexts = option_reach_states.values[j]
                    idx = pos2index_2d(nb_options, self.nb_states, o, nexts)
                    prob = estimated_probabilities_mdp[s][mdp_index_a][nexts]
                    Q_o[i,j] = (1. - self.options_terminating_conditions[idx]) * prob
            e_m = np.ones((opt_nb_states,1))
            q_o = e_m - np.dot(Q_o, e_m)


            self.beta_mu_p[o] =  range_mu_p * np.sqrt(14 * opt_nb_states * log(2 * max_nb_actions
                    * (total_time + 1)/delta)
                                       / np.maximum(1, visits))

            Pprime_o = np.concatenate((q_o, Q_o[:, 1:]), axis=1)
            assert np.allclose(np.sum(Pprime_o, axis=1), np.ones(opt_nb_states))

            Pap = (Pprime_o + np.eye(opt_nb_states)) / 2.
            D, U = np.linalg.eig(np.transpose(Pap))  # eigen decomposition of transpose of P
            sorted_indices = np.argsort(np.real(D))
            mu = np.transpose(np.real(U))[sorted_indices[len(sorted_indices)-1]]
            mu_sum =  np.sum(mu)  # stationary distribution

            for i in range(opt_nb_states):
                self.mu_opt[o].values[i] = mu[i] / mu_sum

            P_star = np.repeat(np.array(mu, ndmin=2), opt_nb_states, axis=0)  # limiting matrix

            # Compute deviation matrix
            I = np.eye(opt_nb_states)  # identity matrix
            Z = np.linalg.inv(I - Pap + P_star)  # fundamental matrix
            H = np.dot(Z, I - P_star)  # deviation matrix

            condition_nb = 0  # condition number of deviation matrix
            for i in range(0, opt_nb_states):  # Seneta's condition number
                for j in range(i + 1, opt_nb_states):
                    condition_nb = max(condition_nb, 0.5 * np.linalg.norm(H[i, :] - H[j, :], ord=1))
            self.cn_opt[o] = condition_nb

    cpdef DTYPE_t run(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] p_hat,
                     DTYPE_t[:,:] r_hat_mdp,
                     DTYPE_t[:,:] beta_p,
                     DTYPE_t[:,:] beta_r_mdp,
                     DTYPE_t r_max,
                     DTYPE_t epsilon):

        cdef SIZE_t s, i, a_idx, action, sub_dim, counter = 0
        cdef SIZE_t first_action, idx, o
        cdef DTYPE_t c1
        cdef DTYPE_t min_u1, max_u1, r_optimal, v, tau_optimal
        cdef SIZE_t nb_states = self.nb_states
        cdef SIZE_t threshold = self.threshold

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef SIZE_t* sorted_indices = self.sorted_indices
        cdef SIZE_t* sorted_indices_mu = self.sorted_indices_mu

        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview

        cdef DoubleVectorStruct* mu_opt = self.mu_opt
        # cdef DoubleVectorStruct* x = self.x
        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DTYPE_t* xx = self.xx


        with nogil:
            for i in range(nb_states):
                u1[i] = 0 #u1[i] - u1[0]
                sorted_indices[i] = i

            while True: #counter < 5:
                for s in prange(nb_states):
                    # printf("state: %d\n", s)
                    first_action = 1
                    for a_idx in range(self.actions_per_state[s].dim):
                        # printf("action_idx: %d\n", a_idx)
                        action = self.actions_per_state[s].values[a_idx]
                        # printf("action: %d [%d]\n", action, threshold)

                        # max_proba_purec
                        # max_proba_reduced
                        max_proba_purec(p_hat[s][a_idx], nb_states,
                                    sorted_indices, beta_p[s][a_idx],
                                    mtx_maxprob_memview[s])

                        mtx_maxprob_memview[s][s] = mtx_maxprob_memview[s][s] - 1.

                        if action <= threshold:
                            r_optimal = min(r_max,
                                            r_hat_mdp[s][a_idx] + beta_r_mdp[s][a_idx])
                            v = r_optimal + dot_prod(mtx_maxprob_memview[s], u1, nb_states)
                        else:
                            o = action - self.threshold - 1 # zero based index
                            # printf("option: %d\n", o)

                            sub_dim = self.reachable_states_per_option[o].dim
                            # printf("sub_dim: %d\n", sub_dim)

                            for i in range(sub_dim):
                                idx = pos2index_2d(nb_states, self.max_reachable_states_per_opt, s, i)
                                sorted_indices_mu[idx] = i
                                # x[o].values[i] = min(r_max, r_tilde_opt[o].values[i])
                                xx[idx] = min(r_max, r_tilde_opt[o].values[i])

                                if s == self.reachable_states_per_option[o].values[i]:
                                    #x[o].values[i] = x[o].values[i] + dot_prod(mtx_maxprob_memview[s], u1, nb_states)
                                    xx[idx] = xx[idx] + dot_prod(mtx_maxprob_memview[s], u1, nb_states)

                            idx = pos2index_2d(nb_states, self.max_reachable_states_per_opt, s, 0)
                            get_sorted_indices(&xx[idx], sub_dim, &sorted_indices_mu[idx])
                            #get_sorted_indices(x[o].values, sub_dim, sorted_indices_mu[idx])

                            # for i in range(sub_dim):
                            #     printf("simu[%d] = %d ", i, sorted_indices_mu[i])
                            #     printf("mu_opt[o].values[%d] = %f ", i, mu_opt[o].values[i])
                            # printf("\n")

                            # printf("\n%f\n", self.cn_opt[o]*beta_mu_p[o])
                            max_proba_purec2(mu_opt[o].values, sub_dim,
                                        &sorted_indices_mu[idx], self.cn_opt[o]*self.beta_mu_p[o],
                                        mtx_maxprob_memview[s])

                            # for i in range(sub_dim):
                            #     printf("mtx_maxprob_memview[%d] = %f", i, mtx_maxprob_memview[s][i])
                            # v = dot_prod(mtx_maxprob_memview[s], x[o].values, sub_dim)
                            v = dot_prod(mtx_maxprob_memview[s], &xx[idx], sub_dim)

                            # free(sorted_indices_mu)

                        c1 = v + u1[s]
                        if first_action or c1 > u2[s] or isclose_c(c1, u2[s]):
                            u2[s] = c1
                            policy_indices[s] = a_idx
                            policy[s] = action

                        first_action = 0
                counter = counter + 1
                # printf("**%d\n", counter)
                # for i in range(nb_states):
                #     printf("%.2f[%.2f]  ", u1[i], u2[i])
                # printf("\n")

                # stopping condition
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    # printf("-- %d\n", counter)
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

    cpdef get_conditioning_numbers(self):
        cn = np.ones((self.nb_options,))
        for i in range(self.nb_options):
            cn[i] = self.cn_opt[i]
        return cn

    cpdef get_beta_mu_p(self):
        beta = np.ones((self.nb_options,))
        for i in range(self.nb_options):
            beta[i] = self.beta_mu_p[i]
        return beta

    cpdef get_mu(self):
        mu = []
        for i in range(self.nb_options):
            L = []
            for j in range(self.mu_opt[i].dim):
                L.append(self.mu_opt[i].values[j])
            mu.append(L)
        return mu

    cpdef get_r_tilde_opt(self):
        r = []
        for i in range(self.nb_options):
            L = []
            for j in range(self.r_tilde_opt[i].dim):
                L.append(self.r_tilde_opt[i].values[j])
            r.append(L)
        return r