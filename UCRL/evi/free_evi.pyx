# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.math cimport sqrt, fabs
from libc.math cimport pow
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdlib cimport rand

from libc.stdio cimport printf
from libc.math cimport log

from cython.parallel import prange

import numpy as np
cimport numpy as np

from ._utils cimport isclose_c
from ._utils cimport get_sorted_indices
from ._utils cimport check_end
from ._utils cimport dot_prod
from ._utils cimport pos2index_2d

from ._max_proba cimport max_proba_purec
from ._max_proba cimport max_proba_purec2
from ._max_proba cimport max_proba_bernstein
from ._max_proba cimport max_proba_bernstein_cin

from ._free_utils cimport get_mu_and_ci_c

from libc.stdio cimport printf

cdef DTYPE_t check_end_opt_evi(DTYPE_t* x, DTYPE_t* y, SIZE_t dim,
                               DTYPE_t* add_res) nogil:
    cdef SIZE_t i, j
    cdef DTYPE_t diff
    cdef DTYPE_t diff_min = x[0]-y[0], diff_max = x[0] - y[0]
    for i in range(0, dim):
        diff = x[i] - y[i]
        if diff_min > diff:
            diff_min = diff
        if diff_max < diff:
            diff_max = diff
    add_res[0] = 0.5 * (diff_max + diff_min)
    return diff_max - diff_min

# =============================================================================
# Extended Value Iteration Class
# =============================================================================

cdef class EVI_FSUCRLv1:

    def __init__(self,
                 int nb_states,
                 int nb_options,
                 list macro_actions_per_state,
                 list mdp_actions_per_state,
                 int threshold,
                 list option_policies,
                 list reachable_states_per_option,
                 list options_terminating_conditions,
                 use_bernstein=0):
        cdef SIZE_t n, m, i, j, idx
        cdef SIZE_t max_states_per_option = 0
        self.nb_states = nb_states
        self.nb_options = nb_options
        self.threshold = threshold
        self.u1 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u2 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))

        self.bernstein_bound = use_bernstein

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
            free(self.mu_opt[i].values)
            free(self.r_tilde_opt[i].values)
        free(self.mu_opt)
        free(self.r_tilde_opt)
        free(self.reachable_states_per_option)
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
                          DTYPE_t alpha_mc,
                          SIZE_t total_time,
                          DTYPE_t delta,
                          SIZE_t max_nb_actions,
                          DTYPE_t r_max):
        cdef SIZE_t nb_options = self.nb_options
        cdef SIZE_t o, opt_nb_states, i, j, s, nexts, idx, mdp_index_a
        cdef DTYPE_t prob, condition_nb, bernstein_bound, bernstein_log
        cdef SIZE_t visits = 0, nb_o

        for o in range(nb_options):
            option_reach_states = self.reachable_states_per_option[o]

            opt_nb_states = option_reach_states.dim

            Q_o = np.zeros((opt_nb_states, opt_nb_states))

            self.beta_mu_p[o] = 0.
            visits = total_time + 2
            bernstein_log = log(6 * max_nb_actions / delta)
            for i in range(option_reach_states.dim):
                s = option_reach_states.values[i]
                # a = self.options_policies[pos2index_2d(nb_options, self.nb_states, o, s)]
                mdp_index_a = self.options_policies_indices_mdp[pos2index_2d(nb_options, self.nb_states, o, s)]

                self.r_tilde_opt[o].values[i] = min(r_max, estimated_rewards_mdp[s, mdp_index_a] + beta_r[s,mdp_index_a])

                if visits > nb_observations_mdp[s, mdp_index_a]:
                    visits = nb_observations_mdp[s, mdp_index_a]

                bernstein_bound = 0.
                nb_o = max(1, nb_observations_mdp[s, mdp_index_a])

                for j in range(option_reach_states.dim):
                    nexts = option_reach_states.values[j]
                    idx = pos2index_2d(nb_options, self.nb_states, o, nexts)
                    prob = estimated_probabilities_mdp[s][mdp_index_a][nexts]
                    Q_o[i,j] = (1. - self.options_terminating_conditions[idx]) * prob
                    bernstein_bound += np.sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o) + bernstein_log * 7 / (3 * nb_o)
                if self.beta_mu_p[o] < alpha_mc * bernstein_bound:
                    self.beta_mu_p[o] = alpha_mc * bernstein_bound
            e_m = np.ones((opt_nb_states,1))
            q_o = e_m - np.dot(Q_o, e_m)

            if self.bernstein_bound == 0:
                self.beta_mu_p[o] =  alpha_mc * np.sqrt(14 * opt_nb_states * log(2 * max_nb_actions
                    * (total_time + 1)/delta)
                                       / np.maximum(1, visits))

            Pprime_o = np.concatenate((q_o, Q_o[:, 1:]), axis=1)
            # assert np.allclose(np.sum(Pprime_o, axis=1), np.ones(opt_nb_states))

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

    cpdef compute_mu_info2(self,
                      DTYPE_t[:,:,:] estimated_probabilities_mdp,
                      DTYPE_t[:,:] estimated_rewards_mdp,
                      DTYPE_t[:,:] beta_r,
                      SIZE_t[:,:] nb_observations_mdp,
                      DTYPE_t alpha_mc,
                      SIZE_t total_time,
                      DTYPE_t delta,
                      SIZE_t max_nb_actions,
                           DTYPE_t r_max):
        cdef SIZE_t nb_options = self.nb_options
        cdef SIZE_t nb_states = self.nb_states
        cdef SIZE_t o, opt_nb_states, i, j, s, nexts, idx, mdp_index_a
        cdef DTYPE_t prob, bernstein_log
        cdef DTYPE_t sum_prob_row, bernstein_bound
        cdef SIZE_t nb_o, l_ij, visits

        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DoubleVectorStruct* mu_opt = self.mu_opt
        cdef DTYPE_t* beta_mu_p = self.beta_mu_p
        cdef DTYPE_t* term_cond = self.options_terminating_conditions
        cdef DTYPE_t* cn_opt = self.cn_opt

        cdef DTYPE_t *Popt

        with nogil:

            for o in prange(nb_options):

                opt_nb_states = self.reachable_states_per_option[o].dim
                bernstein_log = log(6* max_nb_actions / delta)

                # note that this should be row-major (c ordering)
                Popt = <double*>malloc(opt_nb_states * opt_nb_states * sizeof(double))

                visits = total_time + 2
                beta_mu_p[o] = -1.

                for i in range(opt_nb_states):
                    s = self.reachable_states_per_option[o].values[i]
                    # a = self.options_policies[pos2index_2d(nb_options, self.nb_states, o, s)]
                    mdp_index_a = self.options_policies_indices_mdp[pos2index_2d(nb_options, nb_states, o, s)]

                    r_tilde_opt[o].values[i] = min(r_max, estimated_rewards_mdp[s, mdp_index_a] + beta_r[s,mdp_index_a])

                    bernstein_bound = 0.0
                    sum_prob_row = 0.0
                    nb_o = max(1, nb_observations_mdp[s, mdp_index_a])
                    if visits > nb_o:
                        visits = nb_o

                    for j in range(opt_nb_states):
                        nexts = self.reachable_states_per_option[o].values[j]
                        idx = pos2index_2d(nb_options, nb_states, o, nexts)
                        prob = estimated_probabilities_mdp[s][mdp_index_a][nexts]

                        l_ij = pos2index_2d(opt_nb_states, opt_nb_states, i, j)
                        Popt[l_ij] = (1. - term_cond[idx]) * prob
                        sum_prob_row = sum_prob_row + Popt[l_ij]
                        bernstein_bound = bernstein_bound + sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o) + bernstein_log * 7 / (3 * nb_o)

                    if beta_mu_p[o] < alpha_mc * bernstein_bound:
                        beta_mu_p[o] = alpha_mc * bernstein_bound

                    # compute transition matrix of option
                    l_ij = pos2index_2d(opt_nb_states, opt_nb_states, i, 0)
                    Popt[l_ij] = 1. - sum_prob_row

                if self.bernstein_bound == 0:
                    beta_mu_p[o] =  alpha_mc * sqrt(14 * opt_nb_states * log(2 * max_nb_actions
                        * (total_time + 1)/delta) / visits)

                # --------------------------------------------------------------
                #  Aperiodic transformation
                # --------------------------------------------------------------
                for i in range(opt_nb_states):
                    for j in range(opt_nb_states):
                        l_ij = pos2index_2d(opt_nb_states, opt_nb_states, i, j)
                        if i == j:
                            Popt[l_ij] = Popt[l_ij] + 1.0
                        Popt[l_ij] = Popt[l_ij] / 2.0

                get_mu_and_ci_c(Popt, opt_nb_states, &cn_opt[o], mu_opt[o].values)

                free(Popt)
        # free(sum_prob_row)
        return 0

    cpdef DTYPE_t run(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] p_hat,
                     DTYPE_t[:,:] r_hat_mdp,
                     DTYPE_t[:,:,:] beta_p,
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
        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DTYPE_t* xx = self.xx


        with nogil:
            c1 = u1[0]
            for i in range(nb_states):
                u1[i] = u1[i] - c1
                sorted_indices[i] = i
            get_sorted_indices(u1, nb_states, sorted_indices)

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
                        if self.bernstein_bound == 0:
                            max_proba_purec(p_hat[s][a_idx], nb_states,
                                        sorted_indices, beta_p[s][a_idx][0],
                                        mtx_maxprob_memview[s])
                        else:
                            max_proba_bernstein(p_hat[s][a_idx], nb_states,
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

# =============================================================================
# FSUCRLv2
# =============================================================================

cdef class EVI_FSUCRLv2:

    def __init__(self,
             int nb_states,
             int nb_options,
             list macro_actions_per_state,
             list mdp_actions_per_state,
             int threshold,
             list option_policies,
             list reachable_states_per_option,
             list options_terminating_conditions,
             use_bernstein=0):
        cdef SIZE_t n, m, i, j, idx
        cdef SIZE_t max_states_per_option = 0

        self.nb_states = nb_states
        self.nb_options = nb_options
        self.threshold = threshold
        self.bernstein_bound = use_bernstein


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

        # ----------------------------------------------------------------------
        # Copy list of actions per state
        # ----------------------------------------------------------------------
        n = len(macro_actions_per_state)
        assert n == nb_states
        self.actions_per_state = <IntVectorStruct *> malloc(n * sizeof(IntVectorStruct))
        for i in range(n):
            m = len(macro_actions_per_state[i])
            self.actions_per_state[i].dim = m
            self.actions_per_state[i].values = <SIZE_t *> malloc(m * sizeof(SIZE_t))
            for j in range(m):
                self.actions_per_state[i].values[j] = macro_actions_per_state[i][j]

        # ----------------------------------------------------------------------
        # Copy
        #  - option reachable states
        #  - option terminating condition
        #  - option policies
        #  - option policy indices
        # Create
        #  - optimistic reward for options (r_tilde_opt)
        #  - indices for each option (sorted_indices_popt)
        #  - structure for holding transition matrix of options (p_hat_opt)
        #  - structure for holding transition matrix ci of options (beta_opt_p)
        # ----------------------------------------------------------------------
        n = len(reachable_states_per_option)
        assert n == nb_options

        self.reachable_states_per_option = <IntVectorStruct *> malloc(nb_options * sizeof(IntVectorStruct))
        self.r_tilde_opt = <DoubleVectorStruct *> malloc(nb_options * sizeof(DoubleVectorStruct))
        self.sorted_indices_popt = <IntVectorStruct *> malloc(nb_options * sizeof(IntVectorStruct))
        self.beta_opt_p = <DoubleVectorStruct *> malloc(nb_options * sizeof(DoubleVectorStruct))
        self.p_hat_opt = <DoubleVectorStruct *> malloc(nb_options * sizeof(DoubleVectorStruct))
        self.w1 = <DoubleVectorStruct *> malloc(nb_options * sizeof(DoubleVectorStruct))
        self.w2 = <DoubleVectorStruct *> malloc(nb_options * sizeof(DoubleVectorStruct))

        self.options_policies = <SIZE_t *>malloc(nb_options * nb_states * sizeof(SIZE_t)) # (O x S)
        self.options_policies_indices_mdp = <SIZE_t *>malloc(nb_options * nb_states * sizeof(SIZE_t)) # (O x S)
        self.options_terminating_conditions = <DTYPE_t *>malloc(nb_options * nb_states * sizeof(SIZE_t)) # (O x S)

        for i in range(n):
            m = len(reachable_states_per_option[i])
            if m > max_states_per_option:
                max_states_per_option = m
            self.reachable_states_per_option[i].dim = m
            self.reachable_states_per_option[i].values = <SIZE_t *> malloc(m * sizeof(SIZE_t))
            for j in range(m):
                self.reachable_states_per_option[i].values[j] = int(reachable_states_per_option[i][j])

            self.r_tilde_opt[i].dim = m
            self.r_tilde_opt[i].values = <DTYPE_t *> malloc(m * sizeof(DTYPE_t))

            self.sorted_indices_popt[i].dim = m
            self.sorted_indices_popt[i].values = <SIZE_t *> malloc(m * sizeof(SIZE_t))

            self.p_hat_opt[i].dim = m * m
            self.p_hat_opt[i].values = <DTYPE_t *> malloc(m * m * sizeof(DTYPE_t))

            self.beta_opt_p[i].dim = m * m
            self.beta_opt_p[i].values = <DTYPE_t *> malloc(m * m * sizeof(DTYPE_t))

            self.w1[i].dim = m
            self.w1[i].values = <DTYPE_t *> malloc(m * sizeof(DTYPE_t))
            self.w2[i].dim = m
            self.w2[i].values = <DTYPE_t *> malloc(m * sizeof(DTYPE_t))
            for j in range(m):
                self.w1[i].values[j] = 0.
                self.w2[i].values[j] = 0.

            for j in range(nb_states): # index of the state to be considered
                idx = pos2index_2d(nb_options, nb_states, i, j)
                self.options_policies[idx] = option_policies[i][j]
                self.options_terminating_conditions[idx] = options_terminating_conditions[i][j]
                for kk, kk_v in enumerate(mdp_actions_per_state[j]): # save the index of the primitive action
                    if kk_v == self.options_policies[idx]:
                        self.options_policies_indices_mdp[idx] = kk
                        break

        self.max_reachable_states_per_opt = max_states_per_option

        # ----------------------------------------------------------------------
        # Other options information
        # ----------------------------------------------------------------------
        self.mtx_maxprob_opt = <DTYPE_t *>malloc(nb_options * nb_states * sizeof(DTYPE_t))
        self.mtx_maxprob_opt_memview = <DTYPE_t[:nb_options, :nb_states]> self.mtx_maxprob_opt
        self.span_w_opt = <DTYPE_t*> malloc(nb_options * sizeof(DTYPE_t))

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
            free(self.r_tilde_opt[i].values)
            free(self.beta_opt_p[i].values)
            free(self.p_hat_opt[i].values)
            free(self.sorted_indices_popt[i].values)
            free(self.w1[i].values)
            free(self.w2[i].values)
        free(self.reachable_states_per_option)
        free(self.r_tilde_opt)
        free(self.beta_opt_p)
        free(self.p_hat_opt)
        free(self.sorted_indices_popt)
        free(self.w1)
        free(self.w2)
        free(self.options_terminating_conditions)
        free(self.options_policies)
        free(self.options_policies_indices_mdp)
        free(self.mtx_maxprob_opt)
        free(self.span_w_opt)

    cpdef compute_prerun_info(self, DTYPE_t[:,:,:] estimated_probabilities_mdp,
                          DTYPE_t[:,:] estimated_rewards_mdp,
                          DTYPE_t[:,:] beta_r,
                          SIZE_t[:,:] nb_observations_mdp,
                          SIZE_t total_time,
                          DTYPE_t delta,
                          SIZE_t max_nb_actions,
                          DTYPE_t alpha_mc,
                              DTYPE_t r_max):
        cdef SIZE_t nb_options = self.nb_options
        cdef SIZE_t nb_states = self.nb_states
        cdef SIZE_t o, opt_nb_states, i, j, s, nexts, idx, mdp_index_a
        cdef DTYPE_t prob, bernstein_log
        cdef DTYPE_t sum_prob_row
        cdef SIZE_t nb_o, l_ij

        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DoubleVectorStruct* beta_opt_p = self.beta_opt_p
        cdef DoubleVectorStruct* p_hat_opt = self.p_hat_opt

        #cdef IntVectorStruct* option_reach_states = self.reachable_states_per_option
        cdef DTYPE_t* term_cond = self.options_terminating_conditions

        # cdef DTYPE_t* sum_prob_row = <DTYPE_t*> malloc(nb_options * sizeof(DTYPE_t))

        with nogil:

            for o in prange(nb_options):

                opt_nb_states = self.reachable_states_per_option[o].dim
                bernstein_log = log(6* max_nb_actions / delta)

                for i in range(opt_nb_states):
                    s = self.reachable_states_per_option[o].values[i]
                    # a = self.options_policies[pos2index_2d(nb_options, self.nb_states, o, s)]
                    mdp_index_a = self.options_policies_indices_mdp[pos2index_2d(nb_options, nb_states, o, s)]

                    r_tilde_opt[o].values[i] = min(r_max, estimated_rewards_mdp[s, mdp_index_a] + beta_r[s,mdp_index_a])

                    sum_prob_row = 0.
                    nb_o = max(1, nb_observations_mdp[s, mdp_index_a])



                    for j in range(opt_nb_states):
                        nexts = self.reachable_states_per_option[o].values[j]
                        idx = pos2index_2d(nb_options, nb_states, o, nexts)
                        prob = estimated_probabilities_mdp[s][mdp_index_a][nexts]

                        l_ij = pos2index_2d(opt_nb_states, opt_nb_states, i, j)
                        p_hat_opt[o].values[l_ij] = (1. - term_cond[idx]) * prob
                        sum_prob_row = sum_prob_row + p_hat_opt[o].values[l_ij]

                        if self.bernstein_bound == 0:
                            beta_opt_p[o].values[l_ij] = alpha_mc * sqrt(14 * opt_nb_states * log(2 * max_nb_actions
                                                                                              * (total_time + 1)/ delta) / nb_o)
                        else:
                            beta_opt_p[o].values[l_ij] = alpha_mc * (sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o)
                                                                       + bernstein_log * 7 / (3 * nb_o))
                    # compute transition matrix of option
                    l_ij = pos2index_2d(opt_nb_states, opt_nb_states, i, 0)
                    p_hat_opt[o].values[l_ij] = 1. - sum_prob_row
        # free(sum_prob_row)
        return 0

    cpdef DTYPE_t run(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] p_hat,
                     DTYPE_t[:,:] r_hat_mdp,
                     DTYPE_t[:,:,:] beta_p,
                     DTYPE_t[:,:] beta_r_mdp,
                     DTYPE_t r_max,
                     DTYPE_t epsilon):
        cdef SIZE_t nb_states = self.nb_states
        cdef SIZE_t nb_options = self.nb_options
        cdef SIZE_t threshold = self.threshold
        cdef SIZE_t continue_flag, nb_states_per_options
        cdef SIZE_t idx, s, sprime, first_action, action, o, a_idx
        cdef SIZE_t option_act, option_idx
        cdef SIZE_t outer_evi_it, inner_it_opt, i, j,k
        cdef DTYPE_t epsilon_opt
        cdef DTYPE_t v, r_optimal
        cdef DTYPE_t min_u1=0, max_u1=0, max_prob_opt
        cdef DTYPE_t min_w1, max_w1, cv, center_value

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef DoubleVectorStruct *w1 = self.w1
        cdef DoubleVectorStruct *w2 = self.w2
        cdef SIZE_t* sorted_indices = self.sorted_indices
        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview

        cdef IntVectorStruct* reach_states = self.reachable_states_per_option
        cdef DTYPE_t[:,:] mtx_maxprob_opt_memview = self.mtx_maxprob_opt_memview #(|O|x|S|)

        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DoubleVectorStruct* beta_opt_p = self.beta_opt_p
        cdef DoubleVectorStruct* p_hat_opt = self.p_hat_opt
        cdef IntVectorStruct* sorted_indices_popt = self.sorted_indices_popt

        cdef DTYPE_t* span_w_opt = self.span_w_opt

        with nogil:
            center_value = u1[0]
            for i in range(nb_states):
                u1[i] = u1[i] - center_value
                sorted_indices[i] = i
            get_sorted_indices(u1, nb_states, sorted_indices)

            for o in range(nb_options):
                for i in range(w1[o].dim):
                    w1[o].values[i] = 0.0
                    w2[o].values[i] = 0.0
                    sorted_indices_popt[o].values[i] = i


            outer_evi_it = 0
            while True:
                outer_evi_it += 1
                if outer_evi_it > 50000:
                    printf("%f", max_u1 - min_u1)
                    return -5

                # epsilon_opt = 1. / pow(2, outer_evi_it)
                # epsilon_opt = 1. / (outer_evi_it * outer_evi_it)
                epsilon_opt = 1. / (outer_evi_it)

                # --------------------------------------------------------------
                # Estimate value function of each option
                # --------------------------------------------------------------
                for o in prange(nb_options):
                    continue_flag = 1

                    nb_states_per_options = reach_states[o].dim
                    center_value = w1[o].values[0]
                    for i in range(nb_states_per_options):
                        # sorted_indices_popt[o].values[i] = i
                        w1[o].values[i] = w1[o].values[i] - center_value # 0.0
                        w2[o].values[i] = 0.0
                    # sort indices
                    get_sorted_indices(w1[o].values, nb_states_per_options, sorted_indices_popt[o].values)

                    #------------------------------------------------------
                    # compute b*v -v(s)
                    #------------------------------------------------------
                    sprime = reach_states[o].values[0]
                    option_act = o + self.threshold + 1
                    option_idx = -1
                    # get index of the option in the current state
                    for j in range(self.actions_per_state[sprime].dim):
                        if self.actions_per_state[sprime].values[j] == option_act:
                            option_idx = j
                            break
                    if option_idx == -1:
                        return -2
                    if self.bernstein_bound == 0:
                        max_proba_purec(p_hat[sprime][option_idx], nb_states,
                                    sorted_indices, beta_p[sprime][option_idx][0],
                                    mtx_maxprob_opt_memview[o])
                    else:
                        max_proba_bernstein(p_hat[sprime][option_idx], nb_states,
                                    sorted_indices, beta_p[sprime][option_idx],
                                    mtx_maxprob_opt_memview[o])

                    mtx_maxprob_opt_memview[o][sprime] = mtx_maxprob_opt_memview[o][sprime] - 1.
                    max_prob_opt = dot_prod(mtx_maxprob_opt_memview[o], u1, nb_states)

                    inner_it_opt = 0
                    while continue_flag:
                        inner_it_opt = inner_it_opt + 1
                        if inner_it_opt > 500000:
                            return -1

                        for i in range(nb_states_per_options):
                            sprime = reach_states[o].values[i]
                            if i == 0:
                                # this is the starting state
                                # option_act = o + self.threshold + 1
                                # option_idx = -1
                                # # get index of the option in the current state
                                # for j in range(self.actions_per_state[sprime].dim):
                                #     if self.actions_per_state[sprime].values[j] == option_act:
                                #         option_idx = j
                                #         break
                                # if option_idx == -1:
                                #     return -2
                                # if self.bernstein_bound == 0:
                                #     max_proba_purec(p_hat[sprime][option_idx], nb_states,
                                #                 sorted_indices, beta_p[sprime][option_idx][0],
                                #                 mtx_maxprob_opt_memview[o])
                                # else:
                                #     max_proba_bernstein(p_hat[sprime][option_idx], nb_states,
                                #                 sorted_indices, beta_p[sprime][option_idx],
                                #                 mtx_maxprob_opt_memview[o])
                                #
                                # mtx_maxprob_opt_memview[o][sprime] = mtx_maxprob_opt_memview[o][sprime] - 1.
                                # v = dot_prod(mtx_maxprob_opt_memview[o], u1, nb_states)
                                v = max_prob_opt

                            else:
                                v = 0.0

                            #r_optimal = min(r_max, r_tilde_opt[o].values[i])
                            r_optimal = r_tilde_opt[o].values[i]

                            idx = pos2index_2d(nb_states_per_options, nb_states_per_options, i, 0)
                            max_proba_bernstein_cin(
                                &(p_hat_opt[o].values[idx]),
                                nb_states_per_options,
                                sorted_indices_popt[o].values,
                                &(beta_opt_p[o].values[idx]),
                                mtx_maxprob_opt_memview[o])

                            v = r_optimal + v + dot_prod(mtx_maxprob_opt_memview[o], w1[o].values, nb_states_per_options)
                            w2[o].values[i] = v
                            if v > 50000:
                                return -4

                        # stopping condition
                        if check_end_opt_evi(w2[o].values, w1[o].values, nb_states_per_options, &span_w_opt[o]) < epsilon_opt:
                            # printf("-- %d", inner_it_opt)
                            continue_flag = 0
                            #printf("[%d, %.2f] - ", inner_it_opt, span_w_opt[o])
                        else:
                            memcpy(w1[o].values, w2[o].values, nb_states_per_options * sizeof(DTYPE_t))
                            get_sorted_indices(w1[o].values, nb_states_per_options, sorted_indices_popt[o].values)


                # --------------------------------------------------------------
                # Estimate value function of SMDP
                # --------------------------------------------------------------
                for s in range(nb_states):
                    first_action = 1
                    for a_idx in range(self.actions_per_state[s].dim):
                        # printf("action_idx: %d\n", a_idx)
                        action = self.actions_per_state[s].values[a_idx]
                        # printf("action: %d [%d]\n", action, threshold)


                        if action <= threshold:
                            # max_proba_purec
                            # max_proba_reduced
                            if self.bernstein_bound == 0:
                                max_proba_purec(p_hat[s][a_idx], nb_states,
                                            sorted_indices, beta_p[s][a_idx][0],
                                            mtx_maxprob_memview[s])
                            else:
                                max_proba_bernstein(p_hat[s][a_idx], nb_states,
                                            sorted_indices, beta_p[s][a_idx],
                                            mtx_maxprob_memview[s])

                            mtx_maxprob_memview[s][s] = mtx_maxprob_memview[s][s] - 1.
                            r_optimal = min(r_max,
                                            r_hat_mdp[s][a_idx] + beta_r_mdp[s][a_idx])
                            v = r_optimal + dot_prod(mtx_maxprob_memview[s], u1, nb_states)
                        else:
                            o = action - self.threshold - 1 # zero based index
                            v = span_w_opt[o]

                        v = v + u1[s]
                        if v > 50000:
                            printf("%f", v)
                            return -60
                        if first_action or v > u2[s] or isclose_c(v, u2[s]):
                            u2[s] = v
                            policy_indices[s] = a_idx
                            policy[s] = action
                        first_action = 0

                # printf("\n")
                # for i in range(nb_states):
                #     printf("%.2f[%.2f] ", u1[i], u2[i])
                # printf("\n")

                # stopping condition
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    # printf("%d\n", outer_evi_it)
                    return max_u1 - min_u1
                else:
                    # printf("------------\n")
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)
                    # center_value = u1[0]
                    # for j in range(nb_states):
                    #     u1[j] = u1[j] - center_value

                if v > 50000:
                    return -70

                # printf("[%d]u1\n", outer_evi_it)
                # for j in range(nb_states):
                #     printf("%.2f ", u1[j])
                # printf("\n")


    cpdef get_r_tilde_opt(self):
        r = []
        for i in range(self.nb_options):
            L = []
            for j in range(self.r_tilde_opt[i].dim):
                L.append(self.r_tilde_opt[i].values[j])
            r.append(L)
        return r

    cpdef get_opt_p_and_beta(self):
        cdef SIZE_t x, y, dim
        beta = []
        p = []
        for i in range(self.nb_options):
            dim = <SIZE_t>sqrt(self.beta_opt_p[i].dim)
            L = np.zeros((dim, dim))
            P = np.zeros((dim, dim))
            for j in range(self.beta_opt_p[i].dim):
                x = j // dim
                y = j % dim
                L[x,y] = self.beta_opt_p[i].values[j]
                P[x,y] = self.p_hat_opt[i].values[j]
            beta.append(L)
            p.append(P)
        return p, beta

