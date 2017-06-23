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
from libc.stdlib cimport rand, RAND_MAX, srand

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
                 bound_type,
                 int random_state):
        cdef SIZE_t n, m, i, j, idx
        cdef SIZE_t max_states_per_option = 0
        self.nb_states = nb_states
        self.nb_options = nb_options
        self.threshold = threshold
        self.u1 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))
        self.u2 = <DTYPE_t *>malloc(nb_states * sizeof(DTYPE_t))

        if bound_type == "chernoff":
            self.bound_type = BoundType.CHERNOFF
        elif bound_type == "chernoff_statedim":
            self.bound_type = BoundType.CHERNOFF_STATEDIM
        elif bound_type == "bernstein":
            self.bound_type = BoundType.BERNSTEIN
        else:
            raise ValueError("Unknown bound type")

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
        self.max_macroactions_per_state = 0
        for i in range(n):
            m = len(macro_actions_per_state[i])
            if m > self.max_macroactions_per_state:
                self.max_macroactions_per_state = m
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
        self.random_state = random_state
        srand(random_state)

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
        cdef SIZE_t options_visited_allsa
        cdef SIZE_t opt_error = 0, local_error

        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DoubleVectorStruct* mu_opt = self.mu_opt
        cdef DTYPE_t* beta_mu_p = self.beta_mu_p
        cdef DTYPE_t* term_cond = self.options_terminating_conditions
        cdef DTYPE_t* cn_opt = self.cn_opt

        cdef DTYPE_t *Popt
        cdef DTYPE_t alpha_aperiodic = 0.5

        with nogil:

            for o in prange(nb_options):

                opt_nb_states = self.reachable_states_per_option[o].dim
                bernstein_log = log(opt_nb_states * log(total_time + 2.) / delta)

                # note that this should be row-major (c ordering)
                Popt = <double*>malloc(opt_nb_states * opt_nb_states * sizeof(double))

                options_visited_allsa = 0
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
                    options_visited_allsa = options_visited_allsa + min(1, nb_observations_mdp[s, mdp_index_a])

                    for j in range(opt_nb_states):
                        nexts = self.reachable_states_per_option[o].values[j]
                        idx = pos2index_2d(nb_options, nb_states, o, nexts)
                        prob = estimated_probabilities_mdp[s][mdp_index_a][nexts]

                        l_ij = pos2index_2d(opt_nb_states, opt_nb_states, i, j)
                        Popt[l_ij] = (1. - term_cond[idx]) * prob
                        sum_prob_row = sum_prob_row + Popt[l_ij]
                        bernstein_bound = bernstein_bound + sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o) + bernstein_log * 7. / (3. * nb_o)

                    bernstein_bound = alpha_mc * bernstein_bound
                    if beta_mu_p[o] < bernstein_bound:
                        beta_mu_p[o] = bernstein_bound

                    # compute transition matrix of option
                    l_ij = pos2index_2d(opt_nb_states, opt_nb_states, i, 0)
                    Popt[l_ij] = 1. - sum_prob_row

                if options_visited_allsa == opt_nb_states:
                    # do this when all the states has been visited at least once
                    if self.bound_type == BoundType.CHERNOFF:
                        beta_mu_p[o] =  alpha_mc * sqrt(14 * opt_nb_states * log(2 * max_nb_actions
                            * (total_time + 1)/delta) / visits)
                    elif self.bound_type == BoundType.CHERNOFF_STATEDIM:
                        beta_mu_p[o] =  alpha_mc * sqrt(14 * nb_states * log(2 * max_nb_actions
                            * (total_time + 1)/delta) / visits)

                    local_error = get_mu_and_ci_c(Popt, opt_nb_states, &cn_opt[o],
                                                mu_opt[o].values, alpha_aperiodic)
                    if local_error == 2:
                        # the matrix is not invertible
                        beta_mu_p[o] = 2.
                        cn_opt[o] = 1.
                        for i in range(opt_nb_states):
                            mu_opt[o].values[i] = 1./ opt_nb_states
                        local_error = 0
                    elif local_error != 0:
                        printf("Error in the computation of mu and ci of option %d (zero-based)\n", o)
                    # cn_opt[o] = 1.
                else:
                    printf("Still exploring opt %d\n", o)
                    beta_mu_p[o] = 2.
                    cn_opt[o] = 1.
                    for i in range(opt_nb_states):
                        mu_opt[o].values[i] = 1./ opt_nb_states
                    local_error = 0
                    
                opt_error += local_error

                free(Popt)
        # free(sum_prob_row)
        return opt_error

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
        cdef SIZE_t max_nb_actions = self.max_macroactions_per_state

        cdef DTYPE_t* u1 = self.u1
        cdef DTYPE_t* u2 = self.u2
        cdef SIZE_t* sorted_indices = self.sorted_indices
        cdef SIZE_t* sorted_indices_mu = self.sorted_indices_mu

        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview

        cdef DoubleVectorStruct* mu_opt = self.mu_opt
        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DTYPE_t* xx = self.xx
        
        cdef SIZE_t* action_argmax
        cdef SIZE_t* action_argmax_indices
        cdef SIZE_t count_equal_actions, picked_idx, new_idx
        
        action_argmax = <SIZE_t*> malloc(nb_states * max_nb_actions * sizeof(SIZE_t))
        action_argmax_indices = <SIZE_t*> malloc(nb_states * max_nb_actions * sizeof(SIZE_t))


        with nogil:
            c1 = u1[0]
            for i in range(nb_states):
                u1[i] = u1[i] - c1
                sorted_indices[i] = i
            get_sorted_indices(u1, nb_states, sorted_indices)

            while True: #counter < 5:
                for s in range(nb_states):
                    # printf("state: %d\n", s)
                    first_action = 1
                    count_equal_actions = 0
                    for a_idx in range(self.actions_per_state[s].dim):
                        # printf("action_idx: %d\n", a_idx)
                        action = self.actions_per_state[s].values[a_idx]
                        # printf("action: %d [%d]\n", action, threshold)

                        # max_proba_purec
                        # max_proba_reduced
                        if self.bound_type != BoundType.BERNSTEIN:
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

                            sub_dim = self.reachable_states_per_option[o].dim
                            for i in range(sub_dim):
                                idx = pos2index_2d(nb_states, self.max_reachable_states_per_opt, s, i)
                                xx[idx] = min(r_max, r_tilde_opt[o].values[i])

                                sorted_indices_mu[idx] = i # fill vector

                                if s == self.reachable_states_per_option[o].values[i]:
                                    xx[idx] = xx[idx] + dot_prod(mtx_maxprob_memview[s], u1, nb_states)

                            idx = pos2index_2d(nb_states, self.max_reachable_states_per_opt, s, 0)
                            get_sorted_indices(&xx[idx], sub_dim, &sorted_indices_mu[idx])

                            max_proba_purec2(mu_opt[o].values, sub_dim,
                                        &sorted_indices_mu[idx], self.cn_opt[o]*self.beta_mu_p[o],
                                        mtx_maxprob_memview[s])

                            v = dot_prod(mtx_maxprob_memview[s], &xx[idx], sub_dim)


                        c1 = v + u1[s]
                        
                        if first_action:
                            u2[s] = c1
                            count_equal_actions = 1
                            
                            new_idx = pos2index_2d(nb_states, max_nb_actions, s, 0)
                            action_argmax[new_idx] = action
                            action_argmax_indices[new_idx] = a_idx
                        elif isclose_c(c1, u2[s]):
                            new_idx = pos2index_2d(nb_states, max_nb_actions, s, count_equal_actions)
                            action_argmax[new_idx] = action
                            action_argmax_indices[new_idx] = a_idx
                            count_equal_actions = count_equal_actions + 1
                        elif c1 > u2[s]:
                            u2[s] = c1
                            count_equal_actions = 1
                            
                            new_idx = pos2index_2d(nb_states, max_nb_actions, s, 0)
                            action_argmax[new_idx] = action
                            action_argmax_indices[new_idx] = a_idx
                        first_action = 0
                        
                    # randomly select action
                    picked_idx = rand() / (RAND_MAX / count_equal_actions + 1)
                    new_idx = pos2index_2d(nb_states, max_nb_actions, s, picked_idx)
                    policy_indices[s] = action_argmax_indices[new_idx]
                    policy[s] = action_argmax[new_idx]
                    # for i in range(count_equal_actions):
                    #     new_idx = pos2index_2d(nb_states, max_nb_actions, s, i  )
                    #     printf("%d, ", action_argmax_indices[new_idx])
                    # printf("\n")
                        
                counter = counter + 1

                # stopping condition
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    # printf("-- %d\n", counter)
                    free(action_argmax)
                    free(action_argmax_indices)
                    return max_u1 - min_u1
                else:
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)

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
    
    cpdef get_mu_tilde(self, DTYPE_t r_max, DTYPE_t[:,:,:] p_hat, DTYPE_t[:,:,:] beta_p):
        cdef SIZE_t i, j, o, s, sub_dim, idx
        cdef SIZE_t nb_states = self.nb_states, nb_options = self.nb_options
        cdef DTYPE_t[:,:] mtx_maxprob_memview = self.mtx_maxprob_memview
        cdef DoubleVectorStruct* mu_opt = self.mu_opt
        cdef DoubleVectorStruct* r_tilde_opt = self.r_tilde_opt
        cdef DTYPE_t* xx = self.xx
        cdef SIZE_t* sorted_indices_mu = self.sorted_indices_mu
        cdef SIZE_t option_act, option_idx
        cdef SIZE_t* sorted_indices = self.sorted_indices
        
        for i in range(self.nb_states):
            sorted_indices[i] = i
        get_sorted_indices(self.u1, self.nb_states, sorted_indices)
        
        mu_tilde = []
        for o in range(nb_options):
            sub_dim = self.reachable_states_per_option[o].dim
            mu_tilde.append(np.zeros(sub_dim))
        with nogil:
            for o in range(nb_options):
                s = self.reachable_states_per_option[o].values[0]
                sub_dim = self.reachable_states_per_option[o].dim
                
                option_act = o + self.threshold + 1
                option_idx = -1
                # get index of the option in the current state
                for j in range(self.actions_per_state[s].dim):
                    if self.actions_per_state[s].values[j] == option_act:
                        option_idx = j
                        break
                
                if self.bound_type != BoundType.BERNSTEIN:
                    max_proba_purec(p_hat[s][option_idx], nb_states,
                                sorted_indices, beta_p[s][option_idx][0],
                                mtx_maxprob_memview[s])
                else:
                    max_proba_bernstein(p_hat[s][option_idx], nb_states,
                                sorted_indices, beta_p[s][option_idx],
                                mtx_maxprob_memview[s])

                mtx_maxprob_memview[s][s] = mtx_maxprob_memview[s][s] - 1.
                
                for i in range(sub_dim):
                    idx = pos2index_2d(nb_states, self.max_reachable_states_per_opt, s, i)
                    xx[idx] = min(r_max, r_tilde_opt[o].values[i])

                    sorted_indices_mu[idx] = i # fill vector

                    if s == self.reachable_states_per_option[o].values[i]:
                        xx[idx] = xx[idx] + dot_prod(mtx_maxprob_memview[s], self.u1, nb_states)
                        # printf("%f\n", dot_prod(mtx_maxprob_memview[s], self.u1, nb_states))

                idx = pos2index_2d(nb_states, self.max_reachable_states_per_opt, s, 0)
                get_sorted_indices(&xx[idx], sub_dim, &sorted_indices_mu[idx])

                max_proba_purec2(mu_opt[o].values, sub_dim,
                            &sorted_indices_mu[idx], self.cn_opt[o]*self.beta_mu_p[o],
                            mtx_maxprob_memview[s])
                with gil:
                    for i in range(sub_dim):
                        mu_tilde[o][i] = mtx_maxprob_memview[s][i]
        return mu_tilde
    
    cpdef get_P_prime(self, DTYPE_t[:,:,:] estimated_probabilities_mdp):
        cdef SIZE_t i, j, o, s, sub_dim, mdp_index_a, opt_nb_states
        cdef DTYPE_t sum_prob_row
        cdef DTYPE_t* term_cond = self.options_terminating_conditions
        
        p_prime = []
        for o in range(self.nb_options):
            sub_dim = self.reachable_states_per_option[o].dim
            p_prime.append(np.zeros((sub_dim, sub_dim)))

        for o in range(self.nb_options):
            opt_nb_states = self.reachable_states_per_option[o].dim
            for i in range(opt_nb_states):
                s = self.reachable_states_per_option[o].values[i]
                mdp_index_a = self.options_policies_indices_mdp[pos2index_2d(self.nb_options, self.nb_states, o, s)]

                sum_prob_row = 0.0

                for j in range(opt_nb_states):
                    nexts = self.reachable_states_per_option[o].values[j]
                    idx = pos2index_2d(self.nb_options, self.nb_states, o, nexts)
                    prob = estimated_probabilities_mdp[s][mdp_index_a][nexts]

                    p_prime[o][i,j] = (1. - term_cond[idx]) * prob
                    sum_prob_row = sum_prob_row + p_prime[o][i,j]
                
                p_prime[o][i,0] = 1 - sum_prob_row
                        
        return p_prime          
    
    cpdef set_seed(self, SIZE_t seed):
        srand(seed)

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
             bound_type,
             int random_state):
        cdef SIZE_t n, m, i, j, idx
        cdef SIZE_t max_states_per_option = 0

        self.nb_states = nb_states
        self.nb_options = nb_options
        self.threshold = threshold

        if bound_type == "chernoff":
            self.bound_type = BoundType.CHERNOFF
        elif bound_type == "chernoff_statedim":
            self.bound_type = BoundType.CHERNOFF_STATEDIM
        elif bound_type == "bernstein":
            self.bound_type = BoundType.BERNSTEIN
        else:
            raise ValueError("Unknown bound type")


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
        self.max_macroactions_per_state = 0
        for i in range(n):
            m = len(macro_actions_per_state[i])
            if m > self.max_macroactions_per_state:
                self.max_macroactions_per_state = m
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
        
        self.random_state = random_state
        srand(random_state)

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
                bernstein_log = log(opt_nb_states * log(total_time + 2.) / delta)

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

                        if self.bound_type == BoundType.CHERNOFF:
                            beta_opt_p[o].values[l_ij] = alpha_mc * sqrt(14 * opt_nb_states * log(2 * max_nb_actions
                                                                                              * (total_time + 1)/ delta) / nb_o)
                        elif self.bound_type == BoundType.CHERNOFF_STATEDIM:
                            beta_opt_p[o].values[l_ij] = alpha_mc * sqrt(14 * nb_states * log(2 * max_nb_actions
                                                                                              * (total_time + 1)/ delta) / nb_o)
                        else:
                            beta_opt_p[o].values[l_ij] = alpha_mc * (sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o)
                                                                       + bernstein_log * 7. / (3. * nb_o))
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
        cdef SIZE_t max_nb_actions = self.max_macroactions_per_state

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

        cdef SIZE_t MAX_IT_INNER = 500000
        cdef SIZE_t MAX_IT_OUTER = 500000
        cdef DTYPE_t MAX_VAL_V = 500000.
        cdef SIZE_t evi_error = 0
        
        cdef SIZE_t* action_argmax
        cdef SIZE_t* action_argmax_indices
        cdef SIZE_t count_equal_actions, picked_idx
        
        action_argmax = <SIZE_t*> malloc(max_nb_actions * sizeof(SIZE_t))
        action_argmax_indices = <SIZE_t*> malloc(max_nb_actions * sizeof(SIZE_t))

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
                if outer_evi_it > MAX_IT_OUTER:
                    printf("%f", max_u1 - min_u1)
                    return -2

                # epsilon_opt = 1. / pow(2, outer_evi_it)
                # epsilon_opt = 1. / (outer_evi_it * outer_evi_it)
                epsilon_opt = 1. / (outer_evi_it)

                # --------------------------------------------------------------
                # Estimate value function of each option
                # --------------------------------------------------------------
                evi_error = 0
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
                        evi_error = 99
                        continue_flag = 0

                    if self.bound_type != BoundType.BERNSTEIN:
                        # chernoff bound
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
                    while continue_flag and inner_it_opt < MAX_IT_INNER:
                        inner_it_opt = inner_it_opt + 1

                        for i in range(nb_states_per_options):
                            sprime = reach_states[o].values[i]
                            if i == 0:
                                v = max_prob_opt
                            else:
                                v = 0.0

                            #r_optimal = min(r_max, r_tilde_opt[o].values[i])
                            r_optimal = r_tilde_opt[o].values[i]

                            idx = pos2index_2d(nb_states_per_options, nb_states_per_options, i, 0)
                            if self.bound_type != BoundType.BERNSTEIN:
                                max_proba_purec2(&(p_hat_opt[o].values[idx]), nb_states_per_options,
                                    sorted_indices_popt[o].values, beta_opt_p[o].values[idx],
                                    mtx_maxprob_opt_memview[o])
                            else:
                                max_proba_bernstein_cin(
                                    &(p_hat_opt[o].values[idx]),
                                    nb_states_per_options,
                                    sorted_indices_popt[o].values,
                                    &(beta_opt_p[o].values[idx]),
                                    mtx_maxprob_opt_memview[o])

                            v = r_optimal + v + dot_prod(mtx_maxprob_opt_memview[o], w1[o].values, nb_states_per_options)
                            w2[o].values[i] = v

                        # stopping condition
                        if check_end_opt_evi(w2[o].values, w1[o].values, nb_states_per_options, &span_w_opt[o]) < epsilon_opt:
                            # printf("-- %d", inner_it_opt)
                            continue_flag = 0
                            #printf("[%d, %.2f] - ", inner_it_opt, span_w_opt[o])
                        else:
                            memcpy(w1[o].values, w2[o].values, nb_states_per_options * sizeof(DTYPE_t))
                            get_sorted_indices(w1[o].values, nb_states_per_options, sorted_indices_popt[o].values)

                    if inner_it_opt == MAX_IT_INNER:
                        evi_error += 1
                if evi_error > 0:
                    return -1

                # --------------------------------------------------------------
                # Estimate value function of SMDP
                # --------------------------------------------------------------
                evi_error = 0
                count_equal_actions = 0
                for s in range(nb_states):
                    first_action = 1
                    for a_idx in range(self.actions_per_state[s].dim):
                        # printf("action_idx: %d\n", a_idx)
                        action = self.actions_per_state[s].values[a_idx]
                        # printf("action: %d [%d]\n", action, threshold)


                        if action <= threshold:
                            # max_proba_purec
                            # max_proba_reduced
                            if self.bound_type != BoundType.BERNSTEIN:
                                # chernoff bound
                                max_proba_purec(p_hat[s][a_idx], nb_states,
                                            sorted_indices, beta_p[s][a_idx][0],
                                            mtx_maxprob_memview[s])
                            else:
                                # bernstein bound
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

                        if first_action:
                            u2[s] = v
                            count_equal_actions = 1
                            action_argmax[0] = action
                            action_argmax_indices[0] = a_idx
                        elif isclose_c(v, u2[s], 1e-09, 0.5*epsilon_opt):
                            action_argmax[count_equal_actions] = action
                            action_argmax_indices[count_equal_actions] = a_idx
                            count_equal_actions = count_equal_actions + 1
                        elif v > u2[s]:
                            u2[s] = v
                            count_equal_actions = 1
                            action_argmax[0] = action
                            action_argmax_indices[0] = a_idx
                        first_action = 0

                        # error check
                        if v > MAX_VAL_V:
                            evi_error += 1
                    
                    # randomly select action
                    picked_idx = rand() / (RAND_MAX / count_equal_actions + 1)
                    policy_indices[s] = action_argmax_indices[picked_idx]
                    policy[s] = action_argmax[picked_idx]
                    
                if evi_error > 0:
                    return -3

                # stopping condition
                if check_end(u2, u1, nb_states, &min_u1, &max_u1) < epsilon:
                    #free(action_argmax)
                    #free(action_argmax_indices)
                    return max_u1 - min_u1
                else:
                    memcpy(u1, u2, nb_states * sizeof(DTYPE_t))
                    get_sorted_indices(u1, nb_states, sorted_indices)


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


    cpdef get_P_prime_tilde(self):
        cdef SIZE_t sprime, idx, i, o, k, nb_states_per_options, nb_options = self.nb_options
        cdef DTYPE_t[:,:] mtx_maxprob_opt_memview = self.mtx_maxprob_opt_memview #(|O|x|S|)
        cdef IntVectorStruct* reach_states = self.reachable_states_per_option
        cdef DoubleVectorStruct* beta_opt_p = self.beta_opt_p
        cdef IntVectorStruct* sorted_indices_popt = self.sorted_indices_popt
        cdef DoubleVectorStruct* p_hat_opt = self.p_hat_opt
        
        ptilde_list = []
        for o in range(nb_options):
            nb_states_per_options = reach_states[o].dim
            ptilde_list.append(np.zeros((nb_states_per_options, nb_states_per_options)))

        with nogil:
            for o in prange(nb_options):
                nb_states_per_options = reach_states[o].dim
                # sort indices
                get_sorted_indices(self.w1[o].values, nb_states_per_options, sorted_indices_popt[o].values)
                for i in range(nb_states_per_options):
                    sprime = reach_states[o].values[i]

                    idx = pos2index_2d(nb_states_per_options, nb_states_per_options, i, 0)
                    if self.bound_type != BoundType.BERNSTEIN:
                        max_proba_purec2(&(p_hat_opt[o].values[idx]), nb_states_per_options,
                            sorted_indices_popt[o].values, beta_opt_p[o].values[idx],
                            mtx_maxprob_opt_memview[o])
                    else:
                        max_proba_bernstein_cin(
                            &(p_hat_opt[o].values[idx]),
                            nb_states_per_options,
                            sorted_indices_popt[o].values,
                            &(beta_opt_p[o].values[idx]),
                            mtx_maxprob_opt_memview[o])
                    with gil:
                        for k in range(nb_states_per_options):
                            ptilde_list[o][i,k] = mtx_maxprob_opt_memview[o][k]
        return ptilde_list

    cpdef get_uvectors(self):
        u1n = -99*np.ones((self.nb_states,))
        u2n = -99*np.ones((self.nb_states,))
        for i in range(self.nb_states):
            u1n[i] = self.u1[i]
            u2n[i] = self.u2[i]
        return u1n, u2n
    
    cpdef set_seed(self, SIZE_t seed):
        srand(seed)
