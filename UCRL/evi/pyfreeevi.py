import numpy as np
import math as m
from ._free_utils import get_mu_and_ci

class PyEVI_FSUCRLv1(object):

    def __init__(self,
                 nb_states,
                 nb_options,
                 macro_actions_per_state,
                 mdp_actions_per_state,
                 threshold,
                 option_policies,
                 reachable_states_per_option,
                 options_terminating_conditions,
                 use_bernstein=0):
        self.nb_states = nb_states
        self.nb_options = nb_options
        self.threshold = threshold
        self.actions_per_state = macro_actions_per_state
        self.mdp_actions_per_state = mdp_actions_per_state
        self.u1 = np.zeros(nb_states)
        self.u2 = np.zeros(nb_states)
        self.option_policies = option_policies
        self.reachable_states_per_option = reachable_states_per_option
        self.options_terminating_conditions = options_terminating_conditions
        self.use_bernstein = use_bernstein

    def compute_mu_info(self,
                        estimated_probabilities_mdp,
                        estimated_rewards_mdp,
                        beta_r,
                        nb_observations_mdp,
                        range_mu_p,
                        total_time,
                        delta,
                        max_nb_actions,
                        r_max):

        nb_options = self.nb_options
        r_tilde_opt = [None] * nb_options
        mu_opt = [None] * nb_options
        condition_numbers_opt = np.empty((nb_options,))

        beta_mu_p = np.zeros((nb_options,))

        for o in range(nb_options):
            option_policy = self.option_policies[o]
            option_reach_states = self.reachable_states_per_option[o]
            term_cond = self.options_terminating_conditions[o]

            opt_nb_states = len(option_reach_states)

            Q_o = np.zeros((opt_nb_states, opt_nb_states))

            # compute the reward and the mu
            r_o = [0] * len(option_reach_states)
            visits = total_time
            bernstein_log = m.log(6* max_nb_actions / delta)
            for i, s in enumerate(option_reach_states):
                option_action = option_policy[s]
                option_action_index = self.mdp_actions_per_state[s].index(option_action)
                r_o[i] = min(r_max, estimated_rewards_mdp[s, option_action_index] + beta_r[s,option_action_index])

                if visits > nb_observations_mdp[s, option_action_index]:
                    visits = nb_observations_mdp[s, option_action_index]

                bernstein_bound = 0.
                nb_o = max(1, nb_observations_mdp[s, option_action_index])

                for j, sprime in enumerate(option_reach_states):
                    prob = estimated_probabilities_mdp[s][option_action_index][sprime]
                    #q_o[i,0] += term_cond[sprime] * prob
                    Q_o[i,j] = (1. - term_cond[sprime]) * prob
                    bernstein_bound += np.sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o) + bernstein_log * 7 / (3 * nb_o)

                if beta_mu_p[o] < bernstein_bound:
                    beta_mu_p[o] = bernstein_bound

            e_m = np.ones((opt_nb_states,1))
            q_o = e_m - np.dot(Q_o, e_m)

            r_tilde_opt[o] = r_o

            if self.use_bernstein == 0:
                beta_mu_p[o] =  range_mu_p * np.sqrt(14 * opt_nb_states * m.log(2 * max_nb_actions
                    * (total_time + 1)/ delta) / max(1, visits))

            Pprime_o = np.concatenate((q_o, Q_o[:, 1:]), axis=1)
            if not np.allclose(np.sum(Pprime_o, axis=1), np.ones(opt_nb_states)):
                print("{}\n{}".format(Pprime_o,Q_o))


            Pap = (Pprime_o + np.eye(opt_nb_states)) / 2.
            D, U = np.linalg.eig(
                np.transpose(Pap))  # eigen decomposition of transpose of P
            sorted_indices = np.argsort(np.real(D))
            mu = np.transpose(np.real(U))[sorted_indices[-1]]
            mu /= np.sum(mu)  # stationary distribution
            mu_opt[o] = mu

            assert len(mu_opt[o]) == len(r_tilde_opt[o])

            P_star = np.repeat(np.array(mu, ndmin=2), opt_nb_states,
                               axis=0)  # limiting matrix

            # Compute deviation matrix
            I = np.eye(opt_nb_states)  # identity matrix
            Z = np.linalg.inv(I - Pap + P_star)  # fundamental matrix
            H = np.dot(Z, I - P_star)  # deviation matrix

            condition_nb = 0  # condition number of deviation matrix
            for i in range(0, opt_nb_states):  # Seneta's condition number
                for j in range(i + 1, opt_nb_states):
                    condition_nb = max(condition_nb,
                                       0.5 * np.linalg.norm(H[i, :] - H[j, :],
                                                            ord=1))
            condition_numbers_opt[o] = condition_nb

            mu_n, cn_n = get_mu_and_ci(Pap)

            assert np.allclose(mu, mu_n)
            assert np.isclose(condition_nb, cn_n)

        self.r_tilde_opt = r_tilde_opt
        self.condition_numbers_opt = condition_numbers_opt
        self.beta_mu_p = beta_mu_p
        self.mu_opt = mu_opt

        return 0

    def run(self, policy_indices, policy,
            p_hat,
            r_hat_mdp,
            beta_p,
            beta_r_mdp,
            r_max,
            epsilon):
        self.u1.fill(0.0)
        nb_states = p_hat.shape[0]
        sorted_indices_u = np.arange(nb_states)
        self.counter = 0
        while True:
            self.counter += 1
            for s in range(nb_states):
                first_action = True
                for action_idx, action in enumerate(self.actions_per_state[s]):
                    if self.use_bernstein == 0:
                        gg = self.max_proba(p_hat[s][action_idx], nb_states,
                                       sorted_indices_u, beta_p[s][action_idx][0])
                        assert len(beta_p[s][action_idx]) == 1
                    else:
                        gg = self.max_proba_bernstein(p_hat[s][action_idx], nb_states,
                                       sorted_indices_u, beta_p[s][action_idx])
                    gg[s] = gg[s] - 1.

                    if action <= self.threshold:
                        #this is an action
                        r_optimal = min(r_max, r_hat_mdp[s, action] + beta_r_mdp[s, action])
                        v = r_optimal + np.dot(gg, self.u1)
                    else:

                        o = action - self.threshold - 1 # zero based index

                        x = np.zeros(len(self.reachable_states_per_option[o]))
                        for kk, kk_v in enumerate(self.reachable_states_per_option[o]):
                            x[kk] = self.r_tilde_opt[o][kk]

                            if s == kk_v:
                                x[kk] += np.dot(gg, self.u1)

                        sorted_indices_mu = np.argsort(x, kind='mergesort')
                        # print(sorted_indices_mu)

                        max_mu = self.max_proba(self.mu_opt[o], len(self.mu_opt[o]),
                                       sorted_indices_mu, self.condition_numbers_opt[o]*self.beta_mu_p[o])
                        v = np.dot(max_mu, x)

                    c1 = v + self.u1[s]
                    if first_action or c1 > self.u2[s] or m.isclose(c1, self.u2[s]):
                        self.u2[s] = c1
                        policy_indices[s] = action_idx
                        policy[s] = action
                    first_action = False

            # print("**%d\n" % self.counter)
            # for i in range(nb_states):
            #     print("{:.2f}[{:.2f}] ".format(self.u1[i], self.u2[i]), end=" ")
            # print("\n")

            if max(self.u2-self.u1)-min(self.u2-self.u1) < epsilon:  # stopping condition
                # print("++ {}\n".format(self.counter))
                return max(self.u1) - min(self.u1), self.u1, self.u2
            else:
                self.u1 = self.u2
                self.u2 = np.empty(nb_states)
                sorted_indices_u = np.argsort(self.u1, kind='mergesort')
                # for i in range(nb_states):
                #     printf("%d , ", sorted_indices[i])
                # printf("\n")

    def max_proba(self, p, n, sorted_indices, beta):
        """
        Use compiled .so file for improved speed.
        :param p: probability distribution with toys support
        :param sorted_indices: argsort of value function
        :param beta: confidence bound on the empirical probability
        :return: optimal probability
        """
        #n = np.size(sorted_indices)
        min1 = min(1, p[sorted_indices[n-1]] + beta/2)
        if min1 == 1:
            p2 = np.zeros(n)
            p2[sorted_indices[n-1]] = 1
        else:
            sorted_p = p[sorted_indices]
            support_sorted_p = np.nonzero(sorted_p)[0]
            restricted_sorted_p = sorted_p[support_sorted_p]
            support_p = sorted_indices[support_sorted_p]
            p2 = np.zeros(n)
            p2[support_p] = restricted_sorted_p
            p2[sorted_indices[n-1]] = min1
            s = 1 - p[sorted_indices[n-1]] + min1
            s2 = s
            for i, proba in enumerate(restricted_sorted_p):
                max1 = max(0, 1 - s + proba)
                s2 += (max1 - proba)
                p2[support_p[i]] = max1
                s = s2
                if s <= 1: break
        return p2

    def max_proba_bernstein(self, p, n, asc_sorted_indices, beta):
        new_p = np.zeros(n)
        delta = 1.
        for i in range(n):
            new_p[i] = max(0, p[i] - beta[i])
            delta -= new_p[i]
        i = n - 1
        while delta > 0 and i >= 0:
            idx = asc_sorted_indices[i]
            new_delta = min(delta, p[idx] + beta[idx] - new_p[idx])
            new_p[idx] += new_delta
            delta -= new_delta
            i -= 1
        return new_p


class PyEVI_FSUCRLv2(object):

    def __init__(self,
                 nb_states,
                 nb_options,
                 macro_actions_per_state,
                 mdp_actions_per_state,
                 threshold,
                 option_policies,
                 reachable_states_per_option,
                 options_terminating_conditions,
                 use_bernstein=0):
        self.nb_states = nb_states
        self.nb_options = nb_options
        self.threshold = threshold
        self.actions_per_state = macro_actions_per_state
        self.mdp_actions_per_state = mdp_actions_per_state
        self.u1 = np.zeros(nb_states)
        self.u2 = np.zeros(nb_states)
        self.option_policies = option_policies
        self.reachable_states_per_option = reachable_states_per_option
        self.options_terminating_conditions = options_terminating_conditions
        self.use_bernstein = use_bernstein

        self.w1 = []
        self.w2 = []
        for o in range(nb_options):
            ns = len(self.reachable_states_per_option[o])
            self.w1.append(np.zeros(ns))
            self.w2.append(np.zeros(ns))

    def compute_prerun_info(self,
                            estimated_probabilities_mdp,
                            estimated_rewards_mdp,
                            beta_r,
                            nb_observations_mdp,
                            total_time,
                            delta,
                            max_nb_actions,
                            range_opt_p):

        nb_options = self.nb_options
        self.r_tilde_opt = [None] * nb_options
        self.beta_opt_p = [None] * nb_options
        self.p_hat_opt = [None] * nb_options

        visits = [None] * nb_options
        for o in range(nb_options):
            option_policy = self.option_policies[o]
            option_reach_states = self.reachable_states_per_option[o]
            term_cond = self.options_terminating_conditions[o]

            opt_nb_states = len(option_reach_states)

            Q_o = np.zeros((opt_nb_states, opt_nb_states))

            # compute the reward and the mu
            r_o = [0] * opt_nb_states
            bernstein_log = m.log(6* max_nb_actions / delta)
            self.beta_opt_p[o] = np.empty((opt_nb_states, opt_nb_states))

            visits[o] = [0] * len(option_reach_states)

            for i, s in enumerate(option_reach_states):
                option_action = option_policy[s]
                option_action_index = self.mdp_actions_per_state[s].index(option_action)
                r_o[i] = estimated_rewards_mdp[s, option_action_index] + beta_r[s,option_action_index]


                bernstein_bound = 0.
                nb_o = max(1, nb_observations_mdp[s, option_action_index])

                visits[o][i] = nb_observations_mdp[s, option_action_index]

                for j, sprime in enumerate(option_reach_states):
                    prob = estimated_probabilities_mdp[s][option_action_index][sprime]
                    #q_o[i,0] += term_cond[sprime] * prob
                    Q_o[i,j] = (1. - term_cond[sprime]) * prob
                    if self.use_bernstein == 0:
                        self.beta_opt_p[o][i, j] = range_opt_p * np.sqrt(14 * opt_nb_states * m.log(2 * max_nb_actions
                                                                                          * (total_time + 1)/ delta) / nb_o)
                    else:
                        self.beta_opt_p[o][i, j] = range_opt_p * (np.sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o)
                                                                   + bernstein_log * 7 / (3 * nb_o))

            e_m = np.ones((opt_nb_states,1))
            q_o = e_m - np.dot(Q_o, e_m)

            if len(option_reach_states) > 1 and np.isclose(q_o[0],1) and visits[o][0]>20:
                raise ValueError

            self.r_tilde_opt[o] = r_o

            Pprime_o = np.concatenate((q_o, Q_o[:, 1:]), axis=1)
            if not np.allclose(np.sum(Pprime_o, axis=1), np.ones(opt_nb_states)):
                print("{}\n{}".format(Pprime_o,Q_o))
            self.p_hat_opt[o] = Pprime_o

        return 0

    def run(self, policy_indices, policy,
            p_hat,
            r_hat_mdp,
            beta_p,
            beta_r_mdp,
            r_max,
            epsilon):
        self.u1.fill(0.0)
        nb_states = self.nb_states
        nb_options = self.nb_options
        sorted_indices_u = np.arange(nb_states)

        for o in range(nb_options):
            for i in range(len(self.w1[o])):
                self.w1[o][i] = 0.0
                self.w2[o][i] = 0.0

        self.outer_evi_it = 0
        while True:
            self.outer_evi_it += 1
            # print(self.outer_evi_it)
            if self.outer_evi_it > 50000:
                print("{}".format(max(self.u1) - min(self.u1)))
                return -5

            # ------------------------------------------------------------------
            # Estimate value function of each option
            # ------------------------------------------------------------------
            for opt in range(nb_options):
                continue_flag = True
                nb_states_per_options = len(self.reachable_states_per_option[opt])
                epsilon_opt = 1 / (2**self.outer_evi_it)

                #initialize
                sorted_indices_popt = np.arange(nb_states_per_options)
                for i in range(nb_states_per_options):
                    self.w1[opt][i] = 0.0#self.w1[opt][i] - self.w1[opt][0]  # 0.0
                self.w2[opt].fill(0.0)
                inner_it_opt = 0
                while continue_flag:
                    inner_it_opt += 1
                    # print(" .{}: {}".format(inner_it_opt, max(self.w2[opt] - self.w1[opt]) - min(self.w2[opt] - self.w1[opt])))
                    for i, sprime in enumerate(self.reachable_states_per_option[opt]):
                        if i == 0:
                            # this is the starting state
                            option_act = opt + self.threshold + 1
                            option_idx = self.actions_per_state[sprime].index(option_act)
                            # print("({}) ".format(option_idx), end=" ")
                            assert option_idx is not None
                            if self.use_bernstein == 0:
                                max_p = self.max_proba(p_hat[sprime][option_idx], nb_states,
                                                sorted_indices_u,
                                                beta_p[sprime][option_idx][0])
                                assert len(beta_p[sprime][option_idx]) == 1
                            else:
                                max_p = self.max_proba_bernstein(p_hat[sprime][option_idx],
                                                          nb_states,
                                                          sorted_indices_u,
                                                          beta_p[sprime][option_idx])
                            max_p[sprime] = max_p[sprime] - 1.
                            v = np.dot(max_p, self.u1)
                        else:
                            v = 0.
                        r_optimal = min(r_max, self.r_tilde_opt[opt][i])

                        max_p = self.max_proba_bernstein(self.p_hat_opt[opt][i, :],
                                                             nb_states_per_options,
                                                sorted_indices_popt,
                                                self.beta_opt_p[opt][i, :])
                        assert np.isclose(np.sum(max_p),1.)
                        v = r_optimal + v + np.dot(max_p, self.w1[opt])
                        # print(" {:.2f}".format(v), end=" ")
                        self.w2[opt][i] = v
                        if v > 50000:
                            return -4

                    if max(self.w2[opt] - self.w1[opt]) - min(self.w2[opt] - self.w1[opt]) < epsilon_opt:  # stopping condition
                        continue_flag = False

                        # print("[{}, {:.2f}] -".format(inner_it_opt, 0.5 * (
                        #     max(self.w2[opt] - self.w1[opt]) + min(
                        #         self.w2[opt] - self.w1[opt]))), end=" ")
                    else:
                        self.w1[opt] = self.w2[opt]
                        self.w2[opt] = np.empty(len(self.w1[opt]))
                        sorted_indices_popt = np.argsort(self.w1[opt], kind='mergesort')

            # ------------------------------------------------------------------
            # Estimate value function of SMDP
            # ------------------------------------------------------------------
            for s in range(nb_states):
                first_action = True
                for action_idx, action in enumerate(self.actions_per_state[s]):
                    if action <= self.threshold:
                        #this is an action
                        if self.use_bernstein == 0:
                            max_p = self.max_proba(p_hat[s][action_idx], nb_states,
                                           sorted_indices_u, beta_p[s][action_idx][0])
                            assert len(beta_p[s][action_idx]) == 1
                        else:
                            max_p = self.max_proba_bernstein(p_hat[s][action_idx], nb_states,
                                           sorted_indices_u, beta_p[s][action_idx])
                        max_p[s] = max_p[s] - 1.
                        r_optimal = min(r_max, r_hat_mdp[s, action] + beta_r_mdp[s, action])
                        v = r_optimal + np.dot(max_p, self.u1)
                    else:

                        o = action - self.threshold - 1 # zero based index
                        v = 0.5 * (max(self.w2[o] - self.w1[o]) + min(self.w2[o] - self.w1[o]))

                    c1 = v + self.u1[s]
                    if first_action or c1 > self.u2[s] or m.isclose(c1, self.u2[s]):
                        self.u2[s] = c1
                        policy_indices[s] = action_idx
                        policy[s] = action
                    first_action = False
            # print()
            # for i in range(nb_states):
            #     print("{:.2f}[{:.2f}]".format(self.u1[i], self.u2[i]), end=" ")
            # print()

            if max(self.u2-self.u1)-min(self.u2-self.u1) < epsilon:  # stopping condition
                return max(self.u1) - min(self.u1)
            else:
                # print("\n+++++++++", end=" ")
                self.u1 = self.u2
                self.u2 = np.empty(nb_states)
                sorted_indices_u = np.argsort(self.u1, kind='mergesort')

    def max_proba(self, p, n, sorted_indices, beta):
        """
        Use compiled .so file for improved speed.
        :param p: probability distribution with toys support
        :param sorted_indices: argsort of value function
        :param beta: confidence bound on the empirical probability
        :return: optimal probability
        """
        #n = np.size(sorted_indices)
        min1 = min(1, p[sorted_indices[n-1]] + beta/2)
        if min1 == 1:
            p2 = np.zeros(n)
            p2[sorted_indices[n-1]] = 1
        else:
            sorted_p = p[sorted_indices]
            support_sorted_p = np.nonzero(sorted_p)[0]
            restricted_sorted_p = sorted_p[support_sorted_p]
            support_p = sorted_indices[support_sorted_p]
            p2 = np.zeros(n)
            p2[support_p] = restricted_sorted_p
            p2[sorted_indices[n-1]] = min1
            s = 1 - p[sorted_indices[n-1]] + min1
            s2 = s
            for i, proba in enumerate(restricted_sorted_p):
                max1 = max(0, 1 - s + proba)
                s2 += (max1 - proba)
                p2[support_p[i]] = max1
                s = s2
                if s <= 1: break
        return p2

    def max_proba_bernstein(self, p, n, asc_sorted_indices, beta):
        new_p = np.zeros(n)
        delta = 1.
        for i in range(n):
            new_p[i] = max(0, p[i] - beta[i])
            delta -= new_p[i]
        i = n - 1
        while delta > 0 and i >= 0:
            idx = asc_sorted_indices[i]
            new_delta = min(delta, p[idx] + beta[idx] - new_p[idx])
            new_p[idx] += new_delta
            delta -= new_delta
            i -= 1
        return new_p