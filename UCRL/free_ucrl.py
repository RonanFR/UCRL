from .Ucrl import AbstractUCRL
from .envs import MixedEnvironment
import numpy as np
import math as m
from .logging import default_logger
from .evi import FreeEVIAlg1
import time

class FreeUCRL_Alg1(AbstractUCRL):
    def __init__(self, environment, r_max,
                 range_r=-1, range_p=-1, range_mu_p=-1,
                 bound_type="hoeffding",
                 verbose = 0, logger=default_logger):
        assert isinstance(environment, MixedEnvironment)

        self.pyevi = EVI_Alg1(nb_states=environment.nb_states,
                              thr=environment.threshold_options,
                              action_per_state=environment.get_state_actions(),
                              reachable_states_per_option=environment.reachable_states_per_option,
                              use_bernstein=1 if bound_type == "bernstein" else 0)

        new_evi = FreeEVIAlg1(nb_states=environment.nb_states,
                              nb_options=environment.nb_options,
                              threshold=environment.threshold_options,
                              macro_actions_per_state=environment.get_state_actions(),
                              reachable_states_per_option=environment.reachable_states_per_option,
                              option_policies=environment.options_policies,
                              options_terminating_conditions=environment.options_terminating_conditions,
                              mdp_actions_per_state=environment.environment.get_state_actions(),
                              use_bernstein=1 if bound_type == "bernstein" else 0)

        super(FreeUCRL_Alg1, self).__init__(environment=environment,
                                            r_max=r_max, range_r=range_r,
                                            range_p=range_p, solver=new_evi,
                                            verbose=verbose,
                                            logger=logger,
                                            bound_type=bound_type)

        nb_states = self.environment.nb_states
        max_nb_mdp_actions = environment.max_nb_mdp_actions_per_state # actions of the mdp
        max_nb_actions = environment.max_nb_actions_per_state # actions of mdp+opt

        # keep track of the transition probabilities for options and actions
        self.estimated_probabilities = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states # available primitive actions + options
        self.estimated_probabilities_mdp = np.ones((nb_states, max_nb_mdp_actions, nb_states)) / nb_states # all primitive actions

        # keep track of visits
        self.nb_observations = np.zeros((nb_states, max_nb_actions), dtype=np.int64)
        self.nu_k = np.zeros((nb_states, max_nb_actions), dtype=np.int64)
        self.nb_observations_mdp = np.zeros((nb_states, max_nb_mdp_actions), dtype=np.int64) # for reward of mdp actions
        self.nu_k_mdp = np.zeros((nb_states, max_nb_mdp_actions), dtype=np.int64) # for reward of mdp actions

        # reward info
        self.estimated_rewards_mdp = np.ones((nb_states, max_nb_mdp_actions)) * (r_max + 1)

        if (np.asarray(range_mu_p) < 0).any():
            self.range_mu_p = 1
        else:
            self.range_mu_p = range_mu_p

    def learn(self, duration, regret_time_step):
        if self.total_time >= duration:
            return
        threshold = self.total_time + regret_time_step
        threshold_span = threshold

        self.solver_times = []
        self.simulation_times = []
        alg_trace = {'span_values': []}

        t_star_all = time.perf_counter()
        while self.total_time < duration:
            self.episode += 1
            # print(self.total_time)

            # initialize the episode
            self.nu_k_mdp.fill(0)
            self.nu_k.fill(0)
            self.delta = 1 / m.sqrt(self.iteration + 1)

            if self.verbose > 0:
                self.logger.info("{}/{} = {:3.2f}%".format(self.total_time, duration, self.total_time / duration *100))

            # solve the optimistic (extended) model
            t0 = time.time()
            span_value = self.solve_optimistic_model()
            t1 = time.time()
            span_value /= self.r_max
            alg_trace['span_values'].append(span_value)
            if self.verbose > 0:
                self.logger.info("span({}): {:.9f}".format(self.episode, span_value))
                self.logger.info("evi time: {:.4f} s".format(t1-t0))

            if self.total_time > threshold_span:
                self.span_values.append(span_value)
                self.span_times.append(self.total_time)
                threshold_span = self.total_time + regret_time_step

            # execute the recovered policy
            t0 = time.perf_counter()
            while self.update() and self.total_time < duration:
                if self.total_time > threshold:
                    self.regret.append(self.total_time * self.environment.max_gain - self.total_reward)
                    self.unit_duration.append(self.total_time / self.iteration)
                    threshold = self.total_time + regret_time_step
            self.nb_observations_mdp += self.nu_k_mdp
            #assert np.sum(self.nb_observations_mdp) == self.total_time
            self.nb_observations += self.nu_k

            t1 = time.perf_counter()
            self.simulation_times.append(t1-t0)
            if self.verbose > 0:
                self.logger.info("expl time: {:.4f} s".format(t1-t0))

        t_end_all = time.perf_counter()
        self.speed = t_end_all - t_star_all
        self.logger.info("TIME: %.5f s" % self.speed)
        return alg_trace

    def beta_r(self):
        nb_states, nb_prim_actions = self.estimated_rewards_mdp.shape
        beta_r =  np.multiply(self.range_r, np.sqrt(7/2 * m.log(2 * nb_states * nb_prim_actions *
                                                (self.total_time + 1)/self.delta) / np.maximum(1, self.nb_observations_mdp)))
        return beta_r

    def beta_p(self):
        nb_states, nb_actions = self.nb_observations.shape
        if self.bound_type == "hoeffding":
            beta = self.range_p * np.sqrt(14 * nb_states * m.log(2 * nb_actions
                        * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations))
            return beta.reshape([nb_states, nb_actions, 1])
        else:
            Z = m.log(6 * nb_actions * (self.iteration + 1) / self.delta)
            n = np.maximum(1, self.nb_observations)
            A = np.sqrt(2 * self.estimated_probabilities * (1-self.estimated_probabilities) * Z / n[:,:,np.newaxis])
            B = Z * 7 / (3 * n)
            return self.range_p * (A + B[:,:,np.newaxis])

    def update(self):
        s = self.environment.state  # current state
        history = self.environment.execute(self.policy[s])
        s2 = self.environment.state  # new state
        r = self.environment.reward
        t = self.environment.holding_time
        self.total_reward += r
        self.total_time += t
        self.iteration += 1
        if history is None:
            # this is a primitive action
            # update mdp information
            return self.update_from_action(s, self.policy[s], s2, r)
        else:
            return self.update_from_option(history)

    def update_from_option(self, history):
        [s, o, r, t, s2] = history.data # first is for sure an option
        option_index = self.environment.get_index_of_action_state(s, o)
        assert option_index is not None

        scale_factor_opt = self.nb_observations[s][option_index] + self.nu_k[s][option_index]

        self.estimated_probabilities[s][option_index] *= scale_factor_opt / (scale_factor_opt + 1.)
        self.estimated_probabilities[s][option_index][s2] += 1. / (scale_factor_opt + 1.)
        self.nu_k[s][option_index] += 1

        # the rest of the actions are for sure primitive actions (1 level hierarchy)
        iterate_more = True
        for node in history.children:
            assert len(node.children) == 0
            [s, a, r, t, s2] = node.data # this is a primitive action which can be valid or not
            # for sure it is a primitive action (and the index corresponds to all the primitive actions
            if self.update_from_action(s, a, s2, r) == False:
                iterate_more = False

        return iterate_more

    def update_from_action(self, s, a, s2, r):
        mdpopt_index = self.environment.get_index_of_action_state(s, a)
        mdp_index = self.environment.get_index_of_action_state_in_mdp(s, a)

        scale_factor = self.nb_observations_mdp[s][mdp_index] + self.nu_k_mdp[s][mdp_index]

        # check before to increment the counter
        iterate_more = (self.nu_k_mdp[s][mdp_index] < max(1, self.nb_observations_mdp[s][mdp_index]))

        self.estimated_rewards_mdp[s][mdp_index] *= scale_factor / (scale_factor + 1.)
        self.estimated_rewards_mdp[s][mdp_index] += 1. / (scale_factor + 1.) * r
        self.estimated_probabilities_mdp[s][mdp_index] *= scale_factor / (scale_factor + 1.)
        self.estimated_probabilities_mdp[s][mdp_index][s2] += 1. / (scale_factor + 1.)
        self.nu_k_mdp[s][mdp_index] += 1
        if mdpopt_index is not None:
            # this means that the primitive action is a valid action for
            # the environment mdp + opt
            scale_factor_mdpopt = self.nb_observations[s][mdpopt_index] + self.nu_k[s][mdpopt_index]

            self.estimated_probabilities[s][mdpopt_index] *= scale_factor_mdpopt / (scale_factor_mdpopt + 1.)
            self.estimated_probabilities[s][mdpopt_index][s2] += 1. / (scale_factor_mdpopt + 1.)
            self.nu_k[s][mdpopt_index] += 1

        return iterate_more

    def solve_optimistic_model(self):
        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_p = self.beta_p()  # confidence bounds on transition probabilities
        max_nb_actions = self.estimated_probabilities.shape[1]

        t0 = time.time()
        nb_options = self.environment.nb_options
        r_tilde_opt = [None] * nb_options
        mu_opt = [None] * nb_options
        condition_numbers_opt = np.empty((nb_options,))

        beta_mu_p = np.zeros((nb_options,))

        for o in range(nb_options):
            option_policy = self.environment.options_policies[o]
            option_reach_states = self.environment.reachable_states_per_option[o]
            term_cond = self.environment.options_terminating_conditions[o]

            opt_nb_states = len(option_reach_states)

            Q_o = np.zeros((opt_nb_states, opt_nb_states))

            # compute the reward and the mu
            r_o = [0] * len(option_reach_states)
            visits = self.total_time
            bernstein_log = m.log(6* max_nb_actions / self.delta)
            for i, s in enumerate(option_reach_states):
                option_action = option_policy[s]
                option_action_index = self.environment.get_index_of_action_state_in_mdp(s, option_action)
                r_o[i] = self.estimated_rewards_mdp[s, option_action_index] + beta_r[s,option_action_index]

                if visits > self.nb_observations_mdp[s, option_action_index]:
                    visits = self.nb_observations_mdp[s, option_action_index]

                bernstein_bound = 0.
                nb_o = max(1, self.nb_observations_mdp[s, option_action_index])

                for j, sprime in enumerate(option_reach_states):
                    prob = self.estimated_probabilities_mdp[s][option_action_index][sprime]
                    #q_o[i,0] += term_cond[sprime] * prob
                    Q_o[i,j] = (1. - term_cond[sprime]) * prob
                    bernstein_bound += np.sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o) + bernstein_log * 7 / (3 * nb_o)

                if beta_mu_p[o] < bernstein_bound:
                    beta_mu_p[o] = bernstein_bound

            e_m = np.ones((opt_nb_states,1))
            q_o = e_m - np.dot(Q_o, e_m)

            r_tilde_opt[o] = r_o

            if self.bound_type == "hoeffding":
                beta_mu_p[o] =  self.range_mu_p * np.sqrt(14 * opt_nb_states * m.log(2 * max_nb_actions
                    * (self.total_time + 1)/self.delta) / max(1, visits))

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

        span_value, u1, u2 = self.pyevi.run(
            policy_indices=self.policy_indices,
            policy=self.policy,
            p_hat=self.estimated_probabilities,
            r_hat_mdp=self.estimated_rewards_mdp,
            r_tilde_opt=r_tilde_opt,
            beta_p=beta_p,
            beta_r_mdp=beta_r,
            mu_opt=mu_opt,
            cn_opt=condition_numbers_opt,
            beta_mu_p=beta_mu_p,
            r_max=self.r_max,
            epsilon=self.r_max / m.sqrt(self.iteration + 1))
        t1 = time.time()
        print("OLD_EVI: {} s [{}]".format(t1-t0, span_value / self.r_max))

        # CHECK WHEN T_MAX=1
        # # # from UCRL.cython import  extended_value_iteration
        # # # import copy
        # # # new_pol = copy.deepcopy(self.policy)
        # # # new_pol_ind = copy.deepcopy(self.policy_indices)
        # # # from UCRL.Ucrl import UcrlMdp
        # # # uuuu = UcrlMdp(self.environment.environment, self.r_max, range_r=self.range_r, range_p=self.range_p)
        # # # uuuu.policy = new_pol
        # # # uuuu.policy_indices = new_pol_ind
        # # # uuuu.estimated_probabilities = copy.deepcopy(self.estimated_probabilities_mdp)
        # # # uuuu.estimated_rewards=copy.deepcopy(self.estimated_rewards_mdp)
        # # # uuuu.estimated_holding_times=np.ones_like(self.estimated_rewards_mdp)
        # # # uuuu.tau = 1.
        # # # uuuu.tau_max= 1.
        # # # uuuu.tau_min = 1.
        # # #
        # # # sp, u1n, u2n = uuuu.extended_value_iteration(
        # # #                          beta_r=beta_r,
        # # #                          beta_p=beta_p,
        # # #                          beta_tau=np.zeros_like(self.estimated_rewards_mdp),
        # # #                          epsilon=self.r_max / m.sqrt(self.iteration + 1))
        # # # assert np.isclose(span_value, sp)
        # # # assert np.allclose(u1, u1n)
        # # # assert np.allclose(uuuu.policy_indices, self.policy_indices)

        t0 = time.perf_counter()
        self.opt_solver.compute_mu_info(  # environment=self.environment,
            estimated_probabilities_mdp=self.estimated_probabilities_mdp,
            estimated_rewards_mdp=self.estimated_rewards_mdp,
            beta_r=beta_r,
            nb_observations_mdp=self.nb_observations_mdp,
            delta=self.delta,
            max_nb_actions=max_nb_actions,
            total_time=self.total_time,
            range_mu_p=self.range_mu_p)
        t1 = time.perf_counter()

        new_span = self.opt_solver.run(
            policy_indices=self.policy_indices,
            policy=self.policy,
            p_hat=self.estimated_probabilities,
            r_hat_mdp=self.estimated_rewards_mdp,
            beta_p=beta_p,
            beta_r_mdp=beta_r,
            r_max=self.r_max,
            epsilon=self.r_max / m.sqrt(self.iteration + 1))
        t2 = time.perf_counter()
        self.solver_times.append((t1-t0, t2-t1))

        # CHECK WITH PYTHON IMPL
        r = self.opt_solver.get_r_tilde_opt()
        assert len(r) == len(r_tilde_opt)
        for x, y in zip(r, r_tilde_opt):
            assert np.allclose(x, y)

        mu_e = self.opt_solver.get_mu()
        assert len(mu_opt) == len(mu_e)
        for x, y in zip(mu_e, mu_opt):
            assert np.allclose(x, y)

        cn_e = self.opt_solver.get_conditioning_numbers()
        assert np.allclose(condition_numbers_opt, cn_e)

        b_mu_e = self.opt_solver.get_beta_mu_p()
        assert np.allclose(b_mu_e, beta_mu_p)

        print("NEW_EVI: {} s ({} + {})".format(t2-t0, t1-t0, t2-t1))
        print(span_value, new_span)
        assert np.isclose(span_value, new_span)
        #new_u1, new_u2 = self.new_evi.get_uvectors()

        # assert np.allclose(u1, new_u1, 1e-5), "{}\n{}".format(u1, new_u1)
        # assert np.allclose(u2, new_u2, 1e-5), "{}\n{}".format(u2, new_u2)
        return new_span


class EVI_Alg1(object):

    def __init__(self, nb_states, thr, action_per_state,
                 reachable_states_per_option, use_bernstein=0):
        self.mtx_maxprob_memview = np.ones((nb_states, nb_states)) * 99
        self.threshold = thr
        self.actions_per_state = action_per_state
        self.u1 = np.zeros(nb_states)
        self.u2 = np.zeros(nb_states)
        self.reachable_states_per_option = reachable_states_per_option
        self.use_bernstein = use_bernstein

    def run(self, policy_indices, policy,
            p_hat,
            r_hat_mdp,
            r_tilde_opt,
            beta_p,
            beta_r_mdp,
            mu_opt,
            cn_opt,
            beta_mu_p,
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
                            x[kk] = min(r_max, r_tilde_opt[o][kk])

                            if s == kk_v:
                                x[kk] += np.dot(gg, self.u1)

                        sorted_indices_mu = np.argsort(x, kind='mergesort')
                        # print(sorted_indices_mu)

                        max_mu = self.max_proba(mu_opt[o], len(mu_opt[o]),
                                       sorted_indices_mu, cn_opt[o]*beta_mu_p[o])
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

    # def max_proba(self, p, n, asc_sorted_indices, beta):
    #     new_p = 99 * np.ones(n)
    #     temp = min(1., p[asc_sorted_indices[n-1]] + beta/2.0)
    #     new_p[asc_sorted_indices[n-1]] = temp
    #     sum_p = temp
    #     if temp < 1.:
    #         for i in range(0, n-1):
    #             temp = p[asc_sorted_indices[i]]
    #             new_p[asc_sorted_indices[i]] = temp
    #             sum_p += temp
    #         i = 0
    #         while sum_p > 1.0 and i < n:
    #             sum_p -= p[asc_sorted_indices[i]]
    #             new_p[asc_sorted_indices[i]] = max(0.0, 1. - sum_p)
    #             sum_p += new_p[asc_sorted_indices[i]]
    #             i += 1
    #     else:
    #         for i in range(0, n):
    #             new_p[i] = 0
    #         new_p[asc_sorted_indices[n-1]] = temp
    #     return new_p

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
