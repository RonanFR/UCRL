from .Ucrl import AbstractUCRL
from .envs import MixedEnvironment
import numpy as np
import math as m
from .logging import default_logger
from .evi import EVI_FSUCRLv1, EVI_FSUCRLv2
from .evi.pyfreeevi import PyEVI_FSUCRLv1, PyEVI_FSUCRLv2
import time


class FSUCRLv1(AbstractUCRL):
    def __init__(self, environment, r_max,
                 range_r=-1, range_p=-1, range_mu_p=-1,
                 bound_type="hoeffding",
                 verbose = 0, logger=default_logger,
                 evi_solver=None):

        assert isinstance(environment, MixedEnvironment)

        # self.pyevi = PyEVI_FSUCRLv1(nb_states=environment.nb_states,
        #                       thr=environment.threshold_options,
        #                       action_per_state=environment.get_state_actions(),
        #                       reachable_states_per_option=environment.reachable_states_per_option,
        #                       use_bernstein=1 if bound_type == "bernstein" else 0)

        if evi_solver is None:
            evi_solver = EVI_FSUCRLv1(nb_states=environment.nb_states,
                                  nb_options=environment.nb_options,
                                  threshold=environment.threshold_options,
                                  macro_actions_per_state=environment.get_state_actions(),
                                  reachable_states_per_option=environment.reachable_states_per_option,
                                  option_policies=environment.options_policies,
                                  options_terminating_conditions=environment.options_terminating_conditions,
                                  mdp_actions_per_state=environment.environment.get_state_actions(),
                                  use_bernstein=1 if bound_type == "bernstein" else 0)

        super(FSUCRLv1, self).__init__(environment=environment,
                                       r_max=r_max, range_r=range_r,
                                       range_p=range_p, solver=evi_solver,
                                       verbose=verbose,
                                       logger=logger,
                                       bound_type=bound_type)

        nb_states = self.environment.nb_states
        max_nb_mdp_actions = environment.max_nb_mdp_actions_per_state # actions of the mdp
        max_nb_actions = environment.max_nb_actions_per_state # actions of mdp+opt

        # keep track of the transition probabilities for options and actions
        # self.estimated_probabilities = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states # available primitive actions + options
        # self.estimated_probabilities_mdp = np.ones((nb_states, max_nb_mdp_actions, nb_states)) / nb_states # all primitive actions

        self.P_mdp_counter = np.zeros((nb_states, max_nb_mdp_actions, nb_states), dtype=np.int64)
        self.P_mdp = np.ones((nb_states, max_nb_mdp_actions, nb_states)) / nb_states
        self.visited_sa_mdp = set()

        self.P_counter = np.zeros((nb_states, max_nb_actions, nb_states), dtype=np.int64)
        self.P = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states
        self.visited_sa = set()


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
        self.regret_unit_time = []

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

            if self.verbose > 2:
                for i, a in enumerate(self.policy_indices):
                    from UCRL.envs.toys.roommaze import state2coord
                    row, col = state2coord(i,
                                           self.environment.dimension)
                    print("{}".format(a), end=" ")
                    if col == self.environment.dimension - 1:
                        print("")
                print("---------------")

            span_value /= self.r_max
            if self.verbose > 0:
                self.logger.info("span({}): {:.9f}".format(self.episode, span_value))
                self.logger.info("evi time: {:.4f} s".format(t1-t0))

            if self.total_time > threshold_span:
                self.span_values.append(span_value)
                self.span_times.append(self.total_time)
                threshold_span = self.total_time + regret_time_step

            # execute the recovered policy
            t0 = time.perf_counter()
            self.visited_sa_mdp.clear()
            self.visited_sa.clear()
            while self.update() and self.total_time < duration:
                if self.total_time > threshold:
                    curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                    self.regret.append(curr_regret)
                    self.regret_unit_time.append(self.total_time)
                    self.unit_duration.append(self.total_time / self.iteration)
                    threshold = self.total_time + regret_time_step
            self.nb_observations_mdp += self.nu_k_mdp
            #assert np.sum(self.nb_observations_mdp) == self.total_time
            self.nb_observations += self.nu_k

            for (s,a) in self.visited_sa_mdp:
                self.P_mdp[s,a]  = self.P_mdp_counter[s,a] / (self.nb_observations_mdp[s,a])
                # assert np.sum(self.P_mdp_counter[s,a]) == self.nb_observations_mdp[s,a]
            # assert np.allclose(self.estimated_probabilities_mdp, self.P_mdp)

            for (s,a) in self.visited_sa:
                self.P[s,a]  = self.P_counter[s,a] / (self.nb_observations[s,a])
                # assert np.sum(self.P_counter[s,a]) == self.nb_observations[s,a]
            # assert np.allclose(self.estimated_probabilities, self.P)

            t1 = time.perf_counter()
            self.simulation_times.append(t1-t0)
            if self.verbose > 0:
                self.logger.info("expl time: {:.4f} s".format(t1-t0))

        t_end_all = time.perf_counter()
        self.speed = t_end_all - t_star_all
        self.logger.info("TIME: %.5f s" % self.speed)

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

        # scale_factor_opt = self.nb_observations[s][option_index] + self.nu_k[s][option_index]
        # self.estimated_probabilities[s][option_index] *= scale_factor_opt / (scale_factor_opt + 1.)
        # self.estimated_probabilities[s][option_index][s2] += 1. / (scale_factor_opt + 1.)

        self.P_counter[s,option_index,s2] += 1
        self.visited_sa.add((s,option_index))

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

        # check before to increment the counter
        nu_k_mdp_temp = self.nu_k_mdp[s, mdp_index]
        nb_observations_mdp_temp = self.nb_observations_mdp[s, mdp_index]

        #iterate_more = (self.nu_k_mdp[s, mdp_index] < max(1, self.nb_observations_mdp[s, mdp_index]))
        iterate_more = (nu_k_mdp_temp < max(1, nb_observations_mdp_temp))

        #scale_factor = self.nb_observations_mdp[s, mdp_index] + self.nu_k_mdp[s, mdp_index]
        scale_factor = nb_observations_mdp_temp + nu_k_mdp_temp

        r_old = self.estimated_rewards_mdp[s, mdp_index]
        self.estimated_rewards_mdp[s, mdp_index] = ( r_old*scale_factor + r ) / (scale_factor + 1.) #+ r / (scale_factor + 1.)
        # self.estimated_rewards_mdp[s, mdp_index] *= scale_factor / (scale_factor + 1.)
        # self.estimated_rewards_mdp[s, mdp_index] += r / (scale_factor + 1.)
        # self.estimated_probabilities_mdp[s][mdp_index] *= scale_factor / (scale_factor + 1.)
        # self.estimated_probabilities_mdp[s][mdp_index][s2] += 1. / (scale_factor + 1.)
        self.P_mdp_counter[s, mdp_index, s2] += 1
        self.visited_sa_mdp.add((s, mdp_index))
        self.nu_k_mdp[s, mdp_index] += 1
        if mdpopt_index is not None:
            # this means that the primitive action is a valid action for
            # the environment mdp + opt

            # scale_factor_mdpopt = self.nb_observations[s][mdpopt_index] + self.nu_k[s][mdpopt_index]
            # self.estimated_probabilities[s][mdpopt_index] *= scale_factor_mdpopt / (scale_factor_mdpopt + 1.)
            # self.estimated_probabilities[s][mdpopt_index][s2] += 1. / (scale_factor_mdpopt + 1.)
            self.P_counter[s, mdpopt_index, s2] += 1
            self.visited_sa.add((s, mdpopt_index))
            self.nu_k[s, mdpopt_index] += 1

        return iterate_more

    def solve_optimistic_model(self):
        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_p = self.beta_p()  # confidence bounds on transition probabilities
        max_nb_actions = self.nb_observations.shape[1]

        #
        # t0 = time.time()
        # nb_options = self.environment.nb_options
        # r_tilde_opt = [None] * nb_options
        # mu_opt = [None] * nb_options
        # condition_numbers_opt = np.empty((nb_options,))
        #
        # beta_mu_p = np.zeros((nb_options,))
        #
        # for o in range(nb_options):
        #     option_policy = self.environment.options_policies[o]
        #     option_reach_states = self.environment.reachable_states_per_option[o]
        #     term_cond = self.environment.options_terminating_conditions[o]
        #
        #     opt_nb_states = len(option_reach_states)
        #
        #     Q_o = np.zeros((opt_nb_states, opt_nb_states))
        #
        #     # compute the reward and the mu
        #     r_o = [0] * len(option_reach_states)
        #     visits = self.total_time
        #     bernstein_log = m.log(6* max_nb_actions / self.delta)
        #     for i, s in enumerate(option_reach_states):
        #         option_action = option_policy[s]
        #         option_action_index = self.environment.get_index_of_action_state_in_mdp(s, option_action)
        #         r_o[i] = self.estimated_rewards_mdp[s, option_action_index] + beta_r[s,option_action_index]
        #
        #         if visits > self.nb_observations_mdp[s, option_action_index]:
        #             visits = self.nb_observations_mdp[s, option_action_index]
        #
        #         bernstein_bound = 0.
        #         nb_o = max(1, self.nb_observations_mdp[s, option_action_index])
        #
        #         for j, sprime in enumerate(option_reach_states):
        #             prob = self.estimated_probabilities_mdp[s][option_action_index][sprime]
        #             #q_o[i,0] += term_cond[sprime] * prob
        #             Q_o[i,j] = (1. - term_cond[sprime]) * prob
        #             bernstein_bound += np.sqrt(bernstein_log * 2 * prob * (1 - prob) / nb_o) + bernstein_log * 7 / (3 * nb_o)
        #
        #         if beta_mu_p[o] < bernstein_bound:
        #             beta_mu_p[o] = bernstein_bound
        #
        #     e_m = np.ones((opt_nb_states,1))
        #     q_o = e_m - np.dot(Q_o, e_m)
        #
        #     r_tilde_opt[o] = r_o
        #
        #     if self.bound_type == "hoeffding":
        #         beta_mu_p[o] =  self.range_mu_p * np.sqrt(14 * opt_nb_states * m.log(2 * max_nb_actions
        #             * (self.total_time + 1)/self.delta) / max(1, visits))
        #
        #     Pprime_o = np.concatenate((q_o, Q_o[:, 1:]), axis=1)
        #     if not np.allclose(np.sum(Pprime_o, axis=1), np.ones(opt_nb_states)):
        #         print("{}\n{}".format(Pprime_o,Q_o))
        #
        #
        #     Pap = (Pprime_o + np.eye(opt_nb_states)) / 2.
        #     D, U = np.linalg.eig(
        #         np.transpose(Pap))  # eigen decomposition of transpose of P
        #     sorted_indices = np.argsort(np.real(D))
        #     mu = np.transpose(np.real(U))[sorted_indices[-1]]
        #     mu /= np.sum(mu)  # stationary distribution
        #     mu_opt[o] = mu
        #
        #     assert len(mu_opt[o]) == len(r_tilde_opt[o])
        #
        #     P_star = np.repeat(np.array(mu, ndmin=2), opt_nb_states,
        #                        axis=0)  # limiting matrix
        #
        #     # Compute deviation matrix
        #     I = np.eye(opt_nb_states)  # identity matrix
        #     Z = np.linalg.inv(I - Pap + P_star)  # fundamental matrix
        #     H = np.dot(Z, I - P_star)  # deviation matrix
        #
        #     condition_nb = 0  # condition number of deviation matrix
        #     for i in range(0, opt_nb_states):  # Seneta's condition number
        #         for j in range(i + 1, opt_nb_states):
        #             condition_nb = max(condition_nb,
        #                                0.5 * np.linalg.norm(H[i, :] - H[j, :],
        #                                                     ord=1))
        #     condition_numbers_opt[o] = condition_nb
        #
        # span_value, u1, u2 = self.pyevi.run(
        #     policy_indices=self.policy_indices,
        #     policy=self.policy,
        #     p_hat=self.estimated_probabilities,
        #     r_hat_mdp=self.estimated_rewards_mdp,
        #     r_tilde_opt=r_tilde_opt,
        #     beta_p=beta_p,
        #     beta_r_mdp=beta_r,
        #     mu_opt=mu_opt,
        #     cn_opt=condition_numbers_opt,
        #     beta_mu_p=beta_mu_p,
        #     r_max=self.r_max,
        #     epsilon=self.r_max / m.sqrt(self.iteration + 1))
        # t1 = time.time()
        # print("OLD_EVI: {} s [{}]".format(t1-t0, span_value / self.r_max))

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
            estimated_probabilities_mdp=self.P_mdp, #self.estimated_probabilities_mdp,
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
            p_hat=self.P,#self.estimated_probabilities,
            r_hat_mdp=self.estimated_rewards_mdp,
            beta_p=beta_p,
            beta_r_mdp=beta_r,
            r_max=self.r_max,
            epsilon=self.r_max / m.sqrt(self.iteration + 1))
        t2 = time.perf_counter()
        self.solver_times.append((t1-t0, t2-t1))

        # # CHECK WITH PYTHON IMPL
        # r = self.opt_solver.get_r_tilde_opt()
        # assert len(r) == len(r_tilde_opt)
        # for x, y in zip(r, r_tilde_opt):
        #     assert np.allclose(x, y)
        #
        # mu_e = self.opt_solver.get_mu()
        # assert len(mu_opt) == len(mu_e)
        # for x, y in zip(mu_e, mu_opt):
        #     assert np.allclose(x, y)
        #
        # cn_e = self.opt_solver.get_conditioning_numbers()
        # assert np.allclose(condition_numbers_opt, cn_e)
        #
        # b_mu_e = self.opt_solver.get_beta_mu_p()
        # assert np.allclose(b_mu_e, beta_mu_p)
        #
        # print("NEW_EVI: {} s ({} + {})".format(t2-t0, t1-t0, t2-t1))
        # print(span_value, new_span)
        # assert np.isclose(span_value, new_span)
        # #new_u1, new_u2 = self.new_evi.get_uvectors()
        #
        # # assert np.allclose(u1, new_u1, 1e-5), "{}\n{}".format(u1, new_u1)
        # # assert np.allclose(u2, new_u2, 1e-5), "{}\n{}".format(u2, new_u2)

        return new_span


class FSUCRLv2(FSUCRLv1):
    def __init__(self, environment, r_max,
                 range_r=-1, range_p=-1, range_opt_p=-1,
                 bound_type="hoeffding",
                 verbose = 0, logger=default_logger):

        py = False
        if py:
            evi_solver = self.pyevi = PyEVI_FSUCRLv2(nb_states=environment.nb_states,
                                    nb_options=environment.nb_options,
                                    threshold=environment.threshold_options,
                                    macro_actions_per_state=environment.get_state_actions(),
                                    reachable_states_per_option=environment.reachable_states_per_option,
                                    option_policies=environment.options_policies,
                                    options_terminating_conditions=environment.options_terminating_conditions,
                                    mdp_actions_per_state=environment.environment.get_state_actions(),
                                    use_bernstein=1 if bound_type == "bernstein" else 0)
        else:
            evi_solver = EVI_FSUCRLv2(nb_states=environment.nb_states,
                                      nb_options=environment.nb_options,
                                      threshold=environment.threshold_options,
                                      macro_actions_per_state=environment.get_state_actions(),
                                      reachable_states_per_option=environment.reachable_states_per_option,
                                      option_policies=environment.options_policies,
                                      options_terminating_conditions=environment.options_terminating_conditions,
                                      mdp_actions_per_state=environment.environment.get_state_actions(),
                                      use_bernstein=1 if bound_type == "bernstein" else 0)

        super(FSUCRLv2, self).__init__(
            environment=environment,
            r_max=r_max,
            range_r=range_r, range_p=range_p, range_mu_p=range_opt_p,
            bound_type=bound_type,
            verbose=verbose, logger=logger,
            evi_solver=evi_solver
        )

    def solve_optimistic_model(self):
        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_p = self.beta_p()  # confidence bounds on transition probabilities
        max_nb_actions = self.nb_observations.shape[1]

        # self.pyevi.compute_prerun_info(
        #     estimated_probabilities_mdp=self.estimated_probabilities,
        #     estimated_rewards_mdp=self.estimated_rewards_mdp,
        #     beta_r=beta_r,
        #     nb_observations_mdp=self.nb_observations_mdp,
        #     total_time=self.total_time,
        #     delta=self.delta,
        #     max_nb_actions=max_nb_actions,
        #     range_opt_p=self.range_mu_p)

        t0 = time.perf_counter()
        gg = self.opt_solver.compute_prerun_info(
            estimated_probabilities_mdp=self.P_mdp, #self.estimated_probabilities_mdp,
            estimated_rewards_mdp=self.estimated_rewards_mdp,
            beta_r=beta_r,
            nb_observations_mdp=self.nb_observations_mdp,
            total_time=self.total_time,
            delta=self.delta,
            max_nb_actions=max_nb_actions,
            range_opt_p=self.range_mu_p)
        t1 = time.perf_counter()
        if np.isclose(gg,-999):

            raise ValueError()

        # p, b = self.opt_solver.get_opt_p_and_beta()
        # rt = self.opt_solver.get_r_tilde_opt()
        # assert len(p) == len(self.pyevi.p_hat_opt)
        # assert len(b) == len(self.pyevi.beta_opt_p)
        # assert len(rt) == len(self.pyevi.r_tilde_opt)
        # for i in range(len(p)):
        #     assert np.allclose(p[i],self.pyevi.p_hat_opt[i])
        #     assert np.allclose(np.sum(p[i], axis=1),1)
        #     assert np.allclose(b[i],self.pyevi.beta_opt_p[i])
        #     assert np.allclose(rt[i],self.pyevi.r_tilde_opt[i])

        new_span = self.opt_solver.run(
            policy_indices=self.policy_indices,
            policy=self.policy,
            p_hat=self.P, #self.estimated_probabilities,
            r_hat_mdp=self.estimated_rewards_mdp,
            beta_p=beta_p,
            beta_r_mdp=beta_r,
            r_max=self.r_max,
            epsilon=self.r_max / m.sqrt(self.iteration + 1))
        t2 = time.perf_counter()
        self.solver_times.append((t1-t0, t2-t1))

        # py_span = self.pyevi.run(
        #     policy_indices=self.policy_indices,
        #     policy=self.policy,
        #     p_hat=self.estimated_probabilities,
        #     r_hat_mdp=self.estimated_rewards_mdp,
        #     beta_p=beta_p,
        #     beta_r_mdp=beta_r,
        #     r_max=self.r_max,
        #     epsilon=self.r_max / m.sqrt(self.iteration + 1))

        # print("{}, {}".format(new_span, py_span))
        # assert np.isclose(new_span, py_span), "{} != {}".format(new_span, py_span)

        if new_span < 0:
            self.logger.info("ERROR {}".format(new_span))

        assert new_span > -0.00001, "{}".format(new_span)

        return new_span
