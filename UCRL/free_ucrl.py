from .Ucrl import AbstractUCRL
from .envs import MixedEnvironment
import numpy as np
import math as m
from .logging import default_logger
from .evi import EVI_FSUCRLv1, EVI_FSUCRLv2
from .evi.pyfreeevi import PyEVI_FSUCRLv1, PyEVI_FSUCRLv2
from . import bounds as bounds
import time


class FSUCRLv1(AbstractUCRL):
    def __init__(self, environment, r_max,
                 alpha_r=None, alpha_p=None, alpha_mc=None,
                 bound_type="hoeffding",
                 verbose = 0, logger=default_logger,
                 evi_solver=None):

        assert isinstance(environment, MixedEnvironment)
        run_py = False
        self.check_with_py = False
        run_py = False if self.check_with_py else run_py

        if evi_solver is None:
            if run_py or self.check_with_py:
                evi_solver = PyEVI_FSUCRLv1(
                    nb_states=environment.nb_states,
                    nb_options=environment.nb_options,
                    threshold=environment.threshold_options,
                    macro_actions_per_state=environment.get_state_actions(),
                    reachable_states_per_option=environment.reachable_states_per_option,
                    option_policies=environment.options_policies,
                    options_terminating_conditions=environment.options_terminating_conditions,
                    mdp_actions_per_state=environment.environment.get_state_actions(),
                    use_bernstein=1 if bound_type == "bernstein" else 0)
                if self.check_with_py:
                    self.pyevi = evi_solver

            if not run_py:
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
                                       r_max=r_max, alpha_r=alpha_r,
                                       alpha_p=alpha_p, solver=evi_solver,
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
        self.estimated_rewards_mdp = np.ones((nb_states, max_nb_mdp_actions)) * (r_max + 99)

        if alpha_mc is None:
            self.alpha_mc = 1.
        else:
            self.alpha_mc = alpha_mc

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

            span_value /= self.r_max
            if self.verbose > 0:
                self.logger.info("span({}): {:.9f}".format(self.episode, span_value))
                curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                self.logger.info("regret: {}, {:.2f}".format(self.total_time, curr_regret))
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
        S, A = self.estimated_rewards_mdp.shape
        ci = bounds.chernoff(it=self.total_time, N=self.nb_observations_mdp,
                             range=self.r_max, delta=self.delta,
                             sqrt_C=3.5, log_C=2*S*A)
        beta = self.r_max * np.sqrt(7 / 2 * m.log(2 * S * A * (self.total_time + 1) / self.delta) / np.maximum(1, self.nb_observations_mdp))
        assert np.allclose(ci, beta)
        beta_r =  np.multiply(self.alpha_r, ci)
        return beta_r

    def beta_p(self):
        S, A = self.nb_observations.shape
        if self.bound_type == "hoeffding":
            beta = bounds.chernoff(it=self.iteration, N=self.nb_observations,
                                   range=1., delta=self.delta,
                                   sqrt_C=14*S, log_C=2*A)
            ci = np.sqrt(14 * S * m.log(2 * A
                        * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations))
            assert np.allclose(ci, beta)
            return self.alpha_p * beta.reshape([S, A, 1])
        else:
            Z = m.log(6 * A * (self.iteration + 1) / self.delta)
            n = np.maximum(1, self.nb_observations)
            # A = np.sqrt(2 * self.estimated_probabilities * (1-self.estimated_probabilities) * Z / n[:,:,np.newaxis])
            Va = np.sqrt(2 * self.P * (1-self.P) * Z / n[:,:,np.newaxis])
            Vb = Z * 7 / (3 * n)
            return self.alpha_p * (Va + Vb[:, :, np.newaxis])

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
        check_v = self.opt_solver.compute_mu_info2(  # environment=self.environment,
            estimated_probabilities_mdp=self.P_mdp, #self.estimated_probabilities_mdp,
            estimated_rewards_mdp=self.estimated_rewards_mdp,
            beta_r=beta_r,
            nb_observations_mdp=self.nb_observations_mdp,
            delta=self.delta,
            max_nb_actions=max_nb_actions,
            total_time=self.total_time,
            range_mu_p=self.alpha_mc,
        r_max=self.r_max)
        t1 = time.perf_counter()

        if check_v != 0:
            raise ValueError("[FSUCRLv1] Error in the computation of mu and ci")

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
        self.logger.info("{}".format((t1-t0, t2-t1)))
        self.solver_times.append((t1-t0, t2-t1))

        # # CHECK WITH PYTHON IMPL
        if self.check_with_py:
            self.pyevi.compute_mu_info(  # environment=self.environment,
                estimated_probabilities_mdp=self.P_mdp, #self.estimated_probabilities_mdp,
                estimated_rewards_mdp=self.estimated_rewards_mdp,
                beta_r=beta_r,
                nb_observations_mdp=self.nb_observations_mdp,
                delta=self.delta,
                max_nb_actions=max_nb_actions,
                total_time=self.total_time,
                range_mu_p=self.alpha_mc,
            r_max=self.r_max)

            py_span = self.pyevi.run(
                policy_indices=self.policy_indices,
                policy=self.policy,
                p_hat=self.P,#self.estimated_probabilities,
                r_hat_mdp=self.estimated_rewards_mdp,
                beta_p=beta_p,
                beta_r_mdp=beta_r,
                r_max=self.r_max,
                epsilon=self.r_max / m.sqrt(self.iteration + 1))

            r = self.opt_solver.get_r_tilde_opt()
            assert len(r) == len(self.pyevi.r_tilde_opt)
            for x, y in zip(r, self.pyevi.r_tilde_opt):
                assert np.allclose(x, y)

            mu_e = self.opt_solver.get_mu()
            assert len(self.pyevi.mu_opt) == len(mu_e)
            for x, y in zip(mu_e, self.pyevi.mu_opt):
                assert np.allclose(x, y)
                assert np.isclose(np.sum(x),1)


            cn_e = self.opt_solver.get_conditioning_numbers()
            assert np.allclose(self.pyevi.condition_numbers_opt, cn_e),\
                "{} != {}".format(cn_e, self.pyevi.condition_numbers_opt)

            b_mu_e = self.opt_solver.get_beta_mu_p()
            assert np.allclose(b_mu_e, self.pyevi.beta_mu_p),\
                "{} != {}".format(b_mu_e, self.pyevi.beta_mu_p)

            print(py_span, new_span)
            assert np.isclose(py_span, new_span)

        return new_span


class FSUCRLv2(FSUCRLv1):
    def __init__(self, environment, r_max,
                 alpha_r=-1, alpha_p=-1, alpha_mc=-1,
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
            alpha_r=alpha_r, alpha_p=alpha_p, alpha_mc=alpha_mc,
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
            range_opt_p=self.alpha_mc,
        r_max=self.r_max)
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
        self.logger.info("{}".format((t1-t0, t2-t1)))

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
