from .Ucrl import AbstractUCRL
from .envs import MixedEnvironment
from .cython import maxProba, extended_value_iteration
from .logging import default_logger
import numpy as np
import math as m
import time


class SMDPUCRL_Mixed(AbstractUCRL):
    def __init__(self, environment, r_max, t_max, t_min=1, range_r=-1,
                 range_r_actions=-1, range_tau=-1, range_p=-1,
                 verbose = 0, logger=default_logger):
        assert isinstance(environment, MixedEnvironment)

        super(SMDPUCRL_Mixed, self).__init__(environment=environment,
                                             r_max=r_max, range_r=range_r,
                                             range_p=range_p, solver=None,
                                             verbose=verbose,
                                             logger=logger)
        self.tau = t_min - 0.1
        self.tau_max = t_max
        self.tau_min = t_min

        nb_states = self.environment.nb_states
        max_nb_actions = environment.max_nb_actions_per_state

        # keep track of the transition probabilities for options and actions
        self.estimated_probabilities = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states

        # keep track of visits
        self.nb_observations = np.zeros((nb_states, max_nb_actions))
        self.nu_k = np.zeros((nb_states, max_nb_actions))

        # reward info
        self.estimated_rewards = np.ones((nb_states, max_nb_actions)) * r_max * t_max
        # duration
        self.estimated_holding_times = np.ones((nb_states, max_nb_actions)) * t_max

        if (np.asarray(range_r) < 0).any():
            self.range_r = r_max * t_max
        else:
            self.range_r = range_r
        if (np.asarray(range_tau) < 0).any():
            self.range_tau = (t_max - t_min)
        else:
            self.range_tau = range_tau
        if (np.asarray(range_r_actions) < 0).any():
            self.range_r_actions = np.copy(self.estimated_rewards)
        else:
            self.range_r_actions = range_r_actions

        # compute mask
        self.options_mask = np.zeros((nb_states, max_nb_actions))
        self.primitive_actions_mask = np.zeros((nb_states, max_nb_actions))
        for s, actions in enumerate(environment.get_state_actions()):
            for idx, action in enumerate(actions):
                if self.environment.is_primitive_action(action):
                    self.primitive_actions_mask[s, idx] = 1
                    self.estimated_rewards[s, idx] = r_max
                    self.estimated_holding_times[s, idx] = 1
                else:
                    self.options_mask[s, idx] = 1

    def learn(self, duration, regret_time_step):
        if self.total_time >= duration:
            return
        threshold = self.total_time + regret_time_step
        threshold_span = threshold

        self.timing = []

        while self.total_time < duration:
            self.episode += 1
            # print(self.total_time)

            # initialize the episode
            self.nu_k.fill(0)
            self.delta = 1 / m.sqrt(self.iteration + 1)

            if self.verbose > 0:
                self.logger.info("#Episode {}".format(self.episode))
                if self.verbose > 1:
                    self.logger.info("P_hat -> {}:\n{}".format(self.estimated_probabilities.shape, self.estimated_probabilities))
                    self.logger.info("R_hat -> {}:\n{}".format(self.estimated_rewards.shape, self.estimated_rewards))
                    self.logger.info("T_hat -> {}:\n{}".format(self.estimated_holding_times.shape, self.estimated_holding_times))
                    self.logger.info("N -> {}:\n{}".format(self.nb_observations.shape, self.nb_observations))
                    self.logger.info("nu_k -> {}:\n{}".format(self.nu_k.shape, self.nu_k))

            # solve the optimistic (extended) model
            span_value = self.solve_optimistic_model()
            if self.verbose > 0:
                self.logger.info("{:.9f}".format(span_value))

            if self.total_time > threshold_span:
                self.span_values.append(span_value / self.r_max)
                self.span_times.append(self.total_time)
                threshold_span = self.total_time + regret_time_step

            # execute the recovered policy
            curr_st = self.environment.state
            curr_ac = self.policy_indices[curr_st]
            #t0 = time.time()
            while self.nu_k[curr_st][curr_ac] < max(1, self.nb_observations[curr_st][curr_ac]) \
                    and self.total_time < duration:
                self.update()
                if self.total_time > threshold:
                    self.regret.append(self.total_time * self.environment.max_gain - self.total_reward)
                    self.unit_duration.append(self.total_time / self.iteration)
                    threshold = self.total_time + regret_time_step
                # get current state and action
                curr_st = self.environment.state
                curr_ac = self.policy_indices[curr_st]
            #t1 = time.time()
            #self.logger.info(t1 - t0)

            self.nb_observations += self.nu_k

    def beta_p(self):
        nb_states, nb_actions = self.estimated_rewards.shape
        return self.range_p * np.sqrt(14 * nb_states * m.log(2 * nb_actions
                    * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations))

    def beta_r(self):
        nb_states, nb_actions = self.estimated_rewards.shape
        normalized_beta = np.sqrt(7/2 * m.log(2 * nb_states * nb_actions *
                                                (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations))
        beta_r = np.multiply(self.range_r, normalized_beta)
        beta_r_actions = np.multiply(self.range_r_actions, normalized_beta)
        return np.add(np.multiply(self.options_mask, beta_r), np.multiply(self.primitive_actions_mask, beta_r_actions))

    def beta_tau(self):
        beta_tau = np.multiply(self.range_tau, np.sqrt(7/2 * m.log(2 * self.environment.nb_states *
                 self.environment.max_nb_actions_per_state * (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations)))
        return np.multiply(self.options_mask, beta_tau)

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
            self.update_from_action(s, self.policy[s], s2, r, a_index=self.policy_indices[s])
        else:
            self.update_from_option(history)

    def update_from_option(self, history):
        [s, o, r, t, s2] = history.data # first is for sure an option
        option_index = self.environment.get_available_actions_state(s).index(o)


        self.estimated_rewards[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
        self.estimated_rewards[s][option_index] += 1 / (self.nb_observations[s][option_index] + 1) * r
        self.estimated_holding_times[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
        self.estimated_holding_times[s][option_index] += 1 / (self.nb_observations[s][option_index] + 1) * t
        self.estimated_probabilities[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
        self.estimated_probabilities[s][option_index][s2] += 1 / (self.nb_observations[s][option_index] + 1)
        self.nu_k[s][option_index] += 1

        # the rest of the actions are for sure primitive actions (1 level hierarchy)
        for node in history.children:
            assert len(node.children) == 0
            [s, a, r, t, s2] = node.data # this is a primitive action which can be valid or not
            # for sure it is a primitive action (and the index corresponds to all the primitive actions
            self.update_from_action(s, a, s2, r)

    def update_from_action(self, s, a, s2, r, a_index=None):
        a_index = self.environment.get_index_of_action_state(s, a) if a_index is None else a_index
        if a_index:
            self.estimated_rewards[s][a_index] *= self.nb_observations[s][a_index] / (self.nb_observations[s][a_index] + 1)
            self.estimated_rewards[s][a_index] += 1 / (self.nb_observations[s][a_index] + 1) * r
            self.estimated_probabilities[s][a_index] *= self.nb_observations[s][a_index] / (self.nb_observations[s][a_index] + 1)
            self.estimated_probabilities[s][a_index][s2] += 1 / (self.nb_observations[s][a_index] + 1)
            self.nu_k[s][a_index] += 1

    def solve_optimistic_model(self):
        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_tau = self.beta_tau()  # confidence bounds on holding times
        beta_p = self.beta_p()  # confidence bounds on transition probabilities

        # span_value = self.extended_value_iteration(beta_r, beta_p, beta_tau, 1 / m.sqrt(self.iteration + 1))  # python implementation: slow
        t0 = time.perf_counter()
        span_value, u1, u2 = extended_value_iteration(
            self.policy_indices, self.policy,
            int(self.environment.nb_states),
            self.environment.get_state_actions(),
            self.estimated_probabilities,
            self.estimated_rewards,
            self.estimated_holding_times,
            beta_r, beta_p, beta_tau,
            self.tau_max, self.r_max,
            self.tau, self.tau_min,
            self.r_max / m.sqrt(self.iteration + 1)
        )
        t1 = time.perf_counter()
        to = t1 - t0
        if self.verbose > 1:
            self.logger.info("[%d]OLD EVI: %.3f seconds" % (self.episode, to))

        t0 = time.perf_counter()
        span_value_new = self.opt_solver.evi(
            self.policy_indices, self.policy,
            self.estimated_probabilities,
            self.estimated_rewards,
            self.estimated_holding_times,
            beta_r, beta_p, beta_tau, self.tau_max,
            self.r_max, self.tau, self.tau_min,
            self.r_max / m.sqrt(self.iteration + 1)
        )
        t1 = time.perf_counter()
        tn = t1 - t0

        if self.verbose > 1:
            self.logger.info("[%d]NEW EVI: %.3f seconds" % (self.episode, tn))

        self.timing.append([to, tn])

        if self.verbose > 1:
            self.logger.info("span: {:.2f} / {:.2f}".format(span_value, span_value_new))
        assert np.abs(span_value - span_value_new) < 1e-8

        new_u1, new_u2 = self.opt_solver.get_uvectors()

        assert np.allclose(u1, new_u1, 1e-5)
        assert np.allclose(u2, new_u2, 1e-5)

        return span_value


