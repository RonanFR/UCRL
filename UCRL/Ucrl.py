from .cython.max_proba import maxProba
from .cython.ExtendedValueIteration import extended_value_iteration
from .evi.evi import EVI
from .logging import default_logger

import math as m
import numpy as np
import time


class AbstractUCRL(object):

    def __init__(self, environment,
                 r_max, range_r=-1, range_p=-1,
                 solver=None,
                 verbose = 0,
                 logger=default_logger):
        self.environment = environment
        self.r_max = float(r_max)

        if (np.asarray(range_r) < 0).any():
            self.range_r = r_max
        else:
            self.range_r = range_r
        if (np.asarray(range_p) < 0).any():
            self.range_p = 1
        else:
            self.range_p = range_p

        if solver is None:
            # create solver for optimistic model
            self.opt_solver = EVI(self.environment.nb_states,
                                  self.environment.get_state_actions())
        else:
            self.opt_solver = solver

        # initialize matrices
        self.policy = np.zeros((self.environment.nb_states,), dtype=np.int_) #[0]*self.environment.nb_states  # initial policy
        self.policy_indices = np.zeros((self.environment.nb_states,), dtype=np.int_) #[0]*self.environment.nb_states

        # initialization
        self.total_reward = 0
        self.total_time = 0
        self.regret = [0]  # cumulative regret of the learning algorithm
        self.unit_duration = [1]  # ratios (nb of time steps)/(nb of decision steps)
        self.span_values = []
        self.span_times = []
        self.iteration = 0
        self.episode = 0
        self.delta = 1.  # confidence

        self.verbose = verbose
        self.logger = logger

    def clear_before_pickle(self):
        del self.opt_solver
        del self.logger


class UcrlMdp(AbstractUCRL):
    """
    Implementation of Upper Confidence Reinforcement Learning (UCRL) algorithm for toys state/actions MDPs with
    positive bounded rewards.
    """

    def __init__(self, environment, r_max, range_r=-1, range_p=-1, solver=None,
                 verbose = 0,
                 logger=default_logger):
        """
        :param environment: an instance of any subclass of abstract class Environment which is an MDP
        :param r_max: upper bound
        :param range_r: multiplicative factor for the concentration bound on rewards (default is r_max)
        :param range_p: multiplicative factor for the concentration bound on transition probabilities (default is 1)
        """
        super(UcrlMdp, self).__init__(environment=environment,
                                      r_max=r_max, range_r=range_r,
                                      range_p=range_p, solver=solver,
                                      verbose=verbose,
                                      logger=logger)
        self.estimated_probabilities = np.ones((self.environment.nb_states, self.environment.max_nb_actions_per_state,
                                                 self.environment.nb_states)) * 1/self.environment.nb_states
        self.estimated_rewards = np.ones((self.environment.nb_states, self.environment.max_nb_actions_per_state)) * (r_max+1)
        self.estimated_holding_times = np.ones((self.environment.nb_states, self.environment.max_nb_actions_per_state))

        self.nb_observations = np.zeros((self.environment.nb_states, self.environment.max_nb_actions_per_state))
        self.nu_k = np.zeros((self.environment.nb_states, self.environment.max_nb_actions_per_state))
        self.tau = 0.9
        self.tau_max = 1
        self.tau_min = 1

    def learn(self, duration, regret_time_step):
        """ Run UCRL on the provided environment
        
        Args:
            duration (int): the algorithm is run until the number of time steps 
                            exceeds "duration"
            regret_time_step (int): the value of the cumulative regret is stored
                                    every "regret_time_step" time steps
        
        """
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
            self.nu_k.fill(0)
            self.delta = 1 / m.sqrt(self.iteration + 1)

            if self.verbose > 0:
                self.logger.info("{}/{} = {:3.2f}%".format(self.total_time, duration, self.total_time / duration *100))
                if self.verbose > 2:
                    self.logger.info("P_hat -> {}:\n{}".format(self.estimated_probabilities.shape, self.estimated_probabilities))
                    self.logger.info("R_hat -> {}:\n{}".format(self.estimated_rewards.shape, self.estimated_rewards))
                    self.logger.info("T_hat -> {}:\n{}".format(self.estimated_holding_times.shape, self.estimated_holding_times))
                    self.logger.info("N -> {}:\n{}".format(self.nb_observations.shape, self.nb_observations))
                    self.logger.info("nu_k -> {}:\n{}".format(self.nu_k.shape, self.nu_k))

            # solve the optimistic (extended) model
            t0 = time.time()
            span_value = self.solve_optimistic_model()
            t1 = time.time()
            span_value *= self.tau / self.r_max
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
            while self.nu_k[self.environment.state][self.policy_indices[self.environment.state]] \
                    < max(1, self.nb_observations[self.environment.state][self.policy_indices[self.environment.state]]) \
                    and self.total_time < duration:
                self.update()
                if self.total_time > threshold:
                    self.regret.append(self.total_time*self.environment.max_gain - self.total_reward)
                    self.unit_duration.append(self.total_time/self.iteration)
                    threshold = self.total_time + regret_time_step
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
        """ Confidence bounds on the reward
        
        Returns:
            np.array: the vector of confidence bounds on the reward function (|S| x |A|)
            
        """
        return np.multiply(self.range_r, np.sqrt(7/2 * m.log(2 * self.environment.nb_states * self.environment.max_nb_actions_per_state *
                                                (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations)))

    def beta_tau(self):
        """ Confidence bounds on holding times
        
        Returns:
            np.array: the vecor of confidence bounds on the holding times (|S| x |A|)
        
        """
        return np.zeros((self.environment.nb_states, self.environment.max_nb_actions_per_state))

    def beta_p(self):
        """ Confidence bounds on transition probabilities
        
        Returns:
            np.array: the vector of confidence bounds on the transition matrix (|S| x |A|)
            
        """
        return self.range_p * np.sqrt(14 * self.environment.nb_states * m.log(2 * self.environment.max_nb_actions_per_state
                    * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations))

    def update(self):
        s = self.environment.state  # current state
        self.environment.execute(self.policy[s])
        s2 = self.environment.state  # new state
        r = self.environment.reward
        t = self.environment.holding_time

        curr_act = self.policy_indices[s]
        scale_f = self.nb_observations[s][curr_act] + self.nu_k[s][curr_act]

        self.estimated_rewards[s][curr_act] *= scale_f / (scale_f + 1.)
        self.estimated_rewards[s][curr_act] += 1. / (scale_f + 1.) * r
        self.estimated_holding_times[s][curr_act] *= scale_f / (scale_f + 1.)
        self.estimated_holding_times[s][curr_act] += 1. / (scale_f + 1) * t
        self.estimated_probabilities[s][curr_act] *= scale_f / (scale_f + 1.)
        self.estimated_probabilities[s][curr_act][s2] += 1. / (scale_f + 1.)
        self.nu_k[s][curr_act] += 1
        self.total_reward += r
        self.total_time += t
        self.iteration += 1


    def solve_optimistic_model(self):

        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_tau = self.beta_tau()  # confidence bounds on holding times
        beta_p = self.beta_p()  # confidence bounds on transition probabilities

        # span_value = self.extended_value_iteration(beta_r, beta_p, beta_tau, 1 / m.sqrt(self.iteration + 1))  # python implementation: slow
        # t0 = time.perf_counter()
        # span_value, u1, u2 = extended_value_iteration(
        #     self.policy_indices, self.policy,
        #     int(self.environment.nb_states),
        #     self.environment.get_state_actions(),
        #     self.estimated_probabilities,
        #     self.estimated_rewards,
        #     self.estimated_holding_times,
        #     beta_r, beta_p, beta_tau,
        #     self.tau_max, self.r_max,
        #     self.tau, self.tau_min,
        #     self.r_max / m.sqrt(self.iteration + 1)
        # )
        # t1 = time.perf_counter()
        # to = t1 - t0
        # if self.verbose > 1:
        #     self.logger.info("[%d]OLD EVI: %.3f seconds" % (self.episode, to))

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
        self.solver_times.append(tn)
        if self.verbose > 1:
            self.logger.info("[%d]NEW EVI: %.3f seconds" % (self.episode, tn))
            if self.verbose > 2:
                self.logger.info(self.policy_indices)

        span_value = span_value_new ##############

        if self.verbose > 1:
            print("{:.2f} / {:.2f}".format(span_value, span_value_new))
        assert np.abs(span_value - span_value_new) < 1e-8

        # new_u1, new_u2 = self.opt_solver.get_uvectors()

        # assert np.allclose(u1, new_u1, 1e-5)
        # assert np.allclose(u2, new_u2, 1e-5)

        return span_value


    def extended_value_iteration(self, beta_r, beta_p, beta_tau, epsilon):
        """
        Use compiled .so file for improved speed.
        :param beta_r: confidence bounds on rewards
        :param beta_p: confidence bounds on transition probabilities
        :param beta_tau: confidence bounds on holding times
        :param epsilon: desired accuracy
        """
        u1 = np.zeros(self.environment.nb_states)
        sorted_indices = np.arange(self.environment.nb_states)
        u2 = np.zeros(self.environment.nb_states)
        p = self.estimated_probabilities
        counter = 0
        while True:
            counter += 1
            for s in range(0, self.environment.nb_states):
                first_action = True
                for c, a in enumerate(self.environment.get_available_actions_state(s)):
                    vec = self.max_proba(p[s][c], sorted_indices, beta_p[s][c])  # python implementation: slow
                    # vec = maxProba(p[s][c], sorted_indices, beta_p[s][c])  # cython implementation: fast
                    vec[s] -= 1
                    r_optimal = min(self.tau_max*self.r_max,
                                    self.estimated_rewards[s][c] + beta_r[s][c])
                    v = r_optimal + np.dot(vec, u1) * self.tau
                    tau_optimal = min(self.tau_max, max(max(self.tau_min, r_optimal/self.r_max),
                                  self.estimated_holding_times[s][c] - np.sign(v) * beta_tau[s][c]))
                    if first_action or v/tau_optimal + u1[s] > u2[s] or m.isclose(v/tau_optimal + u1[s], u2[s]):  # optimal policy = argmax
                        u2[s] = v/tau_optimal + u1[s]
                        self.policy_indices[s] = c
                        self.policy[s] = a
                    first_action = False
            if max(u2-u1)-min(u2-u1) < epsilon:  # stopping condition of EVI
                print("---{}".format(counter))
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.environment.nb_states)
                sorted_indices = np.argsort(u1)

    def max_proba(self, p, sorted_indices, beta):
        """
        Use compiled .so file for improved speed.
        :param p: probability distribution with toys support
        :param sorted_indices: argsort of value function
        :param beta: confidence bound on the empirical probability
        :return: optimal probability
        """
        n = np.size(sorted_indices)
        min1 = min(1, p[sorted_indices[n-1]] + beta/2)
        if min1 == 1:
            p2 = np.zeros(self.environment.nb_states)
            p2[sorted_indices[n-1]] = 1
        else:
            sorted_p = p[sorted_indices]
            support_sorted_p = np.nonzero(sorted_p)[0]
            restricted_sorted_p = sorted_p[support_sorted_p]
            support_p = sorted_indices[support_sorted_p]
            p2 = np.zeros(self.environment.nb_states)
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


class UcrlSmdpBounded(UcrlMdp):
    """
    UCRL for SMDPs with (almost surely) bounded rewards and holding times.
    """
    def __init__(self, environment, r_max, t_max, t_min=1,
                 range_r=-1, range_tau=-1, range_p=-1,
                 verbose = 0, logger=default_logger):
        super().__init__(environment, r_max, range_r, range_p,
                         verbose=verbose, logger=logger)
        self.tau = t_min - 0.1
        self.tau_max = t_max
        self.tau_min = t_min
        self.estimated_rewards = np.ones((self.environment.nb_states, self.environment.max_nb_actions_per_state)) * r_max * t_max
        self.estimated_holding_times = np.ones((self.environment.nb_states, self.environment.max_nb_actions_per_state)) * t_max
        if (np.asarray(range_r) < 0).any():
            self.range_r = r_max * t_max
        else:
            self.range_r = range_r
        if (np.asarray(range_tau) < 0).any():
            self.range_tau = (t_max - t_min)
        else:
            self.range_tau = range_tau

    def beta_r(self):
        return np.multiply(self.range_r, np.sqrt(7/2 * m.log(2 * self.environment.nb_states * self.environment.max_nb_actions_per_state *
                                               (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations)))

    def beta_tau(self):
        return np.multiply(self.range_tau, np.sqrt(7/2 * m.log(2 * self.environment.nb_states *
                self.environment.max_nb_actions_per_state * (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations)))


class UcrlMixedBounded(UcrlSmdpBounded):
    """
    UCRL for SMDPs with (almost surely) bounded rewards and holding times where some actions are identified as primitive
    actions and dealt with as such.
    """
    def __init__(self, environment, r_max, t_max, t_min=1, range_r=-1,
                 range_r_actions=-1, range_tau=-1, range_p=-1,
                 verbose = 0, logger=default_logger):
        super().__init__(environment, r_max, t_max, t_min,
                         range_r, range_tau, range_p,
                         verbose=verbose, logger=logger)
        self.options = np.zeros((self.environment.nb_states, self.environment.max_nb_actions))
        self.primitive_actions = np.zeros((self.environment.nb_states, self.environment.max_nb_actions))
        for s, actions in enumerate(environment.get_state_actions()):
            i = 0
            for action in actions:
                if action > environment.threshold_options:
                    self.options[s][i] = 1
                else:
                    self.primitive_actions[s][i] = 1
                    self.estimated_rewards[s][i] = r_max
                    self.estimated_holding_times[s][i] = 1
                i += 1
        if (np.asarray(range_r_actions) < 0).any():
            self.range_r_actions = np.copy(self.estimated_rewards)
        else:
            self.range_r_actions = range_r_actions

    def beta_r(self):
        normalized_beta = np.sqrt(7/2 * m.log(2 * self.environment.nb_states * self.environment.max_nb_actions *
                                                (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations))
        beta_r = np.multiply(self.range_r, normalized_beta)
        beta_r_actions = np.multiply(self.range_r_actions, normalized_beta)
        return np.add(np.multiply(self.options, beta_r), np.multiply(self.primitive_actions, beta_r_actions))

    def beta_tau(self):
        beta_tau = np.multiply(self.range_tau, np.sqrt(7/2 * m.log(2 * self.environment.nb_states *
                 self.environment.max_nb_actions * (self.iteration+1)/self.delta) / np.maximum(1, self.nb_observations)))
        return np.multiply(self.options, beta_tau)

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
            scale_f = self.nb_observations[s][self.policy_indices[s]] + self.nu_k[s][self.policy_indices[s]]

            self.estimated_rewards[s][self.policy_indices[s]] *= scale_f / (scale_f + 1)
            self.estimated_rewards[s][self.policy_indices[s]] += 1 / (scale_f + 1) * r
            self.estimated_holding_times[s][self.policy_indices[s]] *= scale_f / (scale_f + 1)
            self.estimated_holding_times[s][self.policy_indices[s]] += 1 / (scale_f + 1) * t
            self.estimated_probabilities[s][self.policy_indices[s]] *= scale_f / (scale_f + 1)
            self.estimated_probabilities[s][self.policy_indices[s]][s2] += 1 / (scale_f + 1)
            self.nu_k[s][self.policy_indices[s]] += 1
        else:
            self.update_history(history)

    def update_history(self, history):
        [s, o, r, t, s2] = history.data
        try:
            option_index = self.environment.get_available_actions_state(s).index(o)
            self.estimated_rewards[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
            self.estimated_rewards[s][option_index] += 1 / (self.nb_observations[s][option_index] + 1) * r
            self.estimated_holding_times[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
            self.estimated_holding_times[s][option_index] += 1 / (self.nb_observations[s][option_index] + 1) * t
            self.estimated_probabilities[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
            self.estimated_probabilities[s][option_index][s2] += 1 / (self.nb_observations[s][option_index] + 1)
            self.nu_k[s][option_index] += 1
        except Exception:
            pass
        for sub_history in history.children:
            self.update_history(sub_history)




class UcrlSmdpExp(UcrlMdp):
    """
    UCRL for SMDPs with sub-Exponential rewards and holding times.
    """
    def __init__(self, environment, r_max, tau_max, tau_min=1,
                 verbose = 0, logger=default_logger):
        super().__init__(environment, r_max,
                         verbose=verbose, logger=logger)
        self.tau = tau_min - 0.1
        self.tau_max = tau_max
        self.tau_min = tau_min

    # TODO: implement SMDP-UCRL for sub-Exponential random variables

    def beta_r(self):
        return self.r_max * np.sqrt(7/2 * m.log(2 * self.environment.nb_states * self.environment.max_nb_actions *
                                                self.iteration/self.delta) / np.maximum(1, self.nb_observations))

    def beta_tau(self):
        return 0