import numpy as np
from .rllogging import default_logger
import time
import math as m
from .evi._max_proba import 
from . import __version__ as ucrl_version
from .evi import EVI
from . import bounds as bounds


class OLP(object):
    """
    Optimistic Linear Programming

    Ref.
    ----
    Tewari, Ambuj, and Peter L. Bartlett.
    "Optimistic linear programming gives logarithmic regret for irreducible MDPs."
    Advances in Neural Information Processing Systems. 2008.
    """

    def __init__(self, environment, r_max, alpha_r=None, alpha_p=None,
                 verbose=0,
                 logger=default_logger, random_state=None):
        self.environment = environment
        self.r_max = float(r_max)

        if alpha_r is None:
            self.alpha_r = 1
        else:
            self.alpha_r = alpha_r
        if alpha_p is None:
            self.alpha_p = 1
        else:
            self.alpha_p = alpha_p

        # initialize matrices
        self.policy = np.zeros((self.environment.nb_states,), dtype=np.int_)
        self.policy_indices = np.zeros((self.environment.nb_states,), dtype=np.int_)

        # initialization
        self.total_reward = 0
        self.total_time = 0
        self.regret = [0]  # cumulative regret of the learning algorithm
        self.regret_unit_time = [0]
        self.unit_duration = [1]  # ratios (nb of time steps)/(nb of decision steps)
        self.span_values = []
        self.span_times = []
        self.iteration = 0
        self.episode = 0
        self.delta = 1.  # confidence

        self.verbose = verbose
        self.logger = logger
        self.version = ucrl_version
        self.random_state = random_state
        self.local_random = np.random.RandomState(seed=random_state)

        nb_states = self.environment.nb_states
        max_nb_actions = self.environment.max_nb_actions_per_state
        # self.P_counter = np.zeros((nb_states, max_nb_actions, nb_states), dtype=np.int64)
        self.P = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states

        self.estimated_rewards = np.ones((nb_states, max_nb_actions)) * r_max
        self.variance_proxy_reward = np.zeros((nb_states, max_nb_actions))

        self.nb_observations = np.zeros((nb_states, max_nb_actions), dtype=np.int64)
        self.tau = 0.9
        self.tau_max = 1
        self.tau_min = 1

    def learn(self, duration, regret_time_step):
        if self.total_time >= duration:
            return
        threshold = self.total_time + regret_time_step

        self.solver_times = []
        self.simulation_times = []

        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state

        # get initial state
        curr_state = self.environment.state

        Ns = self.nb_observations.sum(axis=1)
        state_actions_k = [None] * S
        u_k = np.zeros((A,))
        epsilon = 1e-8

        P_k = np.zeros((S,A,S))
        R_k = np.zeros((S,A))

        t_star_all = time.perf_counter()
        while self.total_time < duration:
            # compute mean probability

            A_original_index = []
            max_reduced_actions = 0
            for s in range(S):
                A_s = []
                a_counter = 0
                for a_idx, a in enumerate(self.environment.state_actions[s]):
                    if Ns[s] == 0 or self.nb_observations[s, a_idx] > m.log(Ns[s]) ** 2:
                        P_k[s, a_counter] = self.P[s, a_idx]
                        A_s.append(a)
                        R_k[s, a_counter] = self.estimated_rewards[s, a_idx]
                        if s == curr_state:
                            A_original_index.append(a_idx)
                        a_counter += 1

                max_reduced_actions = max(max_reduced_actions, a_counter)
                state_actions_k[s] = A_s

            # delete actions in excess
            # P_k = P_k[:,0:max_reduced_actions,:]
            # R_k = R_k[:,0:max_reduced_actions]

            t0 = time.time()
            span_k, gain_k, bias_k = self.solve_mean_model(P_k, R_k, state_actions_k, epsilon)
            t1 = time.time()

            if self.verbose > 0:
                self.logger.info("span({}): {:.9f}".format(self.episode, span_k))
                curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                self.logger.info("regret: {}, {:.2f}".format(self.total_time, curr_regret))
                self.logger.info("evi time: {:.4f} s".format(t1 - t0))

            if self.total_time > threshold:
                self.span_values.append(span_k)
                self.span_times.append(self.total_time)
                curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                self.regret.append(curr_regret)
                self.regret_unit_time.append(self.total_time)
                self.unit_duration.append(self.total_time / self.iteration)
                threshold = self.total_time + regret_time_step

            # Optimal actions (for the current problem) that are about to become "undersampled"
            optimal_set_k = self.compute_optimal_actions(P_k[curr_state], R_k[curr_state],
                                                         A_original_index, bias_k, 2 * epsilon)
            # note that the actions have already the original index when inserted in the optimal_set_k

            gamma_1 = [a_idx for a_idx in optimal_set_k if
                       self.nb_observations[curr_state, a_idx] < m.log(Ns[curr_state] + 1)]

            if len(optimal_set_k) == len(gamma_1):
                curr_act_idx = np.random.choice(gamma_1, 1)
            else:
                # Compute utility for ALL actions in the current state using optimistic estimate

                beta_p = self.beta_p(curr_state)
                beta_r = self.beta_r(curr_state)

                r_optimal = min(self.r_max, np.max(self.estimated_rewards[curr_state] + beta_r))

                n_actions = len(self.environment.state_actions[curr_state])
                for a_idx in range(n_actions):
                    if self.bound_type_p == 'bernstein':
                        opt_v = py_max_proba_bernstein(p=self.P[curr_state, a_idx], v=bias_k,
                                                       beta=beta_p[a_idx], reverse=False)
                    else:
                        opt_v = py_max_proba_chernoff(p=self.P[curr_state, a_idx], v=bias_k,
                                                      beta=beta_p[a_idx], reverse=False)
                    u_k[a_idx] = opt_v.dot(bias_k) + r_optimal

                curr_act_idx = np.argmax(u_k[0:n_actions])

            curr_act = self.environment.state_actions[curr_state][curr_act_idx]
            self.update(curr_state=curr_state, curr_act_idx=curr_act_idx, curr_act=curr_act)
            # remember to update Ns
            Ns[curr_state] += 1
            # get new state
            curr_state = self.environment.state

        t_end_all = time.perf_counter()
        self.speed = t_end_all - t_star_all
        self.logger.info("TIME: %.5f s" % self.speed)

    def compute_optimal_actions(self, P, R, actions, h, epsilon=1e-12):
        v = R + P.dot(h)
        mm = v[0]
        L = []
        for el, a in zip(v,actions):
            if np.isclose(el, mm, atol=epsilon):
                L.append(a)
            elif el > mm:
                mm = el
                L = [a]
        return L

    def solve_mean_model(self, P, R, state_actions, epsilon):
        ns, na = R.shape

        t0 = time.perf_counter()
        solver = EVI(nb_states=ns,
                     actions_per_state=state_actions,
                     bound_type=self.bound_type_p,
                     random_state=np.random.randint(1, 1000000, 1),
                     gamma=1.
                     )
        Z = np.zeros((ns, na))
        span_value = solver.run(
            self.policy_indices, self.policy,
            P,
            R,
            Z,
            Z, np.zeros((ns, na, ns)), Z,
            self.tau_max,
            self.r_max,
            self.tau,
            self.tau_min,
            epsilon
        )
        t1 = time.perf_counter()
        tn = t1 - t0
        self.solver_times.append(tn)

        span_value *= self.tau / self.r_max

        u1, u2 = solver.get_uvectors()
        gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
        return span_value, gain, u2

    def beta_r(self, current_state):
        """ Confidence bounds on the reward

        Returns:
            np.array: the vector of confidence bounds on the reward function (|S| x |A|)

        """
        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        if self.bound_type_rew != "bernstein":
            ci = bounds.chernoff(it=self.iteration, N=self.nb_observations[current_state],
                                 range=self.r_max, delta=self.delta,
                                 sqrt_C=3.5, log_C=2 * S * A)
            return self.alpha_r * ci
        else:
            N = np.maximum(1, self.nb_observations[current_state])
            Nm1 = np.maximum(1, self.nb_observations[current_state] - 1)
            var_r = self.variance_proxy_reward[current_state] / Nm1
            log_value = 2.0 * S * A * (self.iteration + 1) / self.delta
            beta = bounds.bernstein2(scale_a=14 * var_r / N,
                                     log_scale_a=log_value,
                                     scale_b=49.0 * self.r_max / (3.0 * Nm1),
                                     log_scale_b=log_value,
                                     alpha_1=m.sqrt(self.alpha_r), alpha_2=self.alpha_r)
            return beta

    def beta_p(self, current_state):
        """ Confidence bounds on transition probabilities

        Returns:
            np.array: the vector of confidence bounds on the transition matrix (|S| x |A|)

        """
        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        if self.bound_type_p != "bernstein":
            beta = bounds.chernoff(it=self.iteration, N=self.nb_observations[current_state],
                                   range=1., delta=self.delta,
                                   sqrt_C=14 * S, log_C=2 * A)
            return self.alpha_p * beta.reshape([S, A, 1])
        else:
            N = np.maximum(1, self.nb_observations[current_state])
            Nm1 = np.maximum(1, self.nb_observations[current_state] - 1)
            var_p = self.P[current_state] * (1. - self.P[current_state])
            log_value = 2.0 * S * A * (self.iteration + 1) / self.delta
            beta = bounds.bernstein2(scale_a=14 * var_p / N[:, np.newaxis],
                                     log_scale_a=log_value,
                                     scale_b=49.0 / (3.0 * Nm1[:, np.newaxis]),
                                     log_scale_b=log_value,
                                     alpha_1=m.sqrt(self.alpha_p), alpha_2=self.alpha_p)
            return beta

    def update(self, curr_state, curr_act_idx, curr_act):
        s = self.environment.state  # current state
        assert np.allclose(curr_state, s)

        # execute the action
        self.environment.execute(curr_act)
        s2 = self.environment.state  # new state
        r = self.environment.reward
        t = self.environment.holding_time

        # updated observations
        scale_f = self.nb_observations[s][curr_act_idx]

        # update reward and variance estimate
        old_estimated_reward = self.estimated_rewards[s, curr_act_idx]
        self.estimated_rewards[s, curr_act_idx] *= scale_f / (scale_f + 1.)
        self.estimated_rewards[s, curr_act_idx] += r / (scale_f + 1.)
        self.variance_proxy_reward[s, curr_act_idx] += (r - old_estimated_reward) * (
                    r - self.estimated_rewards[s, curr_act_idx])

        # self.P_counter[s, curr_act_idx, s2] += 1
        self.P[s, curr_act_idx] *= scale_f / (scale_f + 1.)
        self.P[s, curr_act_idx, s2] += 1 / (scale_f + 1.)
        # self.P[s, curr_act_idx] = (1 + self.P_counter[s, curr_act_idx]) / (len(self.environment.state_actions[curr_state]) + self.nb_observations[s, curr_act_idx])
        # self.P[s, curr_act_idx] = self.P_counter[s, curr_act_idx] / self.nb_observations[s, curr_act_idx]

        self.nb_observations[s][curr_act_idx] += 1
        self.total_reward += r
        self.total_time += t
        self.iteration += 1

    def clear_before_pickle(self):
        del self.logger

    def description(self):
        desc = {
            "alpha_p": self.alpha_p,
            "alpha_r": self.alpha_r,
            "r_max": self.r_max,
            "version": self.version
        }
        return desc
