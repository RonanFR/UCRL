import numpy as np
# import time
import math as m
from .evi import EVI  # , SpanConstrainedEVI
from .logging import default_logger
from .Ucrl import UcrlMdp, EVIException


def sample_normalgamma(m0, l0, a0, b0, mu_hat, s_hat, n, local_random):
    # http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf (Eq. 85-89)
    # https://en.wikipedia.org/wiki/Normal-gamma_distribution#Posterior_distribution_of_the_parameters

    # draw precision from a gamma distribution
    # T|a,b ~ Gamma(a,b)
    a = a0 + n / 2.
    b = b0 + (n * s_hat + (l0 * n * (mu_hat - m0) * (mu_hat - m0)) / (l0 + n)) / 2.
    T = local_random.gamma(shape=a, scale=1. / b)

    # draw mean from a normal distribution conditioned on the sampled precision
    # X|T ~ N(mu, 1 / (lam T))
    mu = (l0 * m0 + n * mu_hat) / (l0 + n)
    lam = l0 + n
    X = local_random.normal(loc=mu, scale=1. / m.sqrt(lam * T))

    return X, T


class TSDE(UcrlMdp):

    # Conjugate prior Table
    # https://en.wikipedia.org/wiki/Conjugate_prior

    def __init__(self, environment, r_max, verbose=0,
                 logger=default_logger, random_state=None, known_reward=False,
                 posterior="Bernoulli", prior_parameters=None):

        assert posterior in ["Bernoulli", "Normal", None]

        super(TSDE, self).__init__(environment=environment,
                                   r_max=r_max,
                                   verbose=verbose,
                                   logger=logger,
                                   random_state=random_state,
                                   known_reward=known_reward)

        self.opt_solver = EVI(nb_states=self.environment.nb_states,
                              actions_per_state=self.environment.get_state_actions(),
                              bound_type="bernstein",
                              random_state=random_state,
                              gamma=1.
                              )

        self.R = np.ones_like(self.estimated_rewards) * r_max

        self.posterior = posterior

        if self.known_reward:
            nb_states = self.environment.nb_states
            max_nb_actions = self.environment.max_nb_actions_per_state

            self.posterior = None

            self.R = np.zeros((nb_states, max_nb_actions))
            for s in range(nb_states):
                for a_idx, a in enumerate(self.environment.state_actions[s]):
                    self.R[s, a_idx] = self.environment.true_reward(s, a_idx)
        else:
            if posterior == "Normal":
                if prior_parameters is None:
                    # chosen such that E[X] = r_max and E[T] = 1/r_max
                    # https://en.wikipedia.org/wiki/Normal-gamma_distribution
                    self.m0 = self.r_max
                    self.l0 = 1.
                    self.a0 = 1. / self.r_max
                    self.b0 = 1.
                else:
                    self.m0, self.l0, self.a0, self.b0 = prior_parameters
            elif posterior == "Bernoulli":
                if prior_parameters is None:
                    self.a = 1.
                    self.b = 1.
                else:
                    self.a, self.b = prior_parameters

        self.episode_length = 0
        self.tk = 0

    def _stopping_rule(self, curr_state, curr_act_idx, next_state):
        A = self.nu_k[curr_state][curr_act_idx] < max(1, self.nb_observations[curr_state][curr_act_idx])
        B = self.total_time <= self.tk + self.episode_length
        return (A or B)

    def update_at_episode_end(self):

        self.nb_observations += self.nu_k
        self.episode_length = self.total_time - self.tk
        self.tk = self.total_time

    def solve_optimistic_model(self, curr_state=None):
        ns, na = self.estimated_rewards.shape
        for s in range(ns):
            for a, _ in enumerate(self.environment.state_actions[s]):
                # sample transition matrix
                self.P[s, a] = self.local_random.dirichlet(1 + self.P_counter[s, a], 1)

                # sample reward
                N = self.nb_observations[s, a]
                if self.posterior == "Normal":
                    var_r = self.variance_proxy_reward[s, a] / max(1, N)
                    if N == 0:
                        self.R[s, a] = self.r_max
                    else:
                        mu, prec = sample_normalgamma(m0=self.m0, l0=self.l0,
                                                      a0=self.a0, b0=self.b0,
                                                      mu_hat=self.estimated_rewards[s, a],
                                                      s_hat=var_r,
                                                      n=N, local_random=self.local_random)
                        self.R[s, a] = mu
                elif self.posterior == "Bernoulli":
                    v = N * self.estimated_rewards[s, a]
                    a0 = self.a + v
                    b0 = self.b + N - v
                    p = np.asscalar(self.local_random.beta(a=a0, b=b0, size=1))
                    self.R[s, a] = p

        Z = np.zeros((ns, na))

        span_value = self.opt_solver.run(
            self.policy_indices, self.policy,
            self.P,
            self.R,
            np.ones((ns, na)),
            Z, np.zeros((ns, na, ns)), Z,
            self.tau_max,
            self.r_max,
            self.tau,
            self.tau_min,
            1e-8
        )

        if span_value < 0:
            raise EVIException(error_value=span_value)

        return span_value

# class OptimisticPS(PS):
#
#     def __init__(self, environment, r_max, verbose=0,
#                  logger=default_logger, random_state=None,
#                  posterior=None, prior_parameters=None):
#         super(OptimisticPS, self).__init__(environment=environment, r_max=r_max, verbose=verbose,
#                                            logger=logger, random_state=random_state,
#                                            posterior=posterior, prior_parameters=prior_parameters)
#         self.opt_solver = EVI(nb_states=self.environment.nb_states,
#                               actions_per_state=self.environment.get_state_actions(),
#                               bound_type="bernstein",
#                               random_state=random_state,
#                               gamma=1.
#                               )
#         ns, na = self.estimated_rewards.shape
#         self.rho = 0.1
#         self.num_proba_samples = m.ceil(ns * m.log(ns * na / self.rho))
#         self.num_reward_samples = m.ceil(self.r_max * m.log(ns * na / self.rho))
#
#     def learn(self, duration, regret_time_step, render=False):
#         ns, na = self.estimated_rewards.shape
#         self.kappa = m.log(duration / self.rho)
#         self.omega = m.log(duration / self.rho)
#         self.eta = m.sqrt(duration * ns / na) + 12 * self.omega * ns * ns
#         super(OptimisticPS, self).learn(duration=duration, regret_time_step=regret_time_step, render=render)
#
#     def simple_optimistic_sampling(self, s, a_idx, ns):
#         mu_k_sa = self.nb_observations[s, a_idx]
#         if mu_k_sa != 0:
#             p_hat = self.P_counter[s, a_idx] / mu_k_sa
#             Delta = [min(m.sqrt(3 * p_hat[i] * m.log(4 * ns) / mu_k_sa) + 3 * m.log(4 * ns) / mu_k_sa, p_hat[i]) for i
#                      in range(ns)]
#             rest = np.sum(Delta)
#             p_hat -= Delta
#             p_hat += .5 * rest * np.ones(ns)
#             beta_p = .5 * rest * np.ones(ns)
#         else:
#             p_hat = .5 * np.ones(ns)
#             beta_p = .5 * np.ones(ns)
#         return p_hat, beta_p
#
#     def solve_optimistic_model(self, curr_state=None):
#
#         # Sample transition probability vectors
#         # for each s,a, generate psi indipendent probability vectors
#         ns, na = self.estimated_rewards.shape
#         beta_p = np.zeros((ns, na, ns))
#         beta_r = np.zeros((ns, na))
#         for s in range(ns):
#             for a_idx, a in enumerate(self.environment.state_actions[s]):
#                 mu_k_sa = self.nb_observations[s, a_idx]
#
#                 if mu_k_sa >= self.eta:
#                     N = self.P_counter[s, a_idx]
#                     M = (N + self.omega) / self.kappa
#                     Q = np.random.dirichlet(alpha=M, size=self.num_proba_samples)
#                     maxi = np.max(Q, axis=0)
#                     mini = np.min(Q, axis=0)
#                     self.P[s, a_idx] = .5 * (maxi + mini)
#                     beta_p[s, a_idx] = .5 * (maxi - mini)
#                 else:
#                     opt_p, opt_beta = self.simple_optimistic_sampling(s=s, a_idx=a_idx, ns=ns)
#                     self.P[s, a_idx] = opt_p
#                     beta_p[s, a_idx] = opt_beta
#
#                 if self.posterior == "Normal":
#                     var_r = self.variance_proxy_reward[s, a] / max(1, N)
#                     if N == 0:
#                         self.R[s, a] = self.r_max
#                     else:
#                         mu, prec = sample_normalgamma(m0=self.m0, l0=self.l0,
#                                                       a0=self.a0, b0=self.b0,
#                                                       mu_hat=self.estimated_rewards[s, a],
#                                                       s_hat=var_r,
#                                                       n=N, local_random=self.local_random)
#                         Qr = np.random.normal(mu, 1 / prec, self.num_reward_samples)
#                     self.R[s, a_idx] = np.mean(Qr)
#                     beta_r[s, a_idx] = np.max(Qr - self.R[s, a_idx])
#
#                 elif self.posterior == "Bernoulli":
#                     v = N * self.estimated_rewards[s, a]
#                     a0 = self.a + v
#                     b0 = self.b + N - v
#                     p = np.asscalar(self.local_random.beta(a=a0, b=b0, size=1))
#                     Qr = np.random.binomial(1, p, self.num_reward_samples)
#                     self.R[s, a_idx] = np.mean(Qr)
#                     beta_r[s, a_idx] = np.max(Qr - self.R[s, a_idx])
#
#                 elif self.posterior is None:
#                     true_rew = self.environment.true_reward(s, a_idx)
#                     self.R[s, a_idx] = .5 * true_rew
#                     beta_r[s, a_idx] = .5 * true_rew
#
#         beta_tau = self.beta_tau()  # confidence bounds on holding times
#
#         t0 = time.perf_counter()
#         span_value = self.opt_solver.run(
#             self.policy_indices, self.policy,
#             self.P,  # self.estimated_probabilities,
#             self.R,
#             self.estimated_holding_times,
#             beta_r, beta_p, beta_tau, self.tau_max,
#             self.r_max, self.tau, self.tau_min,
#             1e-8
#         )
#         t1 = time.perf_counter()
#         tn = t1 - t0
#         self.solver_times.append(tn)
#
#         if span_value < 0:
#             raise EVIException(error_value=span_value)
#
#         return span_value
#
#
# class OptimisticPS_SCAL(PS):
#
#     def __init__(self, environment, r_max, span_constraint,
#                  bound_type_p="bernstain", verbose=0,
#                  augment_reward=True, operator_type='T',
#                  logger=default_logger, random_state=None,
#                  relative_vi=True, posterior=None,
#                  prior_parameters=None):
#
#         super(OptimisticPS_SCAL, self).__init__(environment=environment, r_max=r_max, verbose=verbose,
#                                                 logger=logger, random_state=random_state,
#                                                 posterior=posterior, prior_parameters=prior_parameters)
#         self.opt_solver = SpanConstrainedEVI(nb_states=environment.nb_states,
#                                              actions_per_state=environment.state_actions,
#                                              bound_type=bound_type_p,
#                                              random_state=random_state,
#                                              augmented_reward=1 if augment_reward else 0,
#                                              gamma=1.,
#                                              span_constraint=span_constraint,
#                                              relative_vi=1 if relative_vi else 0,
#                                              operator_type=operator_type)
#         self.policy = np.zeros((self.environment.nb_states, 2), dtype=np.float)
#         self.policy_indices = np.zeros((self.environment.nb_states, 2), dtype=np.int)
#
#         self.augment_reward = augment_reward
#         self.operator_type = operator_type
#         self.span_constraint = span_constraint
#         self.relative_vi = relative_vi
#
#         ns, na = self.estimated_rewards.shape
#         self.rho = 0.1
#         self.num_proba_samples = m.ceil(ns * m.log(ns * na / self.rho))
#         self.num_reward_samples = m.ceil(self.r_max * m.log(ns * na / self.rho))
#
#     def learn(self, duration, regret_time_step, render=False):
#         ns, na = self.estimated_rewards.shape
#         self.kappa = m.log(duration / self.rho)
#         self.omega = m.log(duration / self.rho)
#         self.eta = m.sqrt(duration * ns / na) + 12 * self.omega * ns * ns
#         self.xi = 1. / duration
#         super(OptimisticPS_SCAL, self).learn(duration=duration, regret_time_step=regret_time_step, render=render)
#
#     def simple_optimistic_sampling(self, s, a_idx, ns):
#         mu_k_sa = self.nb_observations[s, a_idx]
#         if mu_k_sa != 0:
#             p_hat = self.P_counter[s, a_idx] / mu_k_sa
#             Delta = [
#                 min(m.sqrt(3 * p_hat[i] * m.log(4 * ns) / mu_k_sa) + 3 * m.log(4 * ns) / mu_k_sa,
#                     p_hat[i]) for i in range(ns)]
#             rest = np.sum(Delta)
#             p_hat -= Delta
#             p_hat[0] += rest * self.xi
#             rest *= (1 - self.xi)
#
#             p_hat += .5 * rest * np.ones(ns)
#             beta_p = .5 * rest * np.ones(ns)
#         else:
#             p_hat = np.zeros(ns)
#             p_hat[0] += self.xi
#             rest = 1 - self.xi
#
#             p_hat += .5 * rest * np.ones(ns)
#             beta_p = .5 * rest * np.ones(ns)
#         return p_hat, beta_p
