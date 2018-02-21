import numpy as np
import time
import math as m
from .evi import EVI
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


class PS(UcrlMdp):

    # Conjugate prior Table
    # https://en.wikipedia.org/wiki/Conjugate_prior

    def __init__(self, environment, r_max, verbose=0,
                 logger=default_logger, random_state=None,
                 posterior="Bernoulli", prior_parameters=None):

        assert posterior in ["Bernoulli", "Normal"]

        super(PS, self).__init__(environment=environment,
                                 r_max=r_max,
                                 verbose=verbose,
                                 logger=logger, random_state=random_state)

        self.opt_solver = EVI(nb_states=self.environment.nb_states,
                              actions_per_state=self.environment.get_state_actions(),
                              bound_type="bernstein",
                              random_state=random_state,
                              gamma=1.
                              )

        self.R = np.ones_like(self.estimated_rewards) * r_max

        self.posterior = posterior
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

    def update_at_episode_end(self):

        self.nb_observations += self.nu_k

        ns, na = self.estimated_rewards.shape
        for s in range(ns):
            for a, _ in enumerate(self.environment.state_actions[s]):
                N = self.nb_observations[s, a]

                if self.posterior == "Normal":
                    var_r = self.variance_proxy_reward[s, a] / max(1, N)
                    self.P[s, a] = self.local_random.dirichlet(1 + self.P_counter[s, a], 1)
                    if N == 0:
                        self.R[s,a] = self.r_max
                    else:
                        mu, prec = sample_normalgamma(m0=self.m0, l0=self.l0,
                                                  a0=self.a0, b0=self.b0,
                                                  mu_hat=self.estimated_rewards[s,a],
                                                  s_hat=var_r,
                                                  n=N, local_random=self.local_random)
                        # self.R[s, a] = self.local_random.normal(mu, 1./prec) if prec > 1e-10 else mu
                        self.R[s, a] = mu
                elif self.posterior == "Bernoulli":
                    v = N * self.estimated_rewards[s,a]
                    a0 = self.a + v
                    b0 = self.b + N - v
                    p = np.asscalar(self.local_random.beta(a=a0, b=b0, size=1))
                    self.R[s,a] = p

                print(self.R[s,a])


    def solve_optimistic_model(self, curr_state=None):
        # note that the first time we are here P and R are
        # respectively uniform and r_max

        ns, na = self.estimated_rewards.shape
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
