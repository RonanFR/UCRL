import numpy as np
import time
import math as m
from .evi import EVI
from .logging import default_logger
from .Ucrl import UcrlMdp, EVIException


class PS(UcrlMdp):

    def __init__(self, environment, r_max, verbose=0,
                 logger=default_logger, random_state=None):

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

    def update_at_episode_end(self):

        self.nb_observations += self.nu_k

        ns, na = self.estimated_rewards.shape
        for s in range(ns):
            for a, _ in enumerate(self.environment.state_actions[s]):
                N = self.nb_observations[s, a]
                var_r = self.variance_proxy_reward[s,a] / max(1, N - 1)
                self.P[s, a] = self.local_random.dirichlet(1 + self.P_counter[s,a], 1)
                self.R[s, a] = self.local_random.normal(self.estimated_rewards[s, a], m.sqrt(var_r))

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
