import numpy as np
from .Ucrl import UcrlMdp, EVIException
from .logging import default_logger
from .evi import STEVI
import math as m
import time


class STUCRL(UcrlMdp):
    """
    UCRL with short-term extended value iteration
    """

    def __init__(self, environment, r_max, alpha_r=None, alpha_p=None,
                 bound_type_p="chernoff", bound_type_rew="chernoff",
                 verbose=0,
                 logger=default_logger, random_state=None):
        solver = STEVI(nb_states=environment.nb_states,
                       actions_per_state=environment.get_state_actions(),
                       bound_type=bound_type_p,
                       random_state=random_state,
                       gamma=1.)
        super(STUCRL, self).__init__(
            environment=environment, r_max=r_max,
            alpha_r=alpha_r, alpha_p=alpha_p, solver=solver,
            bound_type_p=bound_type_p,
            bound_type_rew=bound_type_rew,
            verbose=verbose, logger=logger, random_state=random_state)

        self.prev_state = None
        self.prev_act_idx = None
        self.SA = self.environment.nb_states * self.environment.max_nb_actions_per_state

    def _stopping_rule(self, curr_state, curr_act_idx):
        if self.prev_state is None:
            self.prev_state = curr_state
            self.prev_act_idx = curr_act_idx
            return True

        p = m.log(self.SA * (self.iteration+1) / self.delta) / max(1, self.nb_observations[self.prev_state][
            self.prev_act_idx])
        p = min(1., p)
        Y = np.random.binomial(1, p=p)

        if Y != 1:
            # update internal values
            self.prev_state = curr_state
            self.prev_act_idx = curr_act_idx
            return True
        else:
            self.prev_state = None
            self.prev_act_idx = None
            return False

    def solve_optimistic_model(self, curr_state=None):

        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_tau = self.beta_tau()  # confidence bounds on holding times
        beta_p = self.beta_p()  # confidence bounds on transition probabilities
        eta = self.compute_eta()

        t0 = time.perf_counter()
        span_value = self.opt_solver.run(
            policy_indices=self.policy_indices,
            policy=self.policy,
            estimated_probabilities=self.P,  # self.estimated_probabilities,
            estimated_rewards=self.estimated_rewards,
            estimated_holding_times=self.estimated_holding_times,
            beta_r=beta_r,
            beta_p=beta_p,
            beta_tau=beta_tau,
            tau_max=self.tau_max,
            r_max=self.r_max, tau=self.tau, tau_min=self.tau_min,
            epsilon=self.r_max / m.sqrt(self.iteration + 1),
            eta=eta,
            ref_state=curr_state
        )
        t1 = time.perf_counter()
        tn = t1 - t0
        self.solver_times.append(tn)
        if self.verbose > 1:
            self.logger.info("[%d]NEW EVI: %.3f seconds" % (self.episode, tn))
            if self.verbose > 2:
                self.logger.info(self.policy_indices)

        if span_value < 0:
            raise EVIException(error_value=span_value)

        return span_value

    def compute_eta(self):
        value = m.log(self.SA * (self.total_time+1) / self.delta)
        N = np.maximum(1, self.nb_observations)
        eta = 1. - np.minimum(1, value / N)
        return eta
