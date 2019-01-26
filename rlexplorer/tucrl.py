import numpy as np
import math
import time
from . import Ucrl as ucrl
from .evi import TEVI


class TUCRL(ucrl.UcrlMdp):

    def __init__(self, environment, r_max,
                 alpha_r=None, alpha_p=None,
                 verbose=0, logger=ucrl.default_logger,
                 random_state=None, known_reward=False):
        tevi = TEVI(nb_states=environment.nb_states,
                    actions_per_state=environment.get_state_actions(),
                    bound_type="bernstein",
                    random_state=random_state,
                    gamma=1.)

        super(TUCRL, self).__init__(
            environment=environment, r_max=r_max,
            alpha_r=alpha_r, alpha_p=alpha_p,
            solver=tevi,
            bound_type_p="bernstein", bound_type_rew="bernstein",
            verbose=verbose,
            logger=logger, random_state=random_state, known_reward=known_reward)

        self.nb_state_observations = np.zeros((self.environment.nb_states,))
        self.bad_state_action = 0
        self.bad_state_action_values = [0]

    def _stopping_rule(self, curr_state, curr_act_idx, next_state):
        c2_1 = self.nu_k[curr_state][curr_act_idx] < max(1, self.nb_observations[curr_state][curr_act_idx])
        c2_2 = self.nb_state_observations[next_state] > 0
        return c2_1 and c2_2

    def solve_optimistic_model(self, curr_state=None):
        S, A = self.nb_observations.shape

        mask = self.nb_state_observations > 0
        mask[curr_state] = True
        reachable_states = np.arange(S)[mask]
        unreachable_states = np.arange(S)[np.logical_not(mask)]
        is_truncated_sa = np.zeros((S, A), dtype=np.int)
        num_truncated_sa = None

        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_p = self.beta_p()  # confidence bounds on transition probabilities
        bonus = np.zeros((S, A))

        if self.num_unreachable_states > 0:
            mask = np.maximum(1, self.nb_observations - 1) > math.sqrt((self.total_time + 1) / (S * A))
            num_truncated_sa = np.sum(mask[reachable_states, :])
            mask[unreachable_states] = False
            is_truncated_sa[mask] = 1
            bonus[mask] = beta_p[mask,unreachable_states[0]] * len(unreachable_states)

        if num_truncated_sa == A*len(reachable_states):
            SEVI = self.reachable_states
        else:
            SEVI = np.arange(S)

        # to run evi on full state space (enable this)
        # states_terminalcond = np.arange(S)

        t0 = time.perf_counter()
        span_value = self.opt_solver.run(
            self.policy_indices, self.policy,
            self.P,
            self.estimated_rewards if not self.known_reward else self.true_reward,
            beta_r, beta_p,
            is_truncated_sa,
            reachable_states,
            SEVI,
            self.r_max, self.tau,
            self.r_max / math.sqrt(self.iteration + 1)
        )
        t1 = time.perf_counter()
        tn = t1 - t0
        self.solver_times.append(tn)
        self.logger.info("unreachable states: {}".format(len(unreachable_states)))
        if self.verbose > 1:
            self.logger.info("[%d]NEW EVI: %.3f seconds" % (self.episode, tn))
            if self.verbose > 2:
                self.logger.info(self.policy_indices)

        if span_value < 0:
            raise ucrl.EVIException(error_value=span_value)

        return span_value

    def update_at_episode_end(self):
        super(TUCRL, self).update_at_episode_end()
        self.nb_state_observations = self.nb_observations.sum(axis=1)

    def reset_after_pickle(self, solver=None, logger=ucrl.default_logger):
        if solver is None:
            self.opt_solver = TEVI(nb_states=self.environment.nb_states,
                                  actions_per_state=self.environment.get_state_actions(),
                                  bound_type="bernstein",
                                  random_state = self.random_state,
                                  gamma=1.
                                  )
        else:
            self.opt_solver = solver
        self.logger = logger

    def save_information(self):
        super(TUCRL, self).save_information()
        self.bad_state_action_values.append(self.bad_state_action)

    def update(self, curr_state, curr_act_idx, curr_act):
        S, A = self.nb_observations.shape
        if max(1, self.nb_observations[curr_state, curr_act_idx] - 1) <= math.sqrt((self.total_time + 1) / (S * A)):
            self.bad_state_action += 1
        super(TUCRL, self).update(curr_state, curr_act_idx, curr_act)
