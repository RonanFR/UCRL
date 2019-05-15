import numpy as np
from .rllogging import default_logger
from .scal import SCALPLUS, SCAL
import math as m


class SCCALPlus(SCALPLUS):
    """
    SCALPlus for continuous states and discrete actions
    """

    def __init__(self, discretized_env, r_max, span_constraint,
                 holder_L, holder_alpha,
                 alpha_r=None, alpha_p=None,
                 truncation_level=None,
                 verbose=0, augment_reward=True,
                 logger=default_logger, random_state=None, relative_vi=True,
                 known_reward=False):
        super(SCCALPlus, self).__init__(environment=discretized_env, r_max=r_max, span_constraint=span_constraint,
                                        truncation_level=truncation_level,
                                        alpha_r=alpha_r, alpha_p=alpha_p,
                                        verbose=verbose, augment_reward=augment_reward,
                                        logger=logger, random_state=random_state, relative_vi=relative_vi,
                                        known_reward=known_reward)
        self.holder_alpha = holder_alpha
        self.holder_L = holder_L
        self.r_max_vi = (self.span_constraint + self.r_max) * 3
        self.r_max_vi = float(np.iinfo(np.int32).max)

    def beta_r(self):
        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        N = np.maximum(1, self.nb_observations)
        LOG_TERM = np.log(S * A / self.delta)
        beta_r = np.sqrt(LOG_TERM / N)
        beta_p = np.sqrt(LOG_TERM / N)  # + 1.0 / (self.nb_observations + 1.0)

        L_term = self.holder_L * m.pow(1.0 / S, self.holder_alpha)
        # P_term = self.alpha_p * self.span_constraint * np.minimum(beta_p + L_term, 2.)
        # R_term = self.alpha_r * self.r_max * np.minimum(beta_r + L_term, 1 if not self.known_reward else 0)
        P_term = self.alpha_p * self.span_constraint * (beta_p + L_term)
        R_term = (self.alpha_r * self.r_max * (beta_r + L_term)) if not self.known_reward else 0

        final_bonus = R_term + P_term
        return final_bonus

    # def update_at_episode_end(self):
    #     self.nb_observations += self.nu_k
    #
    #     for (s, a) in self.visited_sa:
    #         Nsa = self.nb_observations[s, a]
    #         self.P[s, a] = self.P_counter[s, a] / Nsa
    #         assert np.isclose(1., np.sum(self.P[s, a]))

    # def _stopping_rule(self, curr_state, curr_act_idx, next_state):
    #     return self.nu_k[curr_state][curr_act_idx] < 0.1 * max(1, self.nb_observations[curr_state][curr_act_idx])
