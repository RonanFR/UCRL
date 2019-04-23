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
                 verbose=0, augment_reward=True,
                 logger=default_logger, random_state=None, relative_vi=True,
                 known_reward=False):
        super(SCCALPlus, self).__init__(environment=discretized_env, r_max=r_max, span_constraint=span_constraint,
                                        alpha_r=alpha_r, alpha_p=alpha_p,
                                        verbose=verbose, augment_reward=augment_reward,
                                        logger=logger, random_state=random_state, relative_vi=relative_vi,
                                        known_reward=known_reward)
        self.holder_alpha = holder_alpha
        self.holder_L = holder_L

    def beta_r(self):

        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        N = np.maximum(1, self.nb_observations)
        beta_r = np.sqrt(np.log(N * 20 * S * A / self.delta) / N)
        beta_p = np.sqrt(np.log(N * 20 * S * A / self.delta) / N) + 1.0 / (self.nb_observations + 1.0)

        L_term = self.holder_L * m.pow(1.0 / S, self.holder_alpha)
        # P_term = self.alpha_p * self.span_constraint * np.minimum(beta_p + L_term, 2.)
        # R_term = self.alpha_r * self.r_max * np.minimum(beta_r + L_term, 1 if not self.known_reward else 0)
        P_term = self.alpha_p * self.span_constraint * (beta_p + L_term)
        R_term = (self.alpha_r * self.r_max * (beta_r + L_term)) if not self.known_reward else 0

        final_bonus = R_term + P_term
        return final_bonus

    @staticmethod
    def nb_discrete_state(T, nA, holder_L, holder_alpha):
        exp_val = 1./(1.+holder_alpha)
        N = m.pow(holder_alpha * holder_L * m.sqrt(T/nA), exp_val)
        return max(10, int(m.ceil(N)))
