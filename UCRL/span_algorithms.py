import numpy as np
from .Ucrl import UcrlMdp
from .logging import default_logger
from .evi import SpanConstrainedEVI
import time


class SCAL(UcrlMdp):
    """
    UCRL with bias span constraint.
    """

    def __init__(self, environment, r_max, span_constraint, alpha_r=None, alpha_p=None,
                 bound_type_p="chernoff", bound_type_rew="chernoff",
                 verbose=0, augment_reward=True, operator_type="T",
                 logger=default_logger, random_state=None, relative_vi=True,
                 known_reward=False):
        solver = SpanConstrainedEVI(nb_states=environment.nb_states,
                                    actions_per_state=environment.state_actions,
                                    bound_type=bound_type_p,
                                    random_state=random_state,
                                    augmented_reward=1 if augment_reward else 0,
                                    gamma=1.,
                                    span_constraint=span_constraint,
                                    relative_vi=1 if relative_vi else 0,
                                    operator_type=operator_type)
        super(SCAL, self).__init__(
            environment=environment, r_max=r_max,
            alpha_r=alpha_r, alpha_p=alpha_p, solver=solver,
            bound_type_p=bound_type_p,
            bound_type_rew=bound_type_rew,
            verbose=verbose, logger=logger, random_state=random_state,
            known_reward=known_reward)

        # we need to change policy structure since it is stochastic
        self.policy = np.zeros((self.environment.nb_states, 2), dtype=np.float)
        self.policy_indices = np.zeros((self.environment.nb_states, 2), dtype=np.int)

        self.augment_reward = augment_reward
        self.operator_type = operator_type
        self.span_constraint = span_constraint
        self.relative_vi = relative_vi

    # @property
    # def span_constraint(self):
    #     return self.opt_solver.get_span_constraint()

    def description(self):
        super_desc = super().description()
        desc = {
            "span_constraint": self.span_constraint,
            "operator_type": self.opt_solver.get_operator_type(),
            "relative_vi": self.opt_solver.use_relative_vi()
        }
        super_desc.update(desc)
        return super_desc

    def reset_after_pickle(self, solver=None, logger=default_logger):
        if solver is None:
            self.opt_solver = SpanConstrainedEVI(nb_states=self.environment.nb_states,
                                    actions_per_state=self.environment.state_actions,
                                    bound_type=self.bound_type_p,
                                    random_state=self.random_state,
                                    augmented_reward=1 if self.augment_reward else 0,
                                    gamma=1.,
                                    span_constraint=self.span_constraint,
                                    relative_vi=1 if self.relative_vi else 0,
                                    operator_type=self.operator_type)
        else:
            self.opt_solver = solver
        self.logger = logger


class SCAL_bonus(SCAL):
    """
    SCAL with Exploration Bonus
    """

    def __init__(self, environment, r_max, span_constraint, alpha_r=None, alpha_p=None,
                 bound_type_p="chernoff", bound_type_rew="chernoff",
                 verbose=0, augment_reward=True, operator_type="T",
                 logger=default_logger, random_state=None, relative_vi=True,
                 known_reward=False):
        super(SCAL_bonus, self).__init__(environment=environment, r_max=r_max, span_constraint=span_constraint, alpha_r=alpha_r, alpha_p=alpha_p,
                 bound_type_p=bound_type_p, bound_type_rew=bound_type_rew,
                 verbose=verbose, augment_reward=augment_reward, operator_type=operator_type,
                 logger=logger, random_state=random_state, relative_vi=relative_vi,
                 known_reward=known_reward)

        self.opt_solver.set_rmax(r_max*100)

    def beta_p(self):
        return np.zeros_like((self.environment.nb_states, self.environment.max_nb_actions_per_state, 1))

    def beta_r(self):
        beta_r = super(SCAL_bonus, self).beta_r()

        beta_p = super(SCAL_bonus, self).beta_p()
        beta_p = np.sum(beta_p, axis=-1) + 2.0 / (self.nb_observations+1.0)

        beta_r_final = np.minimum(beta_r, self.r_max if not self.known_reward else 0) + self.span_constraint * np.minimum(beta_p, 2.)
        return beta_r_final

    def update_at_episode_end(self):
        self.nb_observations += self.nu_k

        for (s, a) in self.visited_sa:
            Nsa = self.nb_observations[s, a]
            self.P[s, a] = self.P_counter[s, a] / Nsa
            self.P[s, a] = Nsa * self.P[s, a] / (Nsa + 1.0)
            self.P[s, a, 0] += 1.0 / (Nsa + 1.0)
            assert np.isclose(1., np.sum(self.P[s,a]))



