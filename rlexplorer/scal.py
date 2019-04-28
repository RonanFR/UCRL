import numpy as np
from .Ucrl import UcrlMdp
from .rllogging import default_logger
from .evi import SCOPT


class SCAL(UcrlMdp):
    """
    UCRL with bias span constraint.
    """

    def __init__(self, environment, r_max, span_constraint, alpha_r=None, alpha_p=None,
                 bound_type_p="bernstein", bound_type_rew="bernstein",
                 verbose=0, augment_reward=True,
                 logger=default_logger, random_state=None, relative_vi=True,
                 known_reward=False):
        solver = SCOPT(nb_states=environment.nb_states,
                       actions_per_state=environment.state_actions,
                       bound_type=bound_type_p,
                       random_state=random_state,
                       augmented_reward=1 if augment_reward else 0,
                       gamma=1.,
                       span_constraint=span_constraint,
                       relative_vi=1 if relative_vi else 0)
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
        self.span_constraint = span_constraint
        self.relative_vi = relative_vi

        # self.r_max_vi = float(np.iinfo(np.int32).max)

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
            self.opt_solver = SCOPT(nb_states=self.environment.nb_states,
                                    actions_per_state=self.environment.state_actions,
                                    bound_type=self.bound_type_p,
                                    random_state=self.random_state,
                                    augmented_reward=1 if self.augment_reward else 0,
                                    gamma=1.,
                                    span_constraint=self.span_constraint,
                                    relative_vi=1 if self.relative_vi else 0)
        else:
            self.opt_solver = solver
        self.logger = logger


class SCALPLUS(SCAL):
    """
    SCAL with Exploration Bonus
    """

    def __init__(self, environment, r_max, span_constraint, alpha_r=None, alpha_p=None,
                 bound_type_p="hoeffding", bound_type_rew="hoeffding",
                 verbose=0, augment_reward=True,
                 logger=default_logger, random_state=None, relative_vi=True,
                 known_reward=False):
        super(SCALPLUS, self).__init__(environment=environment, r_max=r_max, span_constraint=span_constraint,
                                       alpha_r=alpha_r, alpha_p=alpha_p,
                                       bound_type_p=bound_type_p, bound_type_rew=bound_type_rew,
                                       verbose=verbose, augment_reward=augment_reward,
                                       logger=logger, random_state=random_state, relative_vi=relative_vi,
                                       known_reward=known_reward)

        # self.r_max_vi = float(np.iinfo(np.int32).max)
        self.r_max_vi = (self.span_constraint + self.r_max) * 3

    def beta_p(self):
        if self.bound_type_p == "bernstein":
            return np.zeros(
                (self.environment.nb_states, self.environment.max_nb_actions_per_state, self.environment.nb_states))
        else:
            return np.zeros(
                (self.environment.nb_states, self.environment.max_nb_actions_per_state, 1))

    def beta_r(self):

        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        N = np.maximum(1, self.nb_observations)

        if self.bound_type_p == "bernstein":
            # var_r = self.variance_proxy_reward / N[:, :]
            # var_p = self.P * (1. - self.P)
            # L_CONST = 6
            # log_term = np.log(L_CONST * S * A * N / self.delta) / N
            # beta_r = np.sqrt(var_r * log_term) + self.r_max * log_term
            # beta_p = np.sqrt(var_p.sum(axis=-1) * log_term) \
            #          + log_term + 1.0 / (self.nb_observations + 1.0)
            # P_term = self.alpha_p * self.span_constraint * beta_p
            # R_term = self.alpha_r * beta_r if not self.known_reward else 0
            raise ValueError('Not implemented')
        elif self.bound_type_p == "hoeffding":
            L_CONST = 6  # 20
            beta_r = np.sqrt(np.log(L_CONST * S * A * N / self.delta) / N)
            beta_p = np.sqrt(np.log(L_CONST * S * A * N / self.delta) / N) + 1.0 / (self.nb_observations + 1.0)

            P_term = self.alpha_p * self.span_constraint * np.minimum(beta_p, 2.)
            R_term = self.alpha_r * self.r_max * np.minimum(beta_r, 1 if not self.known_reward else 0)
            # P_term = self.alpha_p * self.span_constraint * beta_p
            # R_term = self.alpha_r * self.r_max * beta_r if not self.known_reward else 0
        else:
            raise ValueError("unknown transition bound type: {}".format(self.bound_type_p))

        final_bonus = R_term + P_term
        return final_bonus

    def update_at_episode_end(self):
        self.nb_observations += self.nu_k

        for (s, a) in self.visited_sa:
            Nsa = self.nb_observations[s, a]
            self.P[s, a] = self.P_counter[s, a] / Nsa
            self.P[s, a] = Nsa * self.P[s, a] / (Nsa + 1.0)
            self.P[s, a, 0] += 1.0 / (Nsa + 1.0)
            assert np.isclose(1., np.sum(self.P[s, a]))
