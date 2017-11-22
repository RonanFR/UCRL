import numpy as np
from .Ucrl import UcrlMdp
from .logging import default_logger
from .evi import SpanConstrainedEVI


class SCUCRLMdp(UcrlMdp):
    """
    UCRL with bias span constraint.
    """

    def __init__(self, environment, r_max, span_constraint, alpha_r=None, alpha_p=None,
                 bound_type="chernoff", verbose=0,
                 logger=default_logger, random_state=None, relative_vi = True):
        solver = SpanConstrainedEVI(nb_states=environment.nb_states,
                                    actions_per_state=environment.state_actions,
                                    bound_type=bound_type,
                                    random_state=random_state,
                                    gamma=1.,
                                    span_constraint=span_constraint,
                                    relative_vi=1 if relative_vi else 0)
        super(SCUCRLMdp, self).__init__(
            environment=environment, r_max=r_max,
            alpha_r=alpha_r, alpha_p=alpha_p, solver=solver,
            bound_type=bound_type,
            verbose=verbose, logger=logger, random_state=random_state)

        # we need to change policy structure since it is stochastic
        self.policy = np.zeros((self.environment.nb_states, 2), dtype=np.float)
        self.policy_indices = np.zeros((self.environment.nb_states, 2), dtype=np.int)

    @property
    def span_constraint(self):
        return self.opt_solver.get_span_constraint()

    def description(self):
        super_desc = super().description()
        desc = {
            "span_constraint": self.span_constraint
        }
        super_desc.update(desc)
        return super_desc
