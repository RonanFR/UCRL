from .Ucrl import UcrlMdp
from .logging import default_logger

class UcrlBiasSpanCon(UcrlMdp):
    """
    UCRL with bias span constraint.
    """
    
    def __init__(self, environment, r_max, alpha_r=None, alpha_p=None,
                 bound_type="chernoff", verbose = 0,
                 logger=default_logger, random_state=None):

        solver = SpanConstrainedEVI()
        super(UcrlBiasSpanCon, self).__init__(
            environment=environment, r_max=r_max,
            alpha_r=alpha_r, alpha_p=alpha_p, solver=solver,
            bound_type=bound_type,
            verbose=verbose, logger=logger,random_state=random_state)
