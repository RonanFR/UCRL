import numpy as np
from collections import namedtuple
from UCRL.evi.scevi import SpanConstrainedEVI
from UCRL.evi.evi import EVI
import pytest


@pytest.mark.parametrize("count", range(300))
def test_srevi(count):
    MDP = namedtuple('MDP', 'S,A,P,R,gamma')

    S = 2

    A = [[1, 0], [0, 1]]
    maxa = max(map(len, A))

    P = -np.inf * np.ones((S, maxa, S), dtype=np.float)
    R = -np.inf * np.ones((S, maxa), dtype=np.float)

    P[0, 0, 1] = 1.
    P[0, 0, 0] = 0
    P[0, 1, 0] = 1.
    P[0, 1, 1] = 0
    P[1, 0, 0] = 1.
    P[1, 0, 1] = 0
    P[1, 1, 1] = 1.
    P[1, 1, 0] = 0
    R[0, 0] = 0.
    R[0, 1] = 0.5
    R[1, 0] = 0
    R[1, 1] = 1

    r_max = np.max(R)

    gamma = 1.

    mdp = MDP(S, A, P, R, gamma)

    policy_indices = np.zeros((mdp.S, 2), dtype=np.int)
    policy = np.zeros((mdp.S, 2), dtype=np.float)

    constraints = np.linspace(0, 1.2, 20)

    rs = np.random.randint(1, 2131243)
    print(rs)
    evi = SpanConstrainedEVI(nb_states=mdp.S,
                             actions_per_state=mdp.A,
                             bound_type="chernoff",
                             span_constraint=np.inf,
                             relative_vi=0, random_state=rs,
                             gamma=mdp.gamma)

    epsi = 0.0001
    for c in constraints:
        evi.reset_u(np.zeros((mdp.S,)))
        span_value_new = evi.run(policy_indices=policy_indices,
                                 policy=policy,
                                 estimated_probabilities=mdp.P,
                                 estimated_rewards=mdp.R,
                                 estimated_holding_times=np.ones((mdp.S, maxa)),
                                 beta_r=np.zeros((mdp.S, maxa)),
                                 beta_p=np.zeros((mdp.S, maxa, mdp.S)),
                                 beta_tau=np.zeros((mdp.S, maxa)),
                                 tau_max=1.,
                                 r_max=r_max,
                                 tau=1.,
                                 tau_min=1.,
                                 epsilon=epsi,
                                 span_constraint=c,
                                 initial_recenter=1,
                                 relative_vi=1)

        # estimate gain
        u1, u2 = evi.get_uvectors()
        maxv = np.max(u2 - u1)
        minv = np.min(u2 - u1)
        gain = 0.5 * (maxv + minv)

        print("{:.3f}: {:.3f} [{:.3f}]".format(c, span_value_new, gain))
        if span_value_new < 0:
            assert False
        assert np.isclose(span_value_new, min(1., c)), span_value_new
        assert np.isclose(gain, np.clip(c, 0.5, 1.))

        if c < 0.5:
            x, y = 0, 1. / (2 + 2 * c)
            assert np.allclose(policy_indices, [[0, 1], [0, 1]]), policy_indices
            assert np.allclose(policy, [[x, 1 - x], [y, 1 - y]]), policy
        elif c < 1.:
            x, y = 1, (1 - c) / (1 + c)
            # note that also the indices of the actions are different
            # since max and min action are inverted
            assert np.allclose(policy_indices, [[1, 0], [0, 1]]), policy_indices
            assert np.allclose(policy, [[1 - x, x], [y, 1 - y]]), policy
        else:
            x, y = 1, 0
            assert np.allclose(policy_indices, [[1, 0], [0, 1]]), policy_indices
            assert np.allclose(policy, [[1 - x, x], [y, 1 - y]]), policy

    evi = EVI(nb_states=mdp.S,
              actions_per_state=mdp.A,
              bound_type="chernoff", random_state=np.random.randint(1, 2131243),
              gamma=mdp.gamma)

    policy_indices = np.zeros((mdp.S,), dtype=np.int)
    policy = np.zeros((mdp.S,), dtype=np.int)
    span_value_new = evi.run(policy_indices=policy_indices,
                             policy=policy,
                             estimated_probabilities=mdp.P,
                             estimated_rewards=mdp.R,
                             estimated_holding_times=np.ones((mdp.S, maxa)),
                             beta_r=np.zeros((mdp.S, maxa)),
                             beta_p=np.zeros((mdp.S, maxa, mdp.S)),
                             beta_tau=np.zeros((mdp.S, maxa)),
                             tau_max=1.,
                             r_max=r_max,
                             tau=1.,
                             tau_min=1.,
                             epsilon=epsi,
                             span_constraint=0,
                             initial_recenter=1,
                             relative_vi=0)
    # estimate gain
    u1, u2 = evi.get_uvectors()
    maxv = np.max(u2 - u1)
    minv = np.min(u2 - u1)
    gain = 0.5 * (maxv + minv)
    assert np.isclose(gain, 1.)


if __name__ == '__main__':
    test_srevi(count=1)
