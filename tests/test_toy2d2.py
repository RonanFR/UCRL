# This is to test the difference between operator T and N
import numpy as np
from collections import namedtuple
from UCRL.evi.scevi import SpanConstrainedEVI
from UCRL.evi.evi import EVI
import pytest


MDP = namedtuple('MDP', 'S,A,P,R,gamma')

@pytest.mark.parametrize("params", [[1,0.5,0.1], [10,4,3.9], [7,3,1], [1.1,1.,0.9]])
def test_srevi(params):

    alpha, delta, beta = params
    print(alpha, delta, beta)

    S = 3

    A = [[0], [0, 1], [0]]
    maxa = max(map(len, A))

    P = -np.inf * np.ones((S, maxa, S), dtype=np.float)
    R = -np.inf * np.ones((S, maxa), dtype=np.float)

    P[0, 0, 1] = 1.
    P[0, 0, 0] = 0
    P[0, 0, 2] = 0

    P[1, 0, 0] = 0.
    P[1, 0, 1] = 0.
    P[1, 0, 2] = 1.
    P[1, 1, 0] = 0.
    P[1, 1, 1] = 0.
    P[1, 1, 2] = 1.
    P[2, 0, 0] = 0.
    P[2, 0, 1] = 0.
    P[2, 0, 2] = 1.

    R[0, 0] = alpha
    R[1, 0] = beta
    R[1, 1] = delta
    R[2, 0] = 0

    r_max = np.max(R)

    gamma = 1.

    mdp = MDP(S, A, P, R, gamma)

    policy_indices = np.zeros((mdp.S, 2), dtype=np.int)
    policy = np.zeros((mdp.S, 2), dtype=np.float)

    rs = np.random.randint(1, 2131243)
    print(rs)
    evi = SpanConstrainedEVI(nb_states=mdp.S,
                             actions_per_state=mdp.A,
                             bound_type="chernoff", random_state=rs,
                             gamma=mdp.gamma)

    epsi = 0.000001
    c = alpha + beta
    u0 = np.zeros((mdp.S,))

    for oper in ['T', 'N']:
        evi.reset_u(u0)
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
                                 relative_vi=0,
                                 operator_type=oper)
        # estimate gain
        u1, u2 = evi.get_uvectors()
        print('{}: {}'.format(oper, u1 - u1[2]))
        if oper == 'T':
            assert np.isclose(span_value_new, -21), span_value_new
            assert np.allclose(u1 - u1[2], [alpha + beta, delta, 0])
            assert np.isclose(np.max(u1) - np.min(u1), c)

            assert np.allclose(policy_indices, [[-1, -1], [0, 1], [0, 0]])
            assert np.allclose(policy, [[-1, -1], [0, 1.], [1., 0]], atol=1e-5)\
                   or np.allclose(policy, [[-1, -1], [0, 1.], [0, 1.]], atol=1e-5), policy
        else:
            assert np.allclose(u1 - u1[2], [alpha + delta, delta, 0])
            assert np.isclose(np.max(u1) - np.min(u1), alpha + delta)

            print(policy_indices)
            print(policy)
            assert np.allclose(policy_indices, [[0, 0], [0, 1], [0, 0]])
            assert np.allclose(policy, [[1., 0], [0, 1], [1., 0]], atol=1e-5)\
                   or np.allclose(policy, [[1., 0], [0, 1], [0, 1.]], atol=1e-5), policy

if __name__ == '__main__':
    test_srevi([1.1,1,0.9])