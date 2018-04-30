import numpy as np
from collections import namedtuple
from UCRL.evi import TEVI


def test_tevi(count):
    MDP = namedtuple('MDP', 'S,A,P,R,gamma')

    S = 2

    A = [[1, 0], [0, 1]]
    na = max(map(len, A))

    P = -np.inf * np.ones((S, na, S), dtype=np.float)
    R = -np.inf * np.ones((S, na), dtype=np.float)

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

    rs = np.random.randint(1, 2131243)
    print(rs)

    evi = TEVI(nb_states=mdp.S,
               actions_per_state=mdp.A,
               bound_type="chernoff", random_state=np.random.randint(1, 2131243),
               gamma=mdp.gamma)

    epsi = 0.0000001

    istruncated = np.zeros((mdp.S, na), dtype=np.int)
    unreachable_s = np.array([], dtype=np.int)

    policy_indices = np.zeros((mdp.S,), dtype=np.int)
    policy = np.zeros((mdp.S,), dtype=np.int)

    run_params = {"policy_indices": policy_indices,
                  "policy": policy,
                  "estimated_probabilities": mdp.P,
                  "estimated_rewards": mdp.R,
                  "estimated_holding_times": np.ones((mdp.S, na)),
                  "beta_r": np.zeros((mdp.S, na)),
                  "beta_p": np.zeros((mdp.S, na, mdp.S)),
                  "beta_tau": np.zeros((mdp.S, na)),
                  "is_truncated_sa": istruncated,
                  "unreachable_states": unreachable_s,
                  "tau_max": 1.,
                  "r_max": r_max,
                  "tau": 1.,
                  "tau_min": 1.,
                  "epsilon": epsi,
                  "span_constraint": 0,
                  "initial_recenter": 0,
                  "relative_vi": 0}

    span_value = evi.run(**run_params)
    # estimate gain
    u1, u2 = evi.get_uvectors()
    maxv = np.max(u2 - u1)
    minv = np.min(u2 - u1)
    gain = 0.5 * (maxv + minv)
    assert np.isclose(gain, 1.)

    # in this case is like run the L operator
    assert np.allclose(policy_indices, [0, 1]), policy_indices
    assert np.allclose(policy, [1, 1]), policy

    run_params['is_truncated_sa'] = np.ones((mdp.S, na), dtype=np.int)

    span_value = evi.run(**run_params)
    # estimate gain
    u1, u2 = evi.get_uvectors()
    maxv = np.max(u2 - u1)
    minv = np.min(u2 - u1)
    gain = 0.5 * (maxv + minv)
    assert np.isclose(gain, 1.)

    # in this case is like run the L operator
    assert np.allclose(policy_indices, [0, 1]), policy_indices
    assert np.allclose(policy, [1, 1]), policy

def test_tevi2(count):
    MDP = namedtuple('MDP', 'S,A,P,R,gamma')

    S = 3

    A = [[0, 1], [0, 1], [0, 1]]
    na = max(map(len, A))

    P = -np.inf * np.ones((S, na, S), dtype=np.float)
    R = -np.inf * np.ones((S, na), dtype=np.float)

    P[0, 0, 0] = 0
    P[0, 0, 1] = 1.
    P[0, 0, 2] = 0
    P[0, 1, 0] = 0.8
    P[0, 1, 2] = 0.2
    P[0, 1, 1] = 0
    P[1, 0, 0] = 0.9
    P[1, 0, 1] = 0
    P[1, 0, 2] = 0.1
    P[1, 1, 1] = 0.9
    P[1, 1, 0] = 0
    P[1, 1, 2] = 0.1
    P[2, 0, :] = 0.
    P[2, 0, 2] = 1.
    P[2, 1, :] = 0.
    P[2, 1, 1] = 1.
    R[0, 0] = 0.
    R[0, 1] = 0.5
    R[1, 0] = 0
    R[1, 1] = 0.7
    R[2, 0] = 0.9
    R[2, 1] = 0.

    r_max = np.max(R)

    gamma = 1.

    mdp = MDP(S, A, P, R, gamma)

    rs = np.random.randint(1, 2131243)
    print(rs)

    evi = TEVI(nb_states=mdp.S,
               actions_per_state=mdp.A,
               bound_type="chernoff", random_state=np.random.randint(1, 2131243),
               gamma=mdp.gamma)

    epsi = 0.00001

    istruncated = np.zeros((mdp.S, na), dtype=np.int)
    unreachable_s = np.array([], dtype=np.int)

    policy_indices = np.zeros((mdp.S,), dtype=np.int)
    policy = np.zeros((mdp.S,), dtype=np.int)

    run_params = {"policy_indices": policy_indices,
                  "policy": policy,
                  "estimated_probabilities": mdp.P,
                  "estimated_rewards": mdp.R,
                  "estimated_holding_times": np.ones((mdp.S, na)),
                  "beta_r": np.zeros((mdp.S, na)),
                  "beta_p": np.zeros((mdp.S, na, 1)),
                  "beta_tau": np.zeros((mdp.S, na)),
                  "is_truncated_sa": istruncated,
                  "unreachable_states": unreachable_s,
                  "tau_max": 1.,
                  "r_max": r_max,
                  "tau": 1.,
                  "tau_min": 1.,
                  "epsilon": epsi,
                  "span_constraint": 0,
                  "initial_recenter": 0,
                  "relative_vi": 0}

    span_value = evi.run(**run_params)
    # estimate gain
    u1, u2 = evi.get_uvectors()
    maxv = np.max(u2 - u1)
    minv = np.min(u2 - u1)
    gain = 0.5 * (maxv + minv)
    assert np.isclose(gain, .9)

    # in this case is like run the L operator
    assert np.allclose(policy_indices, [1, 1, 0]), policy_indices
    assert np.allclose(policy, [1, 1, 0]), policy

    A = np.zeros((mdp.S, na, 1))
    A[1, 1, 0] = 0.2

    istruncated[1, 1] = 1
    unreachable_s = np.array([2], dtype=np.int)
    run_params['is_truncated_sa'] = istruncated
    run_params['unreachable_states'] = unreachable_s
    run_params['beta_p'] = A

    evi.reset_u(np.zeros((mdp.S,)))
    span_value = evi.run(**run_params)
    print("span_value: {}".format(span_value))
    reach_s = [0,1]
    # estimate gain
    u1, u2 = evi.get_uvectors()
    maxv = np.max((u2 - u1))
    minv = np.min((u2 - u1))
    gain = 0.5 * (maxv + minv)
    print(u1, u2)
    assert np.isclose(gain, .9, atol=1e-3), gain

    assert np.isclose(np.max(u2) - np.min(u2), 2.9, atol=1e-3), np.max(u2) - np.min(u2)

    # in this case is like run the L operator
    assert np.allclose(policy_indices, [1, 0, 0]), policy_indices
    assert np.allclose(policy, [1, 0, 0]), policy


if __name__ == '__main__':
    test_tevi2(1)
