import numpy as np
from collections import namedtuple
from UCRL.evi.scevi import SpanConstrainedEVI
from UCRL.evi.evi import EVI
import pytest

MDP = namedtuple('MDP', 'S,A,P,R,gamma')

def core_op(mdp, evi, constraint=np.inf):
    r_max = np.max(mdp.R)

    na = max(map(len, mdp.A))
    if isinstance(evi, SpanConstrainedEVI):
        policy_indices = np.zeros((mdp.S, 2), dtype=np.int)
        policy = np.zeros((mdp.S, 2), dtype=np.float)
    else:
        policy_indices = np.zeros((mdp.S,), dtype=np.int)
        policy = np.zeros((mdp.S,), dtype=np.int)

    # real values are obtained from Puterman pag. 165 (table 6.3.1)
    real_v = {
        0: [0, 0],
        1: [10, -1],
        2: [9.275, -1.95],
        3: [8.47937, -2.8525],
        10: [3.40278, -8.02526],
        20: [-1.40171, -12.8303],
        40: [-6.00119, -17.4298],
        80: [-8.24112, -19.6697],
        162: [-8.56651, -19.9951]
    }

    ################################################################################
    # Run Value Iteration with norm-based stopping rule
    ################################################################################
    u0 = np.zeros((mdp.S,))
    evi.reset_u(u0)
    accuracy = 0.01 * (1 - mdp.gamma) / (2 * mdp.gamma)
    stop = False
    stats = {'values': [], 'stopvalues': []}
    stats['values'].append(u0)
    stats['stopvalues'].append(np.nan)
    i = 0
    while not stop:
        span_value_new = evi.run(policy_indices=policy_indices,
                                 policy=policy,
                                 estimated_probabilities=mdp.P,
                                 estimated_rewards=mdp.R,
                                 estimated_holding_times=np.ones((mdp.S, na)),
                                 beta_r=np.zeros((mdp.S, na)),
                                 beta_p=np.zeros((mdp.S, na, mdp.S)),
                                 beta_tau=np.zeros((mdp.S, na)),
                                 tau_max=1.,
                                 r_max=r_max,
                                 tau=1.,
                                 tau_min=1.,
                                 epsilon=100,
                                 initial_recenter=0,
                                 relative_vi=0,
                                 span_constraint=constraint)
        u1, u2 = evi.get_uvectors()
        # print(u1, u2)
        if i in real_v.keys():
            assert np.allclose(u1, real_v[i])
        if i + 1 in real_v.keys():
            assert np.allclose(u2, real_v[i + 1])
        stop = np.linalg.norm(u2 - u1, ord=np.inf) < accuracy
        stats['stopvalues'].append(np.linalg.norm(u2 - u1, ord=np.inf))
        stats['values'].append(u2)
        i += 1

    print("{:^4}   {:^8} {:^8}    {:^8}".format(
        'it', "v(s_0)", "v(s_1)", "max{|v-v_old|}"
    ))

    for i, el in enumerate(zip(stats['values'], stats['stopvalues'])):
        v = el[0]
        st = el[1]
        print("{:3d}   {:8.4f} {:8.4f}    {:8.5f}".format(i, *v, st))
        if i in real_v.keys():
            assert np.allclose(v, real_v[i])
    assert len(stats['values']) == 163

    ################################################################################
    # Run Value Iteration with span-based stopping rule
    ################################################################################
    # real values are obtained from Puterman pag. 205 (table 6.6.1)
    real_spanv = {
        1: 11,
        2: 0.225,
        3: 0.106875,
        4: 0.050765,
        5: 0.024113,
        6: 0.011453,
        10: 0.000583
    }

    evi.reset_u(np.zeros((mdp.S,)))
    stop = False
    stats_span = {'values': [], 'stopvalues': []}
    stats_span['values'].append(u0)
    stats_span['stopvalues'].append(np.nan)
    i = 0
    while not stop:
        span_value_new = evi.run(policy_indices=policy_indices,
                                 policy=policy,
                                 estimated_probabilities=mdp.P,
                                 estimated_rewards=mdp.R,
                                 estimated_holding_times=np.ones((mdp.S, na)),
                                 beta_r=np.zeros((mdp.S, na)),
                                 beta_p=np.zeros((mdp.S, na, mdp.S)),
                                 beta_tau=np.zeros((mdp.S, na)),
                                 tau_max=1.,
                                 r_max=r_max,
                                 tau=1.,
                                 tau_min=1.,
                                 epsilon=100,
                                 initial_recenter=0,
                                 relative_vi=0,
                                 span_constraint=constraint)
        u1, u2 = evi.get_uvectors()
        span_v = (np.max(u2 - u1) - np.min(u2 - u1))
        print(u1, u2, span_v)
        stop = span_v < accuracy
        stats_span['stopvalues'].append(span_v)
        stats_span['values'].append(u2)

        if i + 1 in real_spanv.keys():
            assert np.isclose(span_v, real_spanv[i + 1], atol=1e-6), i + 1

        i += 1

    print("")
    print("{:^4}   {:^8} {:^8}    {:^8}".format(
        "it", "v(s_0)", "v(s_1)", "v(s_2)", "sp{v-v_old}"
    ))
    for i, el in enumerate(zip(stats_span['values'], stats_span['stopvalues'])):
        v = el[0]
        st = el[1]
        print("{:3d}   {:8.4f} {:8.4f}    {:8.6f}".format(i, *v, st))
        if i in real_spanv.keys():
            assert np.isclose(st, real_spanv[i], atol=1e-6)
    assert len(stats_span['values']) == 13

    return u1, policy_indices, policy

@pytest.mark.parametrize("count",range(100))
def test_state_action(count):
    ################################################################################
    # Define the MDP model (Figure 3.3.1 pag 34)
    ################################################################################
    S = 2

    A = [[1, 0], [0]]  #TODO note that position of action a_0 and a_1 has been reverted
    maxa = max(map(len, A))

    P = -np.inf * np.ones((S, maxa, S))
    R = -np.inf * np.ones((S, maxa))

    P[0, 1, 0] = 0.5
    P[0, 1, 1] = 0.5
    P[0, 0, 1] = 1.
    P[0, 0, 0] = 0.
    R[0, 1] = 5
    R[0, 0] = 10

    P[1, 0, 1] = 1.
    P[1, 0, 0] = 0.
    R[1, 0] = -1

    gamma = 0.95

    mdp = MDP(S, A, P, R, gamma)

    evi1 = EVI(nb_states=mdp.S,
               actions_per_state=mdp.A,
               bound_type="chernoff", random_state=0,
               gamma=mdp.gamma)

    u2_L, policy_indices_L, policy_L = core_op(mdp, evi1)

    evi2 = SpanConstrainedEVI(nb_states=mdp.S,
                              actions_per_state=mdp.A,
                              bound_type="chernoff", random_state=0,
                              gamma=mdp.gamma)

    u2_T, policy_indices_T, policy_T = core_op(mdp, evi2)

    assert np.allclose(u2_L, u2_T)

    print("idx_P_L:\n {}".format(policy_indices_L))
    print("P_L:\n {}".format(policy_indices_L))
    print()
    print("idx_P_T:\n {}".format(policy_indices_T))
    print("P_T:\n {}".format(policy_T))

    assert np.allclose(policy_indices_L, [1,0])
    assert np.allclose(policy_L, [0,0])

    assert np.allclose(policy_indices_T, [[0,1], [0,0]])
    assert np.allclose(policy_T, [[0,1],[1,0]])


if __name__ == '__main__':
    test_state_action(count=1)
