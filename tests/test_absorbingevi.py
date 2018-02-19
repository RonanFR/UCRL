import numpy as np
from UCRL.evi.evi import EVI
from UCRL.evi.absorbingevi import AbsorbingEVI
import pytest


def garnet(S, A, B):
    """
    Arguments:
        S (int): number of states
        A (int): number of actions
        B (int): branching factor (B<S)
    """
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            R[s, a] = np.random.rand(1) * 10
            nextS = np.random.choice(np.arange(S), B, replace=False)
            proba = np.random.rand(B)
            P[s, a, nextS] = proba / np.sum(proba)
    return P, R


@pytest.mark.parametrize("count", range(100))
def test_state_action(count):
    ns = np.asscalar(np.random.randint(2, 200, 1))
    na = np.asscalar(np.random.randint(2, 20, 1))
    b = np.asscalar(np.random.randint(1, ns, 1))
    core_op(ns=ns, na=na, b=b)


def core_op(ns, na, b):
    # ns = 2
    # na = 2
    # A = [[0], [0, 1]]
    # P = np.zeros((ns, na, ns))
    # P[0, 0, 0] = 0.1
    # P[0, 0, 1] = 0.9
    # P[1, 0, 1] = 1
    # P[1, 1, 0] = 1
    #
    # R = np.zeros((ns, na))
    # R[0, 0] = 1
    # R[1, 0] = 0.5
    # R[1, 1] = 0
    #
    # eta = np.zeros((ns, na))
    # eta[0, 0] = 0.6
    # eta[1, 1] = 0.4
    # eta[1, 0] = 0.3

    P, R = garnet(ns, na, int(ns / 1.2))

    A = [list(range(na)) for _ in range(ns)]

    r_max = np.max(R)

    eta = np.random.rand(ns, na)

    refstate = 0

    Pmod = np.zeros_like(P)
    for s in range(ns):
        for a in range(na):
            for sp in range(ns):
                Pmod[s, a, sp] = eta[s, a] * P[s, a, sp] + (1 - eta[s, a]) * (sp == refstate)

    evi1 = EVI(nb_states=ns,
               actions_per_state=A,
               bound_type="chernoff", random_state=0,
               gamma=1)

    policy_indices = np.zeros((ns,), dtype=np.int)
    policy = np.zeros((ns,), dtype=np.int)
    run_params = {"policy_indices": policy_indices,
                  "policy": policy,
                  "estimated_probabilities": Pmod,
                  "estimated_rewards": R,
                  "estimated_holding_times": np.ones((ns, na)),
                  "beta_r": np.zeros((ns, na)),
                  "beta_p": np.zeros((ns, na, ns)),
                  "beta_tau": np.zeros((ns, na)),
                  "tau_max": 1.,
                  "r_max": r_max,
                  "tau": 1.,
                  "tau_min": 1.,
                  "epsilon": 100 * r_max,
                  "initial_recenter": 0,
                  "relative_vi": 0}

    span_value = evi1.run(**run_params)
    u11, u21 = evi1.get_uvectors()
    # span_v = (np.max(u2 - u1) - np.min(u2 - u1))

    run_params['estimated_probabilities'] = P
    run_params['eta'] = eta
    run_params['ref_state'] = refstate

    evi2 = AbsorbingEVI(nb_states=ns,
                        actions_per_state=A,
                        bound_type="chernoff", random_state=0,
                        gamma=1)

    span_value2 = evi2.run(**run_params)
    u12, u22 = evi1.get_uvectors()

    assert np.allclose(u21, u22)
    assert np.allclose(u11, u12)
    assert np.isclose(span_value, span_value2)


if __name__ == '__main__':
    core_op(2, 2, 1)
