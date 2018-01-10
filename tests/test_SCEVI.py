from UCRL.evi.scevi import SpanConstrainedEVI
import numpy as np
import pytest


@pytest.mark.parametrize("count", range(300))
def test_SCEVI1(count):
    v = np.array([0, 1.1269345, 0.3739842])

    state_actions = [[0], [0], [0, 1]]

    P = np.array([[[0., 0., 1.],
                   [0.33333333, 0.33333333, 0.33333333]],
                  [[0.33333333, 0.33333333, 0.33333333],
                   [0.33333333, 0.33333333, 0.33333333]],
                  [[1., 0., 0.],
                   [0., 0., 1.]]])

    R = np.array([[0., 100.],
                  [100., 100.],
                  [0.5, 0.5]])

    holding_time = np.ones((3, 2))
    holding_tau = np.zeros((3, 2))

    beta_p = np.array([[[0.74188992],
                        [23.75206651]],
                       [[23.75206651],
                        [23.75206651]],
                       [[0.74225208],
                        [0.74225208]]])

    beta_r = np.array([[0.2227, 7.13151678],
                       [7.13, 7.13],
                       [0.2228, 0.2228]])

    tau_max = tau_min = 1
    tau = 1.
    r_max = 1

    S = len(state_actions)
    c = 1.5

    policy_indices = np.zeros((S, 2), dtype=np.int)
    policy = np.zeros((S, 2), dtype=np.float)

    evi = SpanConstrainedEVI(nb_states=S,
                             actions_per_state=state_actions,
                             bound_type="chernoff",
                             span_constraint=c,
                             relative_vi=0,
                             random_state=np.random.randint(1, 12451326),
                             augmented_reward=0,
                             gamma=1,
                             operator_type='T')
    for opt in ['T', 'N']:
        print(opt)
        c = 1.5
        evi.reset_u(v)
        span_value_new = evi.run(policy_indices=policy_indices,
                                 policy=policy,
                                 estimated_probabilities=P,
                                 estimated_rewards=R,
                                 estimated_holding_times=holding_time,
                                 beta_r=beta_r,
                                 beta_p=beta_p,
                                 beta_tau=holding_tau,
                                 tau_max=tau_max,
                                 r_max=r_max,
                                 tau=tau,
                                 tau_min=tau_min,
                                 epsilon=100,
                                 span_constraint=c,
                                 initial_recenter=0,
                                 relative_vi=1,
                                 operator_type=opt)
        u1, u2 = evi.get_uvectors()
        u3mina, u3 = evi.get_u3vectors()

        assert span_value_new >= 0.
        assert np.allclose(u1, v)
        assert np.allclose(u2, [0.875987318915, 2.1269345, 1.37622366316]), u2
        assert np.allclose(u3, [0.875987318915, 2.1269345, 1.37622366316])
        if opt == 'T':
            assert np.allclose(u3mina, [-1, -1, -1]), u3mina
        else:
            assert np.allclose(u3mina, [0.23525664589, 0, 0.2772]), u3mina
        assert np.allclose(policy_indices, [[0, 0], [0, 0], [1, 1]]), policy_indices
        assert np.allclose(policy, [[0, 1.], [0, 1.], [0, 1]]), policy

        c = 1.2
        evi.reset_u(v)
        span_value_new = evi.run(policy_indices=policy_indices,
                                 policy=policy,
                                 estimated_probabilities=P,
                                 estimated_rewards=R,
                                 estimated_holding_times=holding_time,
                                 beta_r=beta_r,
                                 beta_p=beta_p,
                                 beta_tau=holding_tau,
                                 tau_max=tau_max,
                                 r_max=r_max,
                                 tau=1.,
                                 tau_min=tau_min,
                                 epsilon=100,
                                 span_constraint=c,
                                 initial_recenter=0,
                                 relative_vi=1,
                                 operator_type=opt)
        u1, u2 = evi.get_uvectors()
        u3mina, u3 = evi.get_u3vectors()

        assert span_value_new >= 0.
        assert np.allclose(u1, v)
        assert np.allclose(u2, [0.875987318915, 2.075987318915, 1.37622366316])
        assert np.allclose(u3, [0.875987318915, 2.1269345, 1.37622366316])
        if opt == 'T':
            assert np.allclose(u3mina, [-1, 0, -1])
        else:
            assert np.allclose(u3mina, [0.23525664589, 0, 0.2772]), u3mina
        assert np.allclose(policy_indices, [[0, 0], [0, 0], [1, 1]]), policy_indices
        assert np.allclose(policy, [[0, 1.], [1. - 0.976046662, 0.976046662], [0, 1]]), policy


if __name__ == '__main__':
    test_SCEVI1(count=1)
