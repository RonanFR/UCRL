import UCRL.Ucrl as Ucrl
from UCRL.span_algorithms import SCAL
from UCRL.evi.scevi import SpanConstrainedEVI
import time
import numpy as np
import math as m
import random
from UCRL.envs.toys import Toy3D_1
import pytest


class TUCRL(Ucrl.UcrlMdp):
    def learn(self, duration, regret_time_step, render=False):
        self.scevi = SpanConstrainedEVI(nb_states=self.environment.nb_states,
                                        actions_per_state=self.environment.get_state_actions(),
                                        bound_type=self.bound_type_p, random_state=self.random_state,
                                        augmented_reward=0,
                                        gamma=1., span_constraint=np.inf,
                                        relative_vi=0)
        self.policy_indices_sc = np.zeros((self.environment.nb_states, 2), dtype=np.int)
        self.policy_sc = np.zeros((self.environment.nb_states, 2), dtype=np.float)

        super(TUCRL, self).learn(duration=duration, regret_time_step=regret_time_step, render=False)

    def sample_action(self, s):

        # this is a stochastic policy
        idx = self.local_random.choice(2, p=self.policy_sc[s])
        assert self.policy_sc[s, idx] > 0
        action_idx_sc = self.policy_indices_sc[s, idx]
        action_sc = self.environment.state_actions[s][action_idx_sc]

        action_idx = self.policy_indices[s]
        action = self.policy[s]
        assert action == action_sc
        assert action_idx == action_idx_sc
        return action_idx, action

    def solve_optimistic_model(self):

        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_tau = self.beta_tau()  # confidence bounds on holding times
        beta_p = self.beta_p()  # confidence bounds on transition probabilities

        ##########################################################################################
        # EVI
        ##########################################################################################
        t0 = time.perf_counter()
        span_value = self.opt_solver.run(
            self.policy_indices, self.policy,
            self.P,  # self.estimated_probabilities,
            self.estimated_rewards,
            self.estimated_holding_times,
            beta_r, beta_p, beta_tau, self.tau_max,
            self.r_max, self.tau, self.tau_min,
            self.r_max / m.sqrt(self.iteration + 1),
            initial_recenter=1
        )
        t1 = time.perf_counter()
        tn = t1 - t0

        ##########################################################################################
        # Span Constrained EVI
        ##########################################################################################

        t0_sc = time.perf_counter()
        span_value_sc = self.scevi.run(
            policy_indices=self.policy_indices_sc,
            policy=self.policy_sc,
            estimated_probabilities=self.P,
            estimated_rewards=self.estimated_rewards,
            estimated_holding_times=self.estimated_holding_times,
            beta_r=beta_r,
            beta_p=beta_p,
            beta_tau=beta_tau,
            tau_max=self.tau_max,
            r_max=self.r_max,
            tau=self.tau, tau_min=self.tau_min,
            epsilon=self.r_max / m.sqrt(self.iteration + 1),
            initial_recenter=1, relative_vi=0
        )
        t1_sc = time.perf_counter()
        tn_sc = t1_sc - t0_sc

        det_pol = self.policy_indices_sc[self.policy_sc == 1.]

        u1, u2 = self.opt_solver.get_uvectors()
        u1_sc, u2_sc = self.scevi.get_uvectors()
        assert np.allclose(u1, u1_sc)
        assert np.allclose(u2, u2_sc)
        assert np.isclose(span_value, span_value_sc), '{} != {}'.format(span_value, span_value_sc)
        assert np.allclose(self.policy_indices, det_pol), '{}\n{}'.format(self.policy_indices_sc, self.policy_sc)

        ##########################################################################################

        if self.verbose > 1:
            self.logger.info("[%d]NEW EVI: %.3f seconds" % (self.episode, tn))
            if self.verbose > 2:
                self.logger.info(self.policy_indices)

        self.solver_times.append(tn)
        return span_value


@pytest.mark.parametrize("seed_0", [12345, 5687, 2363, 8939, 36252, 38598125])
@pytest.mark.parametrize("operator_type", ['T', 'N'])
def test_srevi(seed_0, operator_type):
    env = Toy3D_1(delta=0.99)
    r_max = max(1, np.asscalar(np.max(env.R_mat)))

    nb_simulations = 1
    duration = 50000
    regret_time_steps = 1
    np.random.seed(seed_0)
    random.seed(seed_0)
    seed_sequence = [random.randint(0, 2 ** 30) for _ in range(nb_simulations)]

    for rep in range(nb_simulations):
        seed = seed_sequence[rep]  # set seed
        np.random.seed(seed)
        random.seed(seed)

        ofualg = TUCRL(
            env,
            r_max=r_max,
            alpha_r=1,
            alpha_p=1,
            verbose=1,
            bound_type_p="chernoff",
            bound_type_rew="chernoff",
            random_state=seed)  # learning algorithm

        alg_desc = ofualg.description()
        print(alg_desc)

        ofualg.learn(duration, regret_time_steps)  # learn task

        np.random.seed(seed)
        random.seed(seed)
        env.reset()
        ucrl = Ucrl.UcrlMdp(
            env,
            r_max=r_max,
            alpha_r=1,
            alpha_p=1,
            verbose=1,
            bound_type_p="chernoff",
            bound_type_rew="chernoff",
            random_state=seed)  # learning algorithm
        ucrl.learn(duration, regret_time_steps)  # learn task

        np.random.seed(seed)
        random.seed(seed)
        env.reset()
        scucrl = SCAL(
            env,
            r_max=r_max,
            alpha_r=1,
            alpha_p=1,
            verbose=1,
            bound_type_p="chernoff",
            bound_type_rew="chernoff",
            augment_reward=False,
            random_state=seed,
            span_constraint=np.inf,
            relative_vi=True,
            operator_type=operator_type)  # learning algorithm
        scucrl.learn(duration, regret_time_steps)  # learn task

        assert np.allclose(ucrl.regret, scucrl.regret)
        assert np.allclose(ucrl.regret_unit_time, scucrl.regret_unit_time)
        assert np.allclose(ucrl.unit_duration, scucrl.unit_duration)
        assert np.allclose(ucrl.span_values, scucrl.span_values)
        assert np.allclose(ucrl.span_times, scucrl.span_times)
        assert np.allclose(ucrl.estimated_rewards, scucrl.estimated_rewards)
        assert np.allclose(ucrl.estimated_holding_times, scucrl.estimated_holding_times)
        assert np.allclose(ucrl.P, scucrl.P)
        assert np.allclose(ucrl.nb_observations, scucrl.nb_observations)


if __name__ == '__main__':
    test_srevi(seed_0=3179336,operator_type='N')
