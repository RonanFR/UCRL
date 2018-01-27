import numpy as np
from ..Environment import Environment
from ...evi import EVI
from ...utils.shortestpath import dijkstra
from .. import RewardDistributions as Rdists

class Toy3D_1(Environment):
    def __init__(self, delta=0.99, stochastic_reward=False):
        state_actions = [[0], [0], [0, 1]]
        na = max(map(len, state_actions))
        ns = len(state_actions)

        self.P_mat = -np.Inf * np.ones((ns, na, ns))
        self.R_mat = -np.Inf * np.ones((ns, na))

        self.P_mat[0, 0, 0] = 0.
        self.P_mat[0, 0, 1] = delta
        self.P_mat[0, 0, 2] = 1 - delta
        self.R_mat[0, 0] = 0.

        self.P_mat[1, 0, 0] = 1.
        self.P_mat[1, 0, 1] = 0.
        self.P_mat[1, 0, 2] = 0.
        self.R_mat[1, 0] = 0.

        self.P_mat[2, 0, 0] = 1. - delta
        self.P_mat[2, 0, 1] = delta
        self.P_mat[2, 0, 2] = 0.
        self.R_mat[2, 0] = 0.5
        self.P_mat[2, 1, 0] = 0.
        self.P_mat[2, 1, 1] = 0.
        self.P_mat[2, 1, 2] = 1.
        self.R_mat[2, 1] = 0.5

        self.reward_distributions = [[Rdists.BernouilliReward(r_max=self.R_mat[0, 0] / 0.6, proba=0.6)],
                                     [Rdists.BernouilliReward(r_max=self.R_mat[1, 0] / 0.5, proba=0.3)],
                                     [Rdists.BernouilliReward(r_max=self.R_mat[2, 0] / 0.4, proba=0.4),
                                      Rdists.BernouilliReward(r_max=self.R_mat[2, 1] / 0.8, proba=0.8)]]
        self.stochastic_reward = stochastic_reward

        super(Toy3D_1, self).__init__(initial_state=0,
                                      state_actions=state_actions)
        self.holding_time = 1
        self.delta = delta

    def reset(self):
        self.state = 0

    def compute_max_gain(self):
        if not hasattr(self, "max_gain"):
            policy_indices = np.ones(self.nb_states, dtype=np.int)
            policy = np.ones(self.nb_states, dtype=np.int)

            evi = EVI(nb_states=self.nb_states,
                      actions_per_state=self.state_actions,
                      bound_type="chernoff",
                      random_state=123456)

            span = evi.run(policy_indices=policy_indices,
                           policy=policy,
                           estimated_probabilities=self.P_mat,
                           estimated_rewards=self.R_mat,
                           estimated_holding_times=np.ones((self.nb_states, 4)),
                           beta_p=np.zeros((self.nb_states, 2, 1)),
                           beta_r=np.zeros((self.nb_states, 2)),
                           beta_tau=np.zeros((self.nb_states, 2)),
                           tau_max=1, tau_min=1, tau=1,
                           r_max=1.,
                           epsilon=1e-12,
                           initial_recenter=1, relative_vi=0
                           )
            u1, u2 = evi.get_uvectors()
            self.span = span
            self.max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
            self.optimal_policy_indices = policy_indices
            self.optimal_policy = policy

            diameter = -1
            for s in range(self.nb_states):
                dist, _ = dijkstra(self.P_mat, self.state_actions, s)
                diameter = max(diameter, np.max(dist))
            self.diameter = diameter

        return self.max_gain

    def execute(self, action):
        try:
            action_index = self.state_actions[self.state].index(action)
        except:
            raise ValueError("Action {} cannot be executed in this state {} [{}, {}]".format(action, self.state))
        p = self.P_mat[self.state, action_index]
        next_state = np.asscalar(np.random.choice(self.nb_states, 1, p=p))
        assert p[next_state] > 0.
        rdist = self.reward_distributions[self.state][action_index]
        self.reward = rdist.mean if not self.stochastic_reward else rdist.generate()
        self.state = next_state

    def description(self):
        desc = {
            'name': type(self).__name__,
            'delta': self.delta
        }
        return desc
