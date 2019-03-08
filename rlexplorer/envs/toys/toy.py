import numpy as np
from ..Environment import Environment
from ...evi import EVI
from .. import RewardDistributions as Rdists
from ...utils.shortestpath import dpshortestpath


class Toy3D_1(Environment):
    def __init__(self, delta=0.99, stochastic_reward=False,
                 uniform_reward=False, uniform_range=0.1):
        state_actions = [[0], [0], [0, 1]]
        na = max(map(len, state_actions))
        ns = len(state_actions)
        self.delta = delta

        self.r_max = 1.

        self.P_mat = -np.Inf * np.ones((ns, na, ns))
        self.R_mat = -np.Inf * np.ones((ns, na))

        self.P_mat[0, 0, 0] = 0.
        self.P_mat[0, 0, 1] = delta
        self.P_mat[0, 0, 2] = 1 - delta
        self.R_mat[0, 0] = 0.0

        self.P_mat[1, 0, 0] = 1.
        self.P_mat[1, 0, 1] = 0.
        self.P_mat[1, 0, 2] = 0.
        self.R_mat[1, 0] = self.r_max / 3.

        self.P_mat[2, 0, 0] = 1. - delta
        self.P_mat[2, 0, 1] = delta
        self.P_mat[2, 0, 2] = 0.
        self.R_mat[2, 0] = 2. * self.r_max / 3.
        self.P_mat[2, 1, 0] = 0.
        self.P_mat[2, 1, 1] = 0.
        self.P_mat[2, 1, 2] = 1.
        self.R_mat[2, 1] = 2. * self.r_max / 3.

        if uniform_reward:
            epsilon = uniform_range / 2
            self.reward_distributions = [[Rdists.ConstantReward(c=self.R_mat[0, 0])],
                                         [Rdists.UniformReward(a=self.R_mat[1, 0] - epsilon, b=self.R_mat[1, 0] + epsilon)],
                                         [Rdists.UniformReward(a=self.R_mat[2, 0] - epsilon, b=self.R_mat[2, 0] + epsilon),
                                          Rdists.UniformReward(a=self.R_mat[2, 1] - epsilon, b=self.R_mat[2, 1] + epsilon)]]
        else:
            self.reward_distributions = [[Rdists.ConstantReward(c=self.R_mat[0, 0])],
                                         [Rdists.BernouilliReward(r_max=self.r_max, proba=self.R_mat[1, 0] / self.r_max)],
                                         [Rdists.BernouilliReward(r_max=self.r_max, proba=self.R_mat[2, 0] / self.r_max),
                                          Rdists.BernouilliReward(r_max=self.r_max, proba=self.R_mat[2, 1] / self.r_max)]]
        self.stochastic_reward = stochastic_reward
        self.uniform_reward = uniform_reward
        self.uniform_range = uniform_range

        super(Toy3D_1, self).__init__(initial_state=0,
                                      state_actions=state_actions)
        self.holding_time = 1

    def reset(self):
        self.state = 0

    def compute_max_gain(self):
        if not hasattr(self, "max_gain"):
            policy_indices = np.ones(self.nb_states, dtype=np.int)
            policy = np.ones(self.nb_states, dtype=np.int)

            print("creating model")
            evi = EVI(nb_states=self.nb_states,
                      actions_per_state=self.state_actions,
                      random_state=123456)

            print("running model")
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

            print("solved model")

            if self.delta > 0:
                self.diameter = dpshortestpath(self.P_mat, self.state_actions)
            else:
                self.diameter = np.inf

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

    def true_reward(self, state, action):
        return self.R_mat[state, action]

    def description(self):
        desc = {
            'name': type(self).__name__,
            'delta': self.delta,
            'stoch_reward': self.stochastic_reward,
            'uniform_reward': self.uniform_reward,
            'uniform_range': self.uniform_range
        }
        return desc


class Toy3D_2(Toy3D_1):

    def __init__(self, delta=0.005, epsilon=0.001, stochastic_reward=False):
        super(Toy3D_2, self).__init__(delta=delta, stochastic_reward=stochastic_reward)
        self.epsilon = epsilon
        self.P_mat[1, 0, 0] = epsilon
        self.P_mat[1, 0, 1] = 1. - epsilon

        self.R_mat[1, 0] = 3. * self.r_max / 4.
        self.reward_distributions[1][0] = Rdists.BernouilliReward(r_max=self.r_max, proba=self.R_mat[1, 0] / self.r_max)

        del self.max_gain
        self.compute_max_gain()
