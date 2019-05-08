import numpy as np
from ..Environment import Environment
from ...evi import EVI
from .. import RewardDistributions as Rdists
from ...utils.shortestpath import dpshortestpath
import gym
from gym import spaces


class RiverSwim(Environment):
    """
    Osband, Van Roy, Russo. (More) Efficient Reinforcement Learning via Posterior Sampling. NIPS

    actions = [0: left, 1: right]
    """

    def __init__(self, n_states=6, stochastic_reward=False):
        na = 2

        self.r_max = 1.

        self.P_mat = np.zeros((n_states, na, n_states))
        self.R_mat = np.zeros((n_states, na))

        state_actions = []
        for s in range(n_states):
            state_actions.append([0, 1])

            if s == 0:
                self.P_mat[0, 0, 0] = 1
                self.P_mat[0, 1, 0] = 0.4
                self.P_mat[0, 1, 1] = 0.6
                self.R_mat[0, 0] = 5. / 1000.
            elif s == n_states - 1:
                self.P_mat[s, 0, s - 1] = 1
                self.P_mat[s, 1, s - 1] = 0.4
                self.P_mat[s, 1, s] = 0.6
                self.R_mat[s, 1] = self.r_max
            else:
                self.P_mat[s, 0, s - 1] = 1
                self.P_mat[s, 1, s - 1] = 0.05
                self.P_mat[s, 1, s] = 0.6
                self.P_mat[s, 1, s + 1] = 0.35

        self.reward_distributions = [Rdists.BernouilliReward(r_max=1, proba=self.R_mat[0, 0]),
                                     Rdists.BernouilliReward(r_max=self.r_max, proba=0.9 / self.r_max)]
        self.stochastic_reward = stochastic_reward

        super(RiverSwim, self).__init__(initial_state=0, state_actions=state_actions)
        self.holding_time = 1

    def reset(self):
        self.state = 0

    def compute_max_gain(self):
        if not hasattr(self, "max_gain"):
            na = max(map(len, self.state_actions))
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
                           estimated_holding_times=np.ones((self.nb_states, na)),
                           beta_p=np.zeros((self.nb_states, na, self.nb_states)),
                           beta_r=np.zeros((self.nb_states, na)),
                           beta_tau=np.zeros((self.nb_states, na)),
                           tau_max=1, tau_min=1, tau=1,
                           r_max=1.,
                           epsilon=1e-8,
                           initial_recenter=1, relative_vi=0
                           )
            u1, u2 = evi.get_uvectors()
            self.span = span
            self.max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
            self.optimal_policy_indices = policy_indices
            self.optimal_policy = policy

            print("solved model")

            self.diameter = dpshortestpath(self.P_mat, self.state_actions)

        return self.max_gain

    def execute(self, action):
        try:
            action_index = self.state_actions[self.state].index(action)
        except:
            raise ValueError("Action {} cannot be executed in this state {}".format(action, self.state))
        p = self.P_mat[self.state, action_index]
        next_state = np.asscalar(np.random.choice(self.nb_states, 1, p=p))
        assert p[next_state] > 0.
        self.reward = self.R_mat[self.state, action_index]
        if self.stochastic_reward:
            ns = len(self.state_actions)
            if self.state == 0 and action_index == 0:
                self.reward = self.reward_distributions[0].generate()
            elif self.state == ns - 1 and action_index == 1:
                self.reward = self.reward_distributions[1].generate()
        self.state = next_state

    def true_reward(self, state, action_idx):
        return self.R_mat[state, action_idx]

    def description(self):
        desc = {
            'name': type(self).__name__,
            'nstates': len(self.state_actions)
        }
        return desc

    def properties(self):
        self.compute_max_gain()
        props = {
            'gain': self.max_gain,
            'diameter': self.diameter,
            'bias_span': self.span
        }
        return props


class ContinuousRiverSwim(gym.Env):

    def __init__(self, max_pos=6, dx=0.1):
        self.max_pos = float(max_pos)
        self.min_pos = 0.
        self.dx = dx

        self.observation_space = spaces.Box(low=np.array([self.min_pos]), high=np.array([max_pos]))
        self.action_space = spaces.Discrete(2)

        self.pos = 0.
        self.reward = None

        self.seed()
        self.reset()

    def reset(self, state=None):
        self.pos = 0
        return np.array([self.pos])

    def step(self, action):

        DX = self.dx + np.random.randn()*0.001

        reward = 0.
        if action == 0:
            # left (follow the flow)
            next_pos = max(0., self.pos - DX)
            if np.isclose(self.pos, self.min_pos):
                reward = 0.01
        elif action == 1:
            if np.isclose(self.pos, self.min_pos):
                reward = 0.
                poses = [self.min_pos, min(self.max_pos, self.pos + DX)]
                proba = [0.4, 0.6]
            elif np.isclose(self.pos, self.max_pos):
                reward = 1.
                poses = [max(self.min_pos, self.pos - DX), self.max_pos]
                proba = [0.4, 0.6]
            else:
                reward = 0.
                poses = [max(self.min_pos, self.pos - DX),
                         self.pos,
                         min(self.max_pos, self.pos + DX)]
                proba = [0.05, 0.6, 0.35]
            # if self.pos > self.max_pos - 2*self.dx:
            #     reward = 1.
            next_pos = np.asscalar(np.random.choice(a=poses, size=1, p=proba))
        else:
            raise ValueError("Action {} is not valid".format(action))

        self.pos = next_pos
        self.reward = reward
        done = False
        return np.array([self.pos]), self.reward, done, {}
