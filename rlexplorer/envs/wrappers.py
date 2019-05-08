import numpy as np
from ..evi.evi import EVI
from .Environment import Environment
from ..utils.shortestpath import dpshortestpath
from ..utils.discretization import Discretizer
from gym.envs.toy_text.taxi import TaxiEnv
from gym.envs.classic_control import MountainCarEnv, CartPoleEnv
from .mountaincar import MountainCar
from .toys.riverswim import ContinuousRiverSwim
from .puddleworld import PuddleWorld


class GymDiscreteEnvWrapper(Environment):
    def __init__(self, env):
        self.gym_env = env
        nS = self.gym_env.nS
        nA = self.gym_env.nA
        state_actions = [list(range(nA)) for _ in range(nS)]
        super(GymDiscreteEnvWrapper, self).__init__(state_actions=state_actions, initial_state=None)
        self.reset()

        self.holding_time = 1

    def reset(self):
        self.state = self.gym_env.reset()

    def execute(self, action):
        done = False
        next_state, reward, done, _ = self.gym_env.step(action)
        # bound reward between [0,1]
        self.reward = (reward - self.minr) / (self.maxr - self.minr)
        if done:
            self.state = self.gym_env.reset()
        else:
            self.state = next_state

    def compute_matrix_form(self):
        Pgym = self.gym_env.P
        nS = self.gym_env.nS
        nA = self.gym_env.nA
        isd = self.gym_env.isd
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))
        minr, maxr = np.inf, -np.inf
        for s in range(nS):
            for a in range(nA):
                items = Pgym[s][a]
                for item in items:
                    probability, nextstate, reward, done = item
                    maxr = max(reward, maxr)
                    minr = min(reward, minr)
                    P[s, a, nextstate] = probability
                    R[s, a] += probability * reward
                    if done:
                        P[s, a] = isd
                tot = np.sum(P[s, a])
                assert np.isclose(tot, 1)

        R = (R - minr) / (maxr - minr)
        self.maxr = maxr
        self.minr = minr
        return self.state_actions, P, R

    def compute_max_gain(self):
        if not hasattr(self, "max_gain"):
            state_actions, P, R = self.compute_matrix_form()
            nS = self.gym_env.nS
            nA = self.gym_env.nA
            policy_indices = np.ones(nS, dtype=np.int)
            policy = np.ones(nS, dtype=np.int)

            evi = EVI(nb_states=nS,
                      actions_per_state=state_actions,
                      bound_type="hoeffding",
                      random_state=123456)

            span = evi.run(policy_indices=policy_indices,
                           policy=policy,
                           estimated_probabilities=P,
                           estimated_rewards=R,
                           estimated_holding_times=np.ones((nS, nA)),
                           beta_p=np.zeros((nS, nA, 1)),
                           beta_r=np.zeros((nS, nA)),
                           beta_tau=np.zeros((nS, nA)),
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

        return self.max_gain

    def compute_diameter(self):
        if not hasattr(self, "diameter"):
            state_actions, P, R = self.compute_matrix_form()
            diameter = dpshortestpath(P, state_actions)
            self.diameter = diameter
        return self.diameter

    def description(self):
        desc = {
            'name': type(self.gym_env).__name__
        }
        return desc

    def properties(self):
        self.compute_max_gain()
        props = {
            'gain': self.max_gain,
            'diameter': np.inf,
            'bias_span': self.span
        }
        return props


class GymTaxiCom(Environment):
    def __init__(self, taxi_env):
        assert isinstance(taxi_env, TaxiEnv)
        self.gym_env = taxi_env
        nS = self.gym_env.nS - 100
        nA = self.gym_env.nA
        state_actions = [list(range(nA)) for _ in range(nS)]
        super(GymTaxiCom, self).__init__(state_actions=state_actions, initial_state=None)
        self.reset()

        self.fictitious_state = 0
        self.holding_time = 1
        self.precomputed_diameter = 72.02269121380132

    def convert_fictitious_real_state(self, fictitious_state):
        return fictitious_state - fictitious_state // 5 - 1

    def convert_real_fictitious_state(self, real_state):
        return real_state + real_state // 4 + 1

    def reset(self):
        self.fictitious_state = self.gym_env.reset()
        self.state = self.convert_fictitious_real_state(self.fictitious_state)

    def execute(self, action):
        done = False
        next_ficticious_state, reward, done, _ = self.gym_env.step(action)
        # bound reward between [0,1]
        self.reward = (reward - self.minr) / (self.maxr - self.minr)
        if done:
            self.fictitious_state = self.gym_env.reset()
            self.state = self.convert_fictitious_real_state(self.fictitious_state)
        else:
            self.fictitious_state = next_ficticious_state
            self.state = self.convert_fictitious_real_state(self.fictitious_state)

    def compute_matrix_form(self):
        Pgym = self.gym_env.P
        nS = self.gym_env.nS - 100
        nA = self.gym_env.nA
        isd = self.gym_env.isd
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))
        minr, maxr = np.inf, -np.inf
        for s in range(nS):
            for a in range(nA):
                items = Pgym[self.convert_real_fictitious_state(s)][a]
                for item in items:
                    probability, fictitious_nextstate, reward, done = item
                    maxr = max(reward, maxr)
                    minr = min(reward, minr)
                    P[s, a, self.convert_fictitious_real_state(fictitious_nextstate)] = probability
                    R[s, a] += probability * reward
                    if done:
                        P[s, a] = np.delete(isd, range(0, 500, 5))
                tot = np.sum(P[s, a])
                assert np.isclose(tot, 1)

        R = (R - minr) / (maxr - minr)
        self.maxr = maxr
        self.minr = minr
        return self.state_actions, P, R

    def compute_max_gain(self):
        if not hasattr(self, "max_gain"):
            state_actions, P, R = self.compute_matrix_form()
            nS = self.gym_env.nS - 100
            nA = self.gym_env.nA
            policy_indices = np.ones(nS, dtype=np.int)
            policy = np.ones(nS, dtype=np.int)

            evi = EVI(nb_states=nS,
                      actions_per_state=state_actions,
                      bound_type="hoeffding",
                      random_state=123456)

            TAU = 0.9
            span = evi.run(policy_indices=policy_indices,
                           policy=policy,
                           estimated_probabilities=P,
                           estimated_rewards=R,
                           estimated_holding_times=np.ones((nS, nA)),
                           beta_p=np.zeros((nS, nA, 1)),
                           beta_r=np.zeros((nS, nA)),
                           beta_tau=np.zeros((nS, nA)),
                           tau_max=1, tau_min=1, tau=TAU,
                           r_max=1.,
                           epsilon=1e-12,
                           initial_recenter=1, relative_vi=0
                           )
            u1, u2 = evi.get_uvectors()
            self.span = span * TAU
            self.max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
            self.optimal_policy_indices = policy_indices
            self.optimal_policy = policy

        return self.max_gain

    def compute_diameter(self):
        if not hasattr(self, "diameter"):
            state_actions, P, R = self.compute_matrix_form()
            diameter = dpshortestpath(P, state_actions)
            self.diameter = diameter
        return self.diameter

    def description(self):
        desc = {
            'name': "{}_comm".format(type(self.gym_env).__name__)
        }
        return desc

    def properties(self):
        self.compute_max_gain()
        props = {
            'gain': self.max_gain,
            'diameter': self.precomputed_diameter,
            'bias_span': self.span
        }
        return props


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# CONTINUOUS STATE MDPS
# ----------------------------------------------------------------------------------------------------------------------


class GymContinuousWrapper:

    def __init__(self, gym_env, grid, min_reward, max_reward,
                 reward_done=None,
                 max_gain=None,
                 bias_span=None,
                 diameter=None
                 ):
        self.gym_env = gym_env
        self.grid = grid
        self.min_reward = min_reward
        self.max_reward = max_reward

        nS = self.grid.n_bins()
        nA = self.gym_env.action_space.n
        self.state_actions = [list(range(nA)) for _ in range(nS)]
        self.holding_time = 1
        self.nb_states = nS
        self.max_nb_actions_per_state = max(map(len, self.state_actions))
        self.state = None
        self.reward = None
        self.max_gain = max_gain
        self.reward_done = reward_done
        self.bias_span = bias_span
        self.diameter = diameter
        self.done = False
        self.reset()
        self.max_branching = nS

    def reset(self):
        next_state = self.gym_env.reset()
        self.state = np.asscalar(self.grid.dpos(next_state))

    def execute(self, action):
        next_state, reward, done, _ = self.gym_env.step(action)
        self.reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        self.done = done
        if done:
            self.reset()
            if self.reward_done is not None:
                self.reward = self.reward_done
        else:
            self.state = np.asscalar(self.grid.dpos(next_state))

    def compute_max_gain(self):
        return self.max_gain

    def get_state_actions(self):
        return self.state_actions

    def description(self):
        desc = {
            'name': type(self.gym_env).__name__
        }
        return desc

    def properties(self):
        self.compute_max_gain()
        props = {
            'gain': self.max_gain,
            'diameter': self.diameter,
            'bias_span': self.bias_span
        }
        return props


class GymMountainCarWr(GymContinuousWrapper):

    def __init__(self, Nbins, seed=None):
        self.Nbins = Nbins
        env = MountainCar()
        env.seed(seed=seed)
        LM = env.observation_space.low
        HM = env.observation_space.high
        D = len(HM)

        bins = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])
        grid = Discretizer(bins=bins)
        super(GymMountainCarWr, self).__init__(
            gym_env=env, grid=grid,
            min_reward=-1, max_reward=0,
            reward_done=1,
            max_gain=0,  # 0.009054722111333291,
            bias_span=1.0597499799932173,
            diameter=None
        )

    def reset(self):
        next_state = self.gym_env.reset()
        # next_state = np.array([self.gym_env.np_random.uniform(low=-1.2, high=0.45), self.gym_env.np_random.uniform(low=-0.06, high=0.06)])
        # self.gym_env.state = next_state
        self.state = np.asscalar(self.grid.dpos(next_state))


class PuddleWorldWr(GymContinuousWrapper):

    def __init__(self, Nbins, seed=None):
        self.Nbins = Nbins
        env = PuddleWorld()
        env.seed(seed=seed)
        LM = env.observation_space.low
        HM = env.observation_space.high
        D = len(HM)

        bins = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])
        grid = Discretizer(bins=bins)
        super(PuddleWorldWr, self).__init__(
            gym_env=env, grid=grid,
            min_reward=env.reward_range[0], max_reward=env.reward_range[1],
            reward_done=1,
            max_gain=0,
            bias_span=0.,
            diameter=None
        )

    def reset(self):
        next_state = self.gym_env.reset()
        self.state = np.asscalar(self.grid.dpos(next_state))

class GymCartPoleWr(GymContinuousWrapper):

    def __init__(self, Nbins, seed=None):
        self.Nbins = Nbins
        env = CartPoleEnv()
        env.seed(seed=seed)
        HM = np.array([1.5 * env.x_threshold, 2., 1.5 * env.theta_threshold_radians, 3.])
        LM = -HM
        D = len(HM)

        bins = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])
        grid = Discretizer(bins=bins)
        super(GymCartPoleWr, self).__init__(
            gym_env=env, grid=grid,
            min_reward=0, max_reward=1,
            reward_done=0,
            max_gain=0.99999994512314,
            bias_span=1.0641535523,
            diameter=None
        )


class ContRiverSwimWr(GymContinuousWrapper):

    def __init__(self, Nbins, max_pos=6., dx=0.1, seed=None):
        self.Nbins = Nbins
        env = ContinuousRiverSwim(max_pos=max_pos, dx=dx)
        env.seed(seed=seed)
        LM = env.observation_space.low
        HM = env.observation_space.high
        D = len(HM)

        bins = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])
        grid = Discretizer(bins=bins)
        super(ContRiverSwimWr, self).__init__(
            gym_env=env, grid=grid,
            min_reward=0, max_reward=1,
            reward_done=0,
            max_gain=0., #0.3377335,
            bias_span=0.,
            diameter=None
        )

# class GymMCWrapper2:
#     def __init__(self, N):
#         self.Nbins = N
#         self.gym_env = MountainCarEnv()
#
#         minx, minv = self.gym_env.observation_space.low.tolist()
#         maxx, maxv = self.gym_env.observation_space.high.tolist()
#         x_bins = np.linspace(minx, maxx, N)
#         y_bins = np.linspace(minv, maxv, N)
#
#         self.grid = Discretizer(bins=[x_bins[1:-1], y_bins[1:-1]])
#
#         nS = self.grid.n_bins()
#         nA = self.gym_env.action_space.n
#         self.state_actions = [list(range(nA)) for _ in range(nS)]
#         self.holding_time = 1
#         self.nb_states = nS
#         self.max_nb_actions_per_state = max(map(len, self.state_actions))
#         self.state = None
#         self.reward = None
#         self.max_gain = 0
#         self.reset()
#         self.minr = -1
#         self.maxr = 0
#
#     def reset(self):
#         next_state = np.array([np.random.uniform(low=-1, high=0.4), 0])
#         self.gym_env.state = next_state.copy()
#         next_state = self.gym_env.reset()
#         self.state = np.asscalar(self.grid.dpos(next_state))
#
#     def execute(self, action):
#         done = False
#         next_state, reward, done, _ = self.gym_env.step(action)
#         # bound reward between [0,1]
#         self.reward = (reward - self.minr) / (self.maxr - self.minr)
#         if done:
#             self.reset()
#             self.reward = 1
#         else:
#             self.state = np.asscalar(self.grid.dpos(next_state))
#
#     def compute_max_gain(self):
#         self.max_gain = 0.009054722111333291
#         self.bias_span = 1.0597499799932173
#         return self.max_gain
#
#     def get_state_actions(self):
#         return self.state_actions
#
#     def description(self):
#         desc = {
#             'name': type(self.gym_env).__name__
#         }
#         return desc
#
#     def properties(self):
#         self.compute_max_gain()
#         props = {
#             'gain': self.max_gain,
#             'diameter': "!!!",
#             'bias_span': self.bias_span
#         }
#         return props
