import numpy as np
from ..evi.evi import EVI
from .Environment import Environment
from ..utils.shortestpath import dpshortestpath
from ..utils.discretization import Discretizer, UniformDiscWOB
from gym.envs.toy_text.taxi import TaxiEnv
from gym.envs.classic_control import MountainCarEnv, CartPoleEnv
from .mountaincar import MountainCar
from .shipsteering import ShipSteering
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
            self.state = self.grid.dpos(next_state)

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
        if Nbins == 16:
            TOREMOVE = [0,11,12,13,14,27,28,29,43,44,59,74,89,104,119,120,134,135,149,150,151,164,165,166,167,178,179,180,181,182,183,193,194,195,196,197,198,199,207,208,209,210,211,212,213,214,215,222,223,224]
            grid = UniformDiscWOB(bins=bins, binstoremove=TOREMOVE)
        elif Nbins == 25:
            TOREMOVE = [0,1,16,17,18,19,20,21,22,23,24,42,43,44,45,46,47,48,67,68,69,70,71,92,93,94,95,117,118,119,141,142,143,166,167,190,191,215,239,263,264,287,288,311,312,313,334,335,336,337,358,359,360,361,362,382,383,384,385,386,387,405,406,407,408,409,410,411,412,429,430,431,432,433,434,435,436,437,452,453,454,455,456,457,458,459,460,461,462,476,477,478,479,480,481,482,483,484,485,486,487,500,501,502,503,504,505,506,507,508,509,510,511,512,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575]
            grid = UniformDiscWOB(bins=bins, binstoremove=TOREMOVE)
        elif Nbins == 30:
            TOREMOVE =[0,19,20,21,22,23,24,25,26,27,28,29,50,51,52,53,54,55,56,57,80,81,82,83,84,85,86,110,111,112,113,114,115,140,141,142,143,144,170,171,172,173,200,201,202,229,230,231,259,260,288,289,318,34,376,405,406,434,435,463,464,465,492,493,494,521,522,523,524,549,550,551,552,553,554,578,579,580,581,582,583,584,606,607,608,609,610,611,612,613,614,635,636,637,638,639,640,641,642,643,644,664,665,666,667,668,669,670,671,672,673,674,692,693,694,695,696,697,698,699,700,701,702,703,704,721,722,723,724,725,726,727,728,729,730,731,732,733,734,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840]
            grid = UniformDiscWOB(bins=bins, binstoremove=TOREMOVE)
        else:
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
        self.state = self.grid.dpos(next_state)


class ShipSteeringWr(GymContinuousWrapper):

    def __init__(self, Nbins, seed=None):
        self.Nbins = Nbins
        env = ShipSteering()
        env.seed(seed=seed)
        LM = env.observation_space.low
        HM = env.observation_space.high
        D = len(HM)

        bins = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])
        grid = Discretizer(bins=bins)
        super(ShipSteeringWr, self).__init__(
            gym_env=env, grid=grid,
            min_reward=env.reward_range[0], max_reward=env.reward_range[1],
            reward_done=1,
            max_gain=0,
            bias_span=0,
            diameter=None
        )

    def reset(self):
        next_state = self.gym_env.reset()
        self.state = self.grid.dpos(next_state)


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
            max_gain=0.,  # 0.3377335,
            bias_span=0.,
            diameter=None
        )
