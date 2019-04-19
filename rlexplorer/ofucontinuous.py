import numpy as np
from .rllogging import default_logger
from .scal import SCALPLUS
from scipy import stats
from scipy.interpolate import CubicSpline
from gym.envs.classic_control import MountainCarEnv
import math as m


def discretization_2d(x, y, binx, biny):
    _, _, _, binid = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])
    return binid


class GymMCWrapper:

    def __init__(self, N):
        self.gym_env = MountainCarEnv()

        minx, minv = self.gym_env.observation_space.low.tolist()
        maxx, maxv = self.gym_env.observation_space.high.tolist()
        x_bins = np.linspace(minx, maxx, N)
        y_bins = np.linspace(minv, maxv, N)
        xv, yv = np.meshgrid(x_bins, y_bins)
        total_bins = discretization_2d(xv.reshape(-1), yv.reshape(-1), x_bins, y_bins)
        self.x_bins = np.linspace(minx, maxx, N)
        self.y_bins = np.linspace(minv, maxv, N)

        upper_limits = np.array(
            [[-1.2, 0],
             [-1.198, 0.00324189604083537],
             [-1.19675810395916, 0.00648379208167074],
             [-1.19027431187749, 0.0097363414639057],
             [-1.18053797041359, 0.0130095573472062],
             [-1.16752841306638, 0.0163121891363951],
             [-1.15121622392999, 0.0196510558873098],
             [-1.13156516804268, 0.0230303167657155],
             [-1.10853485127696, 0.0264506670188354],
             [-1.08208418425813, 0.0299084608395493],
             [-1.05217572341858, 0.0333947812151384],
             [-1.01878094220344, 0.0368945024205852],
             [-0.981886439782853, 0.0403854235047771],
             [-0.941501016278076, 0.0438375894881081],
             [-0.897663426789968, 0.0472129567621402],
             [-0.850450470027828, 0.0504655921176023],
             [-0.799984877910226, 0.0535426082086968],
             [-0.746442269701529, 0.0563860158875415],
             [-0.690056253813987, 0.0589355995803992],
             [-0.631120654233588, 0.0611327863399545],
             [-0.569987867893634, 0.0629252886277079],
             [-0.507062579265926, 0.0642720848542332],
             [-0.442790494411693, 0.0651481142452053],
             [-0.377642380166487, 0.0655479710753998],
             [-0.312094409091087, 0.0654879444695967],
             [-0.246606464621491, 0.0650059808339361],
             [-0.181600483787555, 0.0641595050158886],
             [-0.117440978771666, 0.0630214295030634],
             [-0.0544195492686026, 0.0616749953796823],
             [0.00725544611107969, 0.0602082381776891],
             [0.0674636842887688, 0.0587088303711636],
             [0.126172514659932, 0.0572598585015724],
             [0.183432373161505, 0.0559368247810568],
             [0.239369197942562, 0.0548059018213784],
             [0.29417509976394, 0.0539232713067886],
             [0.348098371070729, 0.0533352643859802],
             [0.401433635456709, 0.0530789856973222],
             [0.454512621154031, 0.0531831211908513],
             [0.507695742344882, 0.0536686766603606]]
        )
        cs = CubicSpline(upper_limits[:, 0], upper_limits[:, 1])

        idxs = discretization_2d(upper_limits[:, 0], upper_limits[:, 1], self.x_bins, self.y_bins)

        xv, yv = np.meshgrid(np.linspace(minx, maxx, 3 * N), np.linspace(minv, maxv, 3 * N))
        M = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1)), axis=1)
        M = M[M[:, 1] >= cs(M[:, 0])]
        idxs2 = discretization_2d(M[:, 0], M[:, 1], self.x_bins, self.y_bins)

        ff = np.setdiff1d(idxs2, idxs) # non reachable set
        valid_bins = np.setdiff1d(total_bins, ff)

        self.map_bins = {}
        for counter, id in enumerate(valid_bins):
            self.map_bins[id] = counter

        nS = len(valid_bins)
        nA = self.gym_env.action_space.n
        self.state_actions = [list(range(nA)) for _ in range(nS)]
        self.holding_time = 1
        self.nb_states = nS
        self.reset()
        self.max_nb_actions_per_state = max(map(len, self.state_actions))
        self.state = None
        self.reward = None

    def reset(self):
        next_state = self.gym_env.reset()
        self.state = discretization_2d([next_state[0]], [next_state[1]], self.x_bins, self.y_bins)

    def execute(self, action):
        next_state, self.reward = self.gym_env.step(action)
        self.state = discretization_2d([next_state[0]], [next_state[1]], self.x_bins, self.y_bins)

    def compute_max_gain(self):
        self.env.compute_max_gain()


class GymMCWrapper2:
    def __init__(self, N):
        self.gym_env = MountainCarEnv()

        minx, minv = self.gym_env.observation_space.low.tolist()
        maxx, maxv = self.gym_env.observation_space.high.tolist()
        self.x_bins = np.linspace(minx, maxx, N)
        self.y_bins = np.linspace(minv, maxv, N)

        nS = (len(self.x_bins) -1) * (len(self.y_bins) -1)
        nA = self.gym_env.action_space.n
        self.state_actions = [list(range(nA)) for _ in range(nS)]
        self.holding_time = 1
        self.nb_states = nS
        self.max_nb_actions_per_state = max(map(len, self.state_actions))
        self.state = None
        self.reward = None
        self.max_gain = 0
        self.reset()

    def reset(self):
        next_state = self.gym_env.reset()
        self.state = np.asscalar(discretization_2d([next_state[0]], [next_state[1]], self.x_bins, self.y_bins))

    def execute(self, action):
        next_state, self.reward, _, _ = self.gym_env.step(action)
        self.state = np.asscalar(discretization_2d([next_state[0]], [next_state[1]], self.x_bins, self.y_bins))

    def compute_max_gain(self):
        self.max_gain = 0
        return self.max_gain


class SCCALPlus(SCALPLUS):
    """
    SCALPlus for continuous states and discrete actions
    """

    def __init__(self, discretized_env, r_max, span_constraint,
                 holder_L, holder_alpha,
                 alpha_r=None, alpha_p=None,
                 verbose=0, augment_reward=True,
                 logger=default_logger, random_state=None, relative_vi=True,
                 known_reward=False):
        super(SCCALPlus, self).__init__(environment=discretized_env, r_max=r_max, span_constraint=span_constraint,
                                        alpha_r=alpha_r, alpha_p=alpha_p,
                                        verbose=verbose, augment_reward=augment_reward,
                                        logger=logger, random_state=random_state, relative_vi=relative_vi,
                                        known_reward=known_reward)
        self.holder_alpha = holder_alpha
        self.holder_L = holder_L

    def beta_r(self):
        # beta_r = super(SCALPLUS, self).beta_r() # this is using bernstein for reward

        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        N = np.maximum(1, self.nb_observations)
        beta_r = self.r_max * np.sqrt(np.log(N * 20 * S * A / self.delta) / N)
        beta_p = np.sqrt(np.log(N * 20 * S * A / self.delta) / N) + 1.0 / (self.nb_observations + 1.0)

        P_term = self.alpha_p * self.span_constraint * np.minimum(beta_p, 2.)
        R_term = self.alpha_r * np.minimum(beta_r, self.r_max if not self.known_reward else 0)

        final_bonus = R_term + P_term + (self.span_constraint + self.r_max) * self.holder_L * np.power(S, 1./self.holder_alpha)
        return final_bonus

    @staticmethod
    def nb_discrete_state(T, nA, holder_L, holder_alpha):
        N = m.pow(holder_alpha * holder_L * m.sqrt(T/nA), 1./(1.+holder_alpha))
        return int(m.ceil(N))
