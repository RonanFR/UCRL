"""classic Linear Quadratic task"""
from numbers import Number
import numpy as np
from ...utils.discretization import Discretizer
from ...evi.evi import EVI


class DiscreteLQR:

    def __init__(self, Ns, Na, max_pos=2, max_action=1, sigma_noise=0.5):
        self.max_pos = max_pos
        self.max_action = max_action
        self.sigma_noise = sigma_noise
        self.A = np.array([1]).reshape((1, 1))
        self.B = np.array([1]).reshape((1, 1))
        self.Q = np.array([0.5]).reshape((1, 1))
        self.R = np.array([0.5]).reshape((1, 1))
        self.Ns = Ns
        self.Na = Na

        bins = []
        V = np.linspace(-max_pos, max_pos, Ns)
        bins.append(V[1:-1])
        grid = Discretizer(bins=bins)
        self.grid = grid

        self.actions = np.linspace(-max_action, max_action, Na)
        self.state_actions = [range(Na) for i in range(grid.nbin[0])]

        self.max_r = 0
        self.min_r = -np.dot(max_pos, np.dot(self.Q, max_pos)) + np.dot(max_action, np.dot(self.R, max_action))

        self.nb_states = self.grid.nbin[0]
        self.max_pos = max_pos
        self.max_action = max_action
        self.max_nb_actions_per_state = Na

        self.compute_max_gain()

        self.reset()

        self.max_branching = self.grid.nbin
        # self.max_gain = 0.
        self.diameter = 0.
        self.bias_span = 0.
        self.holding_time = 1
        self.max_gain = 0

    def execute(self, action):
        u = self.actions[action]
        noise = 0
        if self.sigma_noise > 0:
            noise = np.random.randn() * self.sigma_noise
        xn = np.clip(np.dot(self.A, self.cont_state) + np.dot(self.B, u) + noise, -self.max_pos, self.max_pos)
        cost = np.dot(self.cont_state, np.dot(self.Q, self.cont_state)) + np.dot(u, np.dot(self.R, u))

        self.cont_state = np.array(xn.ravel())
        self.state = np.asscalar(self.grid.dpos(self.cont_state))
        self.reward = -cost
        self.reward = np.asscalar((self.reward - self.min_r) / (self.max_r - self.min_r))

    def reset(self):
        initial_state = np.random.uniform(low=-self.max_pos, high=self.max_pos)
        self.state = np.asscalar(self.grid.dpos(initial_state))
        self.cont_state = initial_state

    def description(self):
        desc = {
            'name': type(self).__name__
        }
        return desc

    def properties(self):
        props = {
            'gain': self.max_gain,
            'diameter': self.diameter,
            'bias_span': self.bias_span
        }
        return props

    def compute_matrix_form(self):
        N = 10
        S = np.asscalar(self.grid.nbin)
        A = self.Na
        P_mtx = np.zeros((S, A, S))
        R_mtx = np.zeros((S, A))
        for x in np.linspace(-self.max_pos, self.max_pos, 5 * self.Ns):
            s = self.grid.dpos(x)
            for a in range(self.Na):
                for n in range(N):
                    self.cont_state = x
                    self.execute(a)
                    sn = self.state
                    P_mtx[s, a, sn] += 1
                    R_mtx[s, a] += self.reward

        for s in range(S):
            for a in range(A):
                Nsa = np.sum(P_mtx[s, a])
                P_mtx[s, a] /= Nsa
                R_mtx[s, a] /= Nsa

        return P_mtx, R_mtx

    def compute_max_gain(self):
        if not hasattr(self, "max_gain"):
            P, R = self.compute_matrix_form()
            nS = np.asscalar(self.grid.nbin)
            nA = self.max_nb_actions_per_state
            policy_indices = np.ones(nS, dtype=np.int)
            policy = np.ones(nS, dtype=np.int)

            evi = EVI(nb_states=nS,
                      actions_per_state=self.state_actions,
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
            self.bias_span = span
            self.max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
            self.optimal_policy_indices = policy_indices
            self.optimal_policy = policy

        return self.max_gain
