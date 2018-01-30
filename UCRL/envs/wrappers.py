import numpy as np
from UCRL.evi.evi import EVI
from . import Environment
from ..utils.shortestpath import dijkstra


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
                      bound_type="chernoff",
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

            diameter = -1
            # for s in range(nS):
            #     print("{} ".format(s), end=' ')
            #     dist, _ = dijkstra(P, state_actions, s)
            #     diameter = max(diameter, np.max(dist))
            self.diameter = diameter

        return self.max_gain

    def description(self):
        desc = {
            'name': type(self.gym_env).__name__
        }
        return desc


class GymDiscreteEnvWrapperTaxi(Environment):
    def __init__(self, taxi_env):
        self.gym_env = taxi_env
        nS = self.gym_env.nS - 100
        nA = self.gym_env.nA
        state_actions = [list(range(nA)) for _ in range(nS)]
        super(GymDiscreteEnvWrapperTaxi, self).__init__(state_actions=state_actions, initial_state=None)
        self.reset()

        self.fictitious_state = 0
        self.holding_time = 1

    def convert_fictitious_real_state(self, fictitious_state):
        return fictitious_state - fictitious_state//5 - 1

    def convert_real_fictitious_state(self, real_state):
        return real_state + real_state//4 + 1

    def reset(self):
        self.fictitious_state = self.gym_env.reset()
        self.state = self.convert_fictitious_real_state()

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
                        P[s, a] = [np.delete(a, range(0,500,5)) for a in isd]
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
                      bound_type="chernoff",
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

            diameter = -1
            # for s in range(nS):
            #     print("{} ".format(s), end=' ')
            #     dist, _ = dijkstra(P, state_actions, s)
            #     diameter = max(diameter, np.max(dist))
            self.diameter = diameter

        return self.max_gain

    def description(self):
        desc = {
            'name': type(self.gym_env).__name__
        }
        return desc
