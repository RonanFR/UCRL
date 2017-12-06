import numpy as np
from UCRL.evi.evi import EVI
from . import Environment


class GymDiscreteEnvWrapper(Environment):
    def __init__(self, env):
        self.gym_env = env
        self.need_reset = False
        nS = self.gym_env.nS
        nA = self.gym_env.nA
        state_actions = [list(range(nA)) for _ in range(nS)]
        super(GymDiscreteEnvWrapper, self).__init__(state_actions=state_actions, initial_state=None)
        self.reset()

        self.holding_time = 1

    def reset(self):
        self.state = self.gym_env.reset()
        self.need_reset = False

    def execute(self, action):
        next_state, self.reward, self.need_reset, _ = self.gym_env.step(action)
        if self.need_reset:
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
                        for aprime in range(nA):
                            P[nextstate, aprime] = isd
        R = (R - minr) / (maxr - minr)
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
            #     dist, _ = dijkstra(P, state_actions, s)
            #     diameter = max(diameter, np.max(dist))
            self.diameter = diameter

        return self.max_gain

    def description(self):
        desc = {
            'name': type(self.gym_env).__name__
        }
        return desc
