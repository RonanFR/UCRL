import numpy as np
from .toy import AbstractDiscreteMDP
from ...utils.shortestpath import dpshortestpath


class Garnet(AbstractDiscreteMDP):

    def __init__(self, Ns, Na, Nb):
        '''
        Parameters NS and NA are respectively the number of states and the number of actions.
        Parameter NB is the branching factor defining the number of possible next states for any state-action pair.
        '''
        self.nb_states = Ns
        self.Na = Na
        self.Nb = Nb

        self.state = 0
        self.reward = None
        self.P_mat = None
        self.R_mat = None
        self.state_actions = None

        self.generate()

        super(Garnet, self).__init__(initial_state=0,
                                     state_actions=self.state_actions)

        self.diameter = dpshortestpath(self.P_mat, self.state_actions)
        self.holding_time = 1

    def generate(self):
        self.P_mat = np.zeros((self.nb_states, self.Na, self.nb_states))
        self.R_mat = np.zeros((self.nb_states, self.Na))

        self.state_actions = []

        for s in range(self.nb_states):
            self.state_actions.append(range(self.Na))
            for a in range(self.Na):
                partition = np.sort(np.random.rand(self.Nb - 1))
                partition = [0] + partition.tolist() + [1]

                next_states = np.random.choice(a=self.nb_states, size=self.Nb, replace=False)

                for i, sn in enumerate(next_states):
                    self.P_mat[s, a, sn] = partition[i + 1] - partition[i]

                # enforce unichain model
                self.P_mat[s, a, 0] += 0.01
                self.P_mat[s, a] /= np.sum(self.P_mat[s, a])

                self.R_mat[s, a] = (np.random.rand() > 0.5) * np.random.rand()

    def reset(self):
        self.state = 0

    def execute(self, action):
        try:
            action_index = self.state_actions[self.state].index(action)
        except:
            raise ValueError("Action {} cannot be executed in this state {}".format(action, self.state))
        p = self.P_mat[self.state, action_index]
        next_state = np.asscalar(np.random.choice(self.nb_states, 1, p=p))
        self.reward = self.R_mat[self.state, action_index]
        self.state = next_state

    def true_reward(self, state, action):
        return self.R_mat[state, action]

    def description(self):
        desc = {
            'name': type(self).__name__
        }
        return desc
