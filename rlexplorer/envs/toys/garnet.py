import numpy as np
from .toy import AbstractDiscreteMDP
from ...utils.shortestpath import dpshortestpath


class Garnet(AbstractDiscreteMDP):

    def __init__(self, Ns, Na, Nb, reward_seed=0):
        '''
        Parameters NS and NA are respectively the number of states and the number of actions.
        Parameter NB is the branching factor defining the number of possible next states for any state-action pair.
        '''

        assert Ns >= 2
        assert Na >= 2
        assert Nb >= 1

        self.nb_states = Ns
        self.Na = Na
        self.Nb = Nb

        self.state = 0
        self.reward = None
        self.P_mat = None
        self.R_mat = None
        self.state_actions = None

        self.nplocal = np.random.RandomState()
        self.nplocal.seed(reward_seed)
        self.reward_states = self.nplocal.choice(a=self.nb_states, size=10, replace=False).tolist()

        self.generate()

        super(Garnet, self).__init__(initial_state=0,
                                     state_actions=self.state_actions)

        print(self.max_gain, self.span)
        # self.diameter = dpshortestpath(self.P_mat, self.state_actions)
        self.diameter = None
        self.holding_time = 1
        self.max_branching = self.Nb

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
                MP = 0.01
                SBAR = 0
                self.P_mat[s, a, SBAR] = max(MP, self.P_mat[s, a, SBAR]) if np.random.rand() > 0.4 else MP
                self.P_mat[s, a] /= np.sum(self.P_mat[s, a])

                if s in self.reward_states:
                    self.R_mat[s, a] = 1.
                    # self.R_mat[s, a] = np.random.rand() # self.nplocal.rand() # (np.random.rand() > 0.5) * np.random.rand()

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


class GarnetChain(Garnet):

    def __init__(self, Ns, Na, Nb, chain_length, chain_proba=0.05):
        super(GarnetChain, self).__init__(Ns, Na, Nb)

        self.chain_length = chain_length
        self.chain_proba = chain_proba
        self.generate_chain()

    def generate_chain(self):
        P = np.zeros((self.nb_states + self.chain_length, self.Na, self.nb_states + self.chain_length))
        R = np.zeros((self.nb_states + self.chain_length, self.Na))
        P[0:self.nb_states, 0:self.Na, 0:self.nb_states] = self.P_mat
        R[0:self.nb_states, 0:self.Na] = self.R_mat
        offset = self.nb_states
        P[0, 0, offset] = 0.01
        P[0, 0] /= np.sum(P[0, 0])
        for i in range(self.chain_length - 1):
            P[offset + i, 0, offset + i] = 1. - self.chain_proba
            P[offset + i, 0, offset + i + 1] = self.chain_proba
            self.state_actions.append([0])
            R[offset + i, 0] = np.random.rand()
        P[offset + self.chain_length - 1, 0, offset + self.chain_length - 1] = 1. - self.chain_proba
        P[offset + self.chain_length - 1, 0, self.nb_states-1] = self.chain_proba
        R[offset + self.chain_length - 1, 0] = np.random.rand()
        self.state_actions.append([0])
        self.P_mat = P
        self.R_mat = R

        self.nb_states += self.chain_length

        del self.max_gain
        self.compute_max_gain()
        print(self.max_gain, self.span)
        self.diameter = None
        #self.diameter = dpshortestpath(self.P_mat, self.state_actions)
