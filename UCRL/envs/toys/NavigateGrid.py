import random

import numpy as np

from UCRL.envs.Environment import Environment


class NavigateGrid(Environment):
    """
    This class implements an MDP consisting of a "dimension-by-dimension" square grid with the four cardinal actions
    left, right, up and down, and a single target state on the bottom right corner. All transitions are deterministic
    (i.e., succeed with probability 1). Any state in the grid is characterized by its coordinates (x,y) where x is the
    index of the column (the leftmost column has index 0) and y is the index of the row (the top row has index 0).
    """

    def __init__(self, dimension, initial_position, reward_distribution_states,
                 reward_distribution_target, nb_target_actions):
        """
        :param dimension: dimension of the grid
        :param initial_position: coordinates of the initial state
        :param reward_distribution_states: an instance of any subclass of RewardDistribution corresponding to the reward
                                           obtained in all states but the target
        :param reward_distribution_target: an instance of any subclass of RewardDistribution corresponding to the reward
                                           obtained in the target state
        :param nb_target_actions: sqrt of the number of actions available in the target state, if 1 then every time the
                                  target is reached the problem is restarted in a state chosen uniformly at random, if 2
                                  the restart state is chosen uniformly at random in one of the 4 subsets of the states,
                                  the four subsets are as follow: states with even x- and y-coordinates, states with odd
                                  x- and y-coordinates, states with even x and odd y, states with odd x and even y;
                                  for higher values of "nb_target_actions", the number of actions available
                                  is nb_target_actions**2 where each action consists in a uniform random restart in one
                                  of the nb_target_actions**2 subsets partitioning the states
        """
        self.dimension = dimension
        self.position_coordinates = initial_position  # coordinates of the current state
        self.reward_distribution_states = reward_distribution_states
        self.reward_distribution_target = reward_distribution_target
        if nb_target_actions >= dimension:  # nb_target_actions should not exceed dimension-1
            self.nb_target_actions = 1
        else:
            self.nb_target_actions = nb_target_actions
        initial_state = initial_position[0]+initial_position[1]*dimension  # the states are indexed by: x + y*dimension
        state_actions = []
        # actions are indexed by: 0=right, 1=down, 2=left, 4=up
        for s in range(0, self.dimension**2-1):
            if s == 0:  # top left corner
                actions = [0, 1]
            elif s == self.dimension-1:  # top right corner
                actions = [1, 2]
            elif s == self.dimension**2-self.dimension:
                actions = [0, 3]
            elif s < self.dimension:  # top row
                actions = [0, 1, 2]
            elif s > self.dimension**2-self.dimension:  # bottom row
                actions = [0, 2, 3]
            elif s % self.dimension == 0:  # leftmost column
                actions = [0, 1, 3]
            elif s % self.dimension == self.dimension-1:  # rightmost column
                actions = [1, 2, 3]
            else:
                actions = [0, 1, 2, 3]
            state_actions += [actions]
        state_actions += [range(0, nb_target_actions**2)]  # target state
        super(NavigateGrid, self).__init__(initial_state=initial_state, state_actions=state_actions)
        self.holding_time = 1

    def compute_max_gain(self):
        """
        The optimal gain can be computed in closed-form.
        :return: value of the optimal average reward
        """
        q = self.dimension // self.nb_target_actions
        k = self.dimension % self.nb_target_actions
        # self.dimension = q*self.nb_target_actions + k (Euclidean division)
        duration_round = self.nb_target_actions/2*q  # expected optimal duration to reach target x or y coordinate
        if k < self.nb_target_actions/2:
            duration_round = duration_round - self.nb_target_actions/2 + k
        return (self.reward_distribution_target.mean + 2*duration_round*
                         self.reward_distribution_states.mean)/(2*duration_round + 1)

    def execute(self, action):
        x, y = self.state%self.dimension, self.state//self.dimension  # current coordinates
        if x == self.dimension-1 and y == self.dimension-1:  # if current state is the target: restart
            x, y = random.randrange(action // self.nb_target_actions, self.dimension, self.nb_target_actions),\
                   random.randrange(action % self.nb_target_actions, self.dimension, self.nb_target_actions)
            self.position_coordinates = [x, y]
            self.state = x + y*self.dimension
            self.reward = self.reward_distribution_target.generate()
        else:  # if current state is not the target: move up, left, right or down
            if action == 0:  # right
                x += 1
                x = min(x, self.dimension-1)
            elif action == 1:  # down
                y += 1
                y = min(y, self.dimension-1)
            elif action == 2:  # left
                x -= 1
                x = max(x, 0)
            elif action == 3:  # up
                y -= 1
                y = max(y, 0)
            self.position_coordinates = [x, y]
            self.state = x + y*self.dimension
            self.reward = self.reward_distribution_states.generate()

    @property
    def prob_kernel(self):
        if not hasattr(self, 'Pkernel'):
            self.Pkernel = np.zeros((self.nb_states, self.max_nb_actions_per_state, self.nb_states))
            for s in range(self.nb_states):
                for aidx, action in enumerate(self.state_actions[s]):
                    self.state = s
                    self.execute(action)
                    next_state = self.state
                    self.Pkernel[s, aidx, next_state] = 1.
        return self.Pkernel


class NavigateGrid2(Environment):

    def __init__(self, dimension, initial_position, reward_distribution_states,
                 reward_distribution_target, nb_reset_actions):
        self.dimension = dimension  # the MDP is a dimension x dimension square with deterministic transitions
        self.position_coordinates = initial_position
        self.reward_distribution_states = reward_distribution_states
        self.reward_distribution_target = reward_distribution_target
        if nb_reset_actions >= dimension:
            self.nb_reset_actions = 1
        else:
            self.nb_reset_actions = nb_reset_actions
        initial_state = initial_position[0]+initial_position[1]*dimension
        state_actions = []
        for s in range(0, self.dimension**2-1):
            if s == 0:
                actions = range(0, nb_reset_actions ** 2)
            elif s == self.dimension-1:
                actions = [1, 2]
            elif s == self.dimension**2-self.dimension:
                actions = [0, 3]
            elif s < self.dimension:
                actions = [0, 1, 2]
            elif s > self.dimension**2-self.dimension:
                actions = [0, 2, 3]
            elif s % self.dimension == 0:
                actions = [0, 1, 3]
            elif s % self.dimension == self.dimension-1:
                actions = [1, 2, 3]
            else:
                actions = [0, 1, 2, 3]
            state_actions += [actions]
        state_actions += [[0]]  # target state
        super().__init__(initial_state, state_actions)
        self.holding_time = 1

    def compute_max_gain(self):
        return self.reward_distribution_target.mean/self.dimension + \
                         self.reward_distribution_states.mean*(1 - 1/self.dimension)

    def execute(self, action):
        x, y =  self.state%self.dimension, self.state//self.dimension  # current coordinates
        if x == self.dimension-1 and y == self.dimension-1:
            x, y = random.randrange(action // self.nb_reset_actions, self.dimension, self.nb_reset_actions),\
                   random.randrange(action % self.nb_reset_actions, self.dimension, self.nb_reset_actions)
            self.position_coordinates = [x, y]
            self.state = x + y*self.dimension
            self.reward = self.reward_distribution_states.generate()
        elif x == 0 and y == 0:
            bernouilli_trial = np.random.binomial(n=1, p=0.5)
            if bernouilli_trial:
                x, y = self.dimension-1, 0
            else:
                x, y = 0, self.dimension-1
            self.position_coordinates = [x, y]
            self.state = x + y*self.dimension
            self.reward = self.reward_distribution_target.generate()
        else:
            if action == 0:  # right
                x += 1
                x = min(x, self.dimension-1)
            elif action == 1:  # down
                y += 1
                y = min(y, self.dimension-1)
            elif action == 2:  # left
                x -= 1
                x = max(x, 0)
            elif action == 3:  # up
                y -= 1
                y = max(y, 0)
            self.position_coordinates = [x, y]
            self.state = x + y*self.dimension
            self.reward = self.reward_distribution_states.generate()


class NavigateGrid3(Environment):

    def __init__(self, dimension, initial_position, reward_distribution_states, reward_distribution_target):
        """
        :param dimension: dimension should be a prime number to avoid problems when defining options
        :param initial_position:
        :param reward_distribution_states:
        :param reward_distribution_target:
        :return:
        """
        self.dimension = dimension  # the MDP is a dimension x dimension square with deterministic transitions
        self.position_coordinates = initial_position
        self.reward_distribution_states = reward_distribution_states
        self.reward_distribution_target = reward_distribution_target
        initial_state = initial_position[0]+initial_position[1]*dimension
        state_actions = [[0, 1, 2, 3]]*(self.dimension**2)  # four cardinal actions in every state
        super().__init__(initial_state, state_actions)
        self.holding_time = 1

    def compute_max_gain(self):
        return self.reward_distribution_target.mean

    def execute(self, action):
        x, y = self.state%self.dimension, self.state//self.dimension  # current coordinates
        if action == 0:  # right
            x += 1
            x %= self.dimension
        elif action == 1:  # down
            y += 1
            y %= self.dimension
        elif action == 2:  # left
            x -= 1
            x %= self.dimension
        elif action == 3:  # up
            y -= 1
            y %= self.dimension
        self.position_coordinates = [x, y]
        self.state = x + y*self.dimension
        if y == (self.dimension-1)//2 and action == 2:
            self.reward = self.reward_distribution_target.generate()
        else:
            self.reward = self.reward_distribution_states.generate()

