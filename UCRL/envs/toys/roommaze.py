from ..Environment import Environment
import numpy as np

def coord2state(row, col, dimension):
    state = row * dimension + col
    return state


def state2coord(state, dimension):
    col, row = state % dimension, state // dimension
    return row, col

class FourRoomsMaze(Environment):
    """
    Four-room maze grid.
    """

    def __init__(self, dimension, initial_position,
                 reward_distribution_states, reward_distribution_target,
                 target_coordinates, success_probability=0.8):
        self.tartget_coordinates = target_coordinates
        self.target_state = coord2state(row=target_coordinates[0],
                                        col=target_coordinates[1],
                                        dimension=dimension)
        self.dimension = dimension
        self.curr_coordinates = initial_position  # coordinates of the current state
        self.reward_distribution_states = reward_distribution_states
        self.reward_distribution_target = reward_distribution_target
        self.p_success = min(1., max(0.0001, success_probability))

        state_actions, P = self.construct_transition_matrix(dimension=dimension,
                                                            p_succ=self.p_success)
        self.prob_kernel = P

        initial_state = coord2state(*initial_position, dimension)
        super().__init__(initial_state, state_actions)  # call parent constructor
        self.holding_time = 1

    def construct_transition_matrix(self, dimension, p_succ):
        nb_states = dimension**2

        state_actions = []
        # actions are indexed by: 0=right, 1=down, 2=left, 4=up
        top_left_corners = (0, dimension//2, dimension*(dimension//2),
                            (dimension + 1)*(dimension//2))
        top_right_corners = (dimension//2 - 1, dimension - 1, (dimension + 1)*(dimension//2) - 1,
                             dimension*(dimension//2 + 1) - 1)
        bottom_left_corners = (dimension*(dimension - 1), dimension*(dimension//2-1),
                               dimension*(dimension - 1) + dimension//2,
                               dimension*(dimension//2 - 1) + dimension//2)
        bottom_right_corners = (dimension**2 - 1, (dimension//2)*dimension - 1,
                                dimension*(dimension//2 - 1) + dimension//2 - 1,
                                dimension*(dimension - 1) + dimension//2 - 1)

        Pkernel = -np.inf * np.ones((nb_states, 4, nb_states))
        reachable_states = []
        for s in range(nb_states):
            reset_action = False
            row, col = state2coord(s, dimension)
            if s == self.target_state:  # target state
                # random reset
                actions = [0]
                reachable_states = []
                reset_action = True
                Pkernel[s][0] = np.ones(nb_states) / nb_states
            elif s in top_left_corners:  # top left corner of a room
                actions = [0, 1]
                reachable_states = [coord2state(row, col + 1, dimension), #right
                                    coord2state(row + 1, col, dimension)  #down
                                    ]
            elif s in top_right_corners:  # top right corner of a room
                actions = [1, 2]
                reachable_states = [coord2state(row, col - 1, dimension), #left
                                    coord2state(row + 1, col, dimension)  #down
                                    ]
            elif s in bottom_left_corners:  # bottom left corner of a room
                actions = [0, 3]
                reachable_states = [coord2state(row, col + 1, dimension), #right
                                    coord2state(row - 1, col, dimension)  #up
                                    ]
            elif s in bottom_right_corners:  # bottom right corner of a room
                actions = [2, 3]
                reachable_states = [coord2state(row, col - 1, dimension), #left
                                    coord2state(row - 1, col, dimension)  #up
                                    ]
            elif s < self.dimension or (row == self.dimension//2 and col != self.dimension//4 and col != self.dimension -
                                        self.dimension//4 - 1):  # top row of a room
                actions = [0, 1, 2]
                reachable_states = [coord2state(row, col + 1, dimension),  #right
                                    coord2state(row + 1, col, dimension),  #down
                                    coord2state(row, col - 1, dimension)   # left
                                    ]
            elif s > self.dimension**2-self.dimension or (row == self.dimension//2 - 1 and col != self.dimension//4 and
                                                   col != self.dimension - self.dimension//4 - 1):  # bottom row of a room
                actions = [0, 2, 3]
                reachable_states = [coord2state(row, col + 1, dimension),  # right
                                    coord2state(row, col - 1, dimension), # left
                                    coord2state(row - 1, col, dimension)  #up
                                    ]
            elif s % self.dimension == 0 or (col == self.dimension//2 and row != self.dimension//4 and row != self.dimension -
                                             self.dimension//4 - 1):  # leftmost column of a room
                actions = [0, 1, 3]
                reachable_states = [coord2state(row, col + 1, dimension),  # right
                                    coord2state(row + 1, col, dimension),  # down
                                    coord2state(row - 1, col, dimension)  #up
                                    ]
            elif s % self.dimension == self.dimension-1 or (col == self.dimension//2 - 1 and row != self.dimension//4 and
                                             row != self.dimension - self.dimension//4 - 1):  # rightmost column of a room
                actions = [1, 2, 3]
                reachable_states = [coord2state(row + 1, col, dimension),  # down
                                    coord2state(row, col - 1, dimension), # left
                                    coord2state(row - 1, col, dimension)  #up
                                    ]
            else:
                actions = [0, 1, 2, 3]
                reachable_states = [coord2state(row, col + 1, dimension),  # right
                                    coord2state(row + 1, col, dimension),  # down
                                    coord2state(row, col - 1, dimension), # left
                                    coord2state(row - 1, col, dimension)  #up
                                    ]
            # ------------------------------------------------------------------
            # define transition kernel
            # ------------------------------------------------------------------
            if not reset_action:
                assert len(reachable_states) > 1, "{}".format(reachable_states)
                if 0 in actions:
                    # right action
                    sprime_curr_act = coord2state(row, col + 1, dimension)
                    action_idx = actions.index(0)

                    Pkernel[s][action_idx] = np.zeros(nb_states)
                    Pkernel[s][action_idx][sprime_curr_act] = p_succ
                    n_reach_states = len(reachable_states) - 1
                    for el in reachable_states:
                        if sprime_curr_act != el:
                            Pkernel[s][action_idx][el] = 1. - p_succ/n_reach_states
                if 1 in actions:
                    # down action
                    sprime_curr_act = coord2state(row + 1, col, dimension)
                    action_idx = actions.index(1)

                    Pkernel[s][action_idx] = np.zeros(nb_states)
                    Pkernel[s][action_idx][sprime_curr_act] = p_succ
                    n_reach_states = len(reachable_states) - 1
                    for el in reachable_states:
                        if sprime_curr_act != el:
                            Pkernel[s][action_idx][el] = 1. - p_succ/n_reach_states
                if 2 in actions:
                    # left action
                    sprime_curr_act = coord2state(row, col - 1, dimension)
                    action_idx = actions.index(2)

                    Pkernel[s][action_idx] = np.zeros(nb_states)
                    Pkernel[s][action_idx][sprime_curr_act] = p_succ
                    n_reach_states = len(reachable_states) - 1
                    for el in reachable_states:
                        if sprime_curr_act != el:
                            Pkernel[s][action_idx][el] = 1. - p_succ/n_reach_states
                if 3 in actions:
                    # up action
                    sprime_curr_act = coord2state(row - 1, col, dimension)
                    action_idx = actions.index(3)

                    Pkernel[s][action_idx] = np.zeros(nb_states)
                    Pkernel[s][action_idx][sprime_curr_act] = p_succ
                    n_reach_states = len(reachable_states) - 1
                    for el in reachable_states:
                        if sprime_curr_act != el:
                            Pkernel[s][action_idx][el] = 1. - p_succ/n_reach_states

            # save actions
            state_actions += [actions]
        return state_actions, Pkernel


    def execute(self, action):
        row, col = state2coord(self.state, self.dimension) # current coordinates
        try:
            action_index = self.state_actions[self.state].index(action)
        except:
            raise ValueError("Action {} cannot be executed in this state {} [{}, {}]".format(action, self.state, row, col))

        p = self.prob_kernel[self.state,action_index]
        next_state = np.random.choice(self.nb_states, 1, p=p)
        self.position_coordinates = state2coord(next_state, self.dimension)
        self.state = next_state
        if self.state == self.target_state:  # if current state is the target: restart
            self.reward = self.reward_distribution_target.generate()
        else:
            self.reward = self.reward_distribution_states.generate()

    def compute_max_gain(self):
        """
        The optimal gain can be computed in closed-form.
        :return: value of the optimal average reward
        """
        if not hasattr(self, "max_gain"):
            from ....cython import extended_value_iteration
            policy_indices = np.zeros(self.nb_states)
            policy = np.zeros(self.nb_states)
            estimated_rewards = -np.inf * np.ones((self.nb_states, 4))
            for s, actions in enumerate(self.state_actions):
                for a_idx, a_val in enumerate(actions):
                    if s == self.target_state:
                        estimated_rewards[s,a_idx] = self.reward_distribution_target.mean
                    else:
                        estimated_rewards[s, a_idx] = self.reward_distribution_states.mean
            span, u1, u2 = extended_value_iteration(
                policy_indices, policy,
                self.nb_states, self.state_actions,
                self.prob_kernel, estimated_rewards,
                np.ones((self.nb_states, 4)),
                np.zeros((self.nb_states, 4)),
                np.zeros((self.nb_states, 4)),
                np.zeros((self.nb_states, 4)),
                1., self.reward_distribution_target.mean,
                1., 1., 0.0001)

            self.max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
            self.optimal_policy_indices = policy_indices
            self.optimal_policy = policy

        return self.max_gain
