from .. import Environment, MixedEnvironment
from ...cython import extended_value_iteration
from ...evi import EVI
import numpy as np
import copy
from tqdm import tqdm


def coord2state(row, col, dimension):
    assert 0<= row < dimension
    assert 0<= col < dimension
    state = row * dimension + col
    return state


def state2coord(state, dimension):
    assert state < dimension**2
    col, row = state % dimension, state // dimension
    return row, col


class FourRoomsMaze(Environment):
    """
    Four-room maze grid.
    """

    def __init__(self, dimension, initial_position,
                 reward_distribution_states, reward_distribution_target,
                 target_coordinates, success_probability=0.8,
                 reset_rule="uniform"):
        assert reset_rule in ["uniform", "brroom"]
        self.reset_rule = reset_rule

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
        super(FourRoomsMaze, self).__init__(initial_state, state_actions)  # call parent constructor
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

        self.top_left_corners = top_left_corners
        self.top_right_corners = top_right_corners
        self.bottom_left_corners = bottom_left_corners
        self.bottom_right_corners = bottom_right_corners

        Pkernel = -9 * np.ones((nb_states, 4, nb_states))
        reachable_states = []
        for s in range(nb_states):
            reset_action = False
            row, col = state2coord(s, dimension)
            if s == self.target_state:  # target state
                # random reset
                actions = [0]
                reachable_states = []
                reset_action = True
                # reset uniformly
                if self.reset_rule == "uniform":
                    Pkernel[s][0] = np.ones(nb_states) / nb_states
                elif self.reset_rule == "brroom":
                    # reset in bottom right room
                    room_states = []
                    for row in range(self.dimension//2, self.dimension):
                        for col in range(self.dimension//2, self.dimension):
                            room_states.append(coord2state(row, col, self.dimension))
                    assert len(room_states) == (self.dimension//2)**2
                    Pkernel[s][0] = np.zeros(nb_states)
                    Pkernel[s][0][room_states] = 1/len(room_states)
                else:
                    raise ValueError("Unknown reset rule")

                assert np.isclose(np.sum(Pkernel[s][0]), 1.0)

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
                            Pkernel[s][action_idx][el] = (1. - p_succ)/n_reach_states
                    assert np.isclose(np.sum(Pkernel[s][action_idx]), 1.0)
                if 1 in actions:
                    # down action
                    sprime_curr_act = coord2state(row + 1, col, dimension)
                    action_idx = actions.index(1)

                    Pkernel[s][action_idx] = np.zeros(nb_states)
                    Pkernel[s][action_idx][sprime_curr_act] = p_succ
                    n_reach_states = len(reachable_states) - 1
                    for el in reachable_states:
                        if sprime_curr_act != el:
                            Pkernel[s][action_idx][el] = (1. - p_succ)/n_reach_states
                    assert np.isclose(np.sum(Pkernel[s][action_idx]), 1.0)
                if 2 in actions:
                    # left action
                    sprime_curr_act = coord2state(row, col - 1, dimension)
                    action_idx = actions.index(2)

                    Pkernel[s][action_idx] = np.zeros(nb_states)
                    Pkernel[s][action_idx][sprime_curr_act] = p_succ
                    n_reach_states = len(reachable_states) - 1
                    for el in reachable_states:
                        if sprime_curr_act != el:
                            Pkernel[s][action_idx][el] = (1. - p_succ)/n_reach_states
                    assert np.isclose(np.sum(Pkernel[s][action_idx]), 1.0)
                if 3 in actions:
                    # up action
                    sprime_curr_act = coord2state(row - 1, col, dimension)
                    action_idx = actions.index(3)

                    Pkernel[s][action_idx] = np.zeros(nb_states)
                    Pkernel[s][action_idx][sprime_curr_act] = p_succ
                    n_reach_states = len(reachable_states) - 1
                    for el in reachable_states:
                        if sprime_curr_act != el:
                            Pkernel[s][action_idx][el] = (1. - p_succ)/n_reach_states
                    assert np.isclose(np.sum(Pkernel[s][action_idx]), 1.0)

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
        next_state = np.asscalar(np.random.choice(self.nb_states, 1, p=p))
        assert p[next_state] > 0.
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
            nb_states = self.nb_states
            policy_indices = np.zeros(nb_states, dtype=np.int)
            policy = np.zeros(nb_states, dtype=np.int)
            estimated_rewards = -9. * np.ones((nb_states, 4))
            for s, actions in enumerate(self.state_actions):
                for a_idx, a_val in enumerate(actions):
                    if s == self.target_state:
                        estimated_rewards[s,a_idx] = self.reward_distribution_target.mean
                    else:
                        estimated_rewards[s, a_idx] = self.reward_distribution_states.mean

            evi = EVI(nb_states=nb_states,
                           actions_per_state=self.state_actions,
                           use_bernstein=0)

            span = evi.evi(
                policy_indices=policy_indices,
                policy=policy,
                estimated_probabilities=self.prob_kernel,
                estimated_rewards=estimated_rewards,
                estimated_holding_times=np.ones((nb_states, 4)),
                beta_p=np.zeros((nb_states, 4, 1)),
                beta_r=np.zeros((nb_states, 4)),
                beta_tau=np.zeros((nb_states, 4)),
                tau_max=1, tau_min=1, tau=1,
                r_max=self.dimension,
                epsilon=0.01
            )

            u1, u2 = evi.get_uvectors()

            # span, u1, u2 = extended_value_iteration(
            #     policy_indices, policy,
            #     self.nb_states, self.state_actions,
            #     self.prob_kernel, estimated_rewards,
            #     np.ones((self.nb_states, 4)),
            #     np.zeros((self.nb_states, 4)),
            #     np.zeros((self.nb_states, 4)),
            #     np.zeros((self.nb_states, 4)),
            #     1.0, 50.0,
            #     1.0, 1.0, 0.01)

            self.span = span
            self.max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))
            self.optimal_policy_indices = policy_indices
            self.optimal_policy = policy

        return self.max_gain

    def get_rooms_states(self, index=None):
        """ Return the state (indices) of each room. Rooms are ordered per row and then per column
        +---+---+
        | 0 | 2 |
        +---+---+
        | 1 | 4 |
        +---+---+
        
        Returns:
            list
        """
        assert index is None or 0<=index<4
        room_states = [[],[],[],[]]
        if index is None:
            loop = range(4)
        else:
            loop = [index]
        dim = self.dimension//2
        for i in loop:
            for row in range((i%2)*dim, dim + (i%2)*dim):
                for col in range(i//2*dim, dim + i//2*dim):
                    room_states[i].append(coord2state(row, col, self.dimension))
        return room_states

    def get_room_number(self, state):
        dim = self.dimension
        dim_2 = dim//2
        row, col = state2coord(state, dim)
        if row < dim_2 and col < dim_2:
            return 0
        elif row < dim_2 and col >= dim_2:
            return 2
        elif row >= dim_2 and col < dim_2:
            return 1
        else:
            return 3


class EscapeRoom(MixedEnvironment):

    def __init__(self, maze, target_states="corner_doors_center"):
        assert isinstance(maze, FourRoomsMaze)
        assert target_states in ["corner_doors_center"]

        dim = maze.dimension
        nb_states = maze.nb_states

        state_options = []
        options_policies = []
        options_terminating_conditions = []

        rooms = maze.get_rooms_states()

        self.evi = EVI(nb_states=maze.nb_states,
                       actions_per_state=maze.get_state_actions(),
                       use_bernstein=0)

        option_id = 0
        for s in tqdm(range(nb_states)):
            if s == maze.target_state:
                # target state -> reset (only action 0 available)
                options_policies.append([0] * nb_states)
                options_terminating_conditions.append([1] * nb_states)
                state_options.append([option_id])
                option_id += 1
                assert sum(map(len,state_options)) == 1
            else:
                room_nb = maze.get_room_number(s)
                if room_nb==0: # top left room
                    target_states = [
                        coord2state(dim//2, dim//4, dim), # down door
                        coord2state(dim//4, dim//2, dim), # right door
                        coord2state(0, 0, dim), # corner
                        coord2state(dim//4, dim//4, dim), # center
                    ]
                    if dim == 6:
                        assert np.all([el in [7, 19, 9, 0] for el in target_states])
                elif room_nb==2: #top right  room
                    target_states = [
                        coord2state(dim//2, dim//2+dim//4, dim), # down door
                        coord2state(dim//4, dim//2-1, dim), # right door
                        coord2state(0, dim-1, dim), # corner
                        coord2state(dim//4, dim//2+dim//4, dim), # center
                    ]
                    if dim == 6:
                        assert np.all([el in [10, 8, 22, 5] for el in target_states])
                elif room_nb==1:  #bottom left room
                    target_states = [
                        coord2state(dim//2-1, dim//4, dim), # up door
                        coord2state(dim//2+dim//4, dim//2, dim), # right door
                        coord2state(dim-1, 0, dim), # corner
                        coord2state(dim//2+dim//4, dim//4, dim), # center
                    ]
                    if dim == 6:
                        assert np.all([el in [25,27, 13, 30] for el in target_states])
                else:  # bottom right room
                    target_states = [
                        coord2state(dim//2-1, dim//2+dim//4, dim), # up door
                        coord2state(dim//2+dim//4, dim//2-1, dim), # left door
                        coord2state(dim-1, dim-1, dim), # corner
                        coord2state(dim//2+dim//4, dim//2+dim//4, dim), # center
                    ]
                    if dim == 6:
                        assert np.all([el in [28, 26, 16, 35] for el in target_states])

                # set zero in the current room
                o_tc = [1] * nb_states
                for x in rooms[room_nb]:
                    o_tc[x] = 0
                o_tc[s] = 1 # starting state
                o_tc[maze.target_state] = 1 # target state
                assert s in rooms[room_nb]

                if s in target_states:
                    target_states.remove(s)
                assert s not in target_states

                policies = self.solve_forward_model(maze, target_states)
                state_options.append([])
                for i in range(len(target_states)):
                    options_policies.append(policies[i])
                    loc_otc = copy.deepcopy(o_tc)
                    loc_otc[target_states[i]] = 1 # when i reach the target I stop


                    # if target_states[i] in rooms[room_nb]:
                    #     assert np.sum(loc_otc) == 3 * (dim / 2) ** 2 + 2 + offset
                    # else:
                    #     assert np.sum(loc_otc) == 3 * (dim / 2) ** 2 + 1 + offset

                    options_terminating_conditions.append(loc_otc)
                    state_options[-1].append(option_id)
                    option_id += 1
        assert option_id == nb_states*4 - 8 -2 #starting state has just one option
        assert option_id == len(options_policies)
        assert option_id == len(options_terminating_conditions)

        super(EscapeRoom, self).__init__(environment=maze,
                                         state_options=state_options,
                                         options_policies=options_policies,
                                         options_terminating_conditions=options_terminating_conditions,
                                         delete_environment_actions="all",
                                         is_optimal=True)
        self.dimension = maze.dimension

        self.compute_variance_tau()

        del self.evi

    def solve_forward_model(self, maze, targets_states):
        nb_states = maze.nb_states
        optimal_policies = []
        for s_targ in targets_states:
            policy_indices = np.zeros(nb_states, dtype=np.int)
            policy = np.zeros(nb_states, dtype=np.int)
            reward = np.zeros((nb_states, 4))
            P = maze.prob_kernel.copy()
            for a in range(4):
                P[s_targ,a] = np.zeros(nb_states)
                P[s_targ,a,s_targ] = 1.
                reward[s_targ, a] = 1

            span = self.evi.evi(
                policy_indices=policy_indices,
                policy=policy,
                estimated_probabilities=P,
                estimated_rewards=reward,
                estimated_holding_times=np.ones((nb_states, 4)),
                beta_p=np.zeros((nb_states, 4, 1)),
                beta_r=np.zeros((nb_states, 4)),
                beta_tau=np.zeros((nb_states, 4)),
                tau_max=1, tau_min=1, tau=1,
                r_max=maze.dimension,
                epsilon=0.01
            )
            # span, u1, u2 = extended_value_iteration(
            #     policy_indices, policy,
            #     nb_states, maze.state_actions,
            #     P, reward,
            #     np.ones((nb_states, 4)),
            #     np.zeros((nb_states, 4)),
            #     np.zeros((nb_states, 4)),
            #     np.zeros((nb_states, 4)),
            #     1.0, maze.dimension,
            #     1.0, 1.0, 0.01)

            # for i, a in enumerate(policy):
            #     row, col = state2coord(i,maze.dimension)
            #     print("{}".format(a), end=" ")
            #     if col == maze.dimension-1:
            #         print("")
            # print("-"*10)

            optimal_policies.append(policy)
        return optimal_policies


    def compute_variance_tau(self):
        nb_options = self.nb_options
        reachable_states = self.reachable_states_per_option
        self.tau_options = np.empty((nb_options))
        self.tau_variance_options = np.empty((nb_options))
        for o in range(nb_options):
            tau, tau_var = self._tau_var(
                opt=o,
                reachable_states=reachable_states[o],
                option_policy=self.options_policies[o],
                terminal_condition=np.array(self.options_terminating_conditions[o])
            )
            self.tau_options[o] = tau
            self.tau_variance_options[o] = tau_var

    def _tau_var(self, opt, reachable_states,
                             option_policy, terminal_condition):
        nb_states_mc = len(reachable_states)
        Q = np.zeros((nb_states_mc, nb_states_mc))
        for i, s in enumerate(reachable_states):
            option_action = option_policy[s]
            option_action_index = self.environment.get_index_of_action_state(s, option_action)
            prob = self.environment.prob_kernel[s, option_action_index, reachable_states]
            Q[i] = (1. - terminal_condition[reachable_states]) * prob
        A = np.eye(nb_states_mc) - Q
        N = np.linalg.inv(A)
        tau = np.dot(N, np.ones((nb_states_mc,)))
        tau_var = np.dot((2*N - np.eye(nb_states_mc)), tau) - tau * tau
        return tau[0], tau_var[0]

    def reshaped_sigma_tau(self):
        sigmas = np.sqrt(self.tau_variance_options)
        nb_states = self.nb_states
        sigma_tau = np.zeros((nb_states, self.max_nb_actions_per_state))
        for i in range(nb_states):
            for j, opt in enumerate(self.get_available_actions_state(i)):
                zerob_opt = opt - self.threshold_options - 1
                sigma_tau[i,j] = sigmas[zerob_opt]
        return sigma_tau

