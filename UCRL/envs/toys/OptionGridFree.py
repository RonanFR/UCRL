from UCRL.envs.OptionEnvironment import OptionEnvironment

class OptionGrid1(OptionEnvironment):
    """
    This class builds macro-actions (options) over a NavigateGrid instance. Based on an input grid, four types of
    options are constructed: LEFT, RIGHT, UP and DOWN. LEFT consists in executing primitive action left until either a
    leftmost state is reached or the stopping condition is met (see below for the terminating condition).
    """

    def __init__(self, grid, t_max):
        """
        :param grid: an instance of NavigateGrid1
        :param t_max: Maximal number of time steps of the options
        """
        nb_options = 4*grid.dimension**2 - 4*grid.dimension - 2
        self.options_policies = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.options_terminating_conditions = [[9]*grid.nb_states for i in range(0, nb_options)]
        self.state_options = []

        i = 0
        for s, actions in enumerate(grid.state_actions[0:-1]):
            self.state_options += [range(i, i + len(actions))]
            for action in actions:
                self.options_policies[i] = [action] * grid.nb_states
                x, y = s % grid.dimension, s//grid.dimension
                if action == 0:  # right
                    m = min(t_max, grid.dimension-1-x)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x + j + grid.dimension * y] = 1/(m-j+1)
                elif action == 1:  # down
                    m = min(t_max, grid.dimension-1-y)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x + grid.dimension * (y + j)] = 1/(m-j+1)
                elif action == 2:  # left
                    m = min(x, t_max)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x - j + grid.dimension * y] = 1/(m-j+1)
                elif action == 3:  # up
                    m = min(y, t_max)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x + grid.dimension * (y - j)] = 1/(m-j+1)
                self.options_terminating_conditions[i][s] = 1
                i += 1

        self.state_options += [range(i, i + grid.nb_target_actions**2)]
        for k in range(0, grid.nb_target_actions**2):
            self.options_terminating_conditions += [[1] * grid.nb_states]
            self.options_policies += [[k] * grid.nb_states]


class OptionGrid3_free(OptionEnvironment):

    def __init__(self, grid, t_max):
        """
        :param grid: an instance of NavigateGrid2
        :param t_max: Maximal number of time steps of the options
        """
        nb_options = 4*grid.dimension**2 - 4*grid.dimension - 4
        self.options_policies = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.options_terminating_conditions = [[9]*grid.nb_states for i in range(0, nb_options)]
        self.options_initial_states = [0]*nb_options
        self.state_options = []

        self.state_options += [range(0, grid.nb_reset_actions**2)]
        for k in range(0, grid.nb_reset_actions**2):
            self.options_terminating_conditions = [[1] * grid.nb_states] + self.options_terminating_conditions
            self.options_policies = [[k] * grid.nb_states] + self.options_policies
            self.options_initial_states = [0] + self.options_initial_states

        i = grid.nb_reset_actions**2
        for s, actions in enumerate(grid.state_actions[1:-1]):
            s += 1
            self.state_options += [range(i, i + len(actions))]

            for action in actions:
                self.options_policies[i] = [action] * grid.nb_states
                x, y = s % grid.dimension, s//grid.dimension
                if action == 0:  # right
                    m = min(t_max, grid.dimension-1-x)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x + j + grid.dimension * y] = 1/(m-j+1)
                elif action == 1:  # down
                    m = min(t_max, grid.dimension-1-y)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x + grid.dimension * (y + j)] = 1/(m-j+1)
                elif action == 2:  # left
                    m = min(x, t_max)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x - j + grid.dimension * y] = 1/(m-j+1)
                elif action == 3:  # up
                    m = min(y, t_max)
                    for j in range(1, m+1):
                        self.options_terminating_conditions[i][x + grid.dimension * (y - j)] = 1/(m-j+1)
                # initial state should have probability 1
                s_0 = s
                assert len(self.state_options)-1 == s_0
                self.options_initial_states[i] = s_0
                self.options_terminating_conditions[i][s_0] = 1
                i += 1

        self.state_options += [[i]]
        self.options_initial_states += [len(self.state_options)-1]
        self.options_terminating_conditions += [[1] * grid.nb_states]
        self.options_policies += [[0] * grid.nb_states]