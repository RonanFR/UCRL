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
        self.options_terminating_conditions = [[0]*grid.nb_states for i in range(0, nb_options)]
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
                i += 1

        self.state_options += [range(i, i + grid.nb_target_actions**2)]
        for k in range(0, grid.nb_target_actions**2):
            self.options_terminating_conditions += [[1] * grid.nb_states]
            self.options_policies += [[k] * grid.nb_states]


class OptionGrid2(OptionEnvironment):

    def __init__(self, grid, t_max):
        """
        :param grid: an instance of NavigateGrid1
        :param t_max: Maximal number of time steps of the options
        """
        nb_options = 4*grid.dimension**2 - 4*grid.dimension - 2
        self.options_policies = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.options_terminating_conditions = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.state_options = []

        i = 0
        for s, actions in enumerate(grid.state_actions[0:-1]):
            self.state_options += [range(i, i + len(actions))]
            for action in actions:
                self.options_policies[i] = [action] * grid.nb_states
                x, y = s % grid.dimension, s//grid.dimension
                if action == 0:  # right
                    m = min(t_max, grid.dimension-1-x)
                    self.options_terminating_conditions[i][x + m + grid.dimension * y] = 1
                elif action == 1:  # down
                    m = min(t_max, grid.dimension-1-y)
                    self.options_terminating_conditions[i][x + grid.dimension * (y + m)] = 1
                elif action == 2:  # left
                    m = min(x, t_max)
                    self.options_terminating_conditions[i][x - m + grid.dimension * y] = 1
                elif action == 3:  # up
                    m = min(y, t_max)
                    self.options_terminating_conditions[i][x + grid.dimension * (y - m)] = 1
                i += 1

        self.state_options += [range(i, i + grid.nb_target_actions**2)]
        for k in range(0, grid.nb_target_actions**2):
            self.options_terminating_conditions += [[1] * grid.nb_states]
            self.options_policies += [[k] * grid.nb_states]


class OptionGrid3(OptionEnvironment):

    def __init__(self, grid, t_max):
        """
        :param grid: an instance of NavigateGrid2
        :param t_max: Maximal number of time steps of the options
        """
        nb_options = 4*grid.dimension**2 - 4*grid.dimension - 4
        self.options_policies = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.options_terminating_conditions = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.state_options = []

        self.state_options += [range(0, grid.nb_reset_actions**2)]
        for k in range(0, grid.nb_reset_actions**2):
            self.options_terminating_conditions = [[1] * grid.nb_states] + self.options_terminating_conditions
            self.options_policies = [[k] * grid.nb_states] + self.options_policies

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
                i += 1

        self.state_options += [[i]]
        self.options_terminating_conditions += [[1] * grid.nb_states]
        self.options_policies += [[0] * grid.nb_states]


class OptionGrid4(OptionEnvironment):

    def __init__(self, grid, t_max):
        """
        :param grid: an instance of NavigateGrid3
        :param t_max: Maximal number of time steps of the options
        """
        nb_options = 4*grid.dimension**2
        self.options_policies = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.options_terminating_conditions = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.state_options = []

        i = 0
        for s, actions in enumerate(grid.state_actions):
            self.state_options += [range(i, i + len(actions))]
            for action in actions:
                self.options_policies[i] = [action] * grid.nb_states
                x, y = s % grid.dimension, s//grid.dimension
                if action == 0:  # right
                    for j in range(1, t_max+1):
                        x += 1
                        x %= grid.dimension
                        self.options_terminating_conditions[i][x + grid.dimension * y] = 1/(t_max-j+1)
                elif action == 1:  # down
                    for j in range(1, t_max+1):
                        y += 1
                        y %= grid.dimension
                        self.options_terminating_conditions[i][x + grid.dimension * y] = 1/(t_max-j+1)
                elif action == 2:  # left
                    for j in range(1, t_max+1):
                        x -= 1
                        x %= grid.dimension
                        self.options_terminating_conditions[i][x + grid.dimension * y] = 1/(t_max-j+1)
                elif action == 3:  # up
                    for j in range(1, t_max+1):
                        y -= 1
                        y %= grid.dimension
                        self.options_terminating_conditions[i][x + grid.dimension * y] = 1/(t_max-j+1)
                i += 1


class OptionGrid5(OptionEnvironment):

    def __init__(self, grid, t_max):
        """
        :param grid: an instance of NavigateGrid3
        :param t_max: Maximal number of time steps of the options
        """
        nb_options = 4*grid.dimension**2
        self.options_policies = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.options_terminating_conditions = [[0]*grid.nb_states for i in range(0, nb_options)]
        self.state_options = []

        i = 0
        for s, actions in enumerate(grid.state_actions):
            self.state_options += [range(i, i + len(actions))]
            for action in actions:
                self.options_policies[i] = [action] * grid.nb_states
                x, y = s % grid.dimension, s//grid.dimension
                if action == 0:  # right
                    x += t_max
                    x %= grid.dimension
                    self.options_terminating_conditions[i][x + grid.dimension * y] = 1
                elif action == 1:  # down
                    y += t_max
                    y %= grid.dimension
                    self.options_terminating_conditions[i][x + grid.dimension * y] = 1
                elif action == 2:  # left
                    x -= t_max
                    x %= grid.dimension
                    self.options_terminating_conditions[i][x + grid.dimension * y] = 1
                elif action == 3:  # up
                    y -= t_max
                    y %= grid.dimension
                    self.options_terminating_conditions[i][x + grid.dimension * y] = 1
                i += 1


class EscapeRoom(OptionEnvironment):

    def __init__(self, room_maze):
        """
        :param room_maze: an instance of RoomMaze1
        """
        nb_options = 8
        self.options_policies = [[0] * room_maze.nb_states for i in range(0, nb_options)]
        self.options_terminating_conditions = [[0] * room_maze.nb_states for i in range(0, nb_options)]
        self.state_options = [[0, 0] for i in range(0, room_maze.nb_states)]

        self.options_policies[0] = [0] * room_maze.nb_states
        self.options_terminating_conditions[0][room_maze.dimension // 2 + room_maze.dimension * (room_maze.dimension // 4)] = 1
        self.options_terminating_conditions[0][room_maze.target_state] = 1
        x = room_maze.dimension // 2 - 1
        for y in range(0, room_maze.dimension//4):
            self.options_policies[0][x + y * room_maze.dimension] = 1
        for y in range(room_maze.dimension//4 + 1, room_maze.dimension//2):
            self.options_policies[0][x + y * room_maze.dimension] = 3

        self.options_policies[1] = [1] * room_maze.nb_states
        self.options_terminating_conditions[1][room_maze.dimension // 4 + room_maze.dimension * (room_maze.dimension // 2)] = 1
        self.options_terminating_conditions[1][room_maze.target_state] = 1
        y = room_maze.dimension // 2 - 1
        for x in range(0, room_maze.dimension//4):
            self.options_policies[1][x + y * room_maze.dimension] = 0
        for x in range(room_maze.dimension//4 + 1, room_maze.dimension//2):
            self.options_policies[1][x + y * room_maze.dimension] = 2

        self.options_policies[2] = [1] * room_maze.nb_states
        self.options_terminating_conditions[2][room_maze.dimension - room_maze.dimension // 4 - 1 + room_maze.dimension * (room_maze.dimension // 2)] = 1
        self.options_terminating_conditions[2][room_maze.target_state] = 1
        y = room_maze.dimension // 2 - 1
        for x in range(room_maze.dimension-1, room_maze.dimension-room_maze.dimension//4 - 1, -1):
            self.options_policies[2][x + y * room_maze.dimension] = 2
        for x in range(room_maze.dimension-room_maze.dimension//4 - 2, room_maze.dimension//2 - 1, -1):
            self.options_policies[2][x + y * room_maze.dimension] = 0

        self.options_policies[3] = [2] * room_maze.nb_states
        self.options_terminating_conditions[3][room_maze.dimension // 2 - 1 + room_maze.dimension * (room_maze.dimension // 4)] = 1
        self.options_terminating_conditions[3][room_maze.target_state] = 1
        x = room_maze.dimension // 2
        for y in range(0, room_maze.dimension//4):
            self.options_policies[3][x + y * room_maze.dimension] = 1
        for y in range(room_maze.dimension//4 + 1, room_maze.dimension//2):
            self.options_policies[3][x + y * room_maze.dimension] = 3

        self.options_policies[4] = [0] * room_maze.nb_states
        self.options_terminating_conditions[4][room_maze.dimension // 2 + room_maze.dimension * (room_maze.dimension - room_maze.dimension // 4 - 1)] = 1
        self.options_terminating_conditions[4][room_maze.target_state] = 1
        x = room_maze.dimension // 2 - 1
        for y in range(room_maze.dimension-1, room_maze.dimension-room_maze.dimension//4 - 1, -1):
            self.options_policies[4][x + y * room_maze.dimension] = 3
        for y in range(room_maze.dimension-room_maze.dimension//4 - 2, room_maze.dimension//2 - 1, -1):
            self.options_policies[4][x + y * room_maze.dimension] = 1

        self.options_policies[5] = [3] * room_maze.nb_states
        self.options_terminating_conditions[5][room_maze.dimension // 4 + room_maze.dimension * (room_maze.dimension // 2 - 1)] = 1
        self.options_terminating_conditions[5][room_maze.target_state] = 1
        y = room_maze.dimension // 2
        for x in range(0, room_maze.dimension//4):
            self.options_policies[5][x + y * room_maze.dimension] = 0
        for x in range(room_maze.dimension//4 + 1, room_maze.dimension//2):
            self.options_policies[5][x + y * room_maze.dimension] = 2

        self.options_policies[6] = [2] * room_maze.nb_states
        self.options_terminating_conditions[6][room_maze.dimension // 2 - 1 + room_maze.dimension * (room_maze.dimension - room_maze.dimension // 4 - 1)] = 1
        self.options_terminating_conditions[6][room_maze.target_state] = 1
        x = room_maze.dimension // 2
        for y in range(room_maze.dimension-1, room_maze.dimension-room_maze.dimension//4 - 1, -1):
            self.options_policies[6][x + y * room_maze.dimension] = 3
        for y in range(room_maze.dimension-room_maze.dimension//4 - 2, room_maze.dimension//2 - 1, -1):
            self.options_policies[6][x + y * room_maze.dimension] = 1

        self.options_policies[7] = [3] * room_maze.nb_states
        self.options_terminating_conditions[7][room_maze.dimension - room_maze.dimension // 4 - 1 + room_maze.dimension * (room_maze.dimension // 2 - 1)] = 1
        self.options_terminating_conditions[7][room_maze.target_state] = 1
        y = room_maze.dimension // 2
        for x in range(room_maze.dimension-1, room_maze.dimension-room_maze.dimension//4 - 1, -1):
            self.options_policies[7][x + y * room_maze.dimension] = 2
        for x in range(room_maze.dimension-room_maze.dimension//4 - 2, room_maze.dimension//2 - 1, -1):
            self.options_policies[7][x + y * room_maze.dimension] = 0

        for x in range(0, room_maze.dimension):
            for y in range(0, room_maze.dimension):
                if [x, y] != room_maze.target_coordinates:
                    if y < room_maze.dimension//2:
                        if x < room_maze.dimension//2:
                            self.state_options[x + y * room_maze.dimension] = [0, 1]
                        else:
                            self.state_options[x + y * room_maze.dimension] = [2, 3]
                    else:
                        if x < room_maze.dimension//2:
                            self.state_options[x + y * room_maze.dimension] = [4, 5]
                        else:
                            self.state_options[x + y * room_maze.dimension] = [6, 7]
                else:
                    self.state_options[x + y * room_maze.dimension] = []
