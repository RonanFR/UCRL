from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    """
    Environment is an abstract class representing a toys state/action Semi-Markov Decision Process (SMDP).
    It can be used for MDPs by setting self.holding_time = 1 in function "execute".
    All subclasses should implement the functions "execute" and "compute_max_gain".
    Variable "state_actions" should preferably be accessed through the getters defined below to avoid problems.
    This is because when macro-actions are defined on the SMDP, there will be different "state_actions", one for
    primitive actions and one for macro-actions. In this case, the function "get_state_actions" will be redefined (using
    polymorphism) to return "state_actions" for macro-actions (see subclass "OptionEnvironment").
    """

    def __init__(self, initial_state, state_actions):
        """
        :param initial_state: Index of the first state of the SMDP
        :param state_actions: 2D-array of integers representing the indices of actions available in every state
        """
        self.state = initial_state  # Index of the current state
        self.state_actions = state_actions
        self.max_nb_actions_per_state = max(map(len, self.state_actions))  # Maximal number of actions per state
        self.reward = 0   # Last reward obtained while exploring the SMDP (init = 0)
        self.holding_time = 0  # Last holding time obtained while exploring the SMDP (init = 0)
        self.nb_states = len(self.state_actions)  # Total number of states
        self.max_nb_actions_per_state = max(map(len, self.state_actions))
        self.max_gain = self.compute_max_gain()  # Optimal gain of the SMDP

    def get_state_actions(self):
        """
        :return: 2D-array state_actions
        """
        return self.state_actions

    def get_available_actions(self):
        """
        :return: list of indices of actions available in current state
        """
        return self.state_actions[self.state]

    def get_nb_available_actions(self):
        """
        :return: number of actions available in current state
        """
        return len(self.state_actions[self.state])

    def get_index_of_action_state(self, state, action):
        if action in self.state_actions[state]:
            return self.state_actions[state].index(action)
        return None

    def get_available_actions_state(self, state):
        """
        :param state: index of any state in the SMDP
        :return: list of indices of actions available in state "state"
        """
        return self.state_actions[state]

    def get_nb_available_actions_state(self, state):
        """
        :param state: index of any state in the SMDP
        :return: number of actions available in state "state"
        """
        return len(self.state_actions[state])

    @abstractmethod
    def execute(self, action):
        """
        Execute action "action" in current state "state". Attributes "state", "reward" and "holding_time"
        must be updated.
        :param action: index of the action to be executed
        :return: history or None
        """
        pass

    @abstractmethod
    def compute_max_gain(self):
        """
        :return: optimal average reward of the SMDP
        """
        pass
