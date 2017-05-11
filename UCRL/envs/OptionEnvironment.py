from .Environment import Environment
from UCRL.Tree import Node

import numpy as np
import copy


class OptionEnvironment(Environment):
    """
    This is a subclass of abstract class "Environment". It implements an SMDP with macro-actions (also called options).
    The original SMDP with primitive actions is stored in the attribute "environment".
    Note that it is possible for "environment" to be an SMDP with options. Thus, this implementation allows to
    define any arbitrary hierarchy of options (i.e., options over options over options...).
    """

    def __init__(self, environment, state_options, options_policies,
                 options_terminating_conditions, is_optimal=False, index_min_options=None):
        """
        :param environment: instance of any subclass of abstract class "Environment" representing the original SMDP
                            without options, it can thus be an instance of OptionEnvironment (hierarchy of options)
        :param state_options: 2D-array of integers representing the indices of options available in every state
        :param options_policies: 2D-array of integers representing the policies of every option,
                                 for every option a policy is characterized by the index of the primitive action to be
                                 executed in every state of the original environment (|O| x |S|)
        :param options_terminating_conditions: 2D-array of floats in [0,1] representing the termination condition of
                                 every option, for every option a termination condition is characterized by the
                                 probability of ending the option in every state of the original environment
        :param is_optimal: True if and only if there exists an optimal policy using only options, default value is False
        """
        self.environment = copy.deepcopy(environment)
        self.is_optimal = is_optimal
        super().__init__(self.environment.state, self.environment.state_actions)  # call parent constructor
        self.state_options = state_options
        self.options_policies = options_policies
        self.options_terminating_conditions = options_terminating_conditions
        self.max_nb_actions_per_state = max(map(len, self.state_options))  # Maximal number of options per state
        self.nb_states = sum(map(lambda x: x > 0, map(len, self.state_options)))  # Total number of states in the new SMDP
        if index_min_options is None:
            self.index_min_options = min(map(lambda x: min(x, default = float("inf")), self.state_options))
        else:
            self.index_min_options = index_min_options

    def get_state_actions(self):
        """
        :return: 2D-array state_options
        """
        return self.state_options

    def get_available_actions(self):
        """
        :return: list of indices of options (macro-actions) available in current state
        """
        return self.state_options[self.state]

    def get_nb_available_actions(self):
        """
        :return: number of options (macro-actions) available in current state
        """
        return len(self.state_options[self.state])

    def get_available_actions_state(self, state):
        """
        :param state: index of any state in the SMDP with options
        :return: list of indices of options (macro-actions) available in state "state"
        """
        return self.state_options[state]

    def get_index_of_action_state(self, state, action):
        if action in self.state_options[state]:
            return self.state_options[state].index(action)
        return None

    def get_nb_available_actions_state(self, state):
        """
        :param state: index of any state in the SMDP
        :return: number of actions available in state "state"
        """
        return len(self.state_options[state])

    def execute(self, option):
        """
        :param option: index of the option to be executed
        :return: nothing
        """
        self.environment.state = self.state
        stop = 0  # boolean variable encoded as an integer in {0,1}, False/0 if the option should continue, 1 otherwise
        t = 0  # cumulative time spent executing current option
        r = 0  # cumulative reward obtained executing current option
        history_data = [self.environment.state, option]
        history = Node(history_data)
        while not stop:  # continue until stopping condition is met
            s = self.environment.state
            action = self.options_policies[option - self.index_min_options][self.environment.state]  # execute option policy
            sub_history = self.environment.execute(action)  # update original environment
            t += self.environment.holding_time
            r += self.environment.reward
            if sub_history is None:
                history.add_child(Node([s, action, self.environment.reward, self.environment.holding_time, self.environment.state]))
            else:
                history.add_child(sub_history)
            proba_stop = self.options_terminating_conditions[option - self.index_min_options][self.environment.state]
            if 1 > proba_stop > 0:  # Bernouilli trial: stop with probability p=proba_stop
                stop = np.random.binomial(n=1, p=proba_stop)
            else:
                stop = proba_stop
        self.state = self.environment.state  # final state reached when the option ends
        self.holding_time = t
        self.reward = r
        history_data += [r, t, self.environment.state]
        return history

    def compute_max_gain(self):
        """
        :return: maximal average reward that can be obtained using only policies over options
        """
        if self.is_optimal:
            return self.environment.max_gain  # optimal set of options
        else:
            return None  # TODO: implement value iteration to compute maximal gain with non-optimal options

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def nb_options(self):
        return len(self.options_policies)
