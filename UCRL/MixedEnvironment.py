from OptionEnvironment import OptionEnvironment

import numpy as np
import copy


class MixedEnvironment(OptionEnvironment):
    """
    Mixed environment with both primitive actions and options.
    """

    def __init__(self, environment, state_options, options_policies, options_terminating_conditions,
                 delete_state_actions=[], is_optimal=False):
        """
        :param environment: instance of any subclass of abstract class "Environment" representing the original SMDP
                            without options, it can thus be an instance of OptionEnvironment (hierarchy of options)
        :param state_options: 2D-array of integers representing the indices of options available in every state
        :param options_policies: 2D-array of integers representing the policies of every option,
                                 for every option a policy is characterized by the index of the primitive action to be
                                 executed in every state of the original environment
        :param options_terminating_conditions: 2D-array of floats in [0,1] representing the termination condition of
                                 every option, for every option a termination condition is characterized by the
                                 probability of ending the option in every state of the original environment
        :param delete_state_actions:
        :param is_optimal: True if and only if there exists an optimal policy using only options and primitive actions
                           not in delete_state_actions, default value is False
        """
        if delete_state_actions:
            for s, actions in enumerate(environment.get_state_actions()):
                for action in delete_state_actions[s]:
                    actions.remove(action)
        else:
            try:
                is_optimal = environment.is_optimal
            except AttributeError:
                is_optimal = True
        self.new_threshold_options = max(map(max, environment.get_state_actions()))
        try:
            self.threshold_options = environment.threshold_options
        except AttributeError:
            self.threshold_options = self.new_threshold_options
        for s, actions in enumerate(environment.get_state_actions()):
            state_options[s] = list(map(lambda x: x + self.new_threshold_options + 1, state_options[s]))
        self.option_environment = OptionEnvironment(environment, state_options, options_policies,
                                                    options_terminating_conditions)
        state_options = copy.deepcopy(state_options)
        for s, actions in enumerate(environment.get_state_actions()):
            state_options[s] = actions + state_options[s]
        super().__init__(environment, state_options, options_policies, options_terminating_conditions, is_optimal)  # call parent constructor

    def execute(self, option):
        """
        :param option: index of the option to be executed
        :return: nothing
        """
        if option > self.new_threshold_options:
            self.option_environment.state = self.state
            history = self.option_environment.execute(option)
            self.state = self.option_environment.state  # final state reached when the option ends
            self.holding_time = self.option_environment.holding_time
            self.reward = self.option_environment.reward
        else:
            self.environment.state = self.state
            history = self.environment.execute(option)
            self.state = self.environment.state  # final state reached when the option ends
            self.holding_time = self.environment.holding_time
            self.reward = self.environment.reward
        return history
