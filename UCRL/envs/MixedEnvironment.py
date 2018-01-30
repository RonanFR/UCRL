from .OptionEnvironment import OptionEnvironment
from .Environment import Environment

import numpy as np
import copy


class MixedEnvironment(OptionEnvironment):
    """
    Mixed environment with both primitive actions and options.
    """

    def __init__(self, environment,
                 state_options, options_policies, options_terminating_conditions,
                 delete_environment_actions=None, is_optimal=False):
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
        :param delete_environment_actions (list, "all"): do not use actions of the provided environment.
                                Use only given options
        :param is_optimal: True if and only if there exists an optimal policy using only options and primitive actions
                           not in delete_state_actions, default value is False        
        """
        assert isinstance(environment, Environment)

        threshold_options = max(map(max, environment.get_state_actions()))

        if delete_environment_actions is not None:
            if isinstance(delete_environment_actions, list):
                delete_environment_actions = np.unique(delete_environment_actions)
            elif isinstance(delete_environment_actions, str):
                if delete_environment_actions.lower() != "all":
                    raise ValueError(
                        "The only admissible string for delete_environment_actions is 'all'")
                delete_environment_actions = np.arange(threshold_options+1)
            else:
                raise ValueError("Unknown value for delete_environment_actions")

        if hasattr(environment, 'is_optimal'):
            is_optimal = environment.is_optimal or is_optimal

        # increment the identifier of the options and
        # extend it with environment actions
        environment_actions = environment.get_state_actions()
        new_state_options = [[]] * len(state_options)
        for s, actions in enumerate(environment_actions):
            new_state_options[s] = np.setdiff1d(list(actions),
                                            delete_environment_actions).tolist() + \
                               [x + threshold_options + 1 for x in state_options[s]]

        self.threshold_options = threshold_options
        self.delete_environment_actions = delete_environment_actions
        super().__init__(environment=environment,
                         state_options=new_state_options,
                         options_policies=options_policies,
                         options_terminating_conditions=options_terminating_conditions,
                         is_optimal=is_optimal)  # call parent constructor

        self.max_nb_options_per_state = max(map(len, state_options))

    def reset(self):
        self.environment.reset()

    def execute(self, option):
        """
        :param option: index of the option to be executed
        :return: history
        """
        if option > self.threshold_options:
            # call the parent class that performs an option
            history = super(MixedEnvironment, self).execute(option)
        else:
            self.environment.state = self.state
            history  = self.environment.execute(option)
            self.state = self.environment.state  # final state reached when the action ends
            self.holding_time = self.environment.holding_time
            self.reward = self.environment.reward
        return history

    def is_primitive_action(self, act_index):
            return act_index <= self.threshold_options

    def get_index_of_action_state_in_mdp(self, state, action):
        return self.environment.get_index_of_action_state(state, action)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def max_nb_mdp_actions_per_state(self):
        return self.environment.max_nb_actions_per_state

