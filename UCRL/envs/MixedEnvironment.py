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
                           
        Examples
        environment.state_action = [[0,1,3]...]
        state_options = [[0, 1], ....]
        delete_environment_actions = [1]
        
        self.threshold_options = 3 (we have 4 actions)
        self.state_options = [[0,2,3,4]...
        # state_options contains always indices of available actions without jumps
        
        self.map_index2action = {0: 0 
        1: 2
        2: 3
        3: 4 -> first option
        4: 5 -> second option
        ...
        }
        self.min_index_option = 3 (self.map_index2action[lf.threshold_options + 1])
        
        """
        assert isinstance(environment, Environment)

        threshold_options = environment.max_nb_actions-1
        nb_options = len(options_policies)

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

        map_index2action = dict()
        map_action2index = dict()
        index = 0
        for i in range(threshold_options + 1 + nb_options):
            if i not in delete_environment_actions:
                map_index2action[index] = i
                map_action2index[i] = index
                index += 1

        if hasattr(environment, 'is_optimal'):
            is_optimal = environment.is_optimal or is_optimal

        # increment the identifier of the options and
        # extend it with environment actions
        environment_actions = environment.get_state_actions()
        for s, actions in enumerate(environment_actions):
            state_options[s] = np.setdiff1d(list(actions),
                                            delete_environment_actions).tolist() + \
                               [map_action2index[x + threshold_options + 1] for x in state_options[s]]
            # this ensures that there are no jumps

        self.map_index2action = map_index2action
        self.map_action2index = map_action2index
        self.threshold_options = threshold_options
        self.delete_environment_actions = delete_environment_actions
        super().__init__(environment=environment,
                         state_options=state_options,
                         options_policies=options_policies,
                         options_terminating_conditions=options_terminating_conditions,
                         is_optimal=is_optimal,
                         index_min_options=map_action2index[threshold_options + 1])  # call parent constructor

    def execute(self, option):
        """
        :param option: index of the option to be executed
        :return: history
        """
        index_full = self.map_index2action[option] # move to complete list
        if index_full > self.threshold_options:
            # call the parent class that performs an option
            history = super(MixedEnvironment, self).execute(option)
        else:
            self.environment.state = self.state
            history  = self.environment.execute(option)
            self.state = self.environment.state  # final state reached when the action ends
            self.holding_time = self.environment.holding_time
            self.reward = self.environment.reward
        return history

    def is_primitive_action(self, act_index, isIndex=True):
        if isIndex:
            return self.map_index2action[act_index] <= self.threshold_options
        else:
            return act_index <= self.threshold_options

    def get_action_from_index(self, act_index):
        return self.map_index2action[act_index]

    def get_index_from_action(self, action):
        return self.map_action2index[action]

    def get_zerobased_option_index(self, option, isIndex=True):
        if isIndex:
            assert self.map_index2action[option] > self.threshold_options
            return self.map_index2action[option] - self.threshold_options - 1
        else:
            return option - self.threshold_options - 1

    def is_valid_primitive_action(self, act_index, isIndex=True):
        if isIndex:
            assert self.map_index2action[act_index] <= self.threshold_options
            return self.map_index2action[act_index] not in self.delete_environment_actions
        else:
            return act_index not in self.delete_environment_actions

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def nb_actions(self):
        """Gives the number of available actions (options or primitive actions)
        
        Returns:
             int
        """
        return len(self.map_index2action.keys())

    @property
    def nb_total_primitive_actions(self):
        """Gives the number of primitive actions (available and not)
        
        Returns:
             int
        """
        return self.threshold_options + 1

    @property
    def nb_available_primitive_actions(self):
        """Gives the number of primitive actions available
        
        Returns:
            int
        """
        val = self.nb_actions - self.nb_options
        assert val == self.nb_total_primitive_actions - len(self.delete_environment_actions)
        return val

    @property
    def index_first_option(self):
        """Gives the index of the first option in the set of available actions
        
        Returns:
            int
        """
        return self.map_action2index[self.threshold_options + 1]

