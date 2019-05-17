import numpy as np


class InventoryCorrelated():

    def __init__(self, max_good=10, max_action=None):
        self.max_good = max_good
        self.state = 0

        if max_action is None:
            self.max_action = max_good
        else:
            self.max_action = max_action

        self.state_actions = []
        for s in range(max_good):
            self.state_actions.append(range(max_action))

    def reset(self, state=None):
        if state is None:
            self.state = 0
        else:
            self.state = state

    def execute(self, action):
        assert action in self.state_actions[self.state]
        p = q
        demand = np.random.choice(range(self.state), p=
        stp1 = self.state + action - demand