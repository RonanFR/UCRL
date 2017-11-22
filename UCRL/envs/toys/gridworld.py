import numpy as np
import numbers
from ..Environment import Environment
from ...utils import gridrender as gui
from tkinter import Tk
import tkinter.font as tkFont
import copy
from collections import namedtuple

MDP = namedtuple('MDP', 'S,A,P,R,gamma,d0')


class GridWorld(Environment):
    def __init__(self, gamma=1., grid=None, render=False, proba_succ=0.7,
                 reset_distribution=None):
        self.grid = grid

        self.action_names = np.array(['right', 'down', 'left', 'up'])

        self.n_rows, self.n_cols = len(self.grid), max(map(len, self.grid))

        # Create a map to translate coordinates [r,c] to scalar index
        # (i.e., state) and vice-versa
        self.coord2state = np.empty_like(self.grid, dtype=np.int)
        self.n_states = 0
        self.state2coord = []
        for i in range(self.n_rows):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != 'x':
                    self.coord2state[i, j] = self.n_states
                    self.n_states += 1
                    self.state2coord.append([i, j])
                else:
                    self.coord2state[i, j] = -1

        # compute the actions available in each state
        state_actions = self.compute_available_actions()
        self.gamma = gamma
        self.proba_succ = proba_succ
        self.render = render

        if reset_distribution is None:
            self.d0 = np.ones((self.nstates,)) / self.nstates
        else:
            self.d0 = reset_distribution
            assert len(self.d0) == self.n_states

        super(GridWorld, self).__init__(initial_state=0,
                                        state_actions=state_actions)

    def execute(self, action):
        """
        Args:
            action (int): the action to be executed
        """
        ##########################################################################
        ##  If we don't want to use the transition probability matrix
        ##########################################################################
        # r, c = self.state2coord[state]
        # assert action in self.state_actions[state]
        # if isinstance(self.grid[r][c], numbers.Number):
        #     return state, 0, True
        # else:
        #     failed = np.random.rand(1) > self.proba_succ
        #     if action == 0:
        #         c = min(self.n_cols - 1, c + 1) if not failed else max(0, c - 1)
        #     elif action == 1:
        #         r = min(self.n_rows - 1, r + 1) if not failed else max(0, r - 1)
        #     elif action == 2:
        #         c = max(0, c - 1) if not failed else min(self.n_cols - 1, c + 1)
        #     elif action == 3:
        #         r = max(0, r - 1) if not failed else min(self.n_rows - 1, r + 1)
        #
        #     if self.grid[r][c] == 'x':
        #         next_state = state
        #         r, c = self.state2coord[next_state]
        #     else:
        #         next_state = self.coord2state[r, c]
        #     if isinstance(self.grid[r][c], numbers.Number):
        #         reward = self.grid[r][c]
        #         absorb = True
        #     else:
        #         reward = 0.
        #         absorb = False
        #
        # if self.render:
        #     self.show(state, action, next_state, reward)
        #
        # return next_state, reward, absorb

        try:
            action_index = self.state_actions[self.state].index(action)
        except:
            r, c = self.state2coord[self.state]
            raise ValueError("Action {} cannot be executed in this state {} [{}, {}]".format(action, self.state, r, c))
        p = self.P_mat[self.state, action_index]
        next_state = np.asscalar(np.random.choice(self.nb_states, 1, p=p))
        assert p[next_state] > 0.
        self.reward = self.R_mat[self.state, action_index]
        self.state = next_state

    def show(self, state, action, next_state, reward):
        dim = 200
        rows, cols = len(self.grid) + 0.5, max(map(len, self.grid))
        if not hasattr(self, 'window'):
            root = Tk()
            self.window = gui.GUI(root)

            self.window.config(width=cols * (dim + 12), height=rows * (dim + 12))
            my_font = tkFont.Font(family="Arial", size=32, weight="bold")
            for s in range(self.n_states):
                r, c = self.state2coord[s]
                x, y = 10 + c * (dim + 4), 10 + r * (dim + 4)
                if isinstance(self.grid[r][c], numbers.Number):
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='blue', width=2)
                    self.window.create_text(x + dim / 2., y + dim / 2., text="{:.1f}".format(self.grid[r][c]),
                                            font=my_font, fill='white')
                else:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='white', width=2)
            self.window.pack()

        my_font = tkFont.Font(family="Arial", size=32, weight="bold")

        r0, c0 = self.state2coord[state]
        r0, c0 = 10 + c0 * (dim + 4), 10 + r0 * (dim + 4)
        x0, y0 = r0 + dim / 2., c0 + dim / 2.
        r1, c1 = self.state2coord[next_state]
        r1, c1 = 10 + c1 * (dim + 4), 10 + r1 * (dim + 4)
        x1, y1 = r1 + dim / 2., c1 + dim / 2.

        if hasattr(self, 'oval2'):
            # self.window.delete(self.line1)
            # self.window.delete(self.oval1)
            self.window.delete(self.oval2)
            self.window.delete(self.text1)
            self.window.delete(self.text2)

        # self.line1 = self.window.create_arc(x0, y0, x1, y1, dash=(3,5))
        # self.oval1 = self.window.create_oval(x0 - dim / 20., y0 - dim / 20., x0 + dim / 20., y0 + dim / 20., dash=(3,5))
        self.oval2 = self.window.create_oval(x1 - dim / 5., y1 - dim / 5., x1 + dim / 5., y1 + dim / 5., fill='red')
        self.text1 = self.window.create_text(dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="r= {:.1f}".format(reward), anchor='center')
        self.text2 = self.window.create_text(2 * dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="action: {}".format(self.action_names[action]), anchor='center')
        self.window.update()

    def compute_available_actions(self):
        # define available actions in each state
        # actions are indexed by: 0=right, 1=down, 2=left, 3=up
        state_actions = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if isinstance(self.grid[i][j], numbers.Number):
                    state_actions.append([0])
                elif self.grid[i][j] != 'x':
                    actions = [0, 1, 2, 3]
                    if i == 0:
                        actions.remove(3)
                    if j == self.n_cols - 1:
                        actions.remove(0)
                    if i == self.n_rows - 1:
                        actions.remove(1)
                    if j == 0:
                        actions.remove(2)

                    for a in copy.copy(actions):
                        r, c = i, j
                        if a == 0:
                            c = min(self.n_cols - 1, c + 1)
                        elif a == 1:
                            r = min(self.n_rows - 1, r + 1)
                        elif a == 2:
                            c = max(0, c - 1)
                        else:
                            r = max(0, r - 1)
                        if self.grid[r][c] == 'x':
                            actions.remove(a)

                    state_actions.append(actions)
        return state_actions
                    
    def matrix_representation(self):
        """
        Returns:
             A representation of the MDP in matrix form MDP(S, A_s, P, R, gamma) where
             - S is the number of states
             - A_s contains the list of action indices available in each state, i.e.,
                A_s[3] is a list representing the index of actions available in such state
             - P the transition matrix of dimension S x max{|A_s|} x S
             - R the reward matrix of dimension S x max{|A_s|}
             - gamma: discount factor
             - d0: initial state distribution S x 1
        """
        if hasattr(self, 'P_mat'):
            return MDP(self.n_states, self.state_actions, self.P_mat, self.R_mat, self.gamma, self.d0)

        nstates = self.n_states
        nactions = max(map(len, self.state_actions))
        self.P_mat = np.inf * np.ones((nstates, nactions, nstates))
        self.R_mat = np.inf * np.ones((nstates, nactions))
        for s in range(nstates):
            r, c = self.state2coord[s]
            for a_idx, action in enumerate(self.state_actions[s]):
                self.P_mat[s, a_idx].fill(0.)
                if isinstance(self.grid[r][c], numbers.Number):
                    self.P_mat[s, a_idx, :] = self.d0
                    self.R_mat[s, a_idx] = 0.
                else:
                    ns_succ, ns_fail = np.inf, np.inf
                    if action == 0:
                        ns_succ = self.coord2state[r, min(self.n_cols - 1, c + 1)]
                        ns_fail = self.coord2state[r, max(0, c - 1)]
                    elif action == 1:
                        ns_succ = self.coord2state[min(self.n_rows - 1, r + 1), c]
                        ns_fail = self.coord2state[max(0, r - 1), c]
                    elif action == 2:
                        ns_succ = self.coord2state[r, max(0, c - 1)]
                        ns_fail = self.coord2state[r, min(self.n_cols - 1, c + 1)]
                    elif action == 3:
                        ns_succ = self.coord2state[max(0, r - 1), c]
                        ns_fail = self.coord2state[min(self.n_rows - 1, r + 1), c]

                    x, y = self.state2coord[ns_fail]
                    if ns_fail == -1 or self.grid[x][y] == 'x':
                        ns_fail = s

                    self.P_mat[s, a_idx, ns_succ] = self.proba_succ
                    self.P_mat[s, a_idx, ns_fail] = 1. - self.proba_succ

                    x, y = self.state2coord[ns_fail]
                    x2, y2 = self.state2coord[ns_succ]
                    r_succ, r_fail = 0., 0.
                    if isinstance(self.grid[x][y], numbers.Number):
                        r_fail = self.grid[x][y]
                    if isinstance(self.grid[x2][y2], numbers.Number):
                        r_succ = self.grid[x2][y2]

                    self.R_mat[s, a_idx] = self.proba_succ * r_succ + (1 - self.proba_succ) * r_fail


        return MDP(nstates, self.state_actions, self.P_mat, self.R_mat, self.gamma, self.d0)
