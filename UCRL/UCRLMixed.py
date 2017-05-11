from .Ucrl import AbstractUCRL
from .envs import MixedEnvironment
import numpy as np
import math as m


class FreeUCRL_1(AbstractUCRL):

    def __init__(self, environment, r_max,
                 range_r=-1, range_p=-1):

        assert isinstance(environment, MixedEnvironment)

        super(FreeUCRL_1, self).__init__(environment=environment,
                                         r_max=r_max, range_r=range_r,
                                         range_p=range_p, solver=[])

        nb_states = self.environment.nb_states
        # this vector stores all the possible primitive actions

        nb_primitive_actions = environment.threshold_options + 1
        nb_options = environment.max_nb_actions - nb_primitive_actions

        self.estimated_probabilities_opt = np.zeros((nb_states, nb_options))
        self.estimated_probabilities_mdp = np.zeros((nb_states, nb_primitive_actions))
        self.estimated_rewards_mdp = np.ones((nb_states, nb_primitive_actions)) * r_max
        self.nu_k_mdp = np.zeros((nb_states, nb_primitive_actions))
        self.nb_observations_mdp = np.zeros((nb_states, nb_primitive_actions))
        self.nu_k_opt = np.zeros((nb_states, nb_options))
        self.nb_observations_opt = np.zeros((nb_states, nb_options))

    def learn(self, duration, regret_time_step):
        """ Run UCRL on the provided environment

        Args:
            duration (int): the algorithm is run until the number of time steps 
                            exceeds "duration"
            regret_time_step (int): the value of the cumulative regret is stored
                                    every "regret_time_step" time steps

        """

    def beta_r(self):
        nb_states, nb_prim_actions = self.estimated_rewards_mdp.shape
        return np.multiply(self.range_r, np.sqrt(7/2 * m.log(2 * nb_states * nb_prim_actions *
                                                (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations_mdp)))

    def beta_p(self):
        nb_states, nb_prim_actions = self.estimated_rewards_mdp.shape
        return self.range_p * np.sqrt(14 * nb_states * m.log(2 * nb_prim_actions
                    * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations_mdp))

    def update(self):
        s = self.environment.state  # current state
        history = self.environment.execute(self.policy[s])
        s2 = self.environment.state  # new state
        r = self.environment.reward
        t = self.environment.holding_time
        self.total_reward += r
        self.total_time += t
        self.iteration += 1
        if history is None:
            # this is a primitive action
            act_index = self.policy_indices[s]
            self.estimated_rewards_mdp[s][act_index] *= self.nb_observations_mdp[s][act_index] / (self.nb_observations_mdp[s][act_index] + 1)
            self.estimated_rewards_mdp[s][act_index] += 1 / (self.nb_observations_mdp[s][act_index] + 1) * r
            self.estimated_probabilities_mdp[s][act_index] *= self.nb_observations_mdp[s][act_index] / (self.nb_observations_mdp[s][act_index] + 1)
            self.estimated_probabilities_mdp[s][act_index][s2] += 1 / (self.nb_observations_mdp[s][act_index] + 1)

            self.estimated_probabilities[s][act_index] *= self.nb_observations[s][act_index] / (self.nb_observations[s][act_index] + 1)
            self.estimated_probabilities[s][act_index][s2] += 1 / (self.nb_observations[s][act_index] + 1)

            self.nu_k_mdp[s][act_index] += 1
        else:
            self.update_history(history)

    def update_history(self, history):
        [s, o, r, t, s2] = history.data # first is for sure an option
        option_index = o

        self.estimated_probabilities[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
        self.estimated_probabilities[s][option_index][s2] += 1 / (self.nb_observations[s][option_index] + 1)

        # the rest of the actions are for sure primitive actions (1 level hierarchy)
        assert len(history.children) == 1
        self.update_history_pa(history.children[0])

    def update_history_pa(self, node):
        [s, a, r, t, s2] = node.data # this is a primitive action which can be valid or not

        if self.environment.is_valid_pa(a):
            # if it is a valid action for the mdp
            self.estimated_probabilities[s][a] *= self.nb_observations[s][a] / (self.nb_observations[s][a] + 1)
            self.estimated_probabilities[s][a][s2] += 1 / (self.nb_observations[s][a] + 1)

        # for sure it is a primitive action
        self.estimated_rewards_mdp[s][a] *= self.nb_observations_mdp[s][a] / (self.nb_observations_mdp[s][a] + 1)
        self.estimated_rewards_mdp[s][a] += 1 / (self.nb_observations_mdp[s][a] + 1) * r
        self.estimated_probabilities_mdp[s][a] *= self.nb_observations_mdp[s][a] / (self.nb_observations_mdp[s][a] + 1)
        self.estimated_probabilities_mdp[s][a][s2] += 1 / (self.nb_observations_mdp[s][a] + 1)
        self.nu_k_mdp[s][a] += 1

        # the rest of the actions are for sure primitive actions (1 level hierarchy)
        assert len(node.children) <= 1
        if len(node.children) > 0:
            self.update_history_pa(node.children[0])

    def solve_optimistic_model(self):
        nb_options = len(self.environment.options_policies)
        nb_states = self.environment.nb_states

        # compute estimated reward for each option
        r_hat = []
        mu = []
        condition_numbers = []
        for i in range(self.environment.max_nb_actions):
            if self.environment.is_primitive_action(i, isIndex=True):
                # this is a primitive action
                r_hat.append(self.estimated_rewards_mdp[i])
            else:
                # this is an option
                o = self.environment.get_zerobased_option_index(i, isIndex=True)
                L = []
                for s, a in enumerate(self.environment.options_policies[o]):
                    L.append(self.estimated_rewards_mdp[s][a])
                r_hat.append(L)

                # --------------------------------------------------------------
                # Compute condition number
                # --------------------------------------------------------------
                term_cond = self.environment.options_terminating_conditions[o]
                Q_o = np.zeros((nb_states, nb_states))
                q_o = np.zeros((nb_states,))
                for s in range(nb_states):
                    act = self.environment.options_policies[o][s]
                    q_o[s] = 0.0
                    for nexts in range(nb_states):
                        prob = self.estimated_probabilities[s][act][nexts]
                        q_o[s] += term_cond[nexts] * prob
                        Q_o[s,nexts] = (1.0 - term_cond[nexts]) * prob

                Pprime_o = np.concatenate((q_o, Q_o[:,1:]), axis=1)

                Pap = (Pprime_o + np.eye(nb_states)) / 2.
                D, U = np.linalg.eig(np.transpose(Pap))  # eigen decomposition of transpose of P
                sorted_indices = np.argsort(np.real(D))
                mu_o = np.transpose(np.real(U))[sorted_indices[-1]]
                mu_o /= np.sum(mu_o)  # stationary distribution
                mu.append(mu_o)

                P_star = np.repeat(np.array(mu, ndmin=2), nb_states, axis=0)  # limiting matrix

                # Compute deviation matrix
                I = np.eye(nb_states)  # identity matrix
                Z = np.linalg.inv(I - Pap + P_star)  # fundamental matrix
                H = np.dot(Z, I - P_star)  # deviation matrix

                condition_nb = 0  # condition number of deviation matrix
                for i in range(0, nb_states):  # Seneta's condition number
                    for j in range(i + 1, nb_states):
                        condition_nb = max(condition_nb,
                                           0.5 * np.linalg.norm(H[i, :] - H[j, :], ord=1))
                condition_numbers.append(condition_nb)
