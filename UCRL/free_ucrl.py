from .Ucrl import AbstractUCRL
from .envs import MixedEnvironment
import numpy as np
import math as m
from .logging import default_logger

class FreeUCRL_Alg1(AbstractUCRL):
    def __init__(self, environment, r_max, range_r=-1, range_p=-1,
                 logger=default_logger):
        assert isinstance(environment, MixedEnvironment)

        evi = EVI_Alg1(nb_states=environment.nb_states,
                       thr=environment.threshold_options,
                       map_index2action=environment.map_index2action,
                       option_offset=environment.index_first_option,
                       action_per_state=environment.state_actions)

        super(FreeUCRL_Alg1, self).__init__(environment=environment,
                                            r_max=r_max, range_r=range_r,
                                            range_p=range_p, solver=evi,
                                            logger=logger)

        nb_states = self.environment.nb_states
        tot_nb_prim_actions = environment.nb_total_primitive_actions
        nb_options = environment.nb_options

        # keep track of the transition probabilities for options and actions
        self.estimated_probabilities_opt = np.ones((nb_states, nb_options, nb_states)) / nb_states
        self.estimated_probabilities_mdp = np.ones((nb_states, tot_nb_prim_actions, nb_states)) / nb_states

        # keep track of visits
        self.nb_observations_opt = np.zeros((nb_states, nb_options))
        self.nb_observations_mdp = np.zeros((nb_states, tot_nb_prim_actions))
        self.nu_k_opt = np.zeros((nb_states, nb_options))
        self.nu_k_mdp = np.zeros((nb_states, tot_nb_prim_actions))

        # reward info
        self.estimated_rewards_mdp = np.ones((nb_states, tot_nb_prim_actions)) * r_max

    def learn(self, duration, regret_time_step):
        if self.total_time >= duration:
            return
        threshold = self.total_time + regret_time_step
        threshold_span = threshold

        self.timing = []

        while self.total_time < duration:
            self.episode += 1
            # print(self.total_time)

            # initialize the episode
            self.nu_k_mdp.fill(0)
            self.nu_k_opt.fill(0)
            self.delta = 1 / m.sqrt(self.iteration + 1)

            # solve the optimistic (extended) model
            span_value = self.solve_optimistic_model()

            if self.total_time > threshold_span:
                self.span_values.append(span_value / self.r_max)
                self.span_times.append(self.total_time)
                threshold_span = self.total_time + regret_time_step

            # execute the recovered policy
            curr_st = self.environment.state
            curr_ac = self.policy_indices[curr_st]
            while self.nu_k_mdp[curr_st][curr_ac] < max(1, self.nb_observations_mdp[curr_st][curr_ac]) \
                    and self.total_time < duration:
                self.update()
                if self.total_time > threshold:
                    self.regret.append(self.total_time * self.environment.max_gain - self.total_reward)
                    self.unit_duration.append(self.total_time / self.iteration)
                    threshold = self.total_time + regret_time_step
                # get current state and action
                curr_st = self.environment.state
                curr_ac = self.policy_indices[curr_st]

            self.nb_observations_mdp += self.nu_k_mdp

    def beta_r(self):
        nb_states, nb_prim_actions = self.estimated_rewards_mdp.shape
        beta_r =  np.multiply(self.range_r, np.sqrt(7/2 * m.log(2 * nb_states * nb_prim_actions *
                                                (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations_mdp)))
        return beta_r

    def beta_p(self):
        nb_states = self.environment.nb_states
        total_actions = self.environment.nb_actions
        beta_p_opt =  self.range_p * np.sqrt(14 * nb_states * m.log(2 * total_actions
                    * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations_opt))
        beta_p_mdp =  self.range_p * np.sqrt(14 * nb_states * m.log(2 * total_actions
                    * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations_mdp))
        return beta_p_mdp, beta_p_opt

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
            # update mdp information
            pa_index = self.environment.get_action_from_index(self.policy_indices[s]) # convert index to real primitive action
            self.update_from_action(s, pa_index, s2, r)
        else:
            self.update_from_option(history)

    def update_from_option(self, history):
        [s, o, r, t, s2] = history.data # first is for sure an option
        option_index = self.environment.get_zerobased_option_index(o)
        self.estimated_probabilities_opt[s][option_index] *= self.nb_observations_opt[s][option_index] / (self.nb_observations_opt[s][option_index] + 1)
        self.estimated_probabilities_opt[s][option_index][s2] += 1 / (self.nb_observations_opt[s][option_index] + 1)
        self.nu_k_opt[s][option_index] += 1

        # the rest of the actions are for sure primitive actions (1 level hierarchy)
        node = history
        while len(node.children) > 0:
            assert len(node.children) == 1, "history has more than one child"

            # advance node
            node = node.children[0]
            [s, a, r, t, s2] = node.data # this is a primitive action which can be valid or not
            # for sure it is a primitive action (and the index corresponds to all the primitive actions
            self.update_from_action(s, a, s2, r)

    def update_from_action(self, s, pa_index, s2, r):
        self.estimated_rewards_mdp[s][pa_index] *= self.nb_observations_mdp[s][pa_index] / (self.nb_observations_mdp[s][pa_index] + 1)
        self.estimated_rewards_mdp[s][pa_index] += 1 / (self.nb_observations_mdp[s][pa_index] + 1) * r
        self.estimated_probabilities_mdp[s][pa_index] *= self.nb_observations_mdp[s][pa_index] / (self.nb_observations_mdp[s][pa_index] + 1)
        self.estimated_probabilities_mdp[s][pa_index][s2] += 1 / (self.nb_observations_mdp[s][pa_index] + 1)

        self.nu_k_mdp[s][pa_index] += 1

    def solve_optimistic_model(self):
        nb_states = self.environment.nb_states
        nb_options = self.environment.nb_options

        # compute estimated reward, mu and condition number for each option
        r_hat_opt = np.empty((nb_states, nb_options))
        mu_opt = np.empty((nb_states, nb_options))
        condition_numbers_opt = np.empty((nb_options,))
        for i in range(self.environment.nb_actions):
            if self.environment.is_primitive_action(i, isIndex=True):
                # this is a primitive action
                pass
            else:
                # this is an option
                o = self.environment.get_zerobased_option_index(i, isIndex=True)
                for s, a in enumerate(self.environment.options_policies[o]):
                    r_hat_opt[s,o] = self.estimated_rewards_mdp[s][a]

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
                        prob = self.estimated_probabilities_mdp[s][act][nexts]
                        q_o[s] += term_cond[nexts] * prob
                        Q_o[s,nexts] = (1. - term_cond[nexts]) * prob

                Pprime_o = np.concatenate((q_o, Q_o[:,1:]), axis=1)

                Pap = (Pprime_o + np.eye(nb_states)) / 2.
                D, U = np.linalg.eig(np.transpose(Pap))  # eigen decomposition of transpose of P
                sorted_indices = np.argsort(np.real(D))
                mu = np.transpose(np.real(U))[sorted_indices[-1]]
                mu /= np.sum(mu)  # stationary distribution
                mu_opt[:, o] = mu

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
                condition_numbers_opt[o] = condition_nb

class EVI_Alg1(object):

    def __init__(self, nb_states, thr, map_index2action,
                 option_offset, action_per_state):
        self.mtx_maxprob_memview = np.zeros((nb_states, nb_states))
        self.mtx_maxmu_memview = np.zeros((nb_states, nb_states))
        self.threshold = thr
        self.index2action = map_index2action
        self.option_offset = option_offset
        self.actions_per_state = action_per_state

    def run(self, policy_indices, policy,
            p_hat_mdp,
            p_hat_opt,
            r_tilde_mdp,
            r_tilde_opt,
            beta_p_mdp,
            beta_p_opt,
            mu_opt,
            beta_opt,
            r_max,
            epsilon):
        nb_states = p_hat_mdp.shape[0]
        sorted_indices_u = np.arange(nb_states)
        while True:
            for s in range(nb_states):
                first_action = 1
                for action_idx in self.actions_per_state[s]:
                    action = self.index2action[action_idx]
                    if action <= self.threshold:
                        #this is an action
                        self.max_proba(p_hat_mdp[s][action],
                                       sorted_indices_u, beta_p_mdp[s][action],
                                       self.mtx_maxprob_memview[s])
                        self.mtx_maxprob_memview[s][s] = self.mtx_maxprob_memview[s][s] - 1.  # ????

                        r_optimal = min(r_max, r_tilde_mdp[s,action])
                        v = r_optimal + np.dot(self.mtx_maxprob_memview[s], u1)
                    else:
                        o = action - self.option_offset
                        self.max_proba(p_hat_opt[s][o],
                                       sorted_indices_u, beta_p_opt[s][o],
                                       self.mtx_maxprob_memview[s])
                        self.mtx_maxprob_memview[s][s] = self.mtx_maxprob_memview[s][s] - 1.  # ????

                        x  = np.min(r_max, r_tilde_opt[:,o])
                        x[s] += np.dot(self.mtx_maxprob_memview[s], u1)

                        sorted_indices_mu = np.argsort(x)

                        self.max_proba(mu_opt[s,o],
                                       sorted_indices_mu, beta_opt[s][o],
                                       self.mtx_maxmu_memview[s])
                        v = np.dot(self.mtx_maxmu_memview[s], x)

                    c1 = v + u1[s]
                    if first_action or c1 > u2[s] or np.isclose(c1, u2[s]):
                        u2[s] = c1
                        policy_indices[s] = action_idx
                        policy[s] = self.actions_per_state[s][action_idx]
                        assert policy_indices[s] == policy[s]
                    first_action = 0

            if max(u2-u1)-min(u2-u1) < epsilon:  # stopping condition
                # print("%d\n", counter)
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(nb_states)
                sorted_indices_u = np.argsort(u1)
                # for i in range(nb_states):
                #     printf("%d , ", sorted_indices[i])
                # printf("\n")

    def max_proba(self, p, asc_sorted_indices, beta, new_p):
        n = p.shape[0]

        temp = min(1., p[asc_sorted_indices[n-1]] + beta/2.0)
        new_p[asc_sorted_indices[n-1]] = temp
        sum_p = temp
        if temp < 1.:
            for i in range(0, n-1):
                temp = p[asc_sorted_indices[i]]
                new_p[asc_sorted_indices[i]] = temp
                sum_p += temp
            i = 0
            while sum_p > 1.0 and i < n:
                sum_p -= p[asc_sorted_indices[i]]
                new_p[asc_sorted_indices[i]] = max(0.0, 1. - sum_p)
                sum_p += new_p[asc_sorted_indices[i]]
                i += 1
        else:
            for i in range(0, n):
                new_p[i] = 0
            new_p[asc_sorted_indices[n-1]] = temp

