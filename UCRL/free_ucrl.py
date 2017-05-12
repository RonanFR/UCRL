from .Ucrl import AbstractUCRL
from .envs import MixedEnvironment
import numpy as np
import math as m
from .logging import default_logger

class FreeUCRL_Alg1(AbstractUCRL):
    def __init__(self, environment, r_max, range_r=-1, range_p=-1,
                 verbose = 0, logger=default_logger):
        assert isinstance(environment, MixedEnvironment)

        evi = EVI_Alg1(nb_states=environment.nb_states,
                       thr=environment.threshold_options,
                       action_per_state=environment.get_state_actions())

        super(FreeUCRL_Alg1, self).__init__(environment=environment,
                                            r_max=r_max, range_r=range_r,
                                            range_p=range_p, solver=evi,
                                            verbose=verbose,
                                            logger=logger)

        nb_states = self.environment.nb_states
        max_nb_mdp_actions = environment.max_nb_mdp_actions_per_state # actions of the mdp
        max_nb_actions = environment.max_nb_actions_per_state # actions of mdp+opt

        # keep track of the transition probabilities for options and actions
        self.estimated_probabilities = np.ones((nb_states, max_nb_actions, nb_states)) / nb_states
        self.estimated_probabilities_mdp = np.ones((nb_states, max_nb_mdp_actions, nb_states)) / nb_states

        # keep track of visits
        self.nb_observations = np.zeros((nb_states, max_nb_actions))
        self.nu_k = np.zeros((nb_states, max_nb_actions))
        self.nb_observations_mdp = np.zeros((nb_states, max_nb_mdp_actions)) # for reward of mdp actions
        self.nu_k_mdp = np.zeros((nb_states, max_nb_mdp_actions)) # for reward of mdp actions

        # reward info
        self.estimated_rewards_mdp = np.ones((nb_states, max_nb_mdp_actions)) * r_max

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
            self.nu_k.fill(0)
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
            self.nb_observations += self.nu_k

    def beta_r(self):
        nb_states, nb_prim_actions = self.estimated_rewards_mdp.shape
        beta_r =  np.multiply(self.range_r, np.sqrt(7/2 * m.log(2 * nb_states * nb_prim_actions *
                                                (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations_mdp)))
        return beta_r

    def beta_p(self):
        nb_states, nb_actions = self.nb_observations.shape
        return self.range_p * np.sqrt(14 * nb_states * m.log(2 * nb_actions
                    * (self.iteration + 1)/self.delta) / np.maximum(1, self.nb_observations))

    def beta_mu_p(self):
        nb_options = self.environment.nb_options
        nb_actions = self.nb_observations.shape[1]
        beta_mu_p = np.empty((nb_options,))
        for o in range(nb_options):
            reach_s = self.environment.reachable_states_per_option[o]
            s_0 = reach_s[0]
            nb_states = len(reach_s)

            real_opt = o + self.environment.threshold_options + 1
            index = self.environment.get_index_of_action_state(s_0, real_opt)

            v = self.range_p * np.sqrt(14 * nb_states * m.log(2 * nb_actions
                    * (self.iteration + 1)/self.delta)
                                       / np.maximum(1, self.nb_observations[s_0, index]))
            beta_mu_p[o] = v
        return beta_mu_p

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
            self.update_from_action(s, self.policy[s], s2, r)
        else:
            self.update_from_option(history)

    def update_from_option(self, history):
        [s, o, r, t, s2] = history.data # first is for sure an option
        option_index = self.environment.get_index_of_action_state(s, o)
        assert option_index is not None

        self.estimated_probabilities[s][option_index] *= self.nb_observations[s][option_index] / (self.nb_observations[s][option_index] + 1)
        self.estimated_probabilities[s][option_index][s2] += 1 / (self.nb_observations[s][option_index] + 1)
        self.nu_k[s][option_index] += 1

        # the rest of the actions are for sure primitive actions (1 level hierarchy)
        node = history
        while len(node.children) > 0:
            assert len(node.children) == 1, "history has more than one child"

            # advance node
            node = node.children[0]
            [s, a, r, t, s2] = node.data # this is a primitive action which can be valid or not
            # for sure it is a primitive action (and the index corresponds to all the primitive actions
            self.update_from_action(s, a, s2, r)

    def update_from_action(self, s, a, s2, r):
        mdpopt_index = self.environment.get_index_of_action_state(s, a)
        mdp_index = self.environment.get_index_of_action_state_in_mdp(s, a)

        self.estimated_rewards_mdp[s][mdp_index] *= self.nb_observations_mdp[s][mdp_index] / (self.nb_observations_mdp[s][mdp_index] + 1)
        self.estimated_rewards_mdp[s][mdp_index] += 1 / (self.nb_observations_mdp[s][mdp_index] + 1) * r
        self.estimated_probabilities_mdp[s][mdp_index] *= self.nb_observations_mdp[s][mdp_index] / (self.nb_observations[s][mdp_index] + 1)
        self.estimated_probabilities_mdp[s][mdp_index][s2] += 1 / (self.nb_observations_mdp[s][mdp_index] + 1)
        self.nu_k_mdp[s][mdp_index] += 1
        if mdpopt_index is not None:
            # this means that the primitive action is a valid action for
            # the environment mdp + opt
            self.estimated_probabilities[s][mdpopt_index] *= self.nb_observations[s][mdpopt_index] / (self.nb_observations[s][mdpopt_index] + 1)
            self.estimated_probabilities[s][mdpopt_index][s2] += 1 / (self.nb_observations[s][mdpopt_index] + 1)
            self.nu_k[s][mdpopt_index] += 1

    def solve_optimistic_model(self):
        nb_states = self.environment.nb_states
        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_p = self.beta_p()  # confidence bounds on transition probabilities
        beta_mu_p = self.beta_mu_p() # confidence bound on transition model of options for mu

        nb_options = self.environment.nb_options
        r_tilde_opt = [[]] * nb_options
        mu_opt = [[]] * nb_options
        condition_numbers_opt = np.empty(nb_options)
        for o in range(nb_options):
            option_policy = self.environment.options_policies[o]
            option_reach_states = self.environment.reachable_states_per_option[o]
            term_cond = self.environment.options_terminating_conditions[o]

            opt_nb_states = len(option_reach_states)

            Q_o = np.zeros((opt_nb_states, opt_nb_states))
            q_o = np.zeros((opt_nb_states,1))

            # compute the reward and the mu
            for i, s in enumerate(option_reach_states):
                option_action = option_policy[s]
                r_tilde_opt[o].append(
                    self.estimated_rewards_mdp[s, option_action] + beta_r[s,option_action]
                )

                for j, sprime in enumerate(option_reach_states):
                    prob = self.estimated_probabilities_mdp[s][option_action][sprime]
                    #q_o[i,0] += term_cond[sprime] * prob
                    Q_o[i,j] = (1. - term_cond[sprime]) * prob
            e_m = np.ones((opt_nb_states,1))
            q_o = e_m - np.dot(Q_o, e_m)

            Pprime_o = np.concatenate((q_o, Q_o[:, 1:]), axis=1)
            assert np.allclose(np.sum(Pprime_o, axis=1), np.ones(opt_nb_states))

            Pap = (Pprime_o + np.eye(opt_nb_states)) / 2.
            D, U = np.linalg.eig(
                np.transpose(Pap))  # eigen decomposition of transpose of P
            sorted_indices = np.argsort(np.real(D))
            mu = np.transpose(np.real(U))[sorted_indices[-1]]
            mu /= np.sum(mu)  # stationary distribution
            mu_opt[o].append(mu)

            P_star = np.repeat(np.array(mu, ndmin=2), opt_nb_states,
                               axis=0)  # limiting matrix

            # Compute deviation matrix
            I = np.eye(opt_nb_states)  # identity matrix
            Z = np.linalg.inv(I - Pap + P_star)  # fundamental matrix
            H = np.dot(Z, I - P_star)  # deviation matrix

            condition_nb = 0  # condition number of deviation matrix
            for i in range(0, opt_nb_states):  # Seneta's condition number
                for j in range(i + 1, opt_nb_states):
                    condition_nb = max(condition_nb,
                                       0.5 * np.linalg.norm(H[i, :] - H[j, :],
                                                            ord=1))
            condition_numbers_opt[o] = condition_nb

        self.opt_solver.run(policy_indices=self.policy_indices,
                            policy=self.policy,
                            p_hat=self.estimated_probabilities,
                            r_hat_mdp=self.estimated_rewards_mdp,
                            r_tilde_opt=r_tilde_opt,
                            beta_p=beta_p,
                            beta_r_mdp=beta_r,
                            mu_opt=mu_opt,
                            cn_opt=condition_numbers_opt,
                            beta_mu_p=beta_mu_p,
                            r_max=self.r_max,
                            epsilon=self.r_max / m.sqrt(self.iteration + 1))

class EVI_Alg1(object):

    def __init__(self, nb_states, thr, action_per_state):
        self.mtx_maxprob_memview = np.ones((nb_states, nb_states)) * 99
        self.mtx_maxmu_memview = np.ones((nb_states, nb_states)) * 99
        self.threshold = thr
        self.actions_per_state = action_per_state
        self.u1 = np.zeros(nb_states)
        self.u2 = np.zeros(nb_states)

    def run(self, policy_indices, policy,
            p_hat,
            r_hat_mdp,
            r_tilde_opt,
            beta_p,
            beta_r_mdp,
            mu_opt,
            cn_opt,
            beta_mu_p,
            r_max,
            epsilon):
        nb_states = p_hat.shape[0]
        sorted_indices_u = np.arange(nb_states)
        while True:
            for s in range(nb_states):
                first_action = 1
                for action_idx, action in enumerate(self.actions_per_state[s]):
                    if action <= self.threshold:
                        #this is an action
                        self.mtx_maxprob_memview[s] = self.max_proba(p_hat[s][action_idx], nb_states,
                                       sorted_indices_u, beta_p[s][action_idx])
                        self.mtx_maxprob_memview[s][s] = self.mtx_maxprob_memview[s][s] - 1.  # ????

                        r_optimal = min(r_max, r_hat_mdp[s, action] + beta_r_mdp[s, action])
                        v = r_optimal + np.dot(self.mtx_maxprob_memview[s], self.u1)
                    else:
                        self.mtx_maxprob_memview[s] = self.max_proba(p_hat[s][action_idx], nb_states,
                                       sorted_indices_u, beta_p[s][action_idx])
                        self.mtx_maxprob_memview[s][s] = self.mtx_maxprob_memview[s][s] - 1.  # ????

                        o = action - self.threshold - 1 # zero based index

                        x = np.minimum(r_max, r_tilde_opt[o])
                        x[s] += np.dot(self.mtx_maxprob_memview[s], self.u1)

                        sorted_indices_mu = np.argsort(x)

                        self.mtx_maxmu_memview[s] = self.max_proba(mu_opt[o], len(mu_opt[o]),
                                       sorted_indices_mu, cn_opt[o]*beta_mu_p[o])
                        v = np.dot(self.mtx_maxmu_memview[s], x)

                    c1 = v + self.u1[s]
                    if first_action or c1 > self.u2[s] or np.isclose(c1, self.u2[s]):
                        self.u2[s] = c1
                        policy_indices[s] = action_idx
                        policy[s] = self.actions_per_state[s][action_idx]
                        assert policy_indices[s] == policy[s]
                    first_action = 0

            if max(self.u2-self.u1)-min(self.u2-self.u1) < epsilon:  # stopping condition
                # print("%d\n", counter)
                return max(self.u1) - min(self.u1), self.u1, self.u2
            else:
                self.u1 = self.u2
                self.u2 = np.empty(nb_states)
                sorted_indices_u = np.argsort(self.u1)
                # for i in range(nb_states):
                #     printf("%d , ", sorted_indices[i])
                # printf("\n")

    def max_proba(self, p, n, asc_sorted_indices, beta):
        new_p = 99 * np.ones(n)
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
        return new_p

