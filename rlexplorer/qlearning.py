import numpy as np
from .rllogging import default_logger
from . import __version__ as ucrl_version
from visdom import Visdom


class QLearning():

    def __init__(self, environment,
                 r_max, random_state,
                 lr_alpha_init=1.0, exp_epsilon_init=1.0, gamma=1.0, initq=0.0, exp_power=0.5,
                 verbose=0,
                 logger=default_logger,
                 known_reward=False):
        self.environment = environment
        self.r_max = float(r_max)

        self.lr_alpha_init = lr_alpha_init
        self.exp_epsilon_init = exp_epsilon_init
        self.exp_power = exp_power

        self.lr_alpha = lr_alpha_init
        self.exp_epsilon = exp_epsilon_init

        # initialize matrices
        self.policy = np.zeros((self.environment.nb_states,),
                               dtype=np.int_)
        self.policy_indices = np.zeros((self.environment.nb_states,), dtype=np.int_)

        # initialization
        self.total_reward = 0
        self.total_time = 0
        self.regret = [0]  # cumulative regret of the learning algorithm
        self.regret_unit_time = [0]
        self.unit_duration = [1]  # ratios (nb of time steps)/(nb of decision steps)
        self.span_values = []
        self.span_times = []
        self.iteration = 0
        self.episode = 0
        self.delta = 1.  # confidence
        self.gamma = gamma

        self.verbose = verbose
        self.logger = logger
        self.version = ucrl_version
        self.random_state = random_state
        self.local_random = np.random.RandomState(seed=random_state)

        self.nb_states = self.environment.nb_states
        self.max_nb_actions = self.environment.max_nb_actions_per_state
        self.known_reward = known_reward

        self.q = initq * np.ones((self.nb_states, self.max_nb_actions))
        self.nb_observations = np.zeros((self.nb_states, self.max_nb_actions), dtype=np.int64)
        self.viz = None  # for visualization

    def clear_before_pickle(self):
        del self.logger
        del self.viz

    def reset_after_pickle(self, solver=None, logger=default_logger):
        self.logger = logger

    def description(self):
        desc = {
            "alpha_p": self.lr_alpha_init,
            "alpha_r": self.exp_epsilon_init,
            "r_max": self.r_max,
            "version": self.version
        }
        return desc

    def sample_action(self, s):
        """
        Args:
            s (int): a given state index

        Returns:
            action_idx (int): index of the selected action
            action (int): selected action
        """

        rnd = self.local_random.uniform()
        if rnd < self.exp_epsilon:
            # uniform random action
            action_idx = np.asscalar(self.local_random.choice(self.environment.state_actions[s], 1))
            action = self.environment.state_actions[s][action_idx]
        else:
            na = self.environment.state_actions[s]
            b = self.q[s, na]
            # print(s, b)
            idxs = np.flatnonzero(np.isclose(b, b.max()))
            action_idx = np.asscalar(self.local_random.choice(idxs))
            action = self.environment.state_actions[s][action_idx]
        self.policy_indices[s] = action_idx
        self.policy[s] = action
        return action_idx, action

    def learn(self, duration, regret_time_step, span_episode_step=1, render=False):
        """ Run UCRL on the provided environment

        Args:
            duration (int): the algorithm is run until the number of time steps
                            exceeds "duration"
            regret_time_step (int): the value of the cumulative regret is stored
                                    every "regret_time_step" time steps
            render (flag): True for rendering the domain, False otherwise

        """
        if self.total_time >= duration:
            return

        # --------------------------------------------
        self.first_span = True
        self.first_regret = True
        self.viz_plots = {}
        self.viz = Visdom()
        if not self.viz.check_connection(timeout_seconds=3):
            self.viz = None
        # --------------------------------------------

        threshold = self.total_time + regret_time_step
        threshold_span = self.total_time

        # get initial state
        curr_state = self.environment.state
        self.sbar = curr_state
        self.abar = 0

        self.first_regret = True
        while self.total_time < duration:

            if self.verbose > 0 and self.total_time >= threshold_span:
                curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                self.logger.info("{}/{} = {:3.2f}%".format(self.total_time, duration, self.total_time / duration * 100))
                self.logger.info("regret: {}, {:.2f}".format(self.total_time, curr_regret))
                threshold_span += regret_time_step

            curr_state = self.environment.state
            curr_act_idx, curr_act = self.sample_action(curr_state)  # sample action from the policy

            self.environment.execute(curr_act)

            next_state = self.environment.state  # new state
            r = self.environment.reward

            # update Q value
            self.lr_alpha = self.lr_alpha_init / np.sqrt(self.nb_observations[curr_state, curr_act_idx] + 1)
            MIN_EXP = -1  # 200000
            # N0 = 0
            if self.total_time < MIN_EXP:
                self.exp_epsilon = 1.0
                # N0 = self.nb_observations.copy()

            else:
                self.exp_epsilon = self.exp_epsilon_init / np.power(
                    np.maximum(self.nb_observations[curr_state, curr_act_idx], 1), self.exp_power)

            # self.exp_epsilon = self.exp_epsilon_init / np.power(np.maximum(self.nb_observations[curr_state, curr_act_idx] - 1000, 1), 2/3)
            # self.exp_epsilon = 1.0
            self.q[curr_state, curr_act_idx] = (1 - self.lr_alpha) * self.q[
                curr_state, curr_act_idx] + self.lr_alpha * (r + self.gamma * np.max(self.q[next_state, :]) - self.q[self.sbar, self.abar])

            self.nb_observations[curr_state, curr_act_idx] += 1

            # update time
            self.total_time += 1
            self.iteration += 1
            self.total_reward += r

            if self.total_time > threshold:
                self.save_information()
                threshold = self.total_time + regret_time_step

        if self.verbose > 0:
            self.logger.info(self.q)
            self.logger.info(np.sum(self.nb_observations, 1))
            self.logger.info(np.sum(self.nb_observations, 1) / self.total_time)
            self.logger.info(self.lr_alpha)
            self.logger.info(self.exp_epsilon)

    def save_information(self):
        curr_regret = self.total_time * self.environment.max_gain - self.total_reward
        self.regret.append(curr_regret)
        self.regret_unit_time.append(self.total_time)
        self.unit_duration.append(self.total_time / self.iteration)

        if self.viz is not None:
            if self.first_regret:
                self.first_regret = False
                self.viz_plots["regret"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([curr_regret]),
                                                         env="main", opts=dict(
                        title="{} - Regret".format(type(self).__name__),
                        xlabel="Time",
                        ylabel="Cumulative Regret"
                    ))
                self.viz_plots["epsilon"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([self.exp_epsilon]),
                                                          env="main", opts=dict(
                        title="{} - Expl. Epsilon".format(type(self).__name__),
                        xlabel="Time",
                        ylabel="Exploration Epsilon"
                    ))
                self.viz_plots["alpha"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([self.lr_alpha]),
                                                          env="main", opts=dict(
                        title="{} - LR Alpha".format(type(self).__name__),
                        xlabel="Time",
                        ylabel="LR Alpha"
                    ))
                self.viz_plots["reward"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([self.environment.reward]),
                                                          env="main", opts=dict(
                        title="{} - Reward".format(type(self).__name__),
                        xlabel="Time",
                        ylabel="Immediate reward"
                    ))
            else:
                self.viz.line(X=np.array([self.total_time]), Y=np.array([curr_regret]),
                              env="main", win=self.viz_plots["regret"],
                              update='append')
                self.viz.line(X=np.array([self.total_time]), Y=np.array([self.exp_epsilon]),
                              env="main", win=self.viz_plots["epsilon"],
                              update='append')
                self.viz.line(X=np.array([self.total_time]), Y=np.array([self.lr_alpha]),
                              env="main", win=self.viz_plots["alpha"],
                              update='append')
                self.viz.line(X=np.array([self.total_time]), Y=np.array([self.environment.reward]),
                              env="main", win=self.viz_plots["reward"],
                              update='append')


class QLearningUCB(QLearning):

    def __init__(self, environment, r_max, random_state,
                 span_constraint, alpha_r=None, alpha_p=None,
                 lr_alpha_init=1.0, exp_epsilon_init=1.0, gamma=1.0, exp_power=0.5,
                 lipschitz_const=0.0,
                 verbose=0,
                 logger=default_logger,
                 known_reward=False):
        initq = 2*span_constraint
        super(QLearningUCB, self).__init__(environment=environment, r_max=r_max, random_state=random_state,
                                           lr_alpha_init=lr_alpha_init, exp_epsilon_init=exp_epsilon_init,
                                           gamma=gamma, initq=initq, exp_power=exp_power,
                                           verbose=verbose, logger=logger, known_reward=known_reward)
        self.span_constraint = span_constraint
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.lipschitz_const = lipschitz_const

    def beta_r(self, state, action):
        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state
        N = np.maximum(1, self.nb_observations[state, action])

        LOG_TERM = np.log(S * A / self.delta)

        beta_r = np.sqrt(LOG_TERM / N)
        beta_p = np.sqrt(LOG_TERM / N)  # + 1.0 / (self.nb_observations + 1.0)

        P_term = self.alpha_p * self.span_constraint * np.minimum(beta_p, 2.)
        R_term = self.alpha_r * self.r_max * np.minimum(beta_r, 1 if not self.known_reward else 0)
        # P_term = self.alpha_p * self.span_constraint * beta_p
        # R_term = self.alpha_r * self.r_max * beta_r if not self.known_reward else 0

        final_bonus = R_term + P_term
        # print(final_bonus)
        # print(self.nb_observations)
        return final_bonus

    def description(self):
        super_desc = super().description()
        desc = {
            "span_constraint": self.span_constraint,
            "bound_type": "hoeffding",
            "alpha_p": self.alpha_p,
            "alpha_r": self.alpha_r,
        }
        super_desc.update(desc)
        return super_desc

    def learn(self, duration, regret_time_step, span_episode_step=1, render=False):
        """ Run UCRL on the provided environment

        Args:
            duration (int): the algorithm is run until the number of time steps
                            exceeds "duration"
            regret_time_step (int): the value of the cumulative regret is stored
                                    every "regret_time_step" time steps
            render (flag): True for rendering the domain, False otherwise

        """
        if self.total_time >= duration:
            return

        # --------------------------------------------
        self.first_span = True
        self.first_regret = True
        self.viz_plots = {}
        self.viz = Visdom()
        if not self.viz.check_connection(timeout_seconds=3):
            self.viz = None
        # --------------------------------------------

        threshold = self.total_time + regret_time_step
        threshold_span = self.total_time

        # get initial state
        curr_state = self.environment.state
        self.sbar = curr_state
        self.abar = 0
        self.exp_epsilon = -1

        self.first_regret = True
        while self.total_time < duration:
            self.delta = 1 / np.sqrt(self.total_time + 1)

            if self.verbose > 0 and self.total_time >= threshold_span:
                curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                self.logger.info("{}/{} = {:3.2f}%".format(self.total_time, duration, self.total_time / duration * 100))
                self.logger.info("regret: {}, {:.2f}".format(self.total_time, curr_regret))
                threshold_span += regret_time_step


            curr_state = self.environment.state
            curr_act_idx, curr_act = self.sample_action(curr_state)  # sample action from the policy

            self.environment.execute(curr_act)

            next_state = self.environment.state  # new state
            r = self.environment.reward

            # update Q value
            # self.lr_alpha = (self.span_constraint + 1) / (self.span_constraint + np.sqrt(self.nb_observations[curr_state, curr_act_idx] + 1))
            self.lr_alpha = (self.span_constraint + 1) / (self.span_constraint + self.nb_observations[curr_state, curr_act_idx] + 1)
            # self.lr_alpha = self.lr_alpha_init / (np.sqrt(self.nb_observations[curr_state, curr_act_idx]+1))

            self.bonus = self.beta_r(curr_state, curr_act_idx) + self.lipschitz_const / np.sqrt(1 + self.nb_observations[curr_state, curr_act_idx])
            MM = min(self.span_constraint, np.max(self.q[next_state, :]))
            # MM = np.max(self.q[next_state, :])
            self.q[curr_state, curr_act_idx] = (1 - self.lr_alpha) * self.q[
                curr_state, curr_act_idx] + self.lr_alpha * (r + self.bonus + self.gamma * MM - self.q[self.sbar, self.abar])
            # self.q[curr_state, curr_act_idx] = min(self.span_constraint, self.q[curr_state, curr_act_idx])

            self.nb_observations[curr_state, curr_act_idx] += 1

            # update time
            self.total_time += 1
            self.iteration += 1
            self.total_reward += r

            if self.total_time > threshold:
                self.save_information()
                threshold = self.total_time + regret_time_step

        if self.verbose > 0:
            self.logger.info(self.q)
            self.logger.info(np.sum(self.nb_observations, 1))
            self.logger.info(np.sum(self.nb_observations, 1) / self.total_time)
            self.logger.info(self.lr_alpha)
            self.logger.info(self.exp_epsilon)


    def save_information(self):
        if self.viz is not None:
            if self.first_regret:
                self.viz_plots["bonus"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([self.bonus]),
                                                          env="main", opts=dict(
                        title="{} - Expl. bonus".format(type(self).__name__),
                        xlabel="Time",
                        ylabel="Exploration Bonus"
                    ))
            else:
                self.viz.line(X=np.array([self.total_time]), Y=np.array([self.bonus]),
                              env="main", win=self.viz_plots["bonus"],
                              update='append')
        super(QLearningUCB, self).save_information()
                
        # curr_regret = self.total_time * self.environment.max_gain - self.total_reward
        # self.regret.append(curr_regret)
        # self.regret_unit_time.append(self.total_time)
        # self.unit_duration.append(self.total_time / self.iteration)
        #
        # if self.viz is not None:
        #     if self.first_regret:
        #         self.first_regret = False
        #         self.viz_plots["regret"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([curr_regret]),
        #                                                  env="main", opts=dict(
        #                 title="{} - Regret".format(type(self).__name__),
        #                 xlabel="Time",
        #                 ylabel="Cumulative Regret"
        #             ))
        #         self.viz_plots["bonus"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([self.bonus]),
        #                                                   env="main", opts=dict(
        #                 title="{} - Expl. bonus".format(type(self).__name__),
        #                 xlabel="Time",
        #                 ylabel="Exploration Bonus"
        #             ))
        #         self.viz_plots["alpha"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([self.lr_alpha]),
        #                                                   env="main", opts=dict(
        #                 title="{} - LR Alpha".format(type(self).__name__),
        #                 xlabel="Time",
        #                 ylabel="LR Alpha"
        #             ))
        #         self.viz_plots["reward"] = self.viz.line(X=np.array([self.total_time]), Y=np.array([self.environment.reward]),
        #                                                   env="main", opts=dict(
        #                 title="{} - Reward".format(type(self).__name__),
        #                 xlabel="Time",
        #                 ylabel="Immediate reward"
        #             ))
        #     else:
        #         self.viz.line(X=np.array([self.total_time]), Y=np.array([curr_regret]),
        #                       env="main", win=self.viz_plots["regret"],
        #                       update='append')
        #         self.viz.line(X=np.array([self.total_time]), Y=np.array([self.bonus]),
        #                       env="main", win=self.viz_plots["bonus"],
        #                       update='append')
        #         self.viz.line(X=np.array([self.total_time]), Y=np.array([self.lr_alpha]),
        #                       env="main", win=self.viz_plots["alpha"],
        #                       update='append')
        #         self.viz.line(X=np.array([self.total_time]), Y=np.array([self.environment.reward]),
        #                       env="main", win=self.viz_plots["reward"],
        #                       update='append')