import numpy as np
from .rllogging import default_logger
from . import __version__ as ucrl_version
from visdom import Visdom


class QLearning():

    def __init__(self, environment,
                 r_max, random_state,
                 lr_alpha_init=1.0, exp_epsilon_init=1.0, gamma=1.0, initq=0.0,
                 verbose=0,
                 logger=default_logger,
                 known_reward=False):
        self.environment = environment
        self.r_max = float(r_max)

        self.lr_alpha_init = lr_alpha_init
        self.exp_epsilon_init = exp_epsilon_init

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
            action_idx = self.local_random.choice(self.environment.state_actions[s], 1)
            action = action_idx
        else:
            greedy_actions = np.argmax(self.q[s, :])
            if greedy_actions.size > 1:
                # this is a stochastic policy
                action_idx = self.local_random.choice(greedy_actions, 1)
                action = self.environment.state_actions[s][action_idx]
            else:
                action_idx = greedy_actions
                action = greedy_actions

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
        threshold_span = threshold

        # get initial state
        curr_state = self.environment.state

        self.first_regret = True
        while self.total_time < duration:

            if self.verbose > 0:
                curr_regret = self.total_time * self.environment.max_gain - self.total_reward
                self.logger.info("regret: {}, {:.2f}".format(self.total_time, curr_regret))

            curr_state = self.environment.state
            curr_act_idx, curr_act = self.sample_action(curr_state)  # sample action from the policy

            self.environment.execute(np.asscalar(curr_act))

            next_state = self.environment.state  # new state
            r = self.environment.reward

            # update Q value
            self.lr_alpha = self.lr_alpha_init / np.sqrt(self.nb_observations[curr_state, curr_act_idx] + 1)
            MIN_EXP = 200000
            # N0 = 0
            if self.total_time < MIN_EXP:
                self.exp_epsilon = 1.0
                N0 = self.nb_observations.copy()

            else:
                self.exp_epsilon = self.exp_epsilon_init / np.power(
                    np.maximum(self.nb_observations[curr_state, curr_act_idx] - N0[curr_state, curr_act_idx], 1), 1. / 3.)

            # self.exp_epsilon = self.exp_epsilon_init / np.power(np.maximum(self.nb_observations[curr_state, curr_act_idx] - 1000, 1), 2/3)
            # self.exp_epsilon = 1.0
            self.q[curr_state, curr_act_idx] = (1 - self.lr_alpha) * self.q[
                curr_state, curr_act_idx] + self.lr_alpha * (r + self.gamma * np.max(self.q[next_state, :]) - self.q[0, 0])

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
            else:
                self.viz.line(X=np.array([self.total_time]), Y=np.array([curr_regret]),
                              env="main", win=self.viz_plots["regret"],
                              update='append')
                self.viz.line(X=np.array([self.total_time]), Y=np.array([self.exp_epsilon]),
                              env="main", win=self.viz_plots["epsilon"],
                              update='append')

