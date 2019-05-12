from gym.envs.classic_control import MountainCarEnv
import numpy as np
import math


class MountainCar(MountainCarEnv):

    def reset(self):
        # self.state = np.array(
        #     [self.np_random.uniform(low=self.observation_space.low[0], high=self.observation_space.high[0]),
        #      self.np_random.uniform(low=self.observation_space.low[1], high=self.observation_space.high[1])])
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        # dvel = (action - 1) * 0.001 * 0.7 + math.cos(3 * position) * (-0.0025)
        # velocity += dvel + 0.5 * dvel * np.random.randn()
        velocity += (action - 1) * 0.001 * 0.7 + math.cos(3 * position) * (-0.0025) + 0.0005 * np.random.randn()
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        # if position == self.min_position and velocity < 0:
        #     velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}
