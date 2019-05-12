import numpy as np
import gym
from gym.envs.classic_control.acrobot import wrap, bound, rk4
from gym import spaces
from gym.utils import seeding

RAD2DEG = 57.29577951308232
DEG2RAD = 0.01745329251

"""
Barto and Rosenstein
Supervised actor-critic reinforcement learning
Handbook of learning and approximate dynamic programming 2 (2004): 359
"""


class ShipSteering(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    dt = 25
    C = 0.01
    AVAIL_ANGLEDX = [-10 * DEG2RAD, 0, 10 * DEG2RAD]
    torque_noise_max = 2 * DEG2RAD
    torque_noise_min = -2 * DEG2RAD
    goal_radii = 0.15
    goal = np.array([0., 0.])

    def __init__(self):
        self.viewer = None
        high = np.array([6., 2., np.pi])
        low = np.array([0, -2., -np.pi])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()
        self.reward_range = np.array([-1, 0], dtype=np.float)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.pos = np.array([3.66, -1.86], dtype=np.float)
            self.heading = 0
        else:
            self.pos = np.array([state[0], state[1]], dtype=np.float)
            self.heading = wrap(float(state[2]), -np.pi, np.pi)
        return self._get_ob()

    def step(self, a):
        torque = self.AVAIL_ANGLEDX[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        self.heading = wrap(self.heading + torque, -np.pi, np.pi)

        ns = rk4(self._dsdt, self.pos, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = bound(ns[0], self.observation_space.low[0], self.observation_space.high[0])
        ns[1] = bound(ns[1], self.observation_space.low[1], self.observation_space.high[1])
        self.pos = ns
        terminal = self._terminal()
        if terminal:
            reward = 0.
        else:
            if np.isclose(self.pos[0], 0.):
                self.pos = np.array([5.5, 0], dtype=np.float)
                self.heading = 0
            reward = -1.

        return self._get_ob(), reward, terminal, {}

    def _terminal(self):
        return np.linalg.norm(self.pos - self.goal) <= self.goal_radii

    def _dsdt(self, s, t):
        x, y = s
        dx = self.C * np.cos(self.heading - y)
        dy = self.C * np.sin(self.heading)
        return (dx, dy)

    def _get_ob(self):
        return np.array([self.pos[0], self.pos[1], self.heading])

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.observation_space.high[0] - self.observation_space.low[0]
        world_height = self.observation_space.high[1] - self.observation_space.low[1]
        scale_w = screen_width / world_width
        scale_h = screen_height / world_height
        carwidth = 20
        carheight = 10

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            nopts = 200
            angles = np.array([a * 2.0 * np.pi / nopts for a in range(nopts)])
            circle = np.zeros((2, nopts))
            circle[0, :], circle[1, :] = np.cos(angles), np.sin(angles)

            xys = list(zip((circle[0, :] * self.goal_radii + self.goal[0] - self.observation_space.low[0]) * scale_w,
                           (circle[1, :] * self.goal_radii + self.goal[1] - self.observation_space.low[1]) * scale_h))
            aa = rendering.FilledPolygon(xys)
            aa.set_color(153. / 255, 255. / 255, 153. / 255)
            self.viewer.add_geom(aa)

            border = [(0, 0), (0, screen_height), (screen_width, screen_height), (screen_width, 0)]
            border = rendering.make_polyline(border)
            border.set_linewidth(5)
            border.set_color(0. / 255, 102. / 255, 204. / 255)
            self.viewer.add_geom(border)

            point = rendering.make_circle(5.)
            self.dot = rendering.Transform()
            point.add_attr(self.dot)
            point.set_color(204. / 255, 0, 0)
            self.viewer.add_geom(point)

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight / 2, -carheight / 2
            c, ar = 0, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (c, t), (r, ar), (c, b)])
            # car.add_attr(rendering.Transform(translation=(0, clearance)))
            car.set_color(204. / 255, 0, 0)
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

        pos = self._get_ob()
        self.cartrans.set_translation((pos[0] - self.observation_space.low[0]) * scale_w,
                                      (pos[1] - self.observation_space.low[1]) * scale_h)
        self.cartrans.set_rotation(self.heading)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
