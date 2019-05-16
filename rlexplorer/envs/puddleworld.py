import gym
import numpy as np
from gym import spaces
import math
from gym.utils import seeding

"""
The Puddleworld environment

Info
----
  - State space: 2D Box (x,y)
  - Action space: Discrete (UP,DOWN,RIGHT,LEFT)
  - Parameters: goal position x and y, puddle centers, puddle variances

References
----------
  
  - Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, Marcello Restelli.
    Importance Weighted Transfer of Samples in Reinforcement Learning.
    International Conference on Machine Learning. 2018.
    
  - https://github.com/amarack/python-rl/blob/master/pyrl/environments/puddleworld.py
"""


class PuddleWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    # def __init__(self, goal_x=10, goal_y=10, puddle_means=[(1.0, 10.0), (1.0, 8.0), (6.0, 6.0), (6.0, 4.0)],
    #              puddle_var=[(.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),
    #                          (.8, 1.e-5, 1.e-5, .8)],
    #              puddle_slow=True):
    def __init__(self, goal_x=5, goal_y=5, puddle_means=[(3.0, .0), (0.8, 2.), (3.0, 3.0), (3., 1.5)],
                 puddle_var=[(.1, 1.e-5, 1.e-5, .2), (.1, 1.e-5, 1.e-5, .1), (.1, 1.e-5, 1.e-5, .1),
                             (.2, 0.1, 0.1, .4)],
                 puddle_slow=True):

        self.horizon = 50
        self.gamma = 0.99
        self.state_dim = 2
        self.action_dim = 1

        # self.size = np.array([10, 10])
        self.size = np.array([5, 5])
        self.goal = np.array([goal_x, goal_y])
        self.goal_reward = 100.
        self.per_step_reward = -5.
        self.puddle_penalty = -100.0
        self.noise = 0.3
        self.reward_noise = 1
        self.fudge = 1.41
        self.puddle_slow = 10 if puddle_slow else 0

        self.puddle_means = list(map(np.array, puddle_means))

        self.puddle_var = list(map(lambda cov: np.linalg.inv(np.array(cov).reshape((2, 2))), puddle_var))
        self.puddles = list(zip(self.puddle_means, self.puddle_var))

        self.observation_space = spaces.Box(low=np.array([0, 0]), high=self.size, dtype=np.float)

        self.action_space = spaces.Discrete(4)
        self.hasrange = False
        self.get_reward_boundaries()

        self.seed()
        self.reset()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        self._absorbing = False
        if state is None:
            # self.pos = np.array([np.random.uniform(low=0, high=2), np.random.uniform(low=0, high=10)])
            self.pos = np.array([0., 0.])
        else:
            self.pos = np.array(state)
        return self.get_state()

    def isAtGoal(self, pos, multiplier=1.):
        return np.linalg.norm(pos - self.goal) < (self.fudge * multiplier)

    def mvnpdf(self, x, mu, sigma_inv):
        size = len(x)
        if size == len(mu) and sigma_inv.shape == (size, size):
            det = 1.0 / np.linalg.det(sigma_inv)
            norm_const = 1.0 / (math.pow((2 * np.pi), float(size) / 2) * math.pow(det, 0.5))
            x_mu = x - mu
            result = math.pow(math.e, -0.5 * np.dot(x_mu, np.dot(sigma_inv, x_mu)))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")

    def _get_puddle_weight(self, pos):

        weight = 0
        for mu, inv_cov in self.puddles:
            weight += self.mvnpdf(pos, mu, inv_cov)
        return weight

    def step(self, a):

        s = self.get_state()

        puddle_weight = self._get_puddle_weight(self.pos)

        alpha = 0.8 / (1 + (self.puddle_slow * puddle_weight))

        if int(a) == 0:
            self.pos[0] += alpha
        elif int(a) == 1:
            self.pos[0] -= alpha
        elif int(a) == 2:
            self.pos[1] += alpha
        elif int(a) == 3:
            self.pos[1] -= alpha

        if self.noise > 0:
            self.pos += np.random.normal(scale=self.noise, size=(2,))
        self.pos = self.pos.clip([0, 0], self.size)

        base_reward = self.goal_reward if self.isAtGoal(self.pos) else self.per_step_reward
        base_reward += puddle_weight * self.puddle_penalty

        if self.reward_noise > 0:
            base_reward += np.random.normal(scale=self.reward_noise)

        base_reward = np.clip(base_reward, self.reward_range[0], self.reward_range[1])

        if self.isAtGoal(self.pos, multiplier=1.):
            self._absorbing = True
        else:
            self._absorbing = False

        return self.get_state(), base_reward, self._absorbing, {}

    def get_state(self):
        return np.array(self.pos)

    def get_reward_mean(self, s, a):

        puddle_weight = self._get_puddle_weight(s)
        reward = self.goal_reward if self.isAtGoal(s) else self.per_step_reward
        reward += puddle_weight * self.puddle_penalty
        return reward

    def get_reward_boundaries(self):
        if not self.hasrange:
            N = 200
            x = np.linspace(0, 10., N)
            y = np.linspace(0, 10., N)
            rmin = np.inf
            rmax = -np.inf
            for i in range(N):
                for j in range(N):
                    r = self.get_reward_mean(np.array([x[i], y[j]]), 0)
                    rmin = min(rmin, r)
                    rmax = max(rmax, r)
            #rmax = max(rmax, self.goal_reward)
            self.reward_range = (rmin, rmax)
            self.hasrange = True
            print(self.reward_range)
        return self.reward_range

    def get_transition_mean(self, s, a):

        puddle_weight = self._get_puddle_weight(s)
        alpha = 1 / (1 + (self.puddle_slow * puddle_weight))
        next_s = np.array(s)

        if int(a) == 0:
            next_s[0] += alpha
        elif int(a) == 1:
            next_s[0] -= alpha
        elif int(a) == 2:
            next_s[1] += alpha
        elif int(a) == 3:
            next_s[1] -= alpha

        return next_s.clip([0, 0], self.size)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.size[0]
        world_height = self.size[1]
        scale_w = screen_width / world_width
        scale_h = screen_height / world_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            nopts = 200
            angles = np.array([a * 2.0 * np.pi / nopts for a in range(nopts)])
            circle = np.zeros((2, nopts))
            circle[0, :], circle[1, :] = np.cos(angles), np.sin(angles)

            xys = list(zip((circle[0, :] * self.fudge + self.goal[0]) * scale_w,
                           (circle[1, :] * self.fudge + self.goal[1]) * scale_h))
            aa = rendering.FilledPolygon(xys)
            aa.set_color(153. / 255, 255. / 255, 153. / 255)
            self.viewer.add_geom(aa)

            for i in range(1, 10):
                aa = rendering.make_polyline([(i * scale_w, 0), (i * scale_w, screen_height)])
                aa.set_color(225. / 255, 225. / 255, 225. / 255)
                self.viewer.add_geom(aa)
                aa = rendering.make_polyline([(0, i * scale_h), (screen_width, i * scale_h)])
                aa.set_color(225. / 255, 225. / 255, 225. / 255)
                self.viewer.add_geom(aa)

            border = [(0, 0), (0, screen_height), (screen_width, screen_height), (screen_width, 0)]
            border = rendering.make_polyline(border)
            border.set_linewidth(5)
            border.set_color(0. / 255, 102. / 255, 204. / 255)
            self.viewer.add_geom(border)

            # draw puddles
            for pd in self.puddles:
                print(pd)
                mean = pd[0]
                sigma_inv = pd[1]

                point = rendering.make_circle(1.)
                trans = rendering.Transform()
                point.add_attr(trans)
                trans.set_translation(mean[0] * scale_w, mean[1] * scale_h)
                self.viewer.add_geom(point)
                for p in [0.99, 0.75, 0.5, 0.1]:
                    # https://www.xarg.org/2018/04/how-to-plot-a-covariance-error-ellipse/
                    s = -2 * np.log(1. - p)
                    w, vr = np.linalg.eig(s * np.linalg.inv(sigma_inv))
                    D = np.sqrt(np.diag(w))
                    L = mean[:, np.newaxis] + np.dot(vr, np.dot(D, circle))
                    xs = L[0, :]
                    ys = L[1, :]
                    xys = list(zip((xs - 0.0) * scale_w, (ys - 0.0) * scale_h))
                    aa = rendering.make_polyline(xys)
                    aa.set_color(200. / 255, 200. / 255, 200. / 255)
                    self.viewer.add_geom(aa)

            point = rendering.make_circle(5.)
            self.dot = rendering.Transform()
            point.add_attr(self.dot)
            point.set_color(204. / 255, 0, 0)
            self.viewer.add_geom(point)

        pos = self.get_state()
        self.dot.set_translation((pos[0] - 0.) * scale_w, (pos[1] - 0.) * scale_h)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
