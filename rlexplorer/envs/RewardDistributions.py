import random
import numpy as np
from abc import ABCMeta, abstractmethod


class RewardDistribution(metaclass=ABCMeta):
    """
    This abstract class implements random distributions for rewards in SMDPs.
    A RewardDistribution is characterized by its mean (attribute) and a method "generate" used to generate random
    realizations of the distribution.
    """

    def __init__(self, mean):
        self.mean = mean  # attribute "mean" should be specified in descendant classes

    @abstractmethod
    def generate(self):
        """
        You may use numpy.random or random to implement this abstract method.
        :return: a realization of the random reward
        """
        pass


class ConstantReward(RewardDistribution):
    """
    Constant (i.e., deterministic) rewards.
    """

    def __init__(self, c):
        super(ConstantReward, self).__init__(mean=c)

    def generate(self):
        return self.mean


class UniformReward(RewardDistribution):
    """
    Continuous uniformly distributed rewards.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super(UniformReward, self).__init__(mean=(a + b) / 2)

    def generate(self):
        return random.uniform(self.a, self.b)


class BernouilliReward(RewardDistribution):
    """
    Bernouilli distributed reward.
    """

    def __init__(self, r_max, proba):
        self.r_max = r_max
        self.proba = proba
        super(BernouilliReward, self).__init__(mean=proba * r_max)

    def generate(self):
        return self.r_max * np.random.binomial(n=1, p=self.proba)


class BetaReward(RewardDistribution):
    def __init__(self, a, b):
        """
        arm having a Beta distribution
        Args:
             a (float): first parameter
             b (float): second parameter
             random_state (int): seed to make experiments reproducible
        """
        self.a = a
        self.b = b
        super(BetaReward, self).__init__(mean=a / (a + b)
                                         # , variance=(a * b) / ((a + b) ** 2 * (a + b + 1))
                                         )

    def generate(self):
        return np.random.beta(self.a, self.b, 1)


class ExponentialReward(RewardDistribution):
    # https://en.wikipedia.org/wiki/Truncated_distribution
    # https://en.wikipedia.org/wiki/Exponential_distribution
    # http://lagrange.math.siu.edu/Olive/ch4.pdf
    def __init__(self, L, B=1.):
        """
        pdf =
        Args:
            L (float): parameter of the exponential distribution
            B (float): upper bound of the distribution (lower is 0)
            random_state (int): seed to make experiments reproducible
        """
        assert B > 0.
        self.L = L
        self.B = B
        v_m = (1. - np.exp(-B * L) * (1. + B * L)) / L
        super(ExponentialReward, self).__init__(mean=v_m / (1. - np.exp(-L * B))
                                                # , variance=None # compute it yourself!
                                                )

    def cdf(self, x):
        cdf = lambda y: 1. - np.exp(-self.L * y)
        truncated_cdf = (cdf(x) - cdf(0)) / (cdf(self.B) - cdf(0))
        return truncated_cdf

    def inv_cdf(self, q):
        assert 0 <= q <= 1.
        v = - np.log(1. - (1. - np.exp(- self.L * self.B)) * q) / self.L
        return v

    def sample(self):
        # Inverse transform sampling
        # https://en.wikipedia.org/wiki/Inverse_transform_sampling
        q = np.random.random_sample(1)
        x = self.inv_cdf(q=q)
        return x


class GaussianReward(RewardDistribution):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
        super(GaussianReward, self).__init__(mean=mean)

    def generate(self):
        return np.random.normal(self.mean, self.stddev, 1)
