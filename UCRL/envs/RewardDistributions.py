import random
import numpy as np
from abc import ABCMeta, abstractmethod


class RewardDistribution(metaclass=ABCMeta):
    """
    This abstract class implements random distributions for rewards in SMDPs.
    A RewardDistribution is characterized by its mean (attribute) and a method "generate" used to generate random
    realizations of the distribution.
    """

    def __init__(self):
        self.mean = 0  # attribute "mean" should be specified in descendant classes

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
        self.mean = c

    def generate(self):
        return self.mean


class UniformReward(RewardDistribution):
    """
    Continuous uniformly distributed rewards.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.mean = (a+b)/2

    def generate(self):
        return random.uniform(self.a, self.b)


class BernouilliReward(RewardDistribution):
    """
    Bernouilli distributed reward.
    """
    def __init__(self, r_max, proba):
        self.r_max = r_max
        self.proba = proba
        self.mean = proba * r_max

    def generate(self):
        return self.r_max*np.random.binomial(n=1, p=self.proba)