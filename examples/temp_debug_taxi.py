import os
import pickle


path = '/home/ronan/PycharmProjects/UCRL/examples/UCRL_GymDiscreteEnvWrapper_20180130_144612/ucrl_0.pickle'
with open(path, 'rb') as f:
    data = pickle.load(f)
    data.regret