import os
import pickle
import numpy as np


path = '/home/ronan/PycharmProjects/UCRL/examples/UCRL_GymDiscreteEnvWrapper_20180130_144612/ucrl_0.pickle'
with open(path, 'rb') as f:
    data = pickle.load(f)

l = [range(0,3), range(3,6)]
i = range(0,3,2)
l2 = [np.delete(a, i) for a in l]
print(np.array(l))
print(np.array(i))
print(l2)