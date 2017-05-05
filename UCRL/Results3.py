import os
import pickle

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pylab
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif', size=27)

exp_id = 1000  # experiment ID
ratio = []
tmax = []
mdp_regret = 0


exp_id = 1008  # experiment ID
simu_id = 5
path = '/home/lille3inria/Documents/pycharm-community-5.0.3/projects/UCRL/experiments/experiment ' + str(exp_id)
with open(path + '/simu ' + str(simu_id), 'rb') as f:
    unit_duration = pickle.load(f).unit_duration
    plt.plot([1000*x for x in range(0, len(unit_duration))], unit_duration, color='b', linewidth=5)


exp_id = 1004  # experiment ID
simu_id = 5
path = '/home/lille3inria/Documents/pycharm-community-5.0.3/projects/UCRL/experiments/experiment ' + str(exp_id)
with open(path + '/simu ' + str(simu_id), 'rb') as f:
    unit_duration = pickle.load(f).unit_duration
    plt.plot([1000*x for x in range(0, len(unit_duration))], unit_duration, color='r', linewidth=5)

exp_id = 1002  # experiment ID
simu_id = 7
path = '/home/lille3inria/Documents/pycharm-community-5.0.3/projects/UCRL/experiments/experiment ' + str(exp_id)
with open(path + '/simu ' + str(simu_id), 'rb') as f:
    unit_duration = pickle.load(f).unit_duration
    plt.plot([1000*x for x in range(0, len(unit_duration))], unit_duration, color='g', linewidth=5)

plt.plot([0, 30000000], [1, 1], 'k')

# Axis title
plt.xlabel('Duration $T_{n}$', size=38)
plt.ylabel('Duration per decision step $T_n/n$', size=38)

# Caption
blue_line = mlines.Line2D([], [], color='blue', label='$T_{\max} = 8$', linewidth=4)
red_line = mlines.Line2D([], [], color='red', label='$T_{\max} = 4$', linewidth=4)
green_line = mlines.Line2D([], [], color='green', label='$T_{\max} = 2$', linewidth=4)
plt.legend(handles=[blue_line, red_line, green_line], loc='upper right')

pylab.show()