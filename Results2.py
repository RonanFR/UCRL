import matplotlib.pyplot as plt
import pylab
import matplotlib.lines as mlines
import os
import pickle
from operator import add
from matplotlib import rc


rc('text', usetex=True)
rc('font', family='serif', size=27)

exp_id = 1000  # experiment ID
ratio = []
tmax = []
mdp_regret = 0

for j in range(1, 9):
    print(j)
    exp_id += 1
    path = '/home/lille3inria/Documents/pycharm-community-5.0.3/projects/UCRL/experiments/experiment ' + str(exp_id)
    simulations = os.listdir(path)
    # nb_simulations = 3
    nb_simulations = len(os.listdir(path))
    if nb_simulations > 0:
        with open(path + '/' + simulations[0], 'rb') as f:
            average = pickle.load(f).regret[-1]

        for simulation in simulations[1:nb_simulations]:
            with open(path + '/' + simulation, 'rb') as f:
                regret = pickle.load(f).regret[-1]
            average = average + regret
        average = 1/nb_simulations*average
        ratio += [average]
        tmax += [j]

ratio = [1/ratio[0]*x for x in ratio]
# Plot ratio
plt.plot(tmax, ratio, 'bo-', linewidth=5, markersize=18)

exp_id = 2000  # experiment ID
ratio = []
tmax = []
mdp_regret = 0

for j in range(1, 9):
    print(j)
    exp_id += 1
    path = '/home/lille3inria/Documents/pycharm-community-5.0.3/projects/UCRL/experiments/experiment ' + str(exp_id)
    simulations = os.listdir(path)
    # nb_simulations = 3
    nb_simulations = len(os.listdir(path))
    if nb_simulations > 0:
        with open(path + '/' + simulations[0], 'rb') as f:
            average = pickle.load(f).regret[-1]

        for simulation in simulations[1:nb_simulations]:
            with open(path + '/' + simulation, 'rb') as f:
                regret = pickle.load(f).regret[-1]
            average = average + regret
        average = 1/nb_simulations*average
        ratio +=  [average]
        tmax += [j]

ratio = [1/ratio[0]*x for x in ratio]
# Plot ratio
plt.plot(tmax, ratio, 'r*-', linewidth=5, markersize=20)

exp_id = 3000  # experiment ID
ratio = []
tmax = []
mdp_regret = 0

for j in range(1, 9):
    print(j)
    exp_id += 1
    path = '/home/lille3inria/Documents/pycharm-community-5.0.3/projects/UCRL/experiments/experiment ' + str(exp_id)
    simulations = os.listdir(path)
    # nb_simulations = 3
    nb_simulations = len(os.listdir(path))
    if nb_simulations > 0:
        with open(path + '/' + simulations[0], 'rb') as f:
            average = pickle.load(f).regret[-1]

        for simulation in simulations[1:nb_simulations]:
            with open(path + '/' + simulation, 'rb') as f:
                regret = pickle.load(f).regret[-1]
            average = average + regret
        average = 1/nb_simulations*average
        ratio +=  [average]
        tmax += [j]

ratio = [1/ratio[0]*x for x in ratio]
# Plot ratio
plt.plot(tmax, ratio, 'g^-', linewidth=5, markersize=18)

plt.plot([1, 8], [1, 1], 'k', linewidth=3)

# Axis title
plt.xlabel('Maximal duration of options $T_{\max}$', size=38)
plt.ylabel('Ratio of regrets $\mathcal{R}$', size=38)

# Caption
blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=18, label='20x20 Grid', linewidth=4)
red_line = mlines.Line2D([], [], color='red', marker='*', markersize=20, label='25x25 Grid', linewidth=4)
green_line = mlines.Line2D([], [], color='green', marker='^', markersize=18, label='30x30 Grid', linewidth=4)
plt.legend(handles=[blue_line, red_line, green_line], loc='upper left')

pylab.show()