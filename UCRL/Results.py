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
colors = ['k', 'b', 'c', 'g', 'y', 'orange', 'r', 'magenta']
lines = []

for j in range(1, 9):
    print(j)
    exp_id += 1
    path = '/home/lille3inria/Documents/pycharm-community-5.0.3/projects/UCRL/experiments/experiment ' + str(exp_id)
    simulations = os.listdir(path)
    # nb_simulations = 3
    nb_simulations = len(os.listdir(path))
    if nb_simulations > 0:
        with open(path + '/' + simulations[0], 'rb') as f:
            average = pickle.load(f).regret
        maxi = list(average)
        mini = list(average)

        for simulation in simulations[1:nb_simulations]:
            with open(path + '/' + simulation, 'rb') as f:
                regret = pickle.load(f).regret
            average = map(add, average, regret)
            maxi = [max(x, y) for x, y in zip(maxi, regret)]
            mini = [min(x, y) for x, y in zip(mini, regret)]
        average = [1/nb_simulations*x for x in average]
        # Plot regret
        plt.plot([1000*x for x in range(0, len(average))], average, color=colors[j-1], linewidth=5)
        # plt.plot([1000*x for x in range(0, len(average))], maxi, color=colors[j-1], linewidth=2)
        # plt.plot([1000*x for x in range(0, len(average))], mini, color=colors[j-1], linewidth=2)
        # Caption
        lines += [mlines.Line2D([], [], color=colors[j-1], label='$T_{\max}$ = '+str(j), linewidth=4)]

# Plot caption
plt.legend(handles=lines, loc='upper left')

# Axis title
plt.xlabel('Duration $T_{n}$', size=38)
plt.ylabel('Cumulative Regret $\Delta (T_{n})$', size=38)

pylab.show()