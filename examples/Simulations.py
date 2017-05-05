from UCRL.NavigateGrid import NavigateGrid, NavigateGrid2
from UCRL.OptionEnvironment import OptionEnvironment
import UCRL.OptionGrid as OptionGrid
import UCRL.Ucrl as Ucrl
import UCRL.RewardDistributions as RewardDistributions

import cProfile
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random
import pickle
import os
import copy
import time


# Define environment
dimension = 20  # dimension of the grid
initial_position = [dimension-1, dimension-1]  # initial state
nb_target_actions = 1  # number of available actions in the targeted state
nb_reset_actions = 1  # number of available actions in the reset state
reward_distribution_states = RewardDistributions.ConstantReward(1)
reward_distribution_target = RewardDistributions.ConstantReward(dimension)
# grid = NavigateGrid(dimension, initial_position, reward_distribution_states, reward_distribution_target,
#                     nb_target_actions)
grid = NavigateGrid2(dimension, initial_position, reward_distribution_states, reward_distribution_target,
                     nb_reset_actions)

# Add options
t_max = 3
# options = OptionGrid.OptionGrid1(grid, t_max)
# options = OptionGrid.OptionGrid2(grid, t_max)
options = OptionGrid.OptionGrid3(grid, t_max)
option_environment = OptionEnvironment(grid, options.state_options, options.options_policies, options.options_terminating_conditions, True)

# Define learning algorithm
c = 0.8
r_max = dimension  # maximal reward
range_r = t_max*c
range_tau = (t_max - 1)*c
range_p = 0.015
# ucrl = Ucrl.UcrlMdp(grid, r_max, range_r=range_r, range_p=range_p)
# ucrl = Ucrl.UcrlSmdpBounded(options, r_max, t_max, range_r=range_r, range_p=range_p, range_tau = range_tau)

# Set seeds
seed = 1
np.random.seed(seed)
random.seed(seed)

# Learn task
duration = 30000000  # number of decision steps
regret_time_step = 1000

# Main loop
id_experiment = 1203  # experiment ID
nb_simulations = 100
for i in range(0, nb_simulations):
    start_time = time.time()
    seed = i  # set seed
    np.random.seed(seed)
    random.seed(seed)
    ucrl = Ucrl.UcrlSmdpBounded(copy.deepcopy(option_environment), r_max, t_max, range_r=range_r, range_p=range_p, range_tau=range_tau)  # learning algorithm
    ucrl.learn(duration, regret_time_step)  # learn task
    with open(os.getcwd()+'/experiments/experiment '+str(id_experiment)+'/simu '+str(i), 'wb') as f:  # store regret
        pickle.dump(ucrl, f)
    print("--- %s seconds ---" % (time.time() - start_time))