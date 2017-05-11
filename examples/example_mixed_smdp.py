import copy
import os
import pickle
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import UCRL.envs.toys.NavigateGrid as NavigateGrid
import UCRL.envs.toys.OptionGrid as OptionGrid
import UCRL.envs.RewardDistributions as RewardDistributions
from UCRL.smdp_ucrl import SMDPUCRL_Mixed
from UCRL.envs import MixedEnvironment

# Define environment
dimension = 20  # dimension of the grid
initial_position = [dimension - 1, dimension - 1]  # initial state
nb_target_actions = 1  # number of available actions in the targeted state
nb_reset_actions = 1  # number of available actions in the reset state
reward_distribution_states = RewardDistributions.ConstantReward(1)
reward_distribution_target = RewardDistributions.ConstantReward(dimension)
# grid = NavigateGrid(dimension, initial_position, reward_distribution_states, reward_distribution_target,
#                     nb_target_actions)
grid = NavigateGrid.NavigateGrid2(
    dimension=dimension,
    initial_position=initial_position,
    reward_distribution_states=reward_distribution_states,
    reward_distribution_target=reward_distribution_target,
    nb_reset_actions=nb_reset_actions
)

# Add options
t_max = 3
# options = OptionGrid.OptionGrid1(grid, t_max)
# options = OptionGrid.OptionGrid2(grid, t_max)
options = OptionGrid.OptionGrid3(grid=grid, t_max=t_max)
mixed_environment = MixedEnvironment(
    environment=grid,
    state_options=options.state_options,
    options_policies=options.options_policies,
    options_terminating_conditions=options.options_terminating_conditions,
    is_optimal=True,
    delete_environment_actions="all"
)

assert len(mixed_environment.state_options) == len(options.state_options)
for x, y in zip(mixed_environment.state_options, options.state_options):
    for i in y:
        f = False
        for j in x:
            if j == i + mixed_environment.threshold_options + 1:
                f = True
    assert f

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
nb_simulations = 1
for i in range(0, nb_simulations):
    start_time = time.time()
    seed = i  # set seed
    np.random.seed(seed)
    random.seed(seed)
    ucrl = SMDPUCRL_Mixed(copy.deepcopy(mixed_environment), r_max, t_max,
                          range_r=range_r, range_p=range_p, range_tau=range_tau,
                          verbose=1)  # learning algorithm
    ucrl.learn(duration, regret_time_step)  # learn task
    times = np.array(ucrl.timing)
    df = pd.DataFrame(data=times, columns=["old", "new"])
    df.to_csv("timing_{}.csv".format(i))
    plt.figure()
    plt.plot(np.arange(times.shape[0]), times[:,0], 'o-', label="OLD")
    plt.plot(np.arange(times.shape[0]), times[:,1], 'o-', label="NEW")
    plt.xlabel("UCRL Iterations")
    plt.ylabel("Time [s]")
    plt.legend()
    plt.savefig("timing_{}.png".format(i))
    plt.show()
    with open(os.getcwd()+'/experiments/experiment '+str(id_experiment)+'/simu '+str(i), 'wb') as f:  # store regret
        pickle.dump(ucrl, f)
    print("--- %s seconds ---" % (time.time() - start_time))

