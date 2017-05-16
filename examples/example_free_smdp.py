import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
import numpy as np
import UCRL.envs.toys.NavigateGrid as NavigateGrid
import UCRL.envs.toys.OptionGrid as OptionGrid
import UCRL.envs.toys.OptionGridFree as OptionGridFree
import UCRL.envs.RewardDistributions as RewardDistributions
from UCRL.free_ucrl import FreeUCRL_Alg1
from UCRL.envs import OptionEnvironment, MixedEnvironment
import UCRL.logging as ucrl_logger
from optparse import OptionParser

import matplotlib
# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-d", "--dimension", dest="dimension", type="int",
                  help="dimension of the gridworld", default=5)
parser.add_option("-n", "--duration", dest="duration", type="int",
                  help="duration of the experiment", default=5000000)
parser.add_option("-t", "--tmax", dest="t_max", type="int",
                  help="t_max for options", default=3)
parser.add_option("-c", dest="c", type="float",
                  help="c value", default=0.8)
parser.add_option("--rmax", dest="r_max", type="float",
                  help="maximum reward", default=-1)
parser.add_option("--p_range", dest="range_p", type="float",
                  help="range of transition matrix", default=0.1)
parser.add_option("--mu_range", dest="range_mu_p", type="float",
                  help="range for stationary distribution", default=0.1)
parser.add_option("--regret_steps", dest="regret_time_steps", type="int",
                  help="regret time steps", default=1000)
parser.add_option("-r", "--repetitions", dest="nb_simulations", type="int",
                  help="Number of repetitions", default=1)
parser.add_option("--id", dest="id", type="str",
                  help="Identifier of the script", default=None)
parser.add_option("-q", "--quiet",
                  action="store_true", dest="quiet", default=False,
                  help="don't print status messages to stdout")
parser.add_option("--seed", dest="seed_0", default=random.getrandbits(64),
                  help="Seed used to generate the random seed sequence")

(in_options, in_args) = parser.parse_args()

if in_options.r_max < 0:
    in_options.r_max = in_options.dimension

if in_options.id is None:
    in_options.id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())

range_r = in_options.t_max * in_options.c
range_tau = (in_options.t_max - 1) * in_options.c

config = vars(in_options)
config['range_r'] = range_r
config['range_tau'] = range_tau

# Define environment
initial_position = [in_options.dimension - 1, in_options.dimension - 1]  # initial state
nb_target_actions = 1  # number of available actions in the targeted state
nb_reset_actions = 1  # number of available actions in the reset state
reward_distribution_states = RewardDistributions.ConstantReward(0)
reward_distribution_target = RewardDistributions.ConstantReward(in_options.dimension)
grid = NavigateGrid.NavigateGrid(dimension=in_options.dimension,
                                 initial_position=initial_position,
                                 reward_distribution_states=reward_distribution_states,
                                 reward_distribution_target=reward_distribution_target,
                                 nb_target_actions=nb_target_actions)
# grid = NavigateGrid.NavigateGrid2(
#     dimension=dimension,
#     initial_position=initial_position,
#     reward_distribution_states=reward_distribution_states,
#     reward_distribution_target=reward_distribution_target,
#     nb_reset_actions=nb_reset_actions
# )

# Add options
options = OptionGrid.OptionGrid1(grid, in_options.t_max)
# options = OptionGrid.OptionGrid2(grid, t_max)
# options = OptionGrid.OptionGrid3(grid=grid, t_max=t_max)
options_free = OptionGridFree.OptionGrid1(grid=grid, t_max=in_options.t_max)
option_environment = OptionEnvironment(
    environment=grid,
    state_options=options.state_options,
    options_policies=options.options_policies,
    options_terminating_conditions=options.options_terminating_conditions,
    is_optimal=True
)

mixed_environment = MixedEnvironment(
    environment=grid,
    state_options=options_free.state_options,
    options_policies=options_free.options_policies,
    options_terminating_conditions=options_free.options_terminating_conditions,
    is_optimal=True,
    delete_environment_actions="all"
)

nb_options = mixed_environment.nb_options
x = mixed_environment.reachable_states_per_option

assert len(mixed_environment.state_options) == len(options.state_options)
for x, y in zip(mixed_environment.state_options, options.state_options):
    for i in y:
        f = False
        for j in x:
            if j == i + mixed_environment.threshold_options + 1:
                f = True
    assert f

folder_results = os.path.abspath('freesmdp_{}'.format(in_options.id))
if os.path.exists(folder_results):
    shutil.rmtree(folder_results)
os.makedirs(folder_results)

random.seed(in_options.seed_0)
seed_sequence = [random.randint(0, 2**30) for _ in range(in_options.nb_simulations)]

config['seed_sequence'] = seed_sequence

with open(os.path.join(folder_results, 'settings.conf'), 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

# Main loop
for rep in range(in_options.nb_simulations):
    seed = seed_sequence[rep]  # set seed
    np.random.seed(seed)
    random.seed(seed)

    name = "trace_{}".format(rep)
    ucrl_log = ucrl_logger.create_multilogger(console=not in_options.quiet,
                                              filename=name,
                                              path=folder_results)
    ucrl = FreeUCRL_Alg1(
        environment=copy.deepcopy(mixed_environment),
        r_max=in_options.r_max,
        range_r=range_r,
        range_p=in_options.range_p,
        range_mu_p=in_options.range_mu_p,
        verbose=1,
        logger=ucrl_log)  # learning algorithm
    h = ucrl.learn(in_options.duration, in_options.regret_time_steps)  # learn task
    ucrl.clear_before_pickle()

    with open(os.path.join(folder_results, 'ucrl_{}.pickle'.format(rep)), 'wb') as f:
        pickle.dump(ucrl, f)

    plt.figure()
    plt.plot(ucrl.span_values, 'o-')
    plt.xlabel("Points")
    plt.ylabel("Span")
    plt.savefig(os.path.join(folder_results, "span_{}.png".format(rep)))
    plt.figure()
    plt.plot(ucrl.regret, 'o-')
    plt.xlabel("Points")
    plt.ylabel("Regret")
    plt.savefig(os.path.join(folder_results, "regret_{}.png".format(rep)))
    # plt.show()
    plt.close('all')

