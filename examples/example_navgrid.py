import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
import numpy as np
from UCRL.envs.toys import NavigateGrid
import UCRL.envs.toys.OptionGrid as OptionGrid
import UCRL.envs.toys.OptionGridFree as OptionGridFree
import UCRL.envs.RewardDistributions as RewardDistributions
import UCRL.Ucrl as Ucrl
from UCRL.free_ucrl import FSUCRLv1, FSUCRLv2
from UCRL.envs import OptionEnvironment, MixedEnvironment
import UCRL.logging as ucrl_logger
import UCRL.bounds as tuning
from optparse import OptionParser

import matplotlib
# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
# if os.environ.get('DISPLAY') is None:
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-d", "--dimension", dest="dimension", type="int",
                  help="dimension of the gridworld", default=20)
parser.add_option("-n", "--duration", dest="duration", type="int",
                  help="duration of the experiment", default=40000000)
parser.add_option("-a", "--alg", dest="algorithm", type="str",
                  help="Name of the algorith to execute", default="FSUCRLv1") # UCRL, SUCRL, FSUCRLv1, FSUCRLv2
parser.add_option("-t", "--tmax", dest="t_max", type="int",
                  help="t_max for options", default=1)
parser.add_option("-b", "--bernstein", action="store_true", dest="use_bernstein",
                  default=False, help="use Bernstein bound")
parser.add_option("--rmax", dest="r_max", type="float",
                  help="maximum reward", default=-1)
parser.add_option("--p_alpha", dest="alpha_p", type="float",
                  help="range of transition matrix", default=0.01)
parser.add_option("--r_alpha", dest="alpha_r", type="float",
                  help="range of reward", default=0.1)
parser.add_option("--tau_alpha", dest="alpha_tau", type="float",
                  help="range of reward", default=0.8)
parser.add_option("--mc_alpha", dest="alpha_mc", type="float",
                  help="range for stationary distribution", default=0.01)
parser.add_option("--regret_steps", dest="regret_time_steps", type="int",
                  help="regret time steps", default=1000)
parser.add_option("-r", "--repetitions", dest="nb_simulations", type="int",
                  help="Number of repetitions", default=1)
parser.add_option("--env_file", dest="env_pickle_file", type="str",
                  help="Pickle file storing the mixed environment", default=None)#"escape_room_14_0.8.pickle")
parser.add_option("--id", dest="id", type="str",
                  help="Identifier of the script", default=None)
parser.add_option("-q", "--quiet",
                  action="store_true", dest="quiet", default=False,
                  help="don't print status messages to stdout")
parser.add_option("--seed", dest="seed_0", type=int, default=1011005946, #random.getrandbits(16),
                  help="Seed used to generate the random seed sequence")

(in_options, in_args) = parser.parse_args()

assert in_options.algorithm in ["UCRL", "SUCRL", "FSUCRLv1", "FSUCRLv2"]

if in_options.t_max < 1:
    raise ValueError("t_max should be >= 1")

if in_options.r_max < 0:
    in_options.r_max = in_options.dimension

if in_options.id is None:
    in_options.id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())

config = vars(in_options)

# ------------------------------------------------------------------------------
# Relevant code
# ------------------------------------------------------------------------------
# Define environment
initial_position = [in_options.dimension - 1, in_options.dimension - 1]  # initial state
nb_target_actions = 1  # number of available actions in the targeted state
nb_reset_actions = 1  # number of available actions in the reset state
reward_distribution_states = RewardDistributions.ConstantReward(0)
reward_distribution_target = RewardDistributions.ConstantReward(in_options.r_max)
grid = NavigateGrid.NavigateGrid(dimension=in_options.dimension,
                                 initial_position=initial_position,
                                 reward_distribution_states=reward_distribution_states,
                                 reward_distribution_target=reward_distribution_target,
                                 nb_target_actions=nb_target_actions)

# Add options
if in_options.algorithm != 'UCRL':
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

folder_results = os.path.abspath('{}_navgrid_{}'.format(in_options.algorithm,
                                                       in_options.id))
if os.path.exists(folder_results):
    shutil.rmtree(folder_results)
os.makedirs(folder_results)

np.random.seed(in_options.seed_0)
random.seed(in_options.seed_0)
seed_sequence = [random.randint(0, 2**30) for _ in range(in_options.nb_simulations)]

config['seed_sequence'] = seed_sequence

with open(os.path.join(folder_results, 'settings.conf'), 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

# Main loop
for rep in range(in_options.nb_simulations):
    seed = seed_sequence[rep]  # set seed
    #seed = 1011005946
    np.random.seed(seed)
    random.seed(seed)
    print("rep: {}".format(rep))

    name = "trace_{}".format(rep)
    ucrl_log = ucrl_logger.create_multilogger(logger_name=name,
                                              console=not in_options.quiet,
                                              filename=name,
                                              path=folder_results)
    if in_options.algorithm == "UCRL":
        ucrl = Ucrl.UcrlMdp(
            grid,
            r_max=in_options.r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm
    elif in_options.algorithm == "SUCRL":
        ucrl = Ucrl.UcrlSmdpBounded(
            environment=copy.deepcopy(mixed_environment),
            r_max=in_options.r_max,
            t_max=in_options.t_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            alpha_tau=in_options.alpha_tau,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm
    elif in_options.algorithm == "FSUCRLv1":
        ucrl = FSUCRLv1(
            environment=copy.deepcopy(mixed_environment),
            r_max=in_options.r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            alpha_mc=in_options.alpha_mc,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm
    elif in_options.algorithm == "FSUCRLv2":
        ucrl = FSUCRLv2(
            environment=copy.deepcopy(mixed_environment),
            r_max=in_options.r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            alpha_mc=in_options.alpha_mc,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm

    ucrl_log.info("[id: {}] {}".format(in_options.id, type(ucrl).__name__))
    ucrl_log.info("seed: {}".format(seed))
    ucrl_log.info("Config: {}".format(config))

    h = ucrl.learn(in_options.duration, in_options.regret_time_steps)  # learn task
    ucrl.clear_before_pickle()

    with open(os.path.join(folder_results, 'ucrl_{}.pickle'.format(rep)), 'wb') as f:
        pickle.dump(ucrl, f)

    plt.figure()
    plt.plot(ucrl.span_values, '-')
    plt.xlabel("Points")
    plt.ylabel("Span")
    plt.savefig(os.path.join(folder_results, "span_{}.png".format(rep)))
    plt.figure()
    plt.plot(ucrl.regret, '-')
    plt.xlabel("Points")
    plt.ylabel("Regret")
    plt.savefig(os.path.join(folder_results, "regret_{}.png".format(rep)))
    # plt.show()
    plt.close('all')

