import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
import numpy as np
from UCRL.envs.toys import FourRoomsMaze
from UCRL.envs.toys.roommaze import state2coord, coord2state
from UCRL.envs.toys.roommaze import EscapeRoom
import UCRL.envs.RewardDistributions as RewardDistributions
import UCRL.Ucrl as Ucrl
from UCRL.free_ucrl import FSUCRLv1, FSUCRLv2
import UCRL.logging as ucrl_logger
import UCRL.parameters_init as tuning
from optparse import OptionParser

import matplotlib
# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-d", "--dimension", dest="dimension", type="int",
                  help="dimension of the gridworld", default=14)
parser.add_option("-n", "--duration", dest="duration", type="int",
                  help="duration of the experiment", default=30000000)
parser.add_option("-a", "--alg", dest="algorithm", type="str",
                  help="Name of the algorith to execute", default="FSUCRLv2") # UCRL, SUCRL, FSUCRLv1, FSUCRLv2
parser.add_option("-c", dest="c", type="float",
                  help="c value", default=0.8)
parser.add_option("-t", "--tmax", dest="t_max", type="int",
                  help="t_max for options", default=-1)
parser.add_option("-b", "--bernstein", action="store_true", dest="use_bernstein",
                  default=False, help="use Bernstein bound")
parser.add_option("--rmax", dest="r_max", type="float",
                  help="maximum reward", default=-1)
parser.add_option("--p_range", dest="range_p", type="float",
                  help="range of transition matrix", default=-1)
parser.add_option("--r_range", dest="range_r", type="float",
                  help="range of reward", default=-1)
parser.add_option("--mu_range", dest="range_mu_p", type="float",
                  help="range for stationary distribution", default=-1)
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
    in_options.t_max = 1 + in_options.dimension // 2

if in_options.r_max < 0:
    in_options.r_max = in_options.dimension

if in_options.id is None:
    in_options.id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())

range_tau = (in_options.t_max - 1) * in_options.c

if in_options.range_p < 0:
    if not in_options.use_bernstein:
        in_options.range_p = tuning.range_p_from_hoeffding(
            nb_states=in_options.dimension, nb_actions=4, nb_observations=5)
    else:
        in_options.range_p = tuning.range_p_from_bernstein(
            nb_states=in_options.dimension, nb_actions=4, nb_observations=10)
        # in_options.range_p = 1. ##############

if in_options.range_mu_p < 0:
    in_options.range_mu_p = tuning.range_p_from_hoeffding(
        nb_states=2, nb_actions=4, nb_observations=10)
    # in_options.range_mu_p = 1
if in_options.range_r < 0:
    range_r = tuning.range_r_from_hoeffding(
        nb_states=in_options.dimension, nb_actions=4, nb_observations=40)
    # range_r = 1
    range_tau = range_r * (in_options.t_max - 1.)
    in_options.range_r = in_options.t_max * range_r

config = vars(in_options)
if in_options.algorithm == "SUCRL":
    config['range_tau'] = range_tau


# ------------------------------------------------------------------------------
# Relevant code
# ------------------------------------------------------------------------------
reward_distribution_states = RewardDistributions.ConstantReward(0)
reward_distribution_target = RewardDistributions.ConstantReward(in_options.dimension)

success_probability = 0.8

env = FourRoomsMaze(dimension=in_options.dimension,
                    initial_position=[in_options.dimension-1, in_options.dimension-1],
                    reward_distribution_states=reward_distribution_states,
                    reward_distribution_target=reward_distribution_target,
                    target_coordinates= [0,0],
                    success_probability=success_probability)

if in_options.algorithm != "UCRL":
    if in_options.env_pickle_file is not None:
        mixed_env = pickle.load( open( in_options.env_pickle_file, "rb" ) )
        assert np.isclose(mixed_env.environment.p_success, success_probability)
        assert mixed_env.environment.dimension == in_options.dimension
    else:
        mixed_env = EscapeRoom(env)
        fname = "escape_room_{}_{}.pickle".format(in_options.dimension, success_probability)
        with open(fname, "wb") as f:
            pickle.dump(mixed_env, f)
        print("Mixed environment saved in {}".format(fname))


# # check optimal policy
# for i, a in enumerate(env.optimal_policy):
#     row, col = state2coord(i,dimension)
#     print("{}".format(a), end=" ")
#     if col == dimension-1:
#         print("")

folder_results = os.path.abspath('{}_4rooms_{}'.format(in_options.algorithm,
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
            env,
            r_max=in_options.r_max,
            range_r=in_options.range_r,
            range_p=in_options.range_p,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm
    elif in_options.algorithm == "SUCRL":
        ucrl = Ucrl.UcrlSmdpBounded(
            environment=copy.deepcopy(mixed_env),
            r_max=in_options.r_max,
            t_max=in_options.t_max,
            range_r=in_options.range_r,
            range_p=in_options.range_p,
            range_tau=range_tau,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm
    elif in_options.algorithm == "FSUCRLv1":
        ucrl = FSUCRLv1(
            environment=copy.deepcopy(mixed_env),
            r_max=in_options.r_max,
            range_r=in_options.range_r,
            range_p=in_options.range_p,
            range_mu_p=in_options.range_mu_p,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm
    elif in_options.algorithm == "FSUCRLv2":
        ucrl = FSUCRLv2(
            environment=copy.deepcopy(mixed_env),
            r_max=in_options.r_max,
            range_r=in_options.range_r,
            range_p=in_options.range_p,
            range_opt_p=in_options.range_mu_p,
            verbose=1,
            logger=ucrl_log,
            bound_type="bernstein" if in_options.use_bernstein else "hoeffding")  # learning algorithm

    ucrl_log.info("[id: {}] {}".format(in_options.id, type(ucrl).__name__))
    ucrl_log.info("seed: {}".format(seed))
    ucrl_log.info("Using Bernstein: {}".format(in_options.use_bernstein))
    ucrl_log.info("max gain: {}".format(env.max_gain))
    ucrl_log.info("span: {}".format(env.span / in_options.r_max))

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

