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
import UCRL.bounds as tuning
from optparse import OptionParser, OptionGroup

import matplotlib
# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
#if os.environ.get('DISPLAY') is None:
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-d", "--dimension", dest="dimension", type="int",
                  help="dimension of the gridworld", default=14)
parser.add_option("-n", "--duration", dest="duration", type="int",
                  help="duration of the experiment", default=30000000)
parser.add_option("-b", "--boundtype", type="str", dest="bound_type",
                  help="Selects the bound type", default="chernoff")
parser.add_option("--rmax", dest="r_max", type="float",
                  help="maximum reward", default=1)
parser.add_option("--p_alpha", dest="alpha_p", type="float",
                  help="range of transition matrix", default=0.02)
parser.add_option("--r_alpha", dest="alpha_r", type="float",
                  help="range of reward", default=0.8)
parser.add_option("--tau_alpha", dest="alpha_tau", type="float",
                  help="range of reward", default=0.5)
parser.add_option("--mc_alpha", dest="alpha_mc", type="float",
                  help="range for stationary distribution", default=0.02)
parser.add_option("--regret_steps", dest="regret_time_steps", type="int",
                  help="regret time steps", default=1000)
parser.add_option("-r", "--repetitions", dest="nb_simulations", type="int",
                  help="Number of repetitions", default=1)
parser.add_option("--rep_offset", dest="nb_sim_offset", type="int",
                  help="Repetitions starts at the given number", default=0)
parser.add_option("--env_file", dest="env_pickle_file", type="str",
                  help="Pickle file storing the mixed environment", default=None) #"escape_room_14_0.8.pickle")
parser.add_option("--id", dest="id", type="str",
                  help="Identifier of the script", default=None)
parser.add_option("--path", dest="path", type="str",
                  help="Path to the folder where to store results", default=None)
parser.add_option("-q", "--quiet",
                  action="store_true", dest="quiet", default=False,
                  help="don't print status messages to stdout")
parser.add_option("--seed", dest="seed_0", type=int, default=1011005946, #random.getrandbits(16),
                  help="Seed used to generate the random seed sequence")
parser.add_option("--succ_prob", dest="success_probability", type="float",
                  help="Success probability of an action", default=0.8)

alg_desc = """Here the description of the algorithms                                
|- UCRL                                                                       
|- SUCRL                                                                                  
... v2: sigma_tau = max(sigma_tau)                                            
.       sigma_R = r_max sqrt(tau_max + max(sigma_tau)^2)                      
... v3: sigma_tau -> per option                                               
.       sigma_R = r_max sqrt(tau_max + max(sigma_tau)^2) -> per option         
|- FSUCRLv1                                                                      
|- FSUCRLv2                                                                      
"""
group1 = OptionGroup(parser, title='Algorithms', description=alg_desc)
group1.add_option("-a", "--alg", dest="algorithm", type="str",
                  help="Name of the algorith to execute"
                       "[UCRL, SUCRL_v1, SUCRL_v2, FSUCRLv1, FSUCRLv2]",
                  default="FSUCRLv2")
# UCRL, SUCRL_v1, SUCRL_v2, SUCRL_v3, SUCRL_v4, FSUCRLv1, FSUCRLv2

(in_options, in_args) = parser.parse_args()

if in_options.id and in_options.path:
    parser.error("options --id and --path are mutually exclusive")

assert in_options.algorithm in ["UCRL", "SUCRL_v2", "SUCRL_v3", "FSUCRLv1", "FSUCRLv2"]
assert in_options.nb_sim_offset >= 0

if in_options.r_max < 0:
    in_options.r_max = in_options.dimension

if in_options.id is None:
    in_options.id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())

config = vars(in_options)

# ------------------------------------------------------------------------------
# Relevant code
# ------------------------------------------------------------------------------
reward_distribution_states = RewardDistributions.ConstantReward(0)
reward_distribution_target = RewardDistributions.ConstantReward(in_options.r_max)

success_probability = in_options.success_probability

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

if in_options.path is None:
    folder_results = os.path.abspath('{}_4rooms_{}'.format(in_options.algorithm,
                                                           in_options.id))
    if os.path.exists(folder_results):
        shutil.rmtree(folder_results)
    os.makedirs(folder_results)
else:
    folder_results = os.path.abspath(in_options.path)
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)

np.random.seed(in_options.seed_0)
random.seed(in_options.seed_0)
seed_sequence = [random.randint(0, 2**30) for _ in range(in_options.nb_simulations)]

config['seed_sequence'] = seed_sequence

with open(os.path.join(folder_results, 'settings{}.conf'.format(in_options.nb_sim_offset)), 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
start_sim = in_options.nb_sim_offset
end_sim = start_sim + in_options.nb_simulations
for rep in range(start_sim, end_sim):
    seed = seed_sequence[rep-start_sim]  # set seed
    #seed = 1011005946
    np.random.seed(seed)
    random.seed(seed)
    print("rep: {}/{}".format(rep-start_sim, in_options.nb_simulations))

    name = "trace_{}".format(rep)
    ucrl_log = ucrl_logger.create_multilogger(logger_name=name,
                                              console=not in_options.quiet,
                                              filename=name,
                                              path=folder_results)

    if in_options.algorithm == "UCRL":
        ucrl = Ucrl.UcrlMdp(
            copy.deepcopy(env),
            r_max=in_options.r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=ucrl_log,
            bound_type=in_options.bound_type)  # learning algorithm
    elif in_options.algorithm[0:5] == "SUCRL":
        r_max = in_options.r_max
        tau_min = np.min(mixed_env.tau_options)
        tau_max = np.max(mixed_env.tau_options)

        version = in_options.algorithm[-2:]
        if version == "v2":
            sigma_tau = np.max(mixed_env.reshaped_sigma_tau())
            sigma_r = r_max * np.sqrt(tau_max + sigma_tau**2)
        elif version == "v3":
            sigma_tau = mixed_env.reshaped_sigma_tau()
            tau_bar = mixed_env.reshaped_tau_bar()
            sigma_r = r_max * np.sqrt(tau_bar + sigma_tau**2)
        else:
            raise ValueError("Unknown SUCRL version")

        ucrl = Ucrl.UcrlSmdpExp(
            environment=copy.deepcopy(mixed_env),
            r_max=r_max,
            tau_min=tau_min,
            tau_max=tau_max,
            sigma_tau=sigma_tau,
            sigma_r=sigma_r,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            alpha_tau=in_options.alpha_tau,
            verbose=1,
            logger=ucrl_log,
            bound_type=in_options.bound_type)  # learning algorithm
    elif in_options.algorithm == "FSUCRLv1":
        ucrl = FSUCRLv1(
            environment=copy.deepcopy(mixed_env),
            r_max=in_options.r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            alpha_mc=in_options.alpha_mc,
            verbose=1,
            logger=ucrl_log,
            bound_type=in_options.bound_type)  # learning algorithm
    elif in_options.algorithm == "FSUCRLv2":
        ucrl = FSUCRLv2(
            environment=copy.deepcopy(mixed_env),
            r_max=in_options.r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            alpha_mc=in_options.alpha_mc,
            verbose=1,
            logger=ucrl_log,
            bound_type=in_options.bound_type)  # learning algorithm

    ucrl_log.info("[id: {}] {}".format(in_options.id, type(ucrl).__name__))
    ucrl_log.info("seed: {}".format(seed))
    ucrl_log.info("Config: {}\n".format(config))

    alg_desc = ucrl.description()
    ucrl_log.info("alg desc: {}".format(alg_desc))

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

