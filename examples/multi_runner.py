import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
import numpy as np
from rlexplorer.envs.toys import Toy3D_1, Toy3D_2, RiverSwim
import rlexplorer.Ucrl as Ucrl
import rlexplorer.tucrl as tucrl
import rlexplorer.posteriorsampling as psalgs
# from rlexplorer.olp import OLP
import rlexplorer.logging as ucrl_logger
import matplotlib

# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
# if os.environ.get('DISPLAY') is None:
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#####################################################################
# PARAMETERS

from collections import namedtuple
fields = ("nb_sim_offset", "nb_simulations", "id", "path", "algorithm", "domain", "seed_0",  "alpha_r",
          "alpha_p", "posterior", "use_true_reward", "duration", "regret_time_steps", "quiet")
Options = namedtuple("Options", fields)

id_v = None

in_options = Options(
nb_sim_offset = 0,
nb_simulations = 4,
id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now()) if id_v is None else id_v,
path = None,
algorithm = "TSDE",
domain = "RiverSwim",
seed_0 = 3131331,
alpha_r = 1.,
alpha_p = 1.,
posterior = "Bernoulli", #["Bernoulli", "Normal", None]
use_true_reward = True,
duration = 200000,
regret_time_steps=1000,
quiet=False
)

#####################################################################

config = vars(in_options._asdict())

# ------------------------------------------------------------------------------
# Relevant code
# ------------------------------------------------------------------------------

# env = Toy3D_1(delta=in_options.mdp_delta,
#               stochastic_reward=in_options.stochastic_reward,
#               uniform_reward=in_options.uniform_reward,
#               uniform_range=in_options.unifrew_range)
# env = Toy3D_2(delta=in_options.mdp_delta,
#               epsilon=in_options.mdp_delta/5.,
#               stochastic_reward=in_options.stochastic_reward)
env = RiverSwim()
# gym_env = TaxiEnv()
r_max = max(1, np.asscalar(np.max(env.R_mat)))

if in_options.path is None:
    folder_results = os.path.abspath('{}_{}_{}'.format(in_options.algorithm, type(env).__name__,
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
seed_sequence = [random.randint(0, 2 ** 30) for _ in range(in_options.nb_simulations)]

config['seed_sequence'] = seed_sequence

with open(os.path.join(folder_results, 'settings{}.conf'.format(in_options.nb_sim_offset)), 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
start_sim = in_options.nb_sim_offset
end_sim = start_sim + in_options.nb_simulations
for rep in range(start_sim, end_sim):
    seed = seed_sequence[rep - start_sim]  # set seed
    np.random.seed(seed)
    random.seed(seed)
    print("rep: {}/{}".format(rep - start_sim, in_options.nb_simulations))

    env.reset()
    env_desc = env.description()

    name = "trace_{}".format(rep)
    expalg_log = ucrl_logger.create_multilogger(logger_name=name,
                                                console=not in_options.quiet,
                                                filename=name,
                                                path=folder_results)
    expalg_log.info("mdp desc: {}".format(env_desc))
    ofualg = None
    if in_options.algorithm == "UCRL":
        ofualg = Ucrl.UcrlMdp(
            env,
            r_max=r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=expalg_log,
            random_state=seed,
            known_reward=in_options.use_true_reward)  # learning algorithm
    elif in_options.algorithm == "TUCRL":
        ofualg = tucrl.TUCRL(
            environment=env,
            r_max=r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=expalg_log,
            random_state=seed,
            known_reward=in_options.use_true_reward)
    elif in_options.algorithm == "TSDE":
        ofualg = psalgs.TSDE(environment=env,
                             r_max=r_max,
                             verbose=1,
                             logger=expalg_log,
                             random_state=seed,
                             posterior=in_options.posterior,
                             known_reward=in_options.use_true_reward)
    # elif in_options.algorithm == "OLP":
    #     ofualg = OLP(
    #         environment=env,
    #         r_max=r_max,
    #         alpha_r=alpha_r,
    #         alpha_p=alpha_p,
    #         verbose=1,
    #         logger=ucrl_log,
    #         bound_type_p=in_options.bound_type,
    #         bound_type_rew=in_options.bound_type,
    #         random_state=seed)

    expalg_log.info("[id: {}] {}".format(id, type(ofualg).__name__))
    expalg_log.info("seed: {}".format(seed))
    expalg_log.info("Config: {}".format(config))

    alg_desc = ofualg.description()
    expalg_log.info("alg desc: {}".format(alg_desc))

    pickle_name = 'run_{}.pickle'.format(rep)
    try:
        h = ofualg.learn(in_options.duration, in_options.regret_time_steps)  # learn task
    except Ucrl.EVIException as valerr:
        expalg_log.info("EVI-EXCEPTION -> error_code: {}".format(valerr.error_value))
        pickle_name = 'exception_model_{}.pickle'.format(rep)
    except:
        expalg_log.info("EXCEPTION")
        pickle_name = 'exception_model_{}.pickle'.format(rep)

    ofualg.clear_before_pickle()
    with open(os.path.join(folder_results, pickle_name), 'wb') as f:
        pickle.dump(ofualg, f)

    plt.figure()
    plt.plot(ofualg.span_values, '-')
    plt.xlabel("Points")
    plt.ylabel("Span")
    plt.savefig(os.path.join(folder_results, "span_{}.png".format(rep)))
    plt.figure()
    plt.plot(ofualg.regret, '-')
    plt.xlabel("Points")
    plt.ylabel("Regret")
    plt.savefig(os.path.join(folder_results, "regret_{}.png".format(rep)))
    plt.show()
    if in_options.algorithm == "TUCRL":
        plt.plot(ofualg.bad_state_action_values)
        plt.xlabel("Points")
        plt.ylabel("Sum bad states")
        plt.savefig(os.path.join(folder_results, "bad_statessss_{}.png".format(rep)))
    plt.close('all')

    print('policy indices: ')
    print(ofualg.policy_indices)
