import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
import numpy as np
from gym.envs.toy_text.taxi import TaxiEnv
from UCRL.envs.wrappers import GymDiscreteEnvWrapper
import UCRL.Ucrl as Ucrl
from UCRL.tucrl import TUCRL
import UCRL.span_algorithms as spalg
import UCRL.logging as ucrl_logger
from optparse import OptionParser, OptionGroup

import matplotlib
# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
# if os.environ.get('DISPLAY') is None:
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = OptionParser()
parser.add_option("-n", "--duration", dest="duration", type="int",
                  help="duration of the experiment", default=50000000)
parser.add_option("-b", "--boundtype", type="str", dest="bound_type",
                  help="Selects the bound type", default="bernstein")
parser.add_option("-c", "--span_constraint", type="float", dest="span_constraint",
                  help="Uppper bound to the bias span", default=np.inf)
parser.add_option("--operatortype", type="str", dest="operator_type",
                  help="Select the operator to use for SC-EVI", default="T")
parser.add_option("--p_alpha", dest="alpha_p", type="float",
                  help="range of transition matrix", default=0.05)
parser.add_option("--r_alpha", dest="alpha_r", type="float",
                  help="range of reward", default=0.05)
parser.add_option("--regret_steps", dest="regret_time_steps", type="int",
                  help="regret time steps", default=5000)
parser.add_option("-r", "--repetitions", dest="nb_simulations", type="int",
                  help="Number of repetitions", default=1)
parser.add_option("--no_aug_rew", dest="augmented_reward", action="store_false", default="True")
parser.add_option("--stochrew", dest="stochastic_reward", action="store_true", default="False")
parser.add_option("--rep_offset", dest="nb_sim_offset", type="int",
                  help="Repetitions starts at the given number", default=0)
parser.add_option("--id", dest="id", type="str",
                  help="Identifier of the script", default=None)
parser.add_option("--path", dest="path", type="str",
                  help="Path to the folder where to store results", default=None)
parser.add_option("-q", "--quiet",
                  action="store_true", dest="quiet", default=False,
                  help="don't print status messages to stdout")
parser.add_option("--seed", dest="seed_0", type=int, default=1011005946, #random.getrandbits(16),
                  help="Seed used to generate the random seed sequence")

alg_desc = """Here the description of the algorithms                                
|- UCRL                                                                       
|- SCAL                                                                                                                                            
|- TUCRL                                                                                                                                          
"""
group1 = OptionGroup(parser, title='Algorithms', description=alg_desc)
group1.add_option("-a", "--alg", dest="algorithm", type="str",
                  help="Name of the algorith to execute"
                       "[UCRL, SCAL, TUCRL]",
                  default="TUCRL")
parser.add_option_group(group1)

(in_options, in_args) = parser.parse_args()

if in_options.id and in_options.path:
    parser.error("options --id and --path are mutually exclusive")

assert in_options.algorithm in ["UCRL", "SCAL", "TUCRL"]
assert in_options.nb_sim_offset >= 0
assert in_options.operator_type in ['T', 'N']

if in_options.id is None:
    in_options.id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())

config = vars(in_options)

# ------------------------------------------------------------------------------
# Relevant code
# ------------------------------------------------------------------------------

gym_env = TaxiEnv()
env = GymDiscreteEnvWrapper(gym_env)
_, _, Rmat = env.compute_matrix_form()
# r_max = max(1, np.asscalar(np.max(Rmat)))
r_max = 1  # should always be equal to 1 if we rescale


# fps = 2 #gym_env.metadata.get('video.frames_per_second') or 100
# # for i in range(100):
# #     print('-'*70)
# #     s = gym_env.reset()
# #     gym_env.render(mode='human')
# #     while True:
# #         a = env.optimal_policy_indices[s]
# #         s, r, done, _ = gym_env.step(a)
# #         gym_env.render()
# #         time.sleep(1.0 / fps)
# #         if done:
# #             break
# env.reset()
# s = env.state
# gym_env.render(mode='human')
# for i in range(100):
#     a = env.optimal_policy_indices[s]
#     env.execute(a)
#     print("{} {:.3f}".format(a, env.reward))
#     gym_env.render()
#     time.sleep(1.0 / fps)
#     s = env.state
# exit(10)

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
seed_sequence = [random.randint(0, 2**30) for _ in range(in_options.nb_simulations)]

config['seed_sequence'] = seed_sequence

with open(os.path.join(folder_results, 'settings{}.conf'.format(in_options.nb_sim_offset)),'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
start_sim = in_options.nb_sim_offset
end_sim = start_sim + in_options.nb_simulations
for rep in range(start_sim, end_sim):
    env.reset()
    env_desc = env.description()
    seed = seed_sequence[rep-start_sim]  # set seed
    np.random.seed(seed)
    random.seed(seed)
    print("rep: {}/{}".format(rep-start_sim, in_options.nb_simulations))

    name = "trace_{}".format(rep)
    ucrl_log = ucrl_logger.create_multilogger(logger_name=name,
                                              console=not in_options.quiet,
                                              filename=name,
                                              path=folder_results)
    ucrl_log.info("mdp desc: {}".format(env_desc))
    ofualg = None
    if in_options.algorithm == "UCRL":
        ofualg = Ucrl.UcrlMdp(
            env,
            r_max=r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=ucrl_log,
            bound_type_p=in_options.bound_type,
            bound_type_rew=in_options.bound_type,
            random_state=seed)  # learning algorithm
    elif in_options.algorithm == "SCAL":
        ucrl_log.info("Augmented Reward: {}".format(in_options.augmented_reward))
        ofualg = spalg.SCAL(
            environment=env,
            r_max=r_max,
            span_constraint=in_options.span_constraint,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=ucrl_log,
            bound_type_p=in_options.bound_type,
            bound_type_rew=in_options.bound_type,
            random_state=seed,
            operator_type=in_options.operator_type,
            augment_reward=in_options.augmented_reward
        )
    elif in_options.algorithm == "TUCRL":
        ofualg = TUCRL(
            environment=env,
            r_max=r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=ucrl_log,
            random_state=seed)

    ucrl_log.info("[id: {}] {}".format(in_options.id, type(ofualg).__name__))
    ucrl_log.info("seed: {}".format(seed))
    ucrl_log.info("Config: {}".format(config))

    alg_desc = ofualg.description()
    ucrl_log.info("alg desc: {}".format(alg_desc))

    pickle_name = 'ucrl_{}.pickle'.format(rep)
    try:
        h = ofualg.learn(in_options.duration, in_options.regret_time_steps)  # learn task
    except Ucrl.EVIException as valerr:
        ucrl_log.info("EVI-EXCEPTION -> error_code: {}".format(valerr.error_value))
        pickle_name = 'exception_model_{}.pickle'.format(rep)
    except:
        ucrl_log.info("EXCEPTION")
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
    plt.close('all')