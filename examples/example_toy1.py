import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
import numpy as np
from UCRL.envs.toys import Toy3D_1
import UCRL.Ucrl as Ucrl
import UCRL.span_algorithms as spalg
import UCRL.logging as ucrl_logger
from optparse import OptionParser, OptionGroup

import matplotlib
# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
# if os.environ.get('DISPLAY') is None:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


class ExtendedUCRL(Ucrl.UcrlMdp):

    def solve_optimistic_model(self):
        span_value = super(ExtendedUCRL, self).solve_optimistic_model()
        if not hasattr(self, 'policy_history'):
            self.policy_history = {}
            self.value_history = {}
            self.obs_history = {}
            self.P_history = {}
        self.policy_history.update({self.total_time: (copy.deepcopy(self.policy), copy.deepcopy(self.policy_indices))})
        u1, u2 = self.opt_solver.get_uvectors()
        self.value_history.update({self.total_time: u1.copy()})
        self.obs_history.update({self.total_time: self.nb_observations.copy()})
        self.P_history.update({self.total_time: self.P.copy()})
        return span_value

class ExtendedSC_UCRL(spalg.SCUCRLMdp):

    def solve_optimistic_model(self):
        span_value = super(ExtendedSC_UCRL, self).solve_optimistic_model()
        if not hasattr(self, 'policy_history'):
            self.policy_history = {}
            self.value_history = {}
            self.obs_history = {}
            self.P_history = {}
        self.policy_history.update({self.total_time: (copy.deepcopy(self.policy), copy.deepcopy(self.policy_indices))})
        u1, u2 = self.opt_solver.get_uvectors()
        self.value_history.update({self.total_time: u1.copy()})
        self.obs_history.update({self.total_time: self.nb_observations.copy()})
        self.P_history.update({self.total_time: self.P.copy()})
        return span_value

parser = OptionParser()
parser.add_option("-n", "--duration", dest="duration", type="int",
                  help="duration of the experiment", default=10000000)
parser.add_option("-b", "--boundtype", type="str", dest="bound_type",
                  help="Selects the bound type", default="chernoff")
parser.add_option("-c", "--span_constraint", type="float", dest="span_constraint",
                  help="Uppper bound to the bias span", default=10)
parser.add_option("--operatortype", type="str", dest="operator_type",
                  help="Select the operator to use for SC-EVI", default="T")
parser.add_option("--mdp_delta", type="float", dest="mdp_delta",
                  help="Transition probability mdp", default=0)
parser.add_option("--p_alpha", dest="alpha_p", type="float",
                  help="range of transition matrix", default=1.)
parser.add_option("--r_alpha", dest="alpha_r", type="float",
                  help="range of reward", default=1.)
parser.add_option("--regret_steps", dest="regret_time_steps", type="int",
                  help="regret time steps", default=1000)
parser.add_option("-r", "--repetitions", dest="nb_simulations", type="int",
                  help="Number of repetitions", default=1)
parser.add_option("--no_aug_rew", dest="augmented_reward", action="store_false", default="True")
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
|- SCUCRL                                                                                                                                            
"""
group1 = OptionGroup(parser, title='Algorithms', description=alg_desc)
group1.add_option("-a", "--alg", dest="algorithm", type="str",
                  help="Name of the algorith to execute"
                       "[UCRL, SCUCRL]",
                  default="SCUCRL")
parser.add_option_group(group1)

(in_options, in_args) = parser.parse_args()

if in_options.id and in_options.path:
    parser.error("options --id and --path are mutually exclusive")

assert in_options.algorithm in ["UCRL", "SCUCRL"]
assert in_options.nb_sim_offset >= 0
#assert 1e-16 <= in_options.mdp_delta <= 1.-1e-16
assert in_options.operator_type in ['T', 'N']

if in_options.id is None:
    in_options.id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())

config = vars(in_options)

# ------------------------------------------------------------------------------
# Relevant code
# ------------------------------------------------------------------------------

env = Toy3D_1(delta=in_options.mdp_delta)
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
        ofualg = ExtendedUCRL(
            env,
            r_max=r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=ucrl_log,
            bound_type=in_options.bound_type,
            random_state=seed)  # learning algorithm
    elif in_options.algorithm == "SCUCRL":
        ucrl_log.info("Augmented Reward: {}".format(in_options.augmented_reward))
        ofualg = ExtendedSC_UCRL(
            environment=env,
            r_max=r_max,
            span_constraint=in_options.span_constraint,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=1,
            logger=ucrl_log,
            bound_type=in_options.bound_type,
            random_state=seed,
            operator_type=in_options.operator_type,
            augment_reward=in_options.augmented_reward
        )

    ucrl_log.info("[id: {}] {}".format(in_options.id, type(ofualg).__name__))
    ucrl_log.info("seed: {}".format(seed))
    ucrl_log.info("Config: {}".format(config))

    alg_desc = ofualg.description()
    ucrl_log.info("alg desc: {}".format(alg_desc))

    h = ofualg.learn(in_options.duration, in_options.regret_time_steps)  # learn task
    ofualg.clear_before_pickle()

    with open(os.path.join(folder_results, 'ucrl_{}.pickle'.format(rep)), 'wb') as f:
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