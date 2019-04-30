import copy
import os
import pickle
import random
import math
import datetime
import shutil
import json
import numpy as np
from rlexplorer.envs.toys import Toy3D_1, Toy3D_2, RiverSwim, Garnet
from gym.envs.toy_text.taxi import TaxiEnv
import rlexplorer.envs.wrappers as gymwrap
import rlexplorer.Ucrl as Ucrl
import rlexplorer.tucrl as tucrl
import rlexplorer.scal as scal
import rlexplorer.posteriorsampling as psalgs
import rlexplorer.forcedxpl as forced
# from rlexplorer.olp import OLP
import rlexplorer.rllogging as ucrl_logger
import matplotlib
from rlexplorer.ofucontinuous import SCCALPlus
import sys

# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
# if os.environ.get('DISPLAY') is None:
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#####################################################################
# PARAMETERS

from collections import namedtuple

fields = ("nb_sim_offset", "nb_simulations", "id", "path", "algorithm", "domain", "seed_0", "alpha_r",
          "alpha_p", "posterior", "use_true_reward", "duration", "regret_time_steps", "quiet", "bound_type",
          "span_constraint", "augmented_reward", "communicating_version")
Options = namedtuple("Options", fields)
dfields = ("mdp_delta", "stochastic_reward", "uniform_reward", "unifrew_range",
           "garnet_ns", "garnet_na", "garnet_gamma")
DomainOptions = namedtuple("DomainOptions", dfields)

id_v = None
alg_name = "SCALPLUS"
if len(sys.argv) > 1:
    alg_name = sys.argv[1]

in_options = Options(
    nb_sim_offset=0,
    nb_simulations=10,
    id='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now()) if id_v is None else id_v,
    path=None,
    algorithm=alg_name,  # ["UCRL", "TUCRL", "TSDE", "DSPSRL", "BKIA", "SCAL", "SCALPLUS", "SCCALPLUS"]
    domain="T3D1",  # ["RiverSwim", "T3D1", "T3D2", "Taxi", "MountainCar", "CartPole", "Garnet"]
    seed_0=158956,
    alpha_r=1,
    alpha_p=1,
    posterior="Bernoulli",  # ["Bernoulli", "Normal", None]
    use_true_reward=False,
    duration=500000,
    regret_time_steps=1000,
    quiet=False,
    bound_type="bernstein",  # ["hoeffding", "bernstein", "KL"] this works only for UCRL and BKIA
    span_constraint=5,
    augmented_reward=True,
    communicating_version=True,
)

domain_options = DomainOptions(
    mdp_delta=0.005,
    stochastic_reward=True,
    uniform_reward=True,
    unifrew_range=0.2,
    garnet_ns=10,
    garnet_gamma=3,
    garnet_na=5
)

#####################################################################

config = in_options._asdict()
domain_config = domain_options._asdict()

# ------------------------------------------------------------------------------
# Relevant code
# ------------------------------------------------------------------------------
env = None
Hl = Ha = None
if in_options.domain.upper() == "RIVERSWIM":
    env = RiverSwim()
    r_max = max(1, np.asscalar(np.max(env.R_mat)))
elif in_options.domain.upper() == "T3D1":
    env = Toy3D_1(delta=domain_options.mdp_delta,
                  stochastic_reward=domain_options.stochastic_reward,
                  uniform_reward=domain_options.uniform_reward,
                  uniform_range=domain_options.unifrew_range)
    r_max = max(1, np.asscalar(np.max(env.R_mat)))
elif in_options.domain.upper() == "T3D2":
    env = Toy3D_2(delta=domain_options.mdp_delta,
                  epsilon=domain_options.mdp_delta / 5.,
                  stochastic_reward=domain_options.stochastic_reward)
    r_max = max(1, np.asscalar(np.max(env.R_mat)))
elif in_options.domain.upper() == "GARNET":
    env = Garnet(Ns=domain_options.garnet_ns,
                 Nb=domain_options.garnet_gamma,
                 Na=domain_options.garnet_na)
    r_max = max(1, np.asscalar(np.max(env.R_mat)))
elif in_options.domain.upper() == "TAXI":
    gym_env = TaxiEnv()
    if in_options.communicating_version:
        env = gymwrap.GymTaxiCom(gym_env)
    else:
        env = gymwrap.GymDiscreteEnvWrapper(gym_env)
    r_max = 1
elif in_options.domain.upper() == "MOUNTAINCAR":
    Hl = 1
    Ha = 1
    N = SCCALPlus.nb_discrete_state(in_options.duration, 3, Hl, Ha)
    env = gymwrap.GymMountainCarWr(N)
    r_max = 1
elif in_options.domain.upper() == "CARTPOLE":
    Hl = 0.1
    Ha = 1
    N = SCCALPlus.nb_discrete_state(in_options.duration, 3, Hl, Ha)
    N = 6
    env = gymwrap.GymCartPoleWr(N)
    r_max = 1

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
    cc = config.copy()
    cc.update(domain_config)
    json.dump(cc, f, indent=4, sort_keys=True)

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
sum_regret = None
M2 = None
t_step = None

VERBOSITY = 2

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
                                                console=False, #not in_options.quiet,
                                                filename=name,
                                                path=folder_results)
    expalg_log.info("mdp desc: {}".format(env_desc))
    expalg_log.info("mdp prop: {}".format(env.properties()))
    ofualg = None
    if in_options.algorithm == "UCRL":
        ofualg = Ucrl.UcrlMdp(
            env,
            r_max=r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=VERBOSITY,
            logger=expalg_log,
            random_state=seed,
            bound_type_p=in_options.bound_type,
            bound_type_rew=in_options.bound_type,
            known_reward=in_options.use_true_reward)  # learning algorithm
    elif in_options.algorithm == "TUCRL":
        ofualg = tucrl.TUCRL(
            environment=env,
            r_max=r_max,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=VERBOSITY,
            logger=expalg_log,
            random_state=seed,
            known_reward=in_options.use_true_reward)
    elif in_options.algorithm == "TSDE":
        ofualg = psalgs.TSDE(environment=env,
                             r_max=r_max,
                             verbose=VERBOSITY,
                             logger=expalg_log,
                             random_state=seed,
                             posterior=in_options.posterior,
                             known_reward=in_options.use_true_reward)
    elif in_options.algorithm == "DSPSRL":
        ofualg = psalgs.DSPSRL(environment=env,
                               r_max=r_max,
                               verbose=VERBOSITY,
                               logger=expalg_log,
                               random_state=seed,
                               posterior=in_options.posterior,
                               known_reward=in_options.use_true_reward)
    elif in_options.algorithm == "BKIA":
        ofualg = forced.BKIA(environment=env,
                             r_max=r_max,
                             alpha_r=in_options.alpha_r,
                             alpha_p=in_options.alpha_p,
                             verbose=VERBOSITY,
                             logger=expalg_log,
                             bound_type_p=in_options.bound_type,
                             bound_type_rew=in_options.bound_type,
                             random_state=seed)
    elif in_options.algorithm == "SCAL":
        ofualg = scal.SCAL(
            environment=env,
            r_max=r_max,
            span_constraint=in_options.span_constraint,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=VERBOSITY,
            logger=expalg_log,
            bound_type_p=in_options.bound_type,
            bound_type_rew=in_options.bound_type,
            random_state=seed,
            augment_reward=in_options.augmented_reward,
            known_reward=in_options.use_true_reward
        )
    elif in_options.algorithm == "SCALPLUS":
        ofualg = scal.SCALPLUS(
            environment=env,
            r_max=r_max,
            span_constraint=in_options.span_constraint,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=VERBOSITY,
            logger=expalg_log,
            bound_type_p=in_options.bound_type,
            bound_type_rew=in_options.bound_type,
            random_state=seed,
            augment_reward=in_options.augmented_reward,
            known_reward=in_options.use_true_reward
        )
    elif in_options.algorithm == "SCCALPLUS":
        ofualg = SCCALPlus(
            discretized_env=env,
            r_max=r_max,
            holder_L=Hl,
            holder_alpha=Ha,
            span_constraint=in_options.span_constraint,
            alpha_r=in_options.alpha_r,
            alpha_p=in_options.alpha_p,
            verbose=VERBOSITY,
            logger=expalg_log,
            random_state=seed,
            augment_reward=in_options.augmented_reward,
            known_reward=in_options.use_true_reward
        )

    expalg_log.info("[id: {}] {}".format(in_options.id, type(ofualg).__name__))
    expalg_log.info("seed: {}".format(seed))
    expalg_log.info("Config: {}".format(config))

    alg_desc = ofualg.description()
    expalg_log.info("alg desc: {}".format(alg_desc))

    pickle_name = 'run_{}.pickle'.format(rep)
    #    try:
    h = ofualg.learn(in_options.duration, in_options.regret_time_steps)  # learn task
    # except Ucrl.EVIException as valerr:
    #     expalg_log.info("EVI-EXCEPTION -> error_code: {}".format(valerr.error_value))
    #     pickle_name = 'exception_model_{}.pickle'.format(rep)
    # except:
    #     expalg_log.info("EXCEPTION")
    #     pickle_name = 'exception_model_{}.pickle'.format(rep)

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

    print('policy: ')
    print(ofualg.policy)

    if rep == start_sim:
        sum_regret = np.copy(ofualg.regret)
        M2 = np.zeros_like(ofualg.regret)
        t_step = ofualg.regret_unit_time
    else:
        x_nm1 = sum_regret / (rep - start_sim)
        sum_regret += ofualg.regret
        x_n = sum_regret / (rep - start_sim + 1)
        M2 = M2 + (ofualg.regret - x_nm1) * (ofualg.regret - x_n)

plt.figure()
y = sum_regret / in_options.nb_simulations
sigma = np.sqrt(M2 / max(1, in_options.nb_simulations - 1)) / math.sqrt(in_options.nb_simulations)
ax1 = plt.plot(t_step, y, '-')
ax1_col = ax1[0].get_color()
plt.fill_between(t_step, y - 2 * sigma, y + 2 * sigma, facecolor=ax1_col, alpha=0.4)
plt.xlabel("Time")
plt.ylabel("Average Regret")
plt.savefig(os.path.join(folder_results, "avg_regret.png"))
plt.close('all')

# save data for reuse
data = np.zeros(len(t_step), dtype={'names': ('time', 'avg_regret', 'std_regret'),
                                    'formats': ('f8', 'f8', 'f8')})
data['time'] = t_step
data['avg_regret'] = y
data['std_regret'] = sigma
np.savetxt(os.path.join(folder_results, "avg_results.txt"), data)
