import numpy as np
import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
from rlexplorer.Ucrl import UcrlMdp
import rlexplorer.envs.toys as rlenvs
import rlexplorer.logging as loginfo


seed = np.random.randint(1, 987988989)
seed = 970859866
np.random.seed(seed)
random.seed(seed)

print(seed)

env = rlenvs.RiverSwim()


env.reset()
env_desc = env.description()
r_max = 1.

id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
folder_results = os.path.abspath('{}_{}_{}'.format("UCRL", type(env).__name__, id))
if os.path.exists(folder_results):
    shutil.rmtree(folder_results)
os.makedirs(folder_results)

rep = 0
duration = 200000
regret_time_steps = 1
name = "trace_{}".format(rep)
expalg_log = loginfo.create_multilogger(logger_name=name,
                                        console=True,
                                        filename=name,
                                        path=folder_results)
expalg_log.info("seed: {}".format(seed))
expalg_log.info("mdp desc: {}".format(env_desc))

expalg = UcrlMdp(env,
                 r_max, random_state=seed,
                 alpha_r=0.1, alpha_p=0.1,
                 verbose=1,
                 logger=expalg_log)

alg_desc = expalg.description()
expalg_log.info("alg desc: {}".format(alg_desc))

pickle_name = 'ucrl_{}.pickle'.format(rep)
# try:
h = expalg.learn(duration, regret_time_steps)  # learn task
# except:
#     expalg_log.info("EXCEPTION")
#     pickle_name = 'exception_model_{}.pickle'.format(rep)

expalg.clear_before_pickle()
with open(os.path.join(folder_results, pickle_name), 'wb') as f:
    pickle.dump(expalg, f)

print(expalg.policy)
#
# from IPython import embed
# embed()
