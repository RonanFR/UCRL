import copy
import os
import pickle
import random
import time
import datetime
import shutil
import json
import numpy as np
import rlexplorer.Ucrl as Ucrl
import rlexplorer.rllogging as ucrl_logger

import matplotlib
# 'DISPLAY' will be something like this ':0'
# on your local machine, and None otherwise
# if os.environ.get('DISPLAY') is None:
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rep = 0
path = 'folder'
name = 'run_{}.pickle'.format(rep)
with open(os.path.join(path, name), 'rb') as f:
    alg = pickle.load(f)

name = "restart_trace_{}".format(rep)
ucrl_log = ucrl_logger.create_multilogger(logger_name=name,
                                          console=True,
                                          filename=name,
                                          path=path)
alg.reset_after_pickle(logger=ucrl_log)

pickle_name = 'restart_ucrl_{}.pickle'.format(rep)
try:
    h = alg.learn(duration=20000000, regret_time_step=1000)  # learn task
except Ucrl.EVIException as valerr:
    ucrl_log.info("EVI-EXCEPTION -> error_code: {}".format(valerr.error_value))
    pickle_name = 'exception_model_{}.pickle'.format(rep)
except:
    ucrl_log.info("EXCEPTION")
    pickle_name = 'exception_model_{}.pickle'.format(rep)

alg.clear_before_pickle()
with open(os.path.join(path, pickle_name), 'wb') as f:
    pickle.dump(alg, f)

plt.figure()
plt.plot(alg.span_values, '-')
plt.xlabel("Points")
plt.ylabel("Span")
plt.savefig(os.path.join(path, "restart_span_{}.png".format(rep)))
plt.figure()
plt.plot(alg.regret, '-')
plt.xlabel("Points")
plt.ylabel("Regret")
plt.savefig(os.path.join(path, "restart_regret_{}.png".format(rep)))
plt.show()
plt.close('all')