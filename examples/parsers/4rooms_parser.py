import os
import json
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt

from pics_generator import plot_temporal_abstraction, plot_regret

orig_folder = "../../results/0607_morning/rooms_14_20170607_195104"
orig_folder = "../../results/0608_morning/rooms_14_20170608_105332"
orig_folder = "../../results/rooms_14_20170608_172519"
orig_folder = "../../results/rooms_14_20170609_175452"
domain = '4rooms'
dim = 14
configurations = ['c1']
algorithms = ["FSUCRLv2", "SUCRL_v2", "SUCRL_v3"]

for conf in configurations:
    plot_regret(orig_folder, domain, ["UCRL"] + algorithms, conf,
                output_filename="4rooms_{}_regret_{}".format(dim, conf),
                plot_every=1000, log_scale=True, generate_tex=False)
t = range(1, int(5e8), 10000)
sqrtt = np.sqrt(t)
plt.plot(t, sqrtt*80)
plt.show()
