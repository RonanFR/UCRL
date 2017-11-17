import os
import json
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt

from pics_generator import plot_temporal_abstraction, plot_regret


orig_folder = '../../results/navgrid_20_20170612_145532/' # 20 alpha=0.4 rep=1
orig_folder = '../../results/navgrid_20_20170707_154847/' # 20 alpha=0.4 rep=20
orig_folder = '../../results/navgrid_20_20170607_183103/' # 20 alpha=0.02 rep=1
orig_folder = '../../results/navgrid_20_20170904_184919/' # 20 alpha=0.02 rep=20
orig_folder = '../../results/navgrid_20_20170927_152604/' # 20 alpha=0.02 rep=1

orig_folder = '/run/media/matteo/HDD2/navgrid_20_20170928_093755/'
domain = 'navgrid'
# domain = '4rooms'
dim = 20

configurations = ["c{}".format(i) for i in range(1, 13)]
# configurations = ["c{}".format(i) for i in range(1,16,2)]
configurations = ["c{}".format(i) for i in range(1, 19)]

algorithms = ["SUCRL", "SUCRL_subexp", "SUCRL_subexp_tau", "FSUCRLv1", "FSUCRLv2"]
algorithms = ["FSUCRLv1", "FSUCRLv2", "SUCRL_v1", "SUCRL_v2", "SUCRL_v3", "SUCRL_v4", "SUCRL_v5"]
# algorithms = ["FSUCRLv2", "SUCRL_v1", "SUCRL_v2", "SUCRL_v3"]

# orig_folder = "../../results/0607_morning/rooms_14_20170607_195104"
# domain = '4rooms'
# configurations = ['c1']
# algorithms = ["FSUCRLv2", "SUCRL_v2"]

chek_keys = ['alpha_r', 'alpha_p', 't_max', 'alpha_tau', 'r_max', 'alpha_mc']
plot_temporal_abstraction(folder=orig_folder,
                          domain=domain,
                          algorithms=algorithms,
                          configurations=configurations,
                          output_filename="navgrid_{}_temporal_abstraction".format(
                              dim),
                          use_ucrl=True,
                          check_keys=chek_keys)
plt.show()
for conf in configurations:
    plot_regret(orig_folder, domain, ["UCRL"] + algorithms, conf,
                output_filename="navgrid_{}_regret_{}".format(dim, conf),
                plot_every=1000, log_scale=False, generate_tex=False)


plt.show()
