import os
import json
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt

from pics_generator import plot_temporal_abstraction, plot_regret

orig_folder = "../../results/rooms_14_20170621_103609" # 14x14 alpha_p 0.2 alpha_mc 0.2
# # orig_folder = "../../results/rooms_6_20170623_151731" # 6x6 alpha_p 0.2
# orig_folder = "../../results/rooms_14_20170623_145622" # 14x14 alpha_p 0.02 alpha_mc 0.02
# orig_folder = "../../results/rooms_14_20170623_183949" # 14x14 alpha_p 0.02 alpha_mc 0.02
# orig_folder = "../../results/rooms_14_20170627_224944" # 14x14 0.8 Bernstein
domain = '4rooms'
dim = 14
configurations = ['c1']
algorithms = ["FSUCRLv1", "FSUCRLv2", "SUCRL_v2", "SUCRL_v3"]
algorithms = ["FSUCRLv2", "SUCRL_v2", "SUCRL_v3"]
algorithms = ["FSUCRLv2", "SUCRL_v3"]
algorithms = ["FSUCRLv2"]
algorithms = []

for conf in configurations:
    plot_regret(orig_folder, domain, ["UCRL"] + algorithms, conf,
                #output_filename="4rooms_{}_regret_{}".format(dim, conf),
                output_filename="4rooms_{}_avg_regret_BIG_08_CHE_1".format(dim),
                plot_every=240, log_scale=False,
                generate_tex=True,
                gtype="confidence", y_lim=[0,0.047])
# t = range(1, int(5e8), 10000)
# sqrtt = np.sqrt(t)
# plt.plot(t, sqrtt*80)
# plt.ylim([0, 14e6])
# plt.savefig("4rooms_{}_regret_{}_confidence.png".format(dim, configurations[0]))
plt.show()
