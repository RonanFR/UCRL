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
# orig_folder = "../../results/rooms_14_20170613_174609"
orig_folder = "../../results/rooms_14_20170621_103609" # 14x14 alpha_p 0.2
# orig_folder = "../../results/rooms_6_20170623_151731" # 6x6 alpha_p 0.2
domain = '4rooms'
dim = 14
configurations = ['c1']
algorithms = ["FSUCRLv1", "FSUCRLv2", "SUCRL_v2", "SUCRL_v3"]

for conf in configurations:
    plot_regret(orig_folder, domain, ["UCRL"] + algorithms, conf,
                output_filename="4rooms_{}_regret_{}".format(dim, conf),
                plot_every=10, log_scale=False, generate_tex=False,
                gtype="minmax")
# t = range(1, int(5e8), 10000)
# sqrtt = np.sqrt(t)
# plt.plot(t, sqrtt*80)
# plt.ylim([0, 14e6])
# plt.savefig("4rooms_{}_regret_{}_confidence.png".format(dim, configurations[0]))
plt.show()
