import os
import json
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

from pics_generator import *

orig_folder = "../"
domain = 'Toy3D'
configurations = ['c1']
algorithms = ["SCURL"]

output_filename = '{}_CHEB_pure'.format(domain)

data = {}
L = []
i = 0
for x in os.walk(orig_folder):
    if '_{}_'.format(domain) in x[0]:
        L.append(x[0])
        data[i] = {}
        i += 1
print(L)

plot_every = 1
log_scale = False
gtype = None
y_lim = None
generate_tex = False

for i, f in enumerate(L):
    mv = load_mean_values(folder=f, attributes=["regret", "regret_unit_time"])

    data[i] = mv
    data[i].update({'algorithm': ''})

    with open(os.path.join(f, "settings0.conf"), "r") as f:
        settings = json.load(f)
    title = "{} $\\delta$={}:\n $\\alpha_p$={}, $\\alpha_r={}$, bound={}$".format(
        domain, settings['mdp_delta'],
        settings['alpha_p'],
        settings['alpha_r'],
        settings['bound_type']
    )

    if settings['algorithm'] == 'SCUCRL':
        data[i]['algorithm'] = '{}_{}_{}'.format(settings['algorithm'], settings['operator_type'], settings['span_constraint'])
    else:
        data[i]['algorithm'] = settings['algorithm']


plt.figure()
xmin = np.inf
xmax = -np.inf
for k in data.keys():
    el = data[k]
    t = list(range(0, len(el['regret_unit_time_mean']), plot_every)) + [len(el['regret_unit_time_mean'])-1]
    if log_scale:
        ax1 = plt.loglog(el['regret_unit_time_mean'][t], el['regret_mean'][t])
    else:
        # ax1 = plt.plot(el['regret_unit_time'][t], np.gradient(el['regret'][t]- 0.0009 * el['regret_unit_time'][t], 1000*plot_every), **prop)
        # ax1 = plt.plot(el['regret_unit_time'][t],
        #     el['regret'][t] - 0.0009 * el['regret_unit_time'][t], **prop)
        ax1 = plt.plot(el['regret_unit_time_mean'][t], el['regret_mean'][t], label=el['algorithm'])
        ax1_col = ax1[0].get_color()
        if gtype is not None:
            if gtype == "minmax":
                plt.fill_between(el['regret_unit_time_mean'][t],
                              el['regret_min'][t] , el['regret_max'][t], facecolor=ax1_col, alpha=0.4)
                if k == "FSUCRLv2":
                    for v in el['regret']:
                        plt.plot(el['regret_unit_time_mean'][t], np.array(v)[t], '--', c=ax1_col, alpha=0.42)
            else:
                ci = 1.96 * el['regret_std'][t] / np.sqrt(el['regret_num'])
                plt.fill_between(el['regret_unit_time_mean'][t],
                                 el['regret_mean'][t]-ci, el['regret_mean'][t]+ci, facecolor=ax1_col, alpha=0.4)
    xmin = min(xmin, el['regret_unit_time_mean'][0])
    xmax = max(xmax, el['regret_unit_time_mean'][-1])
if not log_scale:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ylabel("Cumulative Regret $\Delta (T_{n})$")
plt.xlabel("Duration $T_{n}$")
plt.title("{}".format(title))
plt.legend(loc=2)
plt.xlim([xmin, xmax])
if y_lim is not None:
    plt.ylim(y_lim)

# save figures
if generate_tex:
    tikz_save('{}.tex'.format(output_filename))
plt.savefig('{}.png'.format(output_filename))
