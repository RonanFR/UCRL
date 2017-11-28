import os
import json
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import UCRL.Ucrl as Ucrl
import UCRL.span_algorithms as spalg

class ExtendedUCRL(Ucrl.UcrlMdp):

    def solve_optimistic_model(self):
        span_value = super(ExtendedUCRL, self).solve_optimistic_model()
        if not hasattr(self, 'policy_history'):
            self.policy_history = {}
        self.policy_history.update({self.episode: (self.policy, self.policy_indices)})
        return span_value

class ExtendedSC_UCRL(spalg.SCUCRLMdp):

    def solve_optimistic_model(self):
        span_value = super(ExtendedSC_UCRL, self).solve_optimistic_model()
        if not hasattr(self, 'policy_history'):
            self.policy_history = {}
        self.policy_history.update({self.episode: (self.policy, self.policy_indices)})
        return span_value

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
        data[i] = None
        i += 1
print(L)

plot_every = 1
log_scale = False
gtype = None
y_lim = None
generate_tex = False

for i, f in enumerate(L):
    mv = load_mean_values(folder=f, attributes=["regret", "regret_unit_time", "span_values", "span_times"])

    if mv is not None:
        data[i] = mv
        data[i].update({'algorithm': ''})

        with open(os.path.join(f, "settings0.conf"), "r") as fil:
            settings = json.load(fil)
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

        onlyfiles = [ff for ff in os.listdir(f) if
                     os.path.isfile(os.path.join(f, ff)) and ".pickle" in ff]

        for ff in onlyfiles:
            model = pickle.load(open(os.path.join(f, ff), "rb"))
            for k in ['policy_history']:
                if k not in data[i].keys():
                    data[i].update({k: []})
                data[i][k].append(getattr(model, k))


plt.figure()
xmin = np.inf
xmax = -np.inf
for k in data.keys():
    el = data[k]
    if el is not None:
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
    tikz_save('REGRET_{}.tex'.format(output_filename))
plt.savefig('REGRET_{}.png'.format(output_filename))


########################################################################
plt.figure()
xmin = np.inf
xmax = -np.inf
for k in data.keys():
    el = data[k]
    if el is not None:
        if el['span_times_mean'][-1] < el['regret_unit_time_mean'][-1]:
            el['span_times_mean'] = np.concatenate([el['span_times_mean'], [el['regret_unit_time_mean'][-1]]])
            el['span_values_mean'] = np.concatenate([el['span_values_mean'], [el['span_values_mean'][-1]]])

        t = list(range(0, len(el['span_times_mean']), plot_every)) + [len(el['span_times_mean'])-1]
        if log_scale:
            ax1 = plt.loglog(el['span_times_mean'][t], el['span_values_mean'][t])
        else:
            # ax1 = plt.plot(el['regret_unit_time'][t], np.gradient(el['regret'][t]- 0.0009 * el['regret_unit_time'][t], 1000*plot_every), **prop)
            # ax1 = plt.plot(el['regret_unit_time'][t],
            #     el['regret'][t] - 0.0009 * el['regret_unit_time'][t], **prop)
            ax1 = plt.step(el['span_times_mean'][t], el['span_values_mean'][t], label=el['algorithm'])
            ax1_col = ax1[0].get_color()
            if gtype is not None:
                if gtype == "minmax":
                    plt.fill_between(el['span_times_mean'][t],
                                  el['span_values_min'][t] , el['span_values_max'][t], facecolor=ax1_col, alpha=0.4)
                    if k == "FSUCRLv2":
                        for v in el['regret']:
                            plt.plot(el['span_times_mean'][t], np.array(v)[t], '--', c=ax1_col, alpha=0.42)
                else:
                    ci = 1.96 * el['span_values_std'][t] / np.sqrt(el['span_values_num'])
                    plt.fill_between(el['span_times_mean'][t],
                                     el['span_values_mean'][t]-ci, el['span_values_mean'][t]+ci, facecolor=ax1_col, alpha=0.4)
        xmin = min(xmin, el['span_times_mean'][0])
        xmax = max(xmax, el['span_times_mean'][-1])
# if not log_scale:
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#     plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.draw()
plt.ylabel("span$(u_{e})$")
plt.xlabel("Duration $T_{n}$")
plt.title("{}".format(title))
plt.legend(loc=2)
plt.xlim([xmin, xmax])
if y_lim is not None:
    plt.ylim(y_lim)

# save figures
if generate_tex:
    tikz_save('SPAN_{}.tex'.format(output_filename))
plt.savefig('SPAN_{}.png'.format(output_filename))


########################################################################
from collections import namedtuple

StateAction = namedtuple('StateAction', 'mina, maxa,times')

plt.figure()
xmin = np.inf
xmax = -np.inf
for k in data.keys():
    el = data[k]
    if el is not None:
        H = el['policy_history'][0]
        states_actions = {}
        ucrl_actions = {}
        ucrl_times = {}
        for s in range(3):
            states_actions[s] = StateAction(mina={0: [], 1: []},
                                            maxa={0: [], 1: []},
                                            times=[])
            ucrl_actions[s] = []
            ucrl_times[s] = []
        for v in H.keys():
            prob, indx = H[v]

            if el['algorithm'] == 'UCRL':

                for s in range(3):
                    sidx = indx[s]
                    ucrl_actions[s].append(sidx)
                    ucrl_times[s].append(v)
            else:
                for s in range(3):
                    sidx = indx[s]
                    sprob = prob[s]
                    try:
                        states_actions[s].mina[0].append(0)
                        states_actions[s].mina[1].append(0)
                        states_actions[s].mina[sidx[0]][-1] = sprob[0]
                        states_actions[s].maxa[0].append(0)
                        states_actions[s].maxa[1].append(0)
                        states_actions[s].maxa[sidx[1]][-1] = sprob[1]
                        states_actions[s].times.append(v)
                    except:
                        print(sidx)
                        print(sprob)

        if el['algorithm'] == 'UCRL':
            for s in range(3):
                if ucrl_times[s][-1] < el['regret_unit_time_mean'][-1]:
                    ucrl_times[s].append(el['regret_unit_time_mean'][-1])
                    ucrl_actions[s].append(ucrl_actions[s][-1])
        else:
            for s in range(3):
                if states_actions[s].times[-1] < el['regret_unit_time_mean'][-1]:
                    states_actions[s].mina[0].append(states_actions[s].mina[0][-1])
                    states_actions[s].mina[1].append(states_actions[s].mina[1][-1])
                    states_actions[s].maxa[0].append(states_actions[s].maxa[0][-1])
                    states_actions[s].maxa[1].append(states_actions[s].maxa[1][-1])
                    states_actions[s].times.append(el['regret_unit_time_mean'][-1])

        if el['algorithm'] == 'UCRL':
            plt.figure()
            for s in range(3):
                plt.step(ucrl_times[s], ucrl_actions[s], label='state_{}'.format(s))
            plt.title(el['algorithm'])
            plt.legend()
        else:

            for s in range(3):
                plt.figure()
                x = states_actions[s].times
                if np.any(states_actions[s].mina[0]):
                    plt.step(x, states_actions[s].mina[0], label='mina_0')
                if np.any(states_actions[s].mina[1]):
                    plt.step(x, states_actions[s].mina[1], label='mina_1')
                if np.any(states_actions[s].maxa[0]):
                    plt.step(x, states_actions[s].maxa[0], label='maxa_0')
                if np.any(states_actions[s].maxa[1]):
                    plt.step(x, states_actions[s].maxa[1], label='maxa_1')
                plt.title('{}: state_{}'.format(el['algorithm'], s))
                plt.legend()

plt.show()