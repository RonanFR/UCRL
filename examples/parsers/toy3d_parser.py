import os
import json
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import matplotlib.colors as pltcolors
import UCRL.Ucrl as Ucrl
import UCRL.span_algorithms as spalg
from UCRL.stucrl import STUCRL
from UCRL.posteriorsampling import PS
from UCRL.olp import OLP
import seaborn as snb
import math
import UCRL.bounds as bounds
from pics_generator import load_mean_values, replace_equal
from cycler import cycler

graph_properties = {}
graph_properties["SCAL_T_2.0"] = {'marker': '*',
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'SCAL $c=2$',
                                'linestyle': ':',
                                # 'color': pltcolors.rgb2hex([0.580392156862745,0.403921568627451,0.741176470588235])
                                # 'color': 'C2'
                            }
graph_properties["SCAL_T_5.0"] = {'marker': "1",
                                'markersize': 12,
                                'markeredgewidth': 2,
                                'linewidth': 2.,
                                'label': 'SCAL $c=5$',
                                'linestyle': ':',
                                # 'color': pltcolors.rgb2hex([0.83921568627451,0.152941176470588,0.156862745098039])
                                # 'color': 'C3'
                            }
graph_properties["SCAL_T_10.0"] = {'marker': "d",
                                'markersize': 8,
                                'linewidth': 2.,
                                'label': 'SCAL $c=10$',
                                'linestyle': '-.',
                                # 'color': pltcolors.rgb2hex([0.172549019607843,0.627450980392157,0.172549019607843])
                                # 'color': 'C4'
                            }
graph_properties["SCAL_T_15.0"] = {'marker': "3",
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'SCAL $c=15$',
                                'linestyle': '--',
                                # 'color': pltcolors.rgb2hex([0.549019607843137,0.337254901960784,0.294117647058824])
                                # 'color': 'C5'
                            }
graph_properties["SCAL_T_20.0"] = {'marker': "3",
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'SCAL $c=20$',
                                'linestyle': '--',
                                # 'color': pltcolors.rgb2hex([0.890196078431372,0.466666666666667,0.76078431372549])
                                # 'color': 'C6'
                            }
graph_properties["SCAL_T_25.0"] = {'marker': '^',
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'SCAL $c=25$',
                                'linestyle': '-',
                                # 'color': pltcolors.rgb2hex([0.12156862745098,0.466666666666667,0.705882352941177])
                                # 'color': 'C7'
                                }
graph_properties["UCRL"] = {'marker': 'o',
                            'markersize': 10,
                            'linewidth': 2.,
                            'label': 'UCRL',
                            'linestyle': '-',
                            # 'color': 'orange'
                            # 'color': 'C0'
                           }
graph_properties["STUCRL"] = {'marker': '^',
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'STUCRL',
                                'linestyle': '-',
                                # 'color': pltcolors.rgb2hex([0.12156862745098,0.466666666666667,0.705882352941177])
                                # 'color': 'C7'
                                }
graph_properties["PS"] = {'marker': "3",
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'PS',
                                'linestyle': '--',
                                # 'color': pltcolors.rgb2hex([0.890196078431372,0.466666666666667,0.76078431372549])
                                # 'color': 'C6'
                            }
graph_properties["OLP"] = {'marker': "3",
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'OLP',
                                'linestyle': '--',
                                # 'color': pltcolors.rgb2hex([0.549019607843137,0.337254901960784,0.294117647058824])
                                # 'color': 'C5'
                            }

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class ExtendedUCRL(Ucrl.UcrlMdp):

    def solve_optimistic_model(self, curr_state=None):
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

class ExtendedSC_UCRL(spalg.SCAL):

    def solve_optimistic_model(self, curr_state=None):
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

class Extended_STUCRL(STUCRL):

    def solve_optimistic_model(self, curr_state=None):
        span_value = super(Extended_STUCRL, self).solve_optimistic_model(curr_state=curr_state)
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
        

# orig_folder = "/home/matteo/Desktop/toy3d_20171128_170346"
# orig_folder = "/home/matteo/Desktop/toy3d_0.0001_20171128_222356"
# orig_folder = "/home/matteo/Desktop/toy3d_0_20171206_023930"
# orig_folder = "/home/matteo/Desktop/toy3d_0.0001_20171206_180633"
orig_folder = "/home/matteo/Desktop/toy3d_0.005_20180129_170516" #bernstein
orig_folder = "/home/matteo/Desktop/toy3d_0._20180129_173355"
orig_folder = '/home/matteo/Desktop/KQ_0.01_20180204_193705'
orig_folder = '/home/matteo/Desktop/toy3d_0.005_20180220_184722'
# domain = 'KQ'
domain = 'Toy3D'
configurations = ['c1']
algorithms = ["STUCRL", "SCAL" , "PS", "OLP"]

output_filename = '{}_BERN_zero'.format(domain)
plot_actions = False
plot_h = False
plot_P = False

plot_every = 100
log_scale = False
gtype = 'confidence'
# gtype = None
y_lim = None
generate_tex = True

data = {}
L = []
i = 0
for x in os.walk(orig_folder):
    if '_{}_'.format(domain) in x[0] or '_{}_'.format(domain.lower()) in x[0]:
        L.append(x[0])
        data[i] = None
        i += 1
print(L)
L.sort(key=natural_keys)
print(L)

#L = [L[l] for l in range(len(L)-1)]


for i, f in enumerate(L):
    mv = load_mean_values(folder=f, attributes=["regret", "regret_unit_time", "span_values", "span_times"])

    if mv is not None:
        data[i] = mv
        data[i].update({'algorithm': ''})

        with open(os.path.join(f, "settings0.conf"), "r") as fil:
            settings = json.load(fil)
        title = "{} $\\delta={}$:\n $\\alpha_p={}$, $\\alpha_r={}$, bound={}".format(
            domain,
            settings['armor_collect_prob'] if domain == "KQ" else settings['mdp_delta'],
            settings['alpha_p'],
            settings['alpha_r'],
            settings['bound_type']
        )
        exp_H = settings['duration']

        if settings['algorithm'] == 'SCAL':
            data[i]['algorithm'] = '{}_{}_{}'.format(settings['algorithm'], settings['operator_type'], settings['span_constraint'])
        else:
            data[i]['algorithm'] = settings['algorithm']

        onlyfiles = [ff for ff in os.listdir(f) if
                     os.path.isfile(os.path.join(f, ff)) and ".pickle" in ff]

        for ff in onlyfiles:
            model = pickle.load(open(os.path.join(f, ff), "rb"))
            for k in ['policy_history', 'value_history', 'obs_history', 'P_history']:
                if hasattr(model, k):
                    if k not in data[i].keys():
                        data[i].update({k: []})
                    data[i][k].append(getattr(model, k))


n_lines = len(data.keys())
cm = plt.get_cmap('bone')
color_cycle = [cm(i) for i in np.linspace(0., .8, n_lines)]
# cm = plt.get_cmap('Dark2')
# cm = plt.get_cmap('YlOrRd')
# color_cycle = [cm(0.2+1-i) for i in np.linspace(0.2, 1., n_lines)]

plt.figure()
ax = plt.axes()
ax.set_prop_cycle(cycler('color', color_cycle))
xmin = np.inf
xmax = -np.inf
for k in data.keys():
    el = data[k]
    print(el['algorithm'])
    print(graph_properties.keys())
    prop = copy.deepcopy(graph_properties[el['algorithm']])
    del prop['marker']
    del prop['markersize']
    if el is not None:
        
        x, y, indices = replace_equal(el['regret_unit_time_mean'], el['regret_mean'])

        # filter = x<=350000
        # x = x[filter]
        # y = y[filter]
        
        t = list(range(0, len(x), plot_every)) + [len(x)-1]
        if log_scale:
            ax1 = plt.loglog(x[t], y[t], **prop)
        else:
            # ax1 = plt.plot(el['regret_unit_time'][t], np.gradient(el['regret'][t]- 0.0009 * el['regret_unit_time'][t], 1000*plot_every), **prop)
            # ax1 = plt.plot(el['regret_unit_time'][t],
            #     el['regret'][t] - 0.0009 * el['regret_unit_time'][t], **prop)
            ax1 = plt.plot(x[t], y[t], **prop)
            ax1_col = ax1[0].get_color()
            if gtype is not None:
                if gtype == "minmax":
                    plt.fill_between(x[t],
                                  el['regret_min'][indices][t] , el['regret_max'][indices][t], facecolor=ax1_col, alpha=0.4)
                    if k == "FSUCRLv2":
                        for v in el['regret']:
                            plt.plot(x[t], np.array(v)[t], '--', c=ax1_col, alpha=0.42)
                else:
                    ci = 1.96 * el['regret_std'][t] / np.sqrt(el['regret_num'])
                    plt.fill_between(x[t],
                                     y[t]-ci, y[t]+ci, facecolor=ax1_col, alpha=0.4)

        xmin = min(xmin, x[0])
        xmax = max(xmax, x[-1])
if not log_scale:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ylabel("Cumulative Regret $\Delta (T)$")
plt.xlabel("Duration $T$")
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
plot_every = 1
plt.figure()
ax = plt.axes()
ax.set_prop_cycle(cycler('color', color_cycle))
xmin = np.inf
xmax = -np.inf
gtype = None
for k in data.keys():
    el = data[k]
    prop = copy.deepcopy(graph_properties[el['algorithm']])
    #del prop['marker']
    #del prop['markersize']
    if el is not None:
        if el['span_times_mean'][-1] < el['regret_unit_time_mean'][-1]:
            el['span_times_mean'] = np.concatenate([el['span_times_mean'], [el['regret_unit_time_mean'][-1]]])
            el['span_values_mean'] = np.concatenate([el['span_values_mean'], [el['span_values_mean'][-1]]])
            el['span_values_min'] = np.concatenate([el['span_values_min'], [el['span_values_min'][-1]]])
            el['span_values_max'] = np.concatenate([el['span_values_max'], [el['span_values_max'][-1]]])
            el['span_values_std'] = np.concatenate([el['span_values_std'], [el['span_values_std'][-1]]])


        x, y, indices = replace_equal(el['span_times_mean'], el['span_values_mean'])
        
        t = list(range(0, len(x), plot_every)) + [len(x)-1]
        if log_scale:
            ax1 = plt.loglog(x[t], y[t], **prop)
        else:
            # ax1 = plt.plot(el['regret_unit_time'][t], np.gradient(el['regret'][t]- 0.0009 * el['regret_unit_time'][t], 1000*plot_every), **prop)
            # ax1 = plt.plot(el['regret_unit_time'][t],
            #     el['regret'][t] - 0.0009 * el['regret_unit_time'][t], **prop)
            ax1 = plt.step(np.log(x[t]), y[t], **prop)
            ax1_col = ax1[0].get_color()
            if gtype is not None:
                if gtype == "minmax":
                    plt.fill_between(x[t],
                                  el['span_values_min'][t] , el['span_values_max'][t], facecolor=ax1_col, alpha=0.4)
                    if k == "FSUCRLv2":
                        for v in el['regret']:
                            plt.plot(x[t], np.array(v)[t], '--', c=ax1_col, alpha=0.42)
                else:
                    ci = 1.96 * el['span_values_std'][t] / np.sqrt(el['span_values_num'])
                    plt.fill_between(x[t],
                                     y[t]-ci, y[t]+ci, facecolor=ax1_col, alpha=0.4)
        xmin = min(xmin, np.log(x[0]))
        xmax = max(xmax, np.log(x[-1]))
# if not log_scale:
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#     plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.draw()
plt.ylabel("span$(u_{e})$")
plt.xlabel("Log-Duration $\ln(T)$")
plt.title("{}".format(title))
plt.legend(loc=0)
plt.xlim([xmin, xmax])
if y_lim is not None:
    plt.ylim(y_lim)

# save figures
if generate_tex:
    tikz_save('SPAN_{}.tex'.format(output_filename))
plt.savefig('SPAN_{}.png'.format(output_filename))

if plot_h:
    for k in data.keys():
        el = data[k]
        if el is not None:
            plt.figure()
            H = el['value_history']
            X, Y = [], []
            for i in sorted(H[0].keys()):
                X.append(i)
                Y.append(H[0][i])
            Y = np.array(Y)
            print(X)
            plt.step(X, Y[:,1]-Y[:,0], 'o-', label='s_1')
            plt.step(X, Y[:,2]-Y[:,0], label='s_2')
            plt.title(el['algorithm'])
            plt.legend()
            H = el['value_history']

if plot_P:
    state_actions = [[0], [0], [0,1]]
    for k in data.keys():
        el = data[k]
        if el is not None:
            P_h = el['P_history'][0]
            Nsa = el['obs_history'][0]
            H = el['value_history'][0]
            iter = 24578 if 24578 in Nsa.keys() else 6992246
            nb_obs= Nsa[iter]
            tr = P_h[iter]
            tr_ci = bounds.chernoff(it=iter, N=nb_obs,
                                 range=1., delta=1 / math.sqrt(iter + 1),
                                 sqrt_C=3.5, log_C=2 * 3 * 2)
            h_v = np.reshape(H[iter] - H[iter][0], (-1,1))

            print(sorted(P_h.keys()))
            P = []
            N = []
            y_labels=[]
            ci = []
            for s in range(3):
                for a in state_actions[s]:
                    P += [tr[s,a]]
                    N += [nb_obs[s,a]]
                    y_labels += ['s_{}/a_{}'.format(s,a)]
                    ci += [tr_ci[s,a]]


            f, ax = plt.subplots(1, 4, figsize=(6, 3))
            snb.heatmap(P, xticklabels=['s_0', 's_1', 's_2'], yticklabels=y_labels,
                        annot=True, linewidths=.5, cmap='viridis', ax=ax[0])

            N =np.array(N).reshape(-1,1)
            snb.heatmap(N, annot=N, linewidths=.5, yticklabels=y_labels, cmap='viridis', ax=ax[1])
            CI_V = np.array(ci).reshape(-1, 1)
            snb.heatmap(CI_V, annot=CI_V, linewidths=.5, yticklabels=y_labels, cmap='viridis', ax=ax[2])
            snb.heatmap(h_v, annot=h_v, linewidths=.5, yticklabels=y_labels, cmap='viridis', ax=ax[3])
            plt.title(el['algorithm'])


for k in data.keys():
    el = data[k]

    if el    is not None:
        if 'policy_history' in el.keys():
            ph = el['policy_history']
            count_a1 = 0
            for exp1 in ph:
                times = sorted(exp1.keys())
                for i, t in enumerate(times):
                    prob, indx = exp1[t]
                    if el['algorithm'] == 'UCRL':
                        action = indx[2]
                    else:
                        sidx = indx[2]
                        sprob = prob[2]
                        action = sidx[sprob==1]

                    nextt = times[i + 1] if i < len(times) - 1 else exp_H
                    duration = nextt - t
                    if action == 1:
                        count_a1 += duration

            count_a1 /= len(el)*exp_H/100.
            print('{}: {}'.format(el['algorithm'], count_a1))




if plot_actions:
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
