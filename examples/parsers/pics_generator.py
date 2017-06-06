import os
import json
import pickle
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

orig_folder = "../../results/navgrid_20_20170602_162754"
# orig_folder = "../../results/navgrid_20_20170602_120209"
domain = 'navgrid'
orig_folder = "../../results/rooms_14_20170602_113507"
domain = '4rooms'

configurations = ["c{}".format(i) for i in range(1,16,2)]
configurations = ["c{}".format(i) for i in range(1,9)]


algorithms = ["SUCRL", "FSUCRLv1", "FSUCRLv2"]

graph_properties = {}
graph_properties["SUCRL"] = {'marker': '*',
                            'markersize': 10,
                                'linewidth': 2.,
                            'label': 'SUCRL',
                                'linestyle': '-',
                            # 'c': 'green'
                            }
graph_properties["FSUCRLv1"] = {'marker': '^',
                            'markersize': 10,
                                'linewidth': 2.,
                            'label': 'FSUCRLv1',
                                'linestyle': '--',
                                # 'c':'orange'
                                }
graph_properties["FSUCRLv2"] = {'marker': 'o',
                            'markersize': 10,
                                'linewidth': 2.,
                            'label': 'FSUCRLv2',
                                'linestyle': '-.',
                                # 'c':'b'
                                }
graph_properties["UCRL"] = {'marker': 'o',
                            'markersize': 10,
                                'linewidth': 2.,
                            'label': 'UCRL',
                                'linestyle': ':',
                           'c':'k'
                           }


lfolder = os.path.join(orig_folder, "{}_{}_{}".format("UCRL", domain, "c1"))
with open(os.path.join(lfolder, "settings.conf"), "r") as f:
    settings = json.load(f)
onlyfiles = [f for f in os.listdir(lfolder) if
             os.path.isfile(os.path.join(lfolder, f)) and ".pickle" in f]
regrets = []
for f in onlyfiles:
    model = pickle.load(open(os.path.join(lfolder, f), "rb"))
    regrets.append(model.regret[-1])

mdp_regret = np.mean(regrets)

data = {}

for alg in algorithms:
    data[alg] = {"x": [], "y": []}
for conf in configurations:
    for alg in algorithms:
        lfolder = os.path.join(orig_folder, "{}_{}_{}".format(alg, domain, conf))
        with open(os.path.join(lfolder, "settings.conf"), "r") as f:
            settings = json.load(f)
        onlyfiles = [f for f in os.listdir(lfolder) if
                     os.path.isfile(os.path.join(lfolder, f)) and ".pickle" in f]
        regrets = []
        for f in onlyfiles:
            model = pickle.load(open(os.path.join(lfolder, f), "rb"))
            regrets.append(model.regret[-1])

        data[alg]['y'].append(np.mean(regrets)/mdp_regret)
        data[alg]['x'].append(settings['t_max'])

        title = "{}-{}: $\\alpha_p$={}, $\\alpha_{{mc}}$={}, $\\alpha_r={}$, $\\alpha_{{tau}}={}$, $r_{{max}}={}$".format(
            domain, settings['dimension'],
            settings['alpha_p'], settings['alpha_mc'],
            settings['alpha_r'], settings['alpha_tau'],
            settings['r_max']
        )

print(data)
plt.figure()
plt.title(title)
plt.plot([1, 8], [1, 1], 'k', linewidth=2, label="UCRL")
for k in algorithms:
    el = data[k]
    plt.plot(el['x'], el['y'], **graph_properties[k])

plt.xlim([1,8])

plt.ylabel("Ratio of regrets $\mathcal{R}$")
plt.xlabel("Maximal duration of options $T_{\max}$")
plt.legend()
tikz_save('temporal_abstraction.tex')
plt.savefig('temporal_abstraction.png')

plt.show()
plt.close()


# ------------------------------------------------------------------------------
def plot_regret(alg, conf, data, key):
    folder = os.path.join(orig_folder, "{}_{}_{}".format(alg, domain, conf))
    with open(os.path.join(folder, "settings.conf"), "r") as f:
        settings = json.load(f)
    onlyfiles = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and ".pickle" in f]
    print(onlyfiles)
    metric = []
    for f in onlyfiles:
        model = pickle.load(open(os.path.join(folder, f), "rb"))
        if key == "Regret":
            metric.append(model.regret)
        elif key == "Duration":
            metric.append(model.unit_duration[1:])
        elif key == "Span":
            metric.append(model.span_values)
    max_common_length = min(map(len, metric))
    regrets = np.array([x[0:max_common_length] for x in metric])
    regret_mean = np.mean(regrets, axis=0)
    regret_std = np.std(regrets, axis=0)
    t = np.arange(max_common_length)
    # ci = 2. * regret_std / np.sqrt(regret_std.shape[0]) * 100
    # print(ci)
    # lbound = regret_mean-ci
    # ubound = regret_mean+ci
    # plt.fill_between(t, lbound[t], ubound[t], facecolor='yellow',
    #                 alpha=0.5,
    #                 label='1 sigma range')
    data[alg]['x'] = t * 1000
    data[alg]['y'] = regret_mean[t]
    if 't_max' in settings.keys():
        data[alg]['t_max'] = settings['t_max']
    return data
# ------------------------------------------------------------------------------
tmax = 5
conf = "c{}".format(1+(tmax-1)*2)
conf = "c{}".format(tmax)
for alg in ['UCRL'] + algorithms:
    data[alg] = {"x": [], "y": []}
key = 'Regret'
data = plot_regret("UCRL", "c1", data, key)
for alg in algorithms:
    data = plot_regret(alg,conf,data, key)

plt.figure()
l = np.inf
for k in ['UCRL'] + algorithms:
    el = data[k]
    l = min(len(el['x']), l)
t = range(0,l,100)
for k in ['UCRL'] + algorithms:
    el = data[k]
    prop = copy.deepcopy(graph_properties[k])
    del prop['marker']
    del prop['markersize']
    if k != "UCRL" and domain=='navgrid':
        assert el['t_max'] == tmax, '{}: {} {}'.format(k, tmax, el['t_max'])
    plt.plot(el['x'][t], el['y'][t], **prop)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel("Cumulative Regret $\Delta (T_{n})$")
plt.xlabel("Duration $T_{n}$ when $T_{\max}=%d$" % tmax)
plt.legend()
tikz_save('regret.tex')
plt.savefig('regret.png')

plt.show()
# # ------------------------------------------------------------------------------
# tmax = 5
# conf = "c{}".format(1+(tmax-1)*2)
# for alg in algorithms:
#     data[alg] = {"x": [], "y": []}
# key = 'Span'
# data = plot_regret("mdp", "c1", data, key)
# for alg in algorithms:
#     data = plot_regret(alg,conf,data, key)
#
# plt.figure()
# for k in algorithms:
#     el = data[k]
#     prop = copy.deepcopy(graph_properties[k])
#     del prop['marker']
#     del prop['markersize']
#     if 't_max' in el.keys():
#         assert el['t_max'] == tmax, '{}'.format(el['t_max'])
#     plt.plot(el['x'], el['y'], **prop)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.ylabel("Duration per decision step $T_n/n$")
# plt.xlabel("Duration $T_{n}$")
# plt.title(tmax)
# plt.legend()
# tikz_save('duration.tex')
# plt.show()
# plt.close()
# ------------------------------------------------------------------------------
# def plot_times(alg, conf, data, key):
#     folder = os.path.join(orig_folder, "{}_{}".format(alg, conf))
#     with open(os.path.join(folder, "settings.conf"), "r") as f:
#         settings = json.load(f)
#     onlyfiles = [f for f in os.listdir(folder) if
#                  os.path.isfile(os.path.join(folder, f)) and ".pickle" in f]
#     print(onlyfiles)
#     sol_times = []
#     epi_times = []
#     for f in onlyfiles:
#         model = pickle.load(open(os.path.join(folder, f), "rb"))
#         epi_times.append(model.simulation_times)
#         sol_times.append(list(map(sum,model.solver_times)))
#     max_common_length = min(map(len, sol_times))
#     sol_times = np.array([x[0:max_common_length] for x in sol_times])
#     epi_times = np.array([x[0:max_common_length] for x in epi_times])
#     sol_times_mean = np.mean(sol_times, axis=0)
#     epi_times_mean = np.mean(epi_times, axis=0)
#     t = np.arange(max_common_length)
#     data[alg]['x'] = t
#     data[alg]['y_sol'] = sol_times_mean[t]
#     data[alg]['y_epi'] = epi_times_mean[t]
#     if 't_max' in settings.keys():
#         data[alg]['t_max'] = settings['t_max']
#     return data
# # ------------------------------------------------------------------------------
# tmax = 5
# conf = "c{}".format(1+(tmax-1)*2)
# algorithms = ['fsucrlv2']
# for alg in algorithms:
#     data[alg] = {"x": [], "y_sol": [], "y_epi": []}
# key = 'Time'
# #data = plot_times("mdp", "c1", data, key)
# for alg in algorithms:
#     data = plot_times(alg,conf,data, key)
#
# plt.figure()
# l = np.inf
# for k in algorithms:
#     el = data[k]
#     l = min(len(el['x']), l)
# t = range(0,l)
# for k in  algorithms:
#     el = data[k]
#     prop = copy.deepcopy(graph_properties[k])
#     del prop['marker']
#     del prop['markersize']
#     if 't_max' in el.keys():
#         assert el['t_max'] == tmax, '{}'.format(el['t_max'])
#     name = prop['label']
#     prop['label'] = '{}_solution_time'.format(name)
#     plt.plot(el['x'][t], el['y_sol'][t], **prop)
#     prop['label'] = '{}_simulation_time'.format(name)
#     plt.plot(el['x'][t], el['y_epi'][t], **prop)
#     print(np.sum(el['y_epi']), np.sum(el['y_sol']))
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.ylabel("Times")
# plt.xlabel("Steps")
# plt.legend()
#
# plt.show()