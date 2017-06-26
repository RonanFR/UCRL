import numpy as np
import pickle
import os
import json
import copy
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

graph_properties = {}
graph_properties["SUCRL_v1"] = {'marker': '*',
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'SUCRL_v1',
                                'linestyle': '-',
                                # 'c': 'green'
                            }
graph_properties["SUCRL"] = graph_properties["SUCRL_v1"]
graph_properties["SUCRL_v2"] = {'marker': "1",
                                'markersize': 12,
                                'markeredgewidth': 2,
                                'linewidth': 2.,
                                'label': 'SUCRL_v2',
                                'linestyle': '-',
                                # 'c': 'green'
                            }
graph_properties["SUCRL_subexp"] = graph_properties["SUCRL_v2"]
graph_properties["SUCRL_v3"] = {'marker': "d",
                                'markersize': 8,
                                'linewidth': 2.,
                                'label': 'SUCRL_v3',
                                'linestyle': '--',
                                # 'c': 'green'
                            }
graph_properties["SUCRL_subexp_tau"] = graph_properties["SUCRL_v2"]
graph_properties["SUCRL_v4"] = {'marker': "3",
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'SUCRL_v4',
                                'linestyle': '--',
                                # 'c': 'green'
                            }
graph_properties["SUCRL_v5"] = {'marker': "3",
                                'markersize': 10,
                                'linewidth': 2.,
                                'label': 'SUCRL_v5',
                                'linestyle': '--',
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
                            'c': 'k'
                           }

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

def load_mean_values(folder, attributes):
    onlyfiles = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and ".pickle" in f]
    print("{}: {} files".format(folder, len(onlyfiles)))
    data = {}
    for f in onlyfiles:
        model = pickle.load(open(os.path.join(folder, f), "rb"))
        for k in attributes:
            if k not in data:
                data[k] = []
            data[k].append(getattr(model, k))

    for k in attributes:
        metric = data[k]
        max_common_length = min(map(len, metric))
        m = np.array([x[0:max_common_length] for x in metric])
        m_mean = np.mean(m, axis=0)
        m_std = np.std(m, axis=0)
        m_max = np.max(m, axis=0)
        m_min = np.min(m, axis=0)
        data['{}_mean'.format(k)] = m_mean
        data['{}_std'.format(k)] = m_std
        data['{}_max'.format(k)] = m_max
        data['{}_min'.format(k)] = m_min
        data['{}_num'.format(k)] = m.shape[0]

    return data


def plot_temporal_abstraction(folder, domain, algorithms, configurations,
                              output_filename, use_ucrl=True, check_keys=[]):
    if use_ucrl:
        lfolder = os.path.join(folder, "{}_{}_{}".format("UCRL", domain, "c1"))
        with open(os.path.join(lfolder, "settings0.conf"), "r") as f:
            mdp_settings = json.load(f)
        mv = load_mean_values(lfolder, ["regret"])
        mdp_regret = mv['regret_mean'][-1]
    else:
        mdp_regret = 1.

    data = {}

    for alg in algorithms:
        data[alg] = {"x": [], "y": []}

    for conf in configurations:
        for i, alg in enumerate(algorithms):
            lfolder = os.path.join(folder,
                                   "{}_{}_{}".format(alg, domain, conf))
            with open(os.path.join(lfolder, "settings0.conf"), "r") as f:
                settings = json.load(f)
                if i == 0:
                    settings0 = settings

            for key in check_keys:
                assert settings[key] == settings0[key]
                if use_ucrl and key != "t_max":
                        assert mdp_settings[key] == settings[key]

            mv = load_mean_values(lfolder, ["regret"])
            mean_regret = mv['regret_mean'][-1]

            data[alg]['y'].append(mean_regret / mdp_regret)
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
    plt.plot([1, len(configurations)], [1, 1], 'k', linewidth=2, label="UCRL")
    for k in algorithms:
        el = data[k]
        plt.plot(el['x'], el['y'], **graph_properties[k])

    plt.xlim([1, len(configurations)])

    plt.ylabel("Ratio of regrets $\mathcal{R}$")
    plt.xlabel("Maximal duration of options $T_{\max}$")
    plt.legend()
    tikz_save("{}.tex".format(output_filename))
    plt.savefig("{}.png".format(output_filename))

    plt.show()
    plt.close()


def plot_regret(folder, domain, algorithms, configuration,
                output_filename, plot_every=1, log_scale=False,
                generate_tex=False, gtype="minmax"):
    assert gtype in ["minmax", "confidence"]
    data = {}
    for a in algorithms:
        data[a] = {}
    for alg in algorithms:
        if alg == "UCRL":
            conf = "c1"
        else:
            conf = configuration
        lfolder = os.path.join(folder,
                           "{}_{}_{}".format(alg, domain, conf))
        mv = load_mean_values(folder=lfolder,
                              attributes=["regret", "regret_unit_time"])
        # data[alg]["regret"] = mv["regret_mean"]
        #data[alg]["regret_error"] = mv["regret_std"]
        # data[alg]["regret_max"] = mv["regret_max"]
        # data[alg]["regret_min"] = mv["regret_min"]
        # data[alg]["regret_unit_time"] = mv["regret_unit_time_mean"]
        data[alg] = mv

        with open(os.path.join(lfolder, "settings0.conf"), "r") as f:
            settings = json.load(f)
        title = "{}-{}: $\\alpha_p$={}, $\\alpha_{{mc}}$={}, $\\alpha_r={}$, $\\alpha_{{tau}}={}$, $r_{{max}}={}$".format(
            domain, settings['dimension'],
            settings['alpha_p'], settings['alpha_mc'],
            settings['alpha_r'], settings['alpha_tau'],
            settings['r_max']
        )

    plt.figure()
    xmin = np.inf
    xmax = -np.inf
    for k in algorithms:
        el = data[k]
        prop = copy.deepcopy(graph_properties[k])
        del prop['marker']
        del prop['markersize']
        # if k != "UCRL" and domain == 'navgrid':
        #     assert el['t_max'] == tmax, '{}: {} {}'.format(k, tmax, el['t_max'])
        t = list(range(0, len(el['regret_unit_time_mean']), plot_every)) + [len(el['regret_unit_time_mean'])-1]
        if log_scale:
            ax1 = plt.loglog(el['regret_unit_time_mean'][t], el['regret_mean'][t], **prop)
        else:
            # ax1 = plt.plot(el['regret_unit_time'][t], np.gradient(el['regret'][t]- 0.0009 * el['regret_unit_time'][t], 1000*plot_every), **prop)
            # ax1 = plt.plot(el['regret_unit_time'][t],
            #     el['regret'][t] - 0.0009 * el['regret_unit_time'][t], **prop)
            ax1 = plt.plot(el['regret_unit_time_mean'][t], el['regret_mean'][t], **prop)
            ax1_col = ax1[0].get_color()
            if gtype == "minmax":
                plt.fill_between(el['regret_unit_time_mean'][t],
                              el['regret_min'][t] , el['regret_max'][t], facecolor=ax1_col, alpha=0.4)
                # if k == "FSUCRLv2":
                #     for v in el['regret']:
                #         plt.plot(el['regret_unit_time_mean'][t], np.array(v)[t], '--', c=ax1_col, alpha=0.45)
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
    plt.title("{} / {}".format(configuration, title))
    plt.legend()
    plt.xlim([xmin, xmax])

    # save figures
    if generate_tex:
        tikz_save('{}.tex'.format(output_filename))
    plt.savefig('{}.png'.format(output_filename))

    #plt.show()
