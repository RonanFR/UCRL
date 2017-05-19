import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
from optparse import OptionParser
from optparse import Option, OptionValueError
from UCRL.Ucrl import UcrlMdp, UcrlSmdpBounded
from UCRL.free_ucrl import FSUCRLv1

class MultipleOption(Option):
    ACTIONS = Option.ACTIONS + ("extend",)
    STORE_ACTIONS = Option.STORE_ACTIONS + ("extend",)
    TYPED_ACTIONS = Option.TYPED_ACTIONS + ("extend",)
    ALWAYS_TYPED_ACTIONS = Option.ALWAYS_TYPED_ACTIONS + ("extend",)

    def take_action(self, action, dest, opt, value, values, parser):
        if action == "extend":
            values.ensure_value(dest, []).append(value)
        else:
            Option.take_action(self, action, dest, opt, value, values, parser)

PROG = os.path.basename(os.path.splitext(__file__)[0])
parser = OptionParser(option_class=MultipleOption,
                      usage='usage: %prog [OPTIONS]')
parser.add_option("-f", "--folder", dest="folder", type="str",
                  help="folder containing results", default="../../results/test_20170518")
parser.add_option('-c', '--categories',
                  action="extend", type="string", default=["c7", "c15"], #["c1", "c3", "c5", "c7", "c9", "c11", "c13", "c15"],
                  dest='categories', metavar='CATEGORIES',
                  help='comma separated list of post categories')

(in_options, in_args) = parser.parse_args()
print("arguments: {}".format(in_args))
print("options: {}".format(in_options))

algorithms = ["smdp", "fsucrlv1", "fsucrlv2", "mdp"]

algorithms = ["fsucrlv2"]
algorithms = ["smdp", "fsucrlv1", "fsucrlv2"]

namedict = {"smdp": "SUCRL",
            "fsucrlv1": "FSUCRLv1",
            "fsucrlv2": "FSUCRLv2",
            "mdp": "UCRL"}

#lineStyles = {
# '-': '_draw_solid',
# '--': '_draw_dashed',
# '-.': '_draw_dash_dot',
# ':': '_draw_dotted',
# 'None': '_draw_nothing', ' ': '_draw_nothing', '': '_draw_nothing'}
linest = {"smdp": "-",
            "fsucrlv1": "--",
            "fsucrlv2": "-.",
            "mdp": ":"}


import json
def plot_config(alg, conf, key):
    folder = os.path.join(in_options.folder, "{}_{}".format(alg, conf))
    with open(os.path.join(folder, "settings.conf"), "r") as f:
        settings = json.load(f)
    onlyfiles = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and ".pickle" in f]
    print(onlyfiles)
    metric = []
    for f in onlyfiles:
        model = pickle.load(open(os.path.join(folder, f), "rb"))
        # print(model)
        # print(len(model.regret))
        if key == "Regret":
            metric.append(model.regret)
        elif key == "Duration":
            metric.append(model.unit_duration[1:])
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
    label = "{}".format(namedict[alg])
    if 't_max' in settings.keys():
        label += " (t_\max = {})".format(settings['t_max'])
    plt.plot(t* 1000, regret_mean[t], linest[alg], label=label, linewidth=1.6)
    plt.text(t[max_common_length*2//3]*1000, regret_mean[max_common_length*2//3], label)
    plt.xlabel("Time")
    plt.ylabel(key)

eq_config = dict()
eq_config['mdp'] = {}
for i in range(16):
    if i%2 == 0:
        eq_config['mdp']["c{}".format(i)] = "c2"
    else:
        eq_config['mdp']["c{}".format(i)] = "c1"
print(eq_config["mdp"])

plt.figure()
#plot_config("mdp", "c1", "Duration")
for conf in in_options.categories:

    for alg in algorithms:
        if alg in eq_config.keys():
            if conf not in ["c1", "c2"]:
                continue
            conf = eq_config[alg][conf]
        plot_config(alg, conf, "Duration")
    plt.legend()


plt.figure()
plot_config("mdp", "c1", "Regret")
for conf in in_options.categories:

    for alg in algorithms:
        if alg in eq_config.keys():
            if conf not in ["c1", "c2"]:
                continue
            conf = eq_config[alg][conf]
        plot_config(alg, conf, "Regret")
    plt.legend()

plt.show()





