import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
from optparse import OptionParser
from optparse import Option, OptionValueError
from UCRL.Ucrl import UcrlMdp, UcrlSmdpBounded
from UCRL.free_ucrl import FreeUCRL_Alg1

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
                  help="folder containing results", default="../res_20170516")
parser.add_option('-c', '--categories',
                  action="extend", type="string", default=["c1"],
                  dest='categories', metavar='CATEGORIES',
                  help='comma separated list of post categories')

(in_options, in_args) = parser.parse_args()
print("arguments: {}".format(in_args))
print("options: {}".format(in_options))

algorithms = ["smdp", "mdp", "freesmdp"]

eq_config = dict()
eq_config['mdp'] = {"c1":"c1", "c2":"c1",
                    "c3":"c3", "c4":"c3"}

for conf in in_options.categories:
    plt.figure()
    for alg in algorithms:
        if alg in eq_config.keys():
            conf = eq_config[alg][conf]
        folder = os.path.join(in_options.folder, "{}_{}".format(alg,conf))
        onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and ".pickle" in f]
        print(onlyfiles)
        regrets = []
        for f in onlyfiles:
            model = pickle.load(open(os.path.join(folder, f), "rb"))
            print(model)
            regrets.append(model.regret)
        regrets = np.array(regrets)
        regret_mean = np.mean(regrets, axis=0)
        regret_std = np.std(regrets, axis=0)
        plt.plot(regret_mean, label=alg)
    plt.legend()
    plt.show()





