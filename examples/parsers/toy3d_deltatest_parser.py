import numpy as np
import os
from pics_generator import load_mean_values
import json
import UCRL.Ucrl as Ucrl
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

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


folder = '/home/matteo/Desktop/toy3d_delta_UCRL'

L = []
Z = []
i = 0
data = {}
for x in os.walk(folder):
    if x[0] !=folder:
        f = x[0]
        mv = load_mean_values(folder=f, attributes=["regret", "regret_unit_time", "span_values", "span_times"])

        if mv is not None:
            data[i] = mv
            data[i].update({'algorithm': ''})

            with open(os.path.join(f, "settings0.conf"), "r") as fil:
                settings = json.load(fil)

            data[i]['algorithm'] = '{}_{}'.format(settings['algorithm'], settings['mdp_delta'])

            if 1e-5< settings['mdp_delta'] <= 0.05:
                L.append([1./settings['mdp_delta'], mv['regret_mean'][-1]])
                ci = 1.96 * mv['regret_std'][-1] / np.sqrt(mv['regret_num'])
                Z.append([1./settings['mdp_delta'], ci])

print(L)
L.sort(key=lambda tup: tup[0])  # sorts in place
Z.sort(key=lambda tup: tup[0])  # sorts in place
print(L)
L = np.array(L)
Z = np.array(Z)
plt.figure()
# ax1 = plt.semilogx(L[:,0], L[:,1], 'o-')
ax1 = plt.loglog(L[:,0], L[:,1], 'o-')
ax1_col = ax1[0].get_color()
plt.fill_between(L[:,0],
                 L[:, 1]-Z[:,1], L[:,1]+Z[:,1], facecolor=ax1_col, alpha=0.4)
plt.xlabel('$1/\delta$')
plt.ylabel('Cumulative Regret $\Delta ({})$'.format(settings['duration']))
tikz_save('UCRL_REGRET_DELTA.tex')
plt.show()
