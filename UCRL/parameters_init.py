import numpy as np
import math as m


def range_r_from_hoeffding(nb_states, nb_actions, nb_observations,
                           iteration=None, delta=1.0, desired_ci=1.):
    if iteration is None:
        D = 2*(np.sqrt(nb_states) - 1)
        iteration = nb_observations * D * nb_states * nb_actions
    ci = np.sqrt(7/2 * m.log(2 * nb_states * nb_actions *(iteration+1)/delta) / nb_observations)
    beta = desired_ci / ci
    return beta


def range_p_from_hoeffding(nb_states, nb_actions, nb_observations,
                           iteration=None, delta=1.0, desired_ci=2.):
    if iteration is None:
        D = 2*(np.sqrt(nb_states) - 1)
        iteration = nb_observations * D * nb_states * nb_actions
    ci = np.sqrt(14 * nb_states * m.log(2 * nb_actions * (iteration + 1) / delta) / nb_observations)
    beta = desired_ci / ci
    return beta