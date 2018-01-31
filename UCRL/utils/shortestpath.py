import numpy as np
from ..evi import EVI
from tqdm import tqdm


def dpshortestpath(P, state_actions):
    Ns = len(state_actions)
    Na = max(map(len, state_actions))

    evi = EVI(nb_states=Ns,
              actions_per_state=state_actions,
              bound_type="chernoff",
              random_state=123456)
    policy_indices = np.ones(Ns, dtype=np.int)
    policy = np.ones(Ns, dtype=np.int)

    diameter = -1

    for target in tqdm(range(Ns)):
        # set reward
        R = -np.ones((Ns, Na))
        R[target] = 0
        # set self-loop
        Pshort = P.copy()
        for a in range(Na):
            Pshort[target, a] = 0.
            Pshort[target, a, target] = 1.

        evi.reset_u(np.zeros((Ns)))

        evi.run(policy_indices=policy_indices,
                policy=policy,
                estimated_probabilities=Pshort,
                estimated_rewards=R,
                estimated_holding_times=np.ones((Ns, Na)),
                beta_p=np.zeros((Ns, Na, 1)),
                beta_r=np.zeros((Ns, Na)),
                beta_tau=np.zeros((Ns, Na)),
                tau_max=1, tau_min=1, tau=1,
                r_max=1,
                epsilon=0.001,
                initial_recenter=1, relative_vi=0
                )

        u1, u2 = evi.get_uvectors()
        max_gain = 0.5 * (max(u2 - u1) + min(u2 - u1))

        v = -np.min(u1)
        source = np.argmin(u1)
        if diameter < v:
            print("{} -> {} : {}".format(source, target, v))
            diameter = v


    return diameter
