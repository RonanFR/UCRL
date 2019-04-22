import numpy as np

def value_iteration(policy_indices,
                    policy,
                    nb_states, state_actions,
                    p, r,
                    epsilon, aper):
    u1 = np.zeros((nb_states,))
    u2 = np.zeros((nb_states,))
    nb_actions = max(map(len, state_actions))

    noise_factor = 0.1 * min(1e-6, epsilon)
    action_noise = np.random.random_sample(nb_actions) * noise_factor

    oldspan = np.inf
    while True:
        for s in range(nb_states):
            for a_idx, action in enumerate(state_actions[s]):
                v = r[s,a_idx] + aper * np.dot(p[s,a_idx], u1) + (1.-aper) * u1[s]
                if a_idx == 0:
                    u2[s] = v
                    policy[s] = action
                    policy_indices[s] = a_idx
                elif v + action_noise[a_idx] > u2[s] + action_noise[policy_indices[s]]:
                    u2[s] = v
                    policy[s] = action
                    policy_indices[s] = a_idx

        maxd = np.max(u2 - u1)
        mind = np.min(u2 - u1)
        # print(oldspan, maxd - mind)
        if (maxd - mind) <= epsilon:
            return 0.5*(maxd + mind), u2

        u1 = u2.copy()
        oldspan = maxd - mind
