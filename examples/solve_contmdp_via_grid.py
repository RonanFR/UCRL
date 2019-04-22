import numpy as np
from gym.envs.classic_control import MountainCarEnv, CartPoleEnv, AcrobotEnv
from rlexplorer.utils.discretization import Discretizer
from rlexplorer.evi.pyvi import value_iteration
# from rlexplorer.evi.vi import value_iteration
from rlexplorer.utils.shortestpath import dpshortestpath
import datetime
import pickle
import os
import sys
from tqdm import tqdm
from scipy.sparse import csr_matrix

print("This is the name of the script: {}".format(sys.argv[0]))


if len(sys.argv) == 1:
    env_name = "MC"  # ["MC", "POLE", "ACROBOT"]
    print("Running {} test".format(env_name))
    path = None
else:
    env_name = None
    if sys.argv[1] in ["MC", "POLE", "ACROBOT"]:
        env_name = sys.argv[1]
    else:
        path = sys.argv[1]

if env_name is None:
    # loading file
    with open("{}.pickle".format(path), 'rb') as f:
        data = pickle.load(f)
    with open("{}_policy.pickle".format(path), 'rb') as f:
        data_policy = pickle.load(f)

    policy = data_policy['policy']
    policy_indices = data_policy['policy_indices']
    print(policy_indices)

    Pnnz = data['Pnnz']
    Pidx = data['Pidx']
    Rnnz = data['Rnnz']
    Ridx = data['Ridx']
    S = data['S']
    A = data['A']

    P = np.zeros((S, A, S))
    P[Pidx] = Pnnz
    R = np.zeros((S, A))
    R[Ridx] = Rnnz

    env_name = data['env_name']
    if env_name == "MC":
        env = MountainCarEnv()
    elif env_name == "POLE":
        env = CartPoleEnv()
    elif env_name == "ACROBOT":
        env = AcrobotEnv

    bins = data['bins']
    grid = Discretizer(bins=bins)

    # state_actions = [list(range(A)) for _ in range(S)]
    #
    #
    #
    # policy_indices = 99*np.ones((S,), dtype=np.int)
    # policy = 99*np.ones((S,), dtype=np.int)
    # gain, u2 = value_iteration(policy_indices,
    #                     policy,
    #                     S, state_actions,
    #                     P, R,
    #                     1e-3, 0.9)
    #
    # print("SPAN: {}".format(np.max(u2)-np.min(u2)))
    # print("GAIN: {}".format(gain))



else:
    if env_name == "POLE":
        env = CartPoleEnv()

        Nbins = 30
        Nsamples = 4

        HM = np.array([1.2 * env.x_threshold, 2., 1.2 * env.theta_threshold_radians, 3.])
        LM = -HM
        D = len(HM)

        bins = []
        points = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])

            V = np.linspace(LM[i], HM[i], 3 * Nbins)
            points.append(V)

        samples = np.meshgrid(*points)
        samples = np.concatenate([M.reshape(-1, 1) for M in samples], axis=1)

        grid = Discretizer(bins=bins)

        S, A = grid.n_bins(), env.action_space.n


    elif env_name == "MC":
        env = MountainCarEnv()

        Nbins = 30
        Nsamples = 4

        LM = env.observation_space.low
        HM = env.observation_space.high
        D = len(HM)

        bins = []
        points = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])

            V = np.linspace(LM[i], HM[i], 7 * Nbins)
            points.append(V)

        samples = np.meshgrid(*points)
        samples = np.concatenate([M.reshape(-1, 1) for M in samples], axis=1)

        grid = Discretizer(bins=bins)
        S, A = grid.n_bins(), env.action_space.n


    elif env_name == "ACROBOT":
        env = AcrobotEnv()

        Nbins = 6
        Nsamples = 1

        LM = env.observation_space.low
        HM = env.observation_space.high
        D = len(HM)

        bins = []
        points = []
        for i in range(D):
            V = np.linspace(LM[i], HM[i], Nbins)
            bins.append(V[1:-1])

            V = np.linspace(LM[i], HM[i], 3 * Nbins)
            points.append(V)

        samples = np.meshgrid(*points)
        samples = np.concatenate([M.reshape(-1, 1) for M in samples], axis=1)

        grid = Discretizer(bins=bins)
        S, A = grid.n_bins(), env.action_space.n

    # --------------
    # sparse P and R
    # --------------


    # P = np.zeros((S, A, S))
    # R = np.zeros((S, A))
    # Nas = np.zeros((S, A))

    # Psparse = [None] * S

    reach_graph = [[] for _ in range(S)]

    Psparse = {}
    Rsparse = {}
    Nsparse = {}
    for point in tqdm(samples):
        sd = np.asscalar(grid.dpos(point))
        for a in range(A):
            r_sa = 0
            for n in range(Nsamples):
                env.state = point.copy()
                nextstate, reward, done, _ = env.step(action=a)

                if done:
                    nextstate = env.reset()
                    if env_name in ["MC", "POLE"]:
                        reward = 0
                nsd = np.asscalar(grid.dpos(nextstate))

                if nsd not in reach_graph[sd]:
                    reach_graph[sd].append(nsd)

                if env_name in ["MC"]:
                    reward += 1

                keyp = "{}_{}_{}".format(sd, a, nsd)
                try:
                    Psparse[keyp] += 1
                except:
                    Psparse[keyp] = 1

                keysa = "{}_{}".format(sd, a)
                try:
                    Rsparse[keysa] += reward
                    Nsparse[keysa] += 1
                except:
                    Rsparse[keysa] = reward
                    Nsparse[keysa] = 1
                # P[sd, a, nsd] += 1
                # R[sd, a] += reward
                # Nas[sd, a] += 1

    r_data = []
    r_row = []
    r_col = []
    for k in Nsparse.keys():
        N = Nsparse[k]
        sa = k.split('_')
        r_row.append(sa[0])
        r_col.append(sa[1])
        r_data.append(Rsparse[k]/N)


    Pnz = [None] * S
    for s in range(S):
        Pnz[s] = [None] * A
        for a in range(A):
            Pnz[s][a] = {}
            Pnz[s][a] = {}
            Pnz[s][a] = {}

    for k in Psparse.keys():
        sas = k.split('_')
        s,a,sn = sas
        s = int(s)
        a = int(a)
        sn = int(sn)
        N = Nsparse['{}_{}'.format(s, a)]

        try:
            Pnz[s][a][sn] += Psparse[k] / N
        except:
            Pnz[s][a][sn] = Psparse[k] / N

    for s in range(S):
        for a in range(A):
            k = list(Pnz[s][a].keys())
            data = list(Pnz[s][a].values())
            Pnz[s][a] = csr_matrix((data, (np.zeros_like(k), k)), shape=(1,S))
            assert np.allclose(Pnz[s][a].sum(), 1)


    Rsparse = csr_matrix((r_data, (r_row, r_col)), shape=(S, A))

    # N = np.maximum(Nas, 1)
    # P = P / N[:, :, np.newaxis]
    # R = R / N

    initial_state = np.asscalar(grid.dpos(env.reset()))
    Rs = []
    queue = [initial_state]
    while len(queue) > 0:
        curr_state = queue.pop()
        Rs.append(curr_state)
        new_states = np.setdiff1d(reach_graph[curr_state], Rs)
        for a in new_states:
            if a not in queue:
                queue.append(a)
    Rs = np.unique(Rs)
    print("Reachable states: {}".format(Rs))
    print("reach {} / tot {}".format(len(Rs), grid.n_bins()))

    # assert np.allclose(np.sum(P, axis=-1), 1)

    # print(P)
    # print(R)

    id = '{:%Y%m%d_%H%M%S}_{}_{}'.format(datetime.datetime.now(), Nbins, Nsamples)
    pickle_name = "{}_{}.pickle".format(env_name, id)
    folder_results = './'
    with open(os.path.join(folder_results, pickle_name), 'wb') as f:

        # Pidx = np.nonzero(P)
        # Pnnz = P[Pidx]
        #
        # Ridx = np.nonzero(R)
        # Rnnz = R[Ridx]

        pickle.dump({"Pnnz": Pnz, "Rnnz": Rsparse,
                     # "Pidx": Pidx, "Ridx": Ridx,
                     "S": S, "A": A, "env_name": env_name,
                     "bins": bins, 'reachable_states': Rs}, f)

    state_actions = [list(range(A)) for _ in range(S)]

    print("solving value iteration")
    policy_indices = 99 * np.ones((S,), dtype=np.int)
    policy = 99 * np.ones((S,), dtype=np.int)
    gain, u2 = value_iteration(policy_indices,
                               policy,
                               S, state_actions,
                               Pnz, Rsparse,
                               # P, R,
                               1e-4, 0.99, verbose=1)

    print("SPAN: {}".format(np.max(u2) - np.min(u2)))
    print("GAIN: {}".format(gain))

    pickle_name = "{}_{}_policy.pickle".format(env_name, id)
    with open(os.path.join(folder_results, pickle_name), 'wb') as f:
        pickle.dump({"policy": policy, "policy_indices": policy_indices}, f)

    if False:
        Preach = P[Rs]
        Preach = Preach[:, :, Rs]
        print(Preach.shape)

        print("Computing Diameter ...")
        diameter = dpshortestpath(Preach, [list(range(A)) for _ in range(len(Rs))])
        print("diameter: {}".format(diameter))


# --------------------------------------------------------------------------------------------------------------
# Execute policy and visualize it
# --------------------------------------------------------------------------------------------------------------
trew = 0
observation = env.reset()
steps = 1000
nb_episodes = 0
for _ in range(steps):
    env.render()
    xd = grid.dpos(observation)
    action = np.asscalar(policy[xd])
    # print(action)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        if env_name in ["MC", "POLE"]:
            reward = 0
        nb_episodes += 1

    if env_name == "MC":
        reward += 1

    trew += reward

env.close()
print("avg_reward: {}".format(trew / steps))
print("#episodes: {}".format(nb_episodes))

# import pickle
#
#
# path = '/Users/pirotta/Documents/Projects/github/UCRL/examples/SCCALPLUS_GymMCWrapper2_20190422_154453/run_0.pickle'
# with open(path, 'rb') as f:
#     data = pickle.load(f)
#
# print(data.nb_observations)
# print(data.beta_r())
#
#
# trew = 0
# observation = env.reset()
# steps = 1000
# for _ in range(steps):
#     env.render()
#     xd = np.asscalar(data.environment.grid.dpos(observation))
#     # print(observation, xd)
#     action = data.sample_action(xd)[1]
#     observation, reward, done, info = env.step(action)
#
#     if done:
#         observation = env.reset()
#         reward = 0
#
#     trew += reward
#
# env.close()
# print(trew/steps)
