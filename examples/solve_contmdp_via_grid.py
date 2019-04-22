import numpy as np
from gym.envs.classic_control import MountainCarEnv, CartPoleEnv, AcrobotEnv
from rlexplorer.utils.discretization import Discretizer
from rlexplorer.evi.pyvi import value_iteration
from rlexplorer.evi.vi import value_iteration
from rlexplorer.utils.shortestpath import dpshortestpath
import datetime
import pickle
import os
import sys
from tqdm import tqdm

print("This is the name of the script: {}".format(sys.argv[0]))

if len(sys.argv) == 1:
    print("Running MountainCar test")
    env_name = "POLE"  # ["MC", "POLE", "ACROBOT"]
    path = None
else:
    env_name = None
    if sys.argv[1] in ["MC", "POLE", "ACROBOT"]:
        env_name = sys.argv[1]
    else:
        path = sys.argv[1]

if env_name is None:
    # loading file
    with open(path, 'rb') as f:
        data = pickle.load(f)

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

    state_actions = [list(range(A)) for _ in range(S)]

    policy_indices = 99*np.ones((S,), dtype=np.int)
    policy = 99*np.ones((S,), dtype=np.int)
    gain, u2 = value_iteration(policy_indices,
                        policy,
                        S, state_actions,
                        P, R,
                        1e-3, 0.9)

    print("SPAN: {}".format(np.max(u2)-np.min(u2)))
    print("GAIN: {}".format(gain))



else:
    if env_name == "POLE":
        env = CartPoleEnv()

        Nbins = 15
        Nsamples = 1

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
        Nsamples = 1

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

    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    Nas = np.zeros((S, A))

    reach_graph = [[] for _ in range(S)]

    for point in tqdm(samples):
        sd = np.asscalar(grid.dpos(point))
        for a in range(A):
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

                P[sd, a, nsd] += 1
                R[sd, a] += reward
                Nas[sd, a] += 1

    N = np.maximum(Nas, 1)

    P = P / N[:, :, np.newaxis]
    R = R / N

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

    assert np.allclose(np.sum(P, axis=-1), 1)

    print(P)
    print(R)

    id = '{:%Y%m%d_%H%M%S}_{}_{}'.format(datetime.datetime.now(), Nbins, Nsamples)
    pickle_name = "{}_{}.pickle".format(env_name, id)
    folder_results = './'
    with open(os.path.join(folder_results, pickle_name), 'wb') as f:

        Pidx = np.nonzero(P)
        Pnnz = P[Pidx]

        Ridx = np.nonzero(R)
        Rnnz = R[Ridx]

        pickle.dump({"Pnnz": Pnnz, "Rnnz": Rnnz, "Pidx": Pidx, "Ridx": Ridx, "S": S, "A": A, "env_name": env_name,
                     "bins": bins, 'reachable_states': Rs}, f)

    state_actions = [list(range(A)) for _ in range(S)]

    policy_indices = 99 * np.ones((S,), dtype=np.int)
    policy = 99 * np.ones((S,), dtype=np.int)
    gain, u2 = value_iteration(policy_indices,
                               policy,
                               S, state_actions,
                               P, R,
                               1e-4, 0.99)

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
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        if env_name in ["MC", "POLE"]:
            reward = 0
        nb_episodes += 1

        break

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
