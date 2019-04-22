import numpy as np
from gym.envs.classic_control import MountainCarEnv
from rlexplorer.utils.discretization import Discretizer
# from rlexplorer.evi.pyvi import value_iteration
from rlexplorer.evi.vi import value_iteration
from rlexplorer.utils.shortestpath import dpshortestpath

env = MountainCarEnv()

Nbins = 30
Nsamples = 1
minx, minv = env.observation_space.low.tolist()
maxx, maxv = env.observation_space.high.tolist()
x_bins = np.linspace(minx, maxx, Nbins)
y_bins = np.linspace(minv, maxv, Nbins)

print(x_bins)
print(y_bins)

grid = Discretizer(bins=[x_bins[1:-1], y_bins[1:-1]])

S, A = grid.n_bins(), env.action_space.n

P = np.zeros((S, A, S))
R = np.zeros((S, A))
Nas = np.zeros((S, A))

reach_graph = [[] for _ in range(S)]

for x in np.linspace(minx, maxx, 7*Nbins):
    for v in np.linspace(minv, maxv, 7*Nbins):
        sd = np.asscalar(grid.dpos([x,v]))
        for a in range(A):
            for n in range(Nsamples):
                env.state = np.array([x,v])
                nextstate, reward, done, _ = env.step(action=a)

                if done:
                    nextstate = env.reset()
                    reward = 0
                nsd = np.asscalar(grid.dpos(nextstate))

                if nsd not in reach_graph[sd]:
                    reach_graph[sd].append(nsd)

                reward += 1

                P[sd, a, nsd] += 1
                R[sd, a] += reward
                Nas[sd, a] += 1

P = P / Nas[:, :, np.newaxis]
R = R / Nas

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

state_actions = [list(range(A)) for _ in range(S)]

policy_indices = 99*np.ones((S,), dtype=np.int)
policy = 99*np.ones((S,), dtype=np.int)
gain, u2 = value_iteration(policy_indices,
                    policy,
                    S, state_actions,
                    P, R,
                    1e-10, 1)

print("SPAN: {}".format(np.max(u2)-np.min(u2)))
print("GAIN: {}".format(gain))


if False:
    Preach = P[Rs]
    Preach = Preach[:,:,Rs]
    print(Preach.shape)

    print("Computing Diameter ...")
    diameter = dpshortestpath(Preach, [list(range(A)) for _ in range(len(Rs))])
    print("diameter: {}".format(diameter))


trew = 0
observation = env.reset()
steps = 1000
for _ in range(steps):
    env.render()
    xd = grid.dpos(observation)
    action = np.asscalar(policy[xd])
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        reward = 0

    trew += reward

env.close()
print(trew/steps)


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
