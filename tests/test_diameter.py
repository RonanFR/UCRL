import numpy as np
from UCRL.utils.shortestpath import dijkstra
from UCRL.envs.toys import Toy3D_1
import pytest


def test_simple_graph():
    state_actions = [[0, 1], [0, 1], [0], [0], [0], [0]]
    P = np.zeros((6, 2, 6))

    P[0, 0, 1] = 1 / 4
    P[0, 1, 2] = 1 / 2
    P[1, 0, 2] = 1 / 5
    P[1, 1, 3] = 1 / 10
    P[2, 0, 4] = 1 / 3
    P[3, 0, 5] = 1 / 11
    P[4, 0, 3] = 1 / 4
    P[5, 0, 5] = 1

    dist, par = dijkstra(P, state_actions, 0)
    print(dist)
    print(par)
    assert np.allclose(dist, [0, 4, 2, 9, 5, 20]), dist
    assert np.allclose(par, [np.nan, 0, 0, 4, 2, 3], equal_nan=True)


@pytest.mark.parametrize("delta", np.linspace(0.01, 0.99, 10))
def test_toy1d(delta):
    env = Toy3D_1(delta=delta)

    diameter = -1
    for s in range(3):
        dist, par = dijkstra(env.P_mat, env.state_actions, s)
        diameter = max(diameter, np.max(dist))
    print(diameter)
    assert np.isclose(diameter, max(1. / delta, 1+1./(1-delta))), max(1. / delta, 1+1./(1-delta))


if __name__ == '__main__':
    for el in np.linspace(0.01, 0.99, 10):
        test_toy1d(el)
