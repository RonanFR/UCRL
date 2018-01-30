import numpy as np
from UCRL.utils.shortestpath import dpshortestpath
from UCRL.envs.toys import Toy3D_1
import pytest

@pytest.mark.parametrize("delta", np.linspace(0.01, 0.99, 10))
def test_toy1d_dp(delta):
    env = Toy3D_1(delta=delta)

    diameter = dpshortestpath(env.P_mat, env.state_actions)
    print(diameter)
    assert np.isclose(diameter, max(1. / delta, 2./(1-delta))), max(1. / delta, 2./(1-delta))

if __name__ == '__main__':
    for el in np.linspace(0.01, 0.99, 10):
        test_toy1d_dp(el)
