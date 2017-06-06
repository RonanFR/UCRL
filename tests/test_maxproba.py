import pytest
from UCRL.evi._max_proba import py_max_proba_chernoff, py_max_proba_bernstein
import numpy as np


@pytest.mark.parametrize("p, beta, v, expected", [
    ([0.2, 0.7, 0.1], 0.1, [10., 4, 32.], [0.2,0.65,0.15]),
    ([0.2, 0.7, 0.1, 0.], 0.5, [10., 4, 32., 122], [0.2, 0.45, 0.1, 0.25])
])
def test_chebishev(p, beta, v, expected):
    p = np.array(p)
    v = np.array(v)

    c_p = py_max_proba_chernoff(p=p, v=v, beta=beta)
    assert np.allclose(c_p, expected)


@pytest.mark.parametrize("p, beta, v, expected", [
    ([0.2, 0.7, 0.1], [0.1, 0.1, 0.1], [10., 4, 32.], [0.2,0.6,0.2]),
    ([0.2, 0.7, 0.1, 0.], [0.5, 0.5, 0.5, 0.5], [10., 4, 32., 122], [0., 0.2, 0.3, 0.5])
])
def test_bernstein(p, beta, v, expected):
    p = np.array(p)
    beta = np.array(beta)
    v = np.array(v)

    b_p = py_max_proba_bernstein(p=p, v=v, beta=beta)
    assert np.allclose(b_p, expected)