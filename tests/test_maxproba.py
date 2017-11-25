from UCRL.evi._max_proba import py_max_proba_chernoff, py_max_proba_bernstein, py_max_proba_chernoff_cin, py_max_proba_bernstein_cin
import numpy as np
import pytest


@pytest.mark.parametrize("p, beta, v, expected, reverse", [
    ([0.2, 0.7, 0.1], 0.1, [10., 4, 32.], [0.2,0.65,0.15], False),
    ([0.2, 0.7, 0.1], 2., [10., 4, 32.], [0.,0,1.], False),
    ([0.2, 0.7, 0.1, 0.], 0.5, [10., 4, 32., 122], [0.2, 0.45, 0.1, 0.25], False),
    ([0.2, 0.7, 0.1, 0.], 1., [10., 4, 32., 122], [0.2, 0.2, 0.1, 0.5], False),
    ([0.2, 0.7, 0.1], 0.1, [10., 4, 32.], [0.2,0.75,0.05], True),
    ([0.2, 0.7, 0.1], 2., [10., 4, 32.], [0.,1,0], True),
    ([0.2, 0.7, 0.1, 0.], 0.5, [10., 4, 32., 122], [0.05, 0.95, 0.0, 0], True),
    ([0.2, 0.7, 0.1, 0.], 0.1, [10., 4, 32., 122], [0.2, 0.75, 0.05, 0.], True),
])
def test_chebishev(p, beta, v, expected, reverse):
    p = np.array(p)
    v = np.array(v)

    c_p = py_max_proba_chernoff(p=p, v=v, beta=beta, reverse=reverse)
    assert np.isclose(np.sum(c_p), 1)
    assert np.allclose(c_p, expected)

    c_p = py_max_proba_chernoff_cin(p=p, v=v, beta=beta, reverse=reverse)
    assert np.isclose(np.sum(c_p), 1)
    assert np.allclose(c_p, expected)


@pytest.mark.parametrize("p, beta, v, expected, reverse", [
    ([0.2, 0.7, 0.1], [0.1, 0.1, 0.1], [10., 4, 32.], [0.2,0.6,0.2], False),
    ([0.2, 0.7, 0.1, 0.], [0.5, 0.5, 0.5, 0.5], [10., 4, 32., 122], [0., 0.2, 0.3, 0.5], False),
    ([0.2, 0.7, 0.1], [0.1, 0.2, 0.1], [10., 4, 32.], [0.1,0.9,0.], True),
    ([0.2, 0.7, 0.1, 0.], [0.5, 0.5, 0.5, 0.5], [10., 4, 32., 122], [0., 1, 0., 0.], True),
    ([0.2, 0.7, 0.1, 0.], [0.1, 0.1, 0.5, 0.5], [10., 4, 32., 122], [0.2, 0.8, 0., 0.], True)
])
def test_bernstein(p, beta, v, expected, reverse):
    p = np.array(p)
    beta = np.array(beta)
    v = np.array(v)

    b_p = py_max_proba_bernstein(p=p, v=v, beta=beta, reverse=reverse)
    assert np.isclose(np.sum(b_p), 1)
    assert np.allclose(b_p, expected)

    b_p = py_max_proba_bernstein_cin(p=p, v=v, beta=beta, reverse=reverse)
    assert np.isclose(np.sum(b_p), 1)
    assert np.allclose(b_p, expected)


if __name__ == '__main__':
    test_chebishev([0.2, 0.7, 0.1], 2., [10., 4, 32.], [0.,1,0], True)