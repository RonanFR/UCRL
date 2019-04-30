from rlexplorer.evi import py_LPPROBA_bernstein, py_LPPROBA_hoeffding
import numpy as np
import pytest
import cvxpy as cp
import warnings


@pytest.mark.parametrize("p, beta, v, expected, reverse", [
    ([0.2, 0.7, 0.1], 0.1, [10., 4, 32.], [0.2, 0.65, 0.15], False),
    ([0.2, 0.7, 0.1], 2., [10., 4, 32.], [0., 0, 1.], False),
    ([0.2, 0.7, 0.1, 0.], 0.5, [10., 4, 32., 122], [0.2, 0.45, 0.1, 0.25], False),
    ([0.2, 0.7, 0.1, 0.], 1., [10., 4, 32., 122], [0.2, 0.2, 0.1, 0.5], False),
    ([0.2, 0.7, 0.1], 0.1, [10., 4, 32.], [0.2, 0.75, 0.05], True),
    ([0.2, 0.7, 0.1], 2., [10., 4, 32.], [0., 1, 0], True),
    ([0.2, 0.7, 0.1, 0.], 0.5, [10., 4, 32., 122], [0.05, 0.95, 0.0, 0], True),
    ([0.2, 0.7, 0.1, 0.], 0.1, [10., 4, 32., 122], [0.2, 0.75, 0.05, 0.], True),
])
def test_chebishev(p, beta, v, expected, reverse):
    p = np.array(p)
    v = np.array(v)
    sorted_idx = np.argsort(v, kind='mergesort').astype(np.int)

    c_p = py_LPPROBA_hoeffding(v=v, p=p, asc_sorted_idx=sorted_idx, beta=beta, reverse=reverse)
    assert np.allclose(c_p, np.dot(expected, v)), "{} - {}".format(c_p, np.dot(expected, v))


@pytest.mark.parametrize("p, beta, v, expected, reverse", [
    ([0.2, 0.7, 0.1], [0.1, 0.1, 0.1], [10., 4, 32.], [0.2, 0.6, 0.2], False),
    ([0.2, 0.7, 0.1, 0.], [0.5, 0.5, 0.5, 0.5], [10., 4, 32., 122], [0., 0.2, 0.3, 0.5], False),
    ([0.2, 0.7, 0.1], [0.1, 0.2, 0.1], [10., 4, 32.], [0.1, 0.9, 0.], True),
    ([0.2, 0.7, 0.1, 0.], [0.5, 0.5, 0.5, 0.5], [10., 4, 32., 122], [0., 1, 0., 0.], True),
    ([0.2, 0.7, 0.1, 0.], [0.1, 0.1, 0.5, 0.5], [10., 4, 32., 122], [0.2, 0.8, 0., 0.], True)
])
def test_bernstein(p, beta, v, expected, reverse):
    p = np.array(p)
    beta = np.array(beta)
    v = np.array(v)
    sorted_idx = np.argsort(v, kind='mergesort').astype(np.int)

    b_p = py_LPPROBA_bernstein(v=v, p=p, asc_sorted_idx=sorted_idx, beta=beta, bplus=0, reverse=reverse)
    assert np.allclose(b_p, np.dot(expected, v)), "{} - {}".format(b_p, np.dot(expected, v))


@pytest.mark.parametrize("n", range(2, 100, 2))
@pytest.mark.parametrize("count", range(10))
def test_random_vector_hoeffding(n, count):
    seed = np.random.randint(0, 41241)
    np.random.seed(seed)
    print(seed)
    # Define and solve the CVXPY problem.
    v = np.random.rand(n) * 100. - 50.
    beta = np.random.rand() * 2
    p = np.random.rand(n)
    p = p / np.sum(p)
    c = np.ones((n,))
    x = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(v.T @ x),
                      [cp.pnorm(x - p, p=1) <= beta, x >= 0, c.T @ x == 1])
    prob.solve(solver=cp.GLPK)

    sorted_idx = np.argsort(v, kind='mergesort').astype(np.int)
    dotp = py_LPPROBA_hoeffding(v=v, p=p, asc_sorted_idx=sorted_idx, beta=beta, reverse=False)

    print(x.value, np.sum(x.value))
    print("{}, {}".format(prob.value, dotp))
    assert np.isclose(prob.value, dotp), "{}, {}".format(prob.value, dotp)


@pytest.mark.parametrize("n", range(2, 100, 2))
@pytest.mark.parametrize("count", range(10))
def test_random_vector_bernstein(n, count):
    warnings.simplefilter("always")
    seed = np.random.randint(0, 41241)
    np.random.seed(seed)
    print(seed)
    # Define and solve the CVXPY problem.
    v = np.random.rand(n) * 100. - 50.
    beta = np.random.rand(n) * 2
    p = np.random.rand(n)
    p = p / np.sum(p)
    c = np.ones((n,))
    x = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(v.T @ x),
                      [x <= p + beta, x >= p - beta, x >= 0, x <= 1, c.T @ x == 1])
    prob.solve(solver=cp.GLPK)

    sorted_idx = np.argsort(v, kind='mergesort').astype(np.int)
    dotp = py_LPPROBA_bernstein(v=v, p=p, asc_sorted_idx=sorted_idx, beta=beta, bplus=0, reverse=False)

    # print(v)
    # print(p)
    # print(beta)
    # print(x.value, np.sum(x.value))
    # print("{}, {}".format(prob.value, dotp))
    assert np.isclose(prob.value, dotp), "{}, {}".format(prob.value, dotp)


# @pytest.mark.parametrize("p, beta, sorted_idx, expected, reverse", [
#     ([0.6, 0, 0.4], 0.1, [0, 2], [0.55,0.45], False),
#     ([0, 1, 0, 0.], 0.5, [1,3], [0.75, 0.25], False),
#     ([0.2, 0., 0.8, 0.], 1., [2,0], [0.3,0.7], False),
#     ([0, 0, 1, 0.], 2, [2,3], [0, 1], False),
#     ([0.3, 0.7, 0.], 0.1, [0,1], [0.35,0.65], True),
#     ([0.1, 0., 0., 0.9], 1., [3,0], [1, 0.], True),
# ])
# def test_chebishev_subvector(p, beta, sorted_idx, expected, reverse):
#     p = np.array(p)
#     sorted_idx = np.array(sorted_idx)
#
#     c_p = py_max_proba_chernoff_indices(p=p, sorted_idx=sorted_idx, beta=beta, reverse=reverse)
#     print(c_p)
#     assert np.isclose(np.sum(c_p[sorted_idx]), 1)
#     assert np.allclose(c_p[sorted_idx], expected)


if __name__ == '__main__':
    # test_chebishev([0.2, 0.7, 0.1], 2., [10., 4, 32.], [0., 1, 0], True)
    # test_bernstein([0.2, 0.7, 0.1], [0.1, 0.1, 0.1], [10., 4, 32.], [0.2, 0.6, 0.2], False)
    # test_chebishev_subvector([0, 0, 1, 0.], 2, [2,3], [0, 1], False)

    test_random_vector_bernstein(68, 0)
