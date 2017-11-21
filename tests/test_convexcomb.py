import numpy as np
import pytest


@pytest.mark.parametrize("count", [-1, -2, -3] + list(range(500)))
def test_1(count):
    a1, b1 = -10, 10
    x1 = np.random.random_sample(1) * (b1 - a1) + a1
    a2, b2 = 10, 30
    x2 = np.random.random_sample(1) * (b2 - a2) + a2

    T = np.random.random_sample(1) * (x2 - x1) + x1

    if count == -1:
        x1, x2, T = 1.1, 1.1, 1.1
    elif count == -2:
        x1, x2, T = 1.1, 1.3, 1.1
    elif count == -3:
        x1, x2, T = 0.1, 1.1, 1.1

    if np.isclose(x1, T):
        w1 = 1.
        w2 = 0.
    elif np.isclose(x2, T):
        w1 = 0
        w2 = 1
    else:
        w1 = (x2 - T) / (x2 - x1)
        w2 = (T - x1) / (x2 - x1)
    est = w1 * x1 + w2 * x2
    print(x1, x2, T)
    print(w1, w2)
    print(est)

    assert np.isclose(w1 + w2, 1.)
    assert 0 <= w1 <= 1.
    assert 0 <= w2 <= 1.
    assert np.isclose(T, est)


if __name__ == '__main__':
    test_1(-2)
