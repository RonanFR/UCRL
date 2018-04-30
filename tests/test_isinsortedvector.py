import numpy as np
from UCRL.evi._utils import py_isinsortedvector
import pytest

@pytest.mark.parametrize("count", range(300))
def test_isinsortedvector(count):
    len =10
    v = np.random.random_integers(-1000, 1000, len)
    idxs = np.argsort(v)
    v = v[idxs]
    if np.random.rand() > 0.5:
        c = np.random.random_integers(0, len-1, 1)
        res = py_isinsortedvector(v[c], v)
        print("{} : {}".format(c,v))
        assert res >= 0
    else:
        c = v[0]
        while c in v:
            c = np.random.random_integers(-4000, 4000, 1)
        res = py_isinsortedvector(c, v)
        assert res == -1


if __name__ == '__main__':
    v = np.array([4,56,2,6,1])
    idxs = np.argsort(v)
    v = v[idxs]
    print(v)
    c = py_isinsortedvector(56, v)
    print(c)
    c = py_isinsortedvector(2, v)
    print(c)
    c = py_isinsortedvector(22, v)
    print(c)

