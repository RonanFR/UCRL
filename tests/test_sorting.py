import numpy as np
import pytest
from UCRL.evi._utils import py_quicksort_indices, py_sorted_indices
import timeit

@pytest.mark.parametrize("count", range(300))
def test_sorting(count):
    v = np.random.rand(40) * 50 - 25
    sorted_idx = np.argsort(v, kind='mergesort').astype(np.int)
    sorted_idx_c = py_quicksort_indices(v)
    sorted_idx_c2 = py_sorted_indices(v)
    assert np.allclose(sorted_idx, sorted_idx_c)
    assert np.allclose(sorted_idx, sorted_idx_c2)

if __name__ == '__main__':
    # code snippet to be executed only once
    mysetup = '''
import numpy as np
from UCRL.evi._utils import py_quicksort_indices, py_sorted_indices
    '''

    # code snippet whose execution time is to be measured
    mycode = '''
def example():
    v = np.rand(350)
    sorted_idx_c = py_quicksort_indices(v)
    '''

    mycode2 = '''
def example2():
    v = np.rand(350)
    sorted_idx_c = py_sorted_indices(v)
        '''
    print(timeit.timeit(setup=mysetup,stmt=mycode,number=10000))
    print(timeit.timeit(setup=mysetup,stmt=mycode2,number=10000))
    # v = np.array([1,54,612,2.1,5,61.])
    # sorted_idx = np.argsort(v, kind='mergesort').astype(np.int)
    # sorted_idx_c = py_quicksort_indices(v)
    # print(sorted_idx, sorted_idx_c)
    # assert np.allclose(sorted_idx, sorted_idx_c)
