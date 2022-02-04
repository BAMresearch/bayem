import bayem
import numpy as np
import pytest

x0 = bayem.MVN([1.], precision=[[1.]])
N_data = 100
_target_x = np.zeros(N_data)

def f1(x):
    buggy_exception = int('I should be a number.')
    return np.full(N_data, x[0]) - _target_x

def f2(x):
    if abs(x[0] - _target_x[0]) < 0.1:
        occasional_exception = int('I should be a number.')
    return np.full(N_data, x[0]) - _target_x

def test_catch_exceptions():
    with pytest.raises(ValueError):
        result = bayem.vba(f1, x0)
    
def test_skip_exceptions():
    # Due to a buggy exception (arising at very first iteration)
    result = bayem.vba(f1, x0, noise0=None, allowed_exceptions=(ValueError,))
    assert (result.success==False)
    # Due an occasional exception at some iteration
    result = bayem.vba(f2, x0, noise0=None, allowed_exceptions=(ValueError,))
    assert (result.success==False)
    assert (abs(result.means[0][0] - _target_x[0]) < 0.1)

if __name__ == "__main__":
    test_catch_exceptions()
    test_skip_exceptions()