import pytest

import bayem

x0 = bayem.MVN()


def f_value_error(x):
    return int("I should be a number.")


i_eval = 0


def f_delayed_value_error(x):
    global i_eval
    if i_eval < 6:
        i_eval += 1
        return x
    else:
        return f_value_error(x)


def test_catch_exceptions():
    with pytest.raises(ValueError):
        result = bayem.vba(f_value_error, x0)


def test_skip_exceptions():
    # Due to a buggy exception (arising at very first iteration)
    result = bayem.vba(f_value_error, x0, noise0=None, allowed_exceptions=ValueError)
    assert result.success == False

    # Due an occasional exception at some iteration
    result = bayem.vba(
        f_delayed_value_error, x0, noise0=None, allowed_exceptions=ValueError
    )
    print(result)
    assert result.success == False


if __name__ == "__main__":
    test_catch_exceptions()
    test_skip_exceptions()
