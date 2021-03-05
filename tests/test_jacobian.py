import unittest
import numpy as np
from bayes.inference_problem import ModelErrorInterface


class TestModelError(ModelErrorInterface):
    def __init__(self):
        super().__init__()
        self.parameter_list.define("A", 17.0)
        self.parameter_list.define("B", 42.0)
        self.xs = np.linspace(0.0, 1.0, 3)

    def __call__(self):
        A, B = self.parameter_list["A"], self.parameter_list["B"]
        return {"out1": self.xs * A + B ** 2, "out2": self.xs * A ** 2 + B * self.xs}


class TestModelErrorPartial(TestModelError):
    def jacobian():
        jac = super().jacobian()
        return jac


class TestJacobian(unittest.TestCase):
    def check_jac(self, me):
        A, B = me.parameter_list["A"], me.parameter_list["B"]
        jac = me.jacobian()
        check = np.testing.assert_array_almost_equal  # just to make it shorter
        check(jac["out1"]["A"], me.xs)
        check(jac["out1"]["B"], 2 * B * np.ones_like(me.xs))
        check(jac["out2"]["A"], me.xs * 2 * A)
        check(jac["out2"]["B"], me.xs)

    def test_scalar_prm(self):
        self.check_jac(TestModelError())


if __name__ == "__main__":
    unittest.main()
