import unittest
import numpy as np
from bayes.inference_problem import ModelErrorInterface


class DummyME(ModelErrorInterface):
    def __init__(self):
        super().__init__()
        self.parameter_list.define("A", 42.0)
        self.parameter_list.define("B", 0.0)
        self.xs = np.linspace(0.0, 1.0, 3)

    def __call__(self):
        A, B = self.parameter_list["A"], self.parameter_list["B"]
        return {"out1": self.xs * A + B ** 2, "out2": self.xs * A ** 2 + B * self.xs}


class DummyMEPartial(ModelErrorInterface):
    def __init__(self):
        super().__init__()
        self.parameter_list.define("A", 42.0)
        self.parameter_list.define("B", 0.0)
        self.xs = np.linspace(0.0, 1.0, 3)

    def __call__(self):
        A, B = self.parameter_list["A"], self.parameter_list["B"]
        return {"out1": self.xs * A + B ** 2, "out2": self.xs * A ** 2 + B * self.xs}

    def jacobian(self):
        """
        We can provide the derivative w.r.t. "A" analytically and use the
        central differences of the superclass for the parameter "B".
        """
        jac = super().jacobian(["B"])
        jac["out1"]["A"] = self.xs
        jac["out2"]["A"] = 2 * self.parameter_list["A"] * self.xs
        return jac


class DummyMEVectorPrm(ModelErrorInterface):
    def __init__(self):
        super().__init__()
        self.parameter_list.define("X", [1.0, 2.0, 3.0])

    def __call__(self):
        x = np.asarray(self.parameter_list["X"])
        return {"out": np.concatenate([x ** 2, x ** 3])}


class DummyMEVectorPrm2(ModelErrorInterface):
    def __init__(self):
        super().__init__()
        self.parameter_list.define("X", [0.0, 42.0])

    def __call__(self):
        return {"out": np.r_[6174 + sum(self.parameter_list["X"])]}


class TestJacobian(unittest.TestCase):
    def test_scalar_prm(self):
        me = DummyME()
        A, B = me.parameter_list["A"], me.parameter_list["B"]
        jac = me.jacobian()
        check = np.testing.assert_array_almost_equal  # just to make it shorter
        check(jac["out1"]["A"], me.xs)
        check(jac["out1"]["B"], 2 * B * np.ones_like(me.xs))
        check(jac["out2"]["A"], me.xs * 2 * A)
        check(jac["out2"]["B"], me.xs)

    def test_vector_prm(self):
        me = DummyMEVectorPrm()
        x = np.asarray(me.parameter_list["X"])
        jac = me.jacobian()

        jac_correct = np.concatenate([np.diag(2 * x), np.diag(3 * x ** 2)])
        np.testing.assert_array_almost_equal(jac["out"]["X"], jac_correct)

    def test_vector_prm2(self):
        me = DummyMEVectorPrm2()
        jac = me.jacobian()

        jac_correct = np.array([[1, 1]])
        np.testing.assert_array_almost_equal(jac["out"]["X"], jac_correct)

    def test_partial_jacobian_definition(self):
        me = DummyMEPartial()
        A, B = me.parameter_list["A"], me.parameter_list["B"]
        jac = me.jacobian()
        check = np.testing.assert_array_almost_equal  # just to make it shorter
        check(jac["out1"]["A"], me.xs)
        check(jac["out1"]["B"], 2 * B * np.ones_like(me.xs))
        check(jac["out2"]["A"], me.xs * 2 * A)
        check(jac["out2"]["B"], me.xs)

if __name__ == "__main__":
    unittest.main()
