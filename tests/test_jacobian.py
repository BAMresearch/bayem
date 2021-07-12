import unittest
import numpy as np
from bayes.inference_problem import  VariationalBayesProblem
from bayes.model_error import ModelErrorInterface
from bayes.noise import UncorrelatedSingleNoise

CHECK = np.testing.assert_array_almost_equal  # just to make it shorter


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
        CHECK(jac["out1"]["A"], me.xs)
        CHECK(jac["out1"]["B"], 2 * B * np.ones_like(me.xs))
        CHECK(jac["out2"]["A"], me.xs * 2 * A)
        CHECK(jac["out2"]["B"], me.xs)

    def test_vector_prm(self):
        me = DummyMEVectorPrm()
        x = np.asarray(me.parameter_list["X"])
        jac = me.jacobian()

        jac_correct = np.concatenate([np.diag(2 * x), np.diag(3 * x ** 2)])
        CHECK(jac["out"]["X"], jac_correct)

    def test_vector_prm2(self):
        me = DummyMEVectorPrm2()
        jac = me.jacobian()

        jac_correct = np.array([[1, 1]])
        CHECK(jac["out"]["X"], jac_correct)

    def test_partial_jacobian_definition(self):
        me = DummyMEPartial()
        A, B = me.parameter_list["A"], me.parameter_list["B"]
        jac = me.jacobian()

        CHECK(jac["out1"]["A"], me.xs)
        CHECK(jac["out1"]["B"], 2 * B * np.ones_like(me.xs))
        CHECK(jac["out2"]["A"], me.xs * 2 * A)
        CHECK(jac["out2"]["B"], me.xs)


class OddEvenME(ModelErrorInterface):
    def __init__(self):
        super().__init__()
        self.parameter_list.define("E_odd", 42.0)
        self.parameter_list.define("E_even", 4.0)
        self.parameter_list.define("E_all", 4.0)
        self.x_odd = np.r_[0, 2, 0, 2, 0, 2]
        self.x_even = np.r_[1, 0, 1, 0, 1, 0]
        self.x_all = np.r_[3, 3, 3, 3, 3, 3]

    def __call__(self):
        return {
            "out": self.x_odd * self.parameter_list["E_odd"]
            + self.x_even * self.parameter_list["E_even"]
            + self.x_all * self.parameter_list["E_all"]
        }


class TestJacobianJointGlobal(unittest.TestCase):
    def test_individual(self):
        me = OddEvenME()
        jac = me.jacobian()
        CHECK(jac["out"]["E_odd"], me.x_odd)
        CHECK(jac["out"]["E_even"], me.x_even)
        CHECK(jac["out"]["E_all"], me.x_all)

    def test_three_joints(self):
        me = OddEvenME()
        p = VariationalBayesProblem()
        p.add_model_error(me)
        p.latent["E"].add(me.parameter_list, "E_odd")
        p.latent["E"].add(me.parameter_list, "E_even")
        p.latent["E"].add(me.parameter_list, "E_all")

        with self.assertRaises(Exception):
            p.jacobian([42.0])  # we have not defined a noise model yet!

        noise_key = p.add_noise_model(UncorrelatedSingleNoise())

        J = p.jacobian([42.0])[noise_key]
        self.assertEqual(J.shape, (6, 1))
        CHECK(J[:, 0], me.x_odd + me.x_even + me.x_all)

    def test_two_joints(self):
        """
        Checks if the joint jacobian is caluclated correctly, if only
        two of the three parameters are defined latent.
        """
        me = OddEvenME()
        p = VariationalBayesProblem()
        p.add_model_error(me)
        p.latent["E"].add(me.parameter_list, "E_odd")
        p.latent["E"].add(me.parameter_list, "E_even")
        noise_key = p.add_noise_model(UncorrelatedSingleNoise())

        J = p.jacobian([42.0])[noise_key]
        CHECK(J[:, 0], me.x_odd + me.x_even)


if __name__ == "__main__":
    unittest.main()
