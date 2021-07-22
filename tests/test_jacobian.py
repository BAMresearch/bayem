import unittest
import numpy as np
from bayes.jacobian import jacobian_cdf
from bayes.inference_problem import *
from bayes.parameters import ParameterList
from bayes.noise import UncorrelatedSingleNoise

CHECK = np.testing.assert_array_almost_equal  # just to make it shorter


class DummyME:
    def __init__(self):
        self.xs = np.linspace(0.0, 1.0, 3)

    def __call__(self, prm):
        A, B = prm["A"], prm["B"]
        return {"out1": self.xs * A + B ** 2, "out2": self.xs * A ** 2 + B * self.xs}


class DummyMEPartial(ModelErrorInterface):
    def __init__(self):
        self.xs = np.linspace(0.0, 1.0, 3)

    def __call__(self, prm):
        A, B = prm["A"], prm["B"]
        return {"out1": self.xs * A + B ** 2, "out2": self.xs * A ** 2 + B * self.xs}

    def jacobian(self, prm):
        """
        We can provide the derivative w.r.t. "A" analytically and use the
        central differences of the superclass for the parameter "B".
        """
        jac = super().jacobian(prm, ["B"])
        jac["out1"]["A"] = self.xs
        jac["out2"]["A"] = 2 * prm["A"] * self.xs
        return jac


def dummy_me_vector_prm(prm):
    x = np.asarray(prm["X"])
    return {"out": np.concatenate([x ** 2, x ** 3])}


def dummy_me_vector_prm2(prm):
    return {"out": np.r_[6174 + sum(prm["X"])]}


class TestJacobian(unittest.TestCase):
    def test_scalar_prm(self):
        A, B = 42.0, 0.0

        me = DummyME()
        prm = ParameterList()
        prm.define("A", A)
        prm.define("B", B)
        jac = jacobian_cdf(me, prm)
        CHECK(jac["out1"]["A"], me.xs)
        CHECK(jac["out1"]["B"], 2 * B * np.ones_like(me.xs))
        CHECK(jac["out2"]["A"], me.xs * 2 * A)
        CHECK(jac["out2"]["B"], me.xs)

    def test_vector_prm(self):
        x = np.r_[1, 2, 3]
        prm = ParameterList()
        prm.define("X", x)
        jac = jacobian_cdf(dummy_me_vector_prm, prm)

        jac_correct = np.concatenate([np.diag(2 * x), np.diag(3 * x ** 2)])
        CHECK(jac["out"]["X"], jac_correct)

    def test_vector_prm2(self):
        prm = ParameterList()
        prm.define("X", [0.0, 42.0])
        jac = jacobian_cdf(dummy_me_vector_prm2, prm)

        jac_correct = np.array([[1, 1]])
        CHECK(jac["out"]["X"], jac_correct)

    def test_partial_jacobian_definition(self):
        A, B = 42.0, 0.0
        me = DummyMEPartial()
        prm = ParameterList()
        prm.define("A", A)
        prm.define("B", B)
        jac = me.jacobian(prm)

        CHECK(jac["out1"]["A"], me.xs)
        CHECK(jac["out1"]["B"], 2 * B * np.ones_like(me.xs))
        CHECK(jac["out2"]["A"], me.xs * 2 * A)
        CHECK(jac["out2"]["B"], me.xs)


class OddEvenME(ModelErrorInterface):
    def __init__(self):
        self.x_odd = np.r_[0, 2, 0, 2, 0, 2]
        self.x_even = np.r_[1, 0, 1, 0, 1, 0]
        self.x_all = np.r_[3, 3, 3, 3, 3, 3]

        self.prms = ParameterList()
        self.prms.define("E_odd", 42.0)
        self.prms.define("E_even", 4.0)
        self.prms.define("E_all", 4.0)

    def __call__(self, latent_prm):
        prm = self.prms.overwrite_with(latent_prm)
        return {
            "out": self.x_odd * prm["E_odd"]
            + self.x_even * prm["E_even"]
            + self.x_all * prm["E_all"]
        }


class TestJacobianJointGlobal(unittest.TestCase):
    def test_individual(self):
        me = OddEvenME()

        jac = me.jacobian(me.prms)
        CHECK(jac["out"]["E_odd"], me.x_odd)
        CHECK(jac["out"]["E_even"], me.x_even)
        CHECK(jac["out"]["E_all"], me.x_all)

    def test_three_joints(self):
        me = OddEvenME()
        p = InferenceProblem()
        me_key = p.add_model_error(me)
        p.latent["E"].add(me, "E_odd")
        p.latent["E"].add(me, "E_even")
        p.latent["E"].add(me, "E_all")

        noise_key = p.add_noise_model(UncorrelatedSingleNoise())

        J = VariationalBayesSolver(p).jacobian([42.0])[noise_key]
        self.assertEqual(J.shape, (6, 1))
        CHECK(J[:, 0], me.x_odd + me.x_even + me.x_all)



    def test_two_joints(self):
        """
        Checks if the joint jacobian is caluclated correctly, if only
        two of the three parameters are defined latent.
        """
        me = OddEvenME()
        p = InferenceProblem()
        p.add_model_error(me)
        p.latent["E"].add(me, "E_odd")
        p.latent["E"].add(me, "E_even")
        noise_key = p.add_noise_model(UncorrelatedSingleNoise())

        J = VariationalBayesSolver(p).jacobian([42.0])[noise_key]
        CHECK(J[:, 0], me.x_odd + me.x_even)


if __name__ == "__main__":
    unittest.main()
