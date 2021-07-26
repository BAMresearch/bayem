import numpy as np
import math
import unittest

from bayes.parameters import ParameterList
from bayes.vb import Gamma
from bayes.noise import UncorrelatedSingleNoise
from bayes.inference_problem import InferenceProblem, ModelErrorInterface
from bayes.solver import *

import scipy.stats

CHECK = np.testing.assert_array_almost_equal  # just to make it shorter


def get_fast_norm(loc, scale):
    dist = scipy.stats.norm(loc=loc, scale=scale)
    prec = 1/scale**2 
    def logpdf(theta):
        return -0.5 * math.log(2 * math.pi / prec) - 0.5 * prec * (theta - loc)**2

    dist.logpdf=logpdf
    return dist

def get_fast_gamma(a, scale):
    dist = scipy.stats.gamma(a=a, scale=scale)

    def logpdf(theta):
        pdf = 1/(math.gamma(a) * scale**a)*theta**(a-1)*math.exp(-theta/scale)
        return math.log(pdf)
    dist.logpdf=logpdf
    return dist


x = np.r_[1, 2]
def dummy_model_error(prms):
    return {"dummy_sensor": prms["B"] * x - 20}


def define_test_problem():
    p = InferenceProblem()
    p.add_model_error(dummy_model_error)
    p.latent["B"].add_shared()
    p.latent["B"].prior = get_fast_norm(loc=10.0, scale=10.0)

    p.add_noise_model(UncorrelatedSingleNoise(), key="noise")
    p.latent["noise"].add("noise", "precision")

    p.latent["noise"].prior = get_fast_gamma(1, 1)
    return p


class TestSolver(unittest.TestCase):
    def test_vb(self):
        p = define_test_problem()
        vbs = VariationalBayesSolver(p)
        result = vbs.estimate_parameters()
        print(result)

    def test_taralli_nested(self):
        p = define_test_problem()
        solver = TaralliSolver(p)
        model = solver.nestle_model()
        model.estimate_parameters()
        model.summary()

    def test_taralli_emcee(self):
        p = define_test_problem()
        solver = TaralliSolver(p)
        model = solver.emcee_model()
        model.estimate_parameters()
        model.summary()

    def test_my_prior(self):
        ref = scipy.stats.norm(loc=20, scale=10)
        my = get_fast_norm(loc=20, scale=10)

        self.assertAlmostEqual(ref.logpdf(10), my.logpdf(10))

        ref = scipy.stats.gamma(a=20, scale=10)
        my = get_fast_gamma(a=20, scale=10)

        self.assertAlmostEqual(ref.logpdf(10), my.logpdf(10))

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
