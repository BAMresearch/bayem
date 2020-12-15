import unittest
from bayes.parameters import *


class TestParameters(unittest.TestCase):
    def test_parameter(self):
        p = ModelParameters()
        p.define("pA", 0.0)
        p.define("pB", 0.0)
        p.update(["pA"], [42.0])
        self.assertEqual(p["pA"], 42.0)
        self.assertEqual(p["pB"], 0.0)


class TestSingleModel(unittest.TestCase):
    def setUp(self):
        self.p = ModelParameters()
        self.p.define("pA", 0.0)
        self.p.define("pB", 0.0)

    def test_joint_list(self):
        l = JointParameterList(self.p)
        l.set_latent("pA")
        l.update([42])
        self.assertEqual(self.p["pA"], 42.0)
        self.assertEqual(self.p["pB"], 0.0)

        # not allowed to set latent, if you use these prior classes
        self.assertRaises(Exception, UncorrelatedNormalPrior, l)

    def test_prior(self):
        l = JointParameterList(self.p)
        prior = UncorrelatedNormalPrior(l)
        prior.add("pA", mean=0, sd=2)
        prior.add("pB", mean=1, sd=4)

        mvn = prior.to_MVN()
        self.assertAlmostEqual(mvn.mean[0], 0.0)
        self.assertAlmostEqual(mvn.std_diag[0], 2.0)

        self.assertAlmostEqual(mvn.mean[1], 1.0)
        self.assertAlmostEqual(mvn.std_diag[1], 4.0)

    def test_define(self):
        self.assertRaises(Exception, self.p.__setitem__, "new_key", 0.2)


class TestTwoModels(unittest.TestCase):
    def setUp(self):
        self.pA = ModelParameters()
        self.pA.define("only_in_A", 0)
        self.pA.define("shared", 2)
        self.keyA = "A"

        self.pB = ModelParameters()
        self.pB.define("only_in_B", 1)
        self.pB.define("shared", 2)
        self.keyB = "B"

        inp = {self.keyA: self.pA, self.keyB: self.pB}
        self.l = JointParameterList(inp, shared=["shared"])

    def test_unknown_parameter(self):
        self.assertRaises(Exception, self.l.set_latent, "only_in_A", self.keyB)

    def test_latent_without_key(self):
        self.assertRaises(Exception, self.l.set_latent, "only_in_A")

    def test_duplicate_latents(self):
        self.l.set_latent("only_in_A", self.keyA)
        self.assertRaises(Exception, self.l.set_latent, "only_in_A", self.keyA)

    def test_shared(self):
        self.l.set_latent("shared")
        self.assertRaises(Exception, self.l.set_latent, "shared", self.keyA)
        self.l.update([6174])
        self.assertEqual(self.pA["shared"], 6174)
        self.assertEqual(self.pB["shared"], 6174)

    def test_global_parameters(self):
        self.l.set_latent("only_in_A", self.keyA)
        self.l.update([17])
        self.assertEqual(self.pA["only_in_A"], 17)

    def test_printable(self):
        self.l.set_latent("shared")
        self.l.set_latent("only_in_A", self.keyA)
        print(self.l)


if __name__ == "__main__":
    unittest.main()
