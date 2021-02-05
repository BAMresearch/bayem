import unittest
from bayes.parameters import *


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.p = ModelParameters()
        self.p.define("pA", 0.0)
        self.p.define("pB", 0.0)

    def test_parameter(self):
        self.p.update(["pA"], [42.0])
        self.assertEqual(self.p["pA"], 42.0)
        self.assertEqual(self.p["pB"], 0.0)

    def test_concat(self):
        p1 = ModelParameters()
        p1.define("A", 17)

        p1 += self.p
        self.assertEqual(p1["A"], 17)
        self.assertEqual(p1["pA"], 0.0)
        self.assertEqual(p1["pB"], 0.0)

    def test_has(self):
        self.assertTrue(self.p.has("pA"))
        self.assertFalse(self.p.has("pC"))

    def test_copy(self):
        import copy

        self.assertEqual(self.p["pA"], 0.0)
        copy_p = copy.deepcopy(self.p)
        copy_p["pA"] = 42.0
        self.assertEqual(copy_p["pA"], 42)
        self.assertEqual(self.p["pA"], 0.0)

    def test_define(self):
        self.assertRaises(Exception, self.p.__setitem__, "new_key", 0.2)

    def test_vector(self):
        p = ModelParameters()
        x = [42.0, 6174.0, 0.0, -2.0]
        p.define("v1", x)
        p.define("v2")
        p["v2"] = x
        p.update(["v2"], [[42.0, 6174.0, 0.0, -3]])
        self.assertListEqual(p["v2"], [42.0, 6174, 0.0, -3])

        # not sure if this (update with length 3 instead of 4) should be an
        # error ...
        p.update(["v1"], [[1, 2, 3]])


class TestSingleModel(unittest.TestCase):
    def setUp(self):
        self.l = LatentParameters()

        self.p = ModelParameters()
        self.p.define("pA", 0.0)
        self.p.define("pB", 0.0)

        # We now need to tell the LatentParameters that some of our
        # ModelParameters are potentially latent. We do not need to provide
        # a key as we only have one ModelParameters.
        self.l.define_model_parameters(self.p)

    def test_joint_list(self):
        self.l.add("pA")
        updated = self.l.update([42.0], return_copy=False)
        self.assertEqual(self.p["pA"], 42.0)
        self.assertEqual(self.p["pB"], 0.0)

    def test_vector_parameter(self):
        self.p.define("v", [0.0, 0.0, 0.0])
        self.l.add("pA")
        self.l.add("v")
        updated = self.l.update([42.0, 1.0, 2.0, 3.0])
        self.assertEqual(self.p["pA"], 42.0)
        self.assertListEqual(self.p["v"], [1.0, 2.0, 3.0])


class TestLatentParameters(unittest.TestCase):
    def setUp(self):
        self.pA = ModelParameters()
        self.pA.define("only_in_A", 0)
        self.pA.define("shared", 2)

        self.pB = ModelParameters()
        self.pB.define("only_in_B", 1)
        self.pB.define("shared", 2)

        self.l = LatentParameters()
        # Parameters from both ModelParameters should be latent. Thus we
        # have to define some keys to uniquely identify them ...
        self.keyA = "A"
        self.keyB = "B"
        # ... and add them to the LatentParameters.
        self.l.define_model_parameters(self.pA, self.keyA)
        self.l.define_model_parameters(self.pB, self.keyB)

    def test_add(self):
        l, keyA, keyB = self.l, self.keyA, self.keyB
        index = l.add("only_in_A", keyA)
        self.assertEqual(index, 0)
        self.assertRaises(Exception, self.l.add, "only_in_A", self.keyA)

        index = l.add("shared", keyB)
        self.assertEqual(index, 1)

        self.assertTrue(l.exists("shared", keyB))
        self.assertRaises(Exception, l.add_shared, index, "shared", keyB)
        l.add_shared(index, "shared", keyA)

        self.assertTrue(l.exists("shared", keyA))

        self.assertListEqual(l.indices_of(keyA), [0, 1])
        self.assertListEqual(l.indices_of(keyB), [1])

        updated_prm = l.update([42, 17])
        self.assertEqual(updated_prm[keyA]["only_in_A"], 42)
        self.assertEqual(updated_prm[keyA]["shared"], 17)
        self.assertEqual(updated_prm[keyB]["shared"], 17)

    def test_add_by_name(self):
        self.l.add_by_name("shared")
        updated_prm = self.l.update([17])
        self.assertEqual(updated_prm[self.keyA]["shared"], 17)
        self.assertEqual(updated_prm[self.keyB]["shared"], 17)


class TestUncorrelatedNormalPrior(unittest.TestCase):
    def setUp(self):
        self.p = ModelParameters()
        self.p.define("pA", 0.0)
        self.p.define("pB", 0.0)
        self.l = LatentParameters()
        self.l.define_model_parameters(self.p)

    def test_existing_latent_parameters(self):
        self.l.add("pA")
        # not allowed to set latent, if you use these prior classes
        self.assertRaises(Exception, UncorrelatedNormalPrior, self.l)

    def test_MVN(self):
        prior = UncorrelatedNormalPrior(self.l)
        prior.add("pA", mean=0, sd=2)
        prior.add("pB", mean=1, sd=4)

        mvn = prior.to_MVN()
        self.assertAlmostEqual(mvn.mean[0], 0.0)
        self.assertAlmostEqual(mvn.std_diag[0], 2.0)

        self.assertAlmostEqual(mvn.mean[1], 1.0)
        self.assertAlmostEqual(mvn.std_diag[1], 4.0)


if __name__ == "__main__":
    unittest.main()
