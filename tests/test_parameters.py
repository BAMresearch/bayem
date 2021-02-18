import unittest
from bayes.parameters import *


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.p = ModelErrorParameters()
        self.p.define("pA", 0.0)
        self.p.define("pB", 0.0)

    def test_parameter(self):
        self.p.update(["pA"], [42.0])
        self.assertEqual(self.p["pA"], 42.0)
        self.assertEqual(self.p["pB"], 0.0)

    def test_concat(self):
        p1 = ModelErrorParameters()
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


class TestSingleModel(unittest.TestCase):
    def setUp(self):
        self.l = LatentParameters()

        self.p = ModelErrorParameters()
        self.p.define("pA", 0.0)
        self.p.define("pB", 0.0)

        # We now need to tell the LatentParameters that some of our
        # ModelErrorParameters are potentially latent. We do not need to provide
        # a key as we only have one ModelErrorParameters.
        self.l.define_parameter_list(self.p)

    def test_joint_list(self):
        self.l.add("latentA","pA")
        updated = self.l.update([42])
        self.assertEqual(self.p["pA"], 42.0)
        self.assertEqual(self.p["pB"], 0.0)


class TestLatentParameters(unittest.TestCase):
    def setUp(self):
        self.pA = ModelErrorParameters()
        self.pA.define("only_in_A", 0)
        self.pA.define("shared", 2)

        self.pB = ModelErrorParameters()
        self.pB.define("only_in_B", 1)
        self.pB.define("shared", 2)

        self.l = LatentParameters()
        # Parameters from both ModelErrorParameters should be latent. Thus we
        # have to define some keys to uniquely identify them ...
        self.keyA = "A"
        self.keyB = "B"
        # ... and add them to the LatentParameters.
        self.l.define_parameter_list(self.pA, self.keyA)
        self.l.define_parameter_list(self.pB, self.keyB)

    def test_add(self):
        l, keyA, keyB = self.l, self.keyA, self.keyB
        l.add("latentA","only_in_A", keyA)
        self.assertRaises(Exception, self.l.add,"latentA", "only_in_A", self.keyA)

        l.add("latentShared", "shared", keyB)

        self.assertTrue(l.exists("shared", keyB))
        l.add("latentShared", "shared", keyA)

        self.assertTrue(l.exists("shared", keyA))

        self.assertListEqual(l.global_range("latentA"), [0])
        self.assertListEqual(l.global_range("latentShared"), [1])

        updated_prm = l.update([42, 17])
        self.assertEqual(updated_prm[keyA]["only_in_A"], 42)
        self.assertEqual(updated_prm[keyA]["shared"], 17)
        self.assertEqual(updated_prm[keyB]["shared"], 17)

    def test_add_by_name(self):
        self.l.add_by_name("shared")
        updated_prm = self.l.update([17])
        self.assertEqual(updated_prm[self.keyA]["shared"], 17)
        self.assertEqual(updated_prm[self.keyB]["shared"], 17)



if __name__ == "__main__":
    unittest.main()
