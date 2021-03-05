import unittest
from bayes.parameters import ParameterList
from bayes.latent import LatentParameters


class TestLatentParameters(unittest.TestCase):
    def setUp(self):
        self.pA = ParameterList()
        self.pA.define("A", 0)
        self.pA.define("shared", 2)

        self.pB = ParameterList()
        self.pB.define("B", 1)
        self.pB.define("shared", 2)

    def test_add(self):
        latent = LatentParameters()
        latent["latentA"].add(self.pA, "A")

    def test_add_twice(self):
        latent = LatentParameters()
        latent["latentA"].add(self.pA, "A")

        another_pA = ParameterList()
        another_pA.define("A", 0)
        another_pA.define("shared", 2)
        latent["latentA"].add(another_pA, "A")

    def test_shared(self):
        latent = LatentParameters()
        latent["shared"].add(self.pA, "shared")
        latent["shared"].add(self.pB, "shared")

        self.assertTrue(latent["shared"].has(self.pA, "shared"))

        self.assertListEqual(latent["shared"].global_index_range(), [0])
        latent.update([42])
        self.assertEqual(self.pA["shared"], 42)
        self.assertEqual(self.pB["shared"], 42)


if __name__ == "__main__":
    unittest.main()
