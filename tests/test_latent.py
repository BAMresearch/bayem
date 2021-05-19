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
        self.pB.define("list", [3, 4])

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

        with self.assertRaises(Exception) as e:
            latent.update([42, 42])
        print("Expected exception: \n", e.exception)

    def test_set_value(self):
        latent = LatentParameters()
        latent["shared"].add(self.pA, "shared")
        latent["shared"].add(self.pB, "shared")
        latent["shared"].set_value(42)

        self.assertEqual(self.pA["shared"], 42)
        self.assertEqual(self.pA["shared"], 42)

        with self.assertRaises(Exception) as e:
            latent["shared"].set_value([1, 2, 3])
        print("Expected exception: \n", e.exception)

    def test_start_vector(self):
        latent = LatentParameters()
        latent["A"].add(self.pA, "A")
        latent["B"].add(self.pB, "B")
        latent["shared"].add(self.pA, "shared")
        latent["shared"].add(self.pB, "shared")
        latent["list"].add(self.pB, "list")

        v = latent.get_vector()
        self.assertListEqual(v, [0, 1, 2, 3, 4])

        v = latent.get_vector({"A": 42, "list": [61, 74]})
        self.assertListEqual(v, [42, 1, 2, 61, 74])

        # check for dimensions errors
        with self.assertRaises(Exception) as e:
            v = latent.get_vector({"A": [1, 2, 3]})
        print("Expected exception: \n", e.exception)

        with self.assertRaises(Exception) as e:
            v = latent.get_vector({"list": [1, 2, 3]})
        print("Expected exception: \n", e.exception)

        # We expect an exception, if a shared parameter is not defined
        # unambiguously
        self.pA["shared"] = 20
        with self.assertRaises(RuntimeError) as e:
            v = latent.get_vector()
        print("Expected exception: \n", e.exception)

        # Provinding a default value for that case is fine though:
        v = latent.get_vector({"shared": 42})


if __name__ == "__main__":
    unittest.main()
