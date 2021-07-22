import unittest
from bayes.latent import LatentParameters, InconsistentLengthException


class VectorModel:
    def get_shape(self, prm):
        if prm == "B":
            return 2
        return 1


class ScalarModel:
    def get_shape(self, prm):
        return 1


class TestLatentParameters(unittest.TestCase):
    def setUp(self):
        self.vector_model = VectorModel()
        self.scalar_model = ScalarModel()
        self.models = {"model1": self.scalar_model, "model2": self.vector_model}

    def test_update(self):
        l = LatentParameters(self.models)

        l["shared"].add(self.scalar_model, "A1")
        l["shared"].add(self.vector_model, "A2")
        l["B"].add(self.vector_model)

        new = l.updated_parameters([42, 6174, 84])

        self.assertEqual(new["model1"]["A1"], 42)
        self.assertEqual(new["model2"]["A2"], 42)
        self.assertEqual(new["model2"]["B"], [6174, 84])

    def test_inconsistent_length(self):
        l = LatentParameters(self.models)
        l["GlobalName"].add(self.vector_model, "B")
        with self.assertRaises(InconsistentLengthException) as e:
            l["GlobalName"].add(self.scalar_model, "B")

        # TODO?
        # msg = str(e.exception)
        # print(msg)
        # self.assertIn("42", msg)
        # self.assertIn("6174", msg)
        # self.assertIn("GlobalName", msg)

        with self.assertRaises(InconsistentLengthException) as e:
            some_vector_with_not_length_2 = [1, 2, 3]
            l.updated_parameters(some_vector_with_not_length_2)

        msg = str(e.exception)
        print(msg)
        self.assertIn("2", msg)
        self.assertIn("3", msg)

    def test_global_latent(self):
        l = LatentParameters(self.models)
        l["A"].add_shared()
        self.assertEqual(len(l), 1)
        self.assertEqual(len(l["A"]), 2)
        
        l["C"].add_shared(prior="something")
        self.assertEqual(len(l), 2)
        self.assertEqual(l["C"].prior, "something")

    def test_prior(self):
        l = LatentParameters(self.models)
        l["A"].add_shared(prior="my normal")

        with self.assertRaises(Exception) as e:
            l["A"].prior = "another prior"

        msg = str(e.exception)
        print(msg)
        self.assertIn("my normal", msg)
        self.assertIn("A", msg)
        self.assertIn("another prior", msg)




    def test_pretty_print(self):
        l = LatentParameters(self.models)
        l["shared"].add("model1", "A1")
        l["shared"].add("model2", "A2")
        l["B"].add("model1")
        print(l)


if __name__ == "__main__":
    unittest.main()
