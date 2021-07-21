import unittest
from bayes.latent import LatentParameters, InconsistentLengthException


class TestLatentParameters(unittest.TestCase):
    def test_update(self):
        l = LatentParameters()

        l.add("shared", "A1", "model1")
        l.add("shared", "A2", "model2")
        l.add("B", "B", "model1", 2)

        new = l.updated_parameters([42, 6174, 84])

        self.assertEqual(new["model1"]["A1"], 42)
        self.assertEqual(new["model2"]["A2"], 42)
        self.assertEqual(new["model1"]["B"], [6174, 84])

    def test_inconsistent_length(self):
        l = LatentParameters()
        l.add("GlobalName", "A", "model", N=42)
        with self.assertRaises(InconsistentLengthException) as e:
            l.add("GlobalName", "who", "cares", N=6174)

        # TODO?
        # msg = str(e.exception)
        # print(msg)
        # self.assertIn("42", msg)
        # self.assertIn("6174", msg)
        # self.assertIn("GlobalName", msg)

        with self.assertRaises(InconsistentLengthException) as e:
            some_vector_with_not_length_42 = [1, 2, 3]
            l.updated_parameters(some_vector_with_not_length_42)

        msg = str(e.exception)
        print(msg)
        self.assertIn("42", msg)
        self.assertIn("3", msg)


if __name__ == "__main__":
    unittest.main()
