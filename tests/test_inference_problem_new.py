# third party imports
import unittest

# local imports
from bayes.inference_problem_new import InferenceProblem


class TestProblem(unittest.TestCase):
    def test_add_calibration_parameter(self):
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', info="Some model parameter",
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        self.assertEqual(p.n_calibration_prms, 1)


if __name__ == "__main__":
    unittest.main()
