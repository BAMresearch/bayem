# third party imports
import unittest

# local imports
from bayes.inference_problem_new import InferenceProblem


class TestProblem(unittest.TestCase):

    def test_add_remove_calibration_parameter(self):
        p = InferenceProblem("TestProblem")
        # adding a parameter without const or prior argument must raise error
        with self.assertRaises(RuntimeError):
            p.add_parameter('a', 'model')
        # adding a parameter with const and prior argument must raise error
        with self.assertRaises(RuntimeError):
            p.add_parameter('a', 'model', const=1.0, prior={})
        # adding a parameter with wrong type must raise an error
        with self.assertRaises(RuntimeError):
            p.add_parameter('a', 'wrong_type')
        # add first calibration parameter; n_calibration_prms should be 1
        p.add_parameter('a', 'model',
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        self.assertEqual(p.n_calibration_prms, 1)
        # add second calibration parameter; n_calibration_prms should be 2
        p.add_parameter('b', 'model',
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        self.assertEqual(p.n_calibration_prms, 2)
        # add second calibration parameter again, which should give an error
        with self.assertRaises(RuntimeError):
            p.add_parameter('b', 'model',
                            prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        # add calibration parameter without prior should raise an error
        with self.assertRaises(RuntimeError):
            p.add_parameter('c', 'model')
        # removing a calibration parameter should reduce n_calibration_prms
        p.remove_parameter('b')
        self.assertEqual(p.n_calibration_prms, 1)
        p.remove_parameter('a')
        self.assertEqual(p.n_calibration_prms, 0)
        # removing a parameter that does not exists should raise an error
        with self.assertRaises(RuntimeError):
            p.remove_parameter('a')
        # check if alias behave correctly when adding/removing cali parameters
        self.assertEqual(len(p._alias_dict.keys()), 0)
        p.add_parameter('a', 'model',
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        self.assertEqual(len(p._alias_dict.keys()), 3)
        p.add_parameter_alias('a', 'a_1')
        self.assertEqual(len(p._alias_dict.keys()), 4)
        self.assertEqual(len(p._prm_dict['a']['alias']), 1)
        p.add_parameter_alias('a', 'a_2')
        self.assertEqual(len(p._alias_dict.keys()), 5)
        self.assertEqual(len(p._prm_dict['a']['alias']), 2)
        p.remove_parameter('a')
        self.assertEqual(len(p._alias_dict.keys()), 0)

    def test_add_remove_const_parameter(self):
        p = InferenceProblem("TestProblem")
        # simply add some parameters with different names
        p.add_parameter('a', 'model', const=1.0)
        p.add_parameter('b', 'prior', const=2.0)
        p.add_parameter('c', 'noise', const=3.0)
        # adding the same parameter again should raise an error
        with self.assertRaises(RuntimeError):
            p.add_parameter('a', 'model', const=1.0)
        # check if removing a parameter and adding it again works
        p.remove_parameter('a')
        p.add_parameter('a', 'model', const=1.0)
        p.remove_parameter('b')
        p.remove_parameter('c')
        # removing a parameter that does not exists should raise an error
        with self.assertRaises(RuntimeError):
            p.remove_parameter('c')
        # check the alias functionality
        self.assertEqual(len(p._alias_dict.keys()), 1)
        self.assertEqual(len(p._prm_dict['a']['alias']), 0)
        p.add_parameter_alias('a', 'a_1')
        self.assertEqual(len(p._alias_dict.keys()), 2)
        self.assertEqual(len(p._prm_dict['a']['alias']), 1)
        p.add_parameter_alias('a', 'a_2')
        self.assertEqual(len(p._alias_dict.keys()), 3)
        self.assertEqual(len(p._prm_dict['a']['alias']), 2)
        p.remove_parameter('a')
        self.assertEqual(len(p._alias_dict.keys()), 0)

    def test_change_parameter_role(self):
        p = InferenceProblem("TestProblem")
        # check simple change of parameter roles
        p.add_parameter('a', 'model',
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        p.change_parameter_role('a', const=1.0)
        p.change_parameter_role('a',
                                prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        # check with additional parameters and aliases
        p.add_parameter('b', 'noise',
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        p.add_parameter_alias('a', ['a_1', 'a_2'])
        self.assertEqual(len(p._alias_dict.keys()), 8)
        p.change_parameter_role('a', const=1.0)
        self.assertEqual(len(p._prm_dict['a']['alias']), 2)
        p.change_parameter_role('a',
                                prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        self.assertEqual(len(p._alias_dict.keys()), 8)


if __name__ == "__main__":
    unittest.main()