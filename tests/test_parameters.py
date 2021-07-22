import unittest
from bayes.parameters import ParameterList


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.p = ParameterList()
        self.p.define("pA", 0.0)
        self.p.define("pB", 0.0)

    def test_concat(self):
        p1 = ParameterList()
        p1.define("A", 17)

        p1 += self.p
        self.assertEqual(p1["A"], 17)
        self.assertEqual(p1["pA"], 0.0)
        self.assertEqual(p1["pB"], 0.0)

    def test_has(self):
        self.assertTrue("pA" in self.p)
        self.assertFalse("pC" in self.p)

    def test_copy(self):
        import copy

        self.assertEqual(self.p["pA"], 0.0)
        copy_p = copy.deepcopy(self.p)
        copy_p["pA"] = 42.0

        self.assertEqual(copy_p["pA"], 42)
        self.assertEqual(self.p["pA"], 0.0)

    def test_define(self):
        self.assertRaises(Exception, self.p.__setitem__, "new_key", 0.2)

    def test_iterate(self):
        for p in self.p.names:
            print(p)

    def test_overwrite(self):
        other = ParameterList()
        other.define("pA", 42)

        p_new = self.p.overwrite_with(other)
        self.assertEqual(p_new["pA"], 42)
        self.assertEqual(p_new["pB"], 0)

        other2 = ParameterList()
        other2.define("pC", 42)
        with self.assertRaises(Exception):
            p_new = self.p.overwrite_with(other2)

    def test_overwrite_one(self):
        p_new = self.p.with_value("pA", 42)
        self.assertEqual(p_new["pA"], 42)
        self.assertEqual(p_new["pB"], 0)


if __name__ == "__main__":
    unittest.main()
