import numpy as np
import time
import unittest

from bayes.vb import check_pattern, split, VB, MVN

class TestSplit(unittest.TestCase):
    
    k = np.r_[0, 1, 2, 3, 4, 5, 6, 7]

    def test_no_split(self):
        k_split = split(self.k, pattern=[[0, 1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(len(k_split), 1)
        self.assertTrue(np.array_equal(self.k, k_split[0]))

        k_split = split(self.k, pattern=None)
        self.assertEqual(len(k_split), 1)
        self.assertTrue(np.array_equal(self.k, k_split[0]))
        
    def test_full_split(self):
        k_split = split(self.k, pattern=[[0, 4], [1, 5], [6, 2], [3, 7]])
        self.assertEqual(len(k_split), 4)
        self.assertListEqual(list(k_split[0]), [0, 4])
        self.assertListEqual(list(k_split[1]), [1, 5])
        self.assertListEqual(list(k_split[2]), [6, 2])
        self.assertListEqual(list(k_split[3]), [3, 7])

    def test_full_split2(self):
        k_split = split(self.k, pattern=[[0, 3, 4, 7], [2, 6], [1, 5]])
        self.assertEqual(len(k_split), 3)
        self.assertListEqual(list(k_split[0]), [0, 3, 4, 7])
        self.assertListEqual(list(k_split[1]), [2, 6])
        self.assertListEqual(list(k_split[2]), [1, 5])


    def test_bad_pattern(self):
        check_pattern([range(8)])
        check_pattern([reversed(range(8))])
        check_pattern([[0,4,3], [2, 1, 5]])

        check_pattern([[0,4,3], [2, 1, 5]], 6)

        self.assertRaises(ValueError, check_pattern, [[0,2]])
        self.assertRaises(ValueError, check_pattern, [[1], [1]])
        self.assertRaises(ValueError, check_pattern, [range(8)], 10)

        def me(prm):
            return 4 - np.ones(10) * prm
        me.noise_pattern = [range(10)]
        VB().run(me, MVN(0,1))


        me.noise_pattern = [range(11)]
        self.assertRaises(ValueError, VB().run, me, MVN(0, 1))


if __name__ == "__main__":
    N = 1_000_000
    k_big = np.linspace(0, 1, N)
    t = time.time()
    pattern = [range(0, N, 3), range(1, N, 3), range(2, N, 3)]

    k_split = split(k_big, pattern)
    print(f"Split {len(k_big)} elements in {time.time()-t} s.")

    unittest.main()
