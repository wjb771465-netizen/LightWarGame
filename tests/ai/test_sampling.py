"""ai/algos/sampling.py 纯数学函数的单元测试。"""

import unittest

import numpy as np

from ai.algos.sampling import logistic_softmax_probs, uniform_probs


class TestUniformProbs(unittest.TestCase):

    def test_sum_is_one(self):
        probs = uniform_probs(5)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0)

    def test_all_equal(self):
        probs = uniform_probs(4)
        np.testing.assert_array_almost_equal(probs, np.full(4, 0.25))

    def test_zero_returns_empty(self):
        probs = uniform_probs(0)
        self.assertEqual(len(probs), 0)


class TestLogisticSoftmaxProbs(unittest.TestCase):

    def test_sum_is_one(self):
        values = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        probs = logistic_softmax_probs(values, lam=1.0, s=100.0, D=400.0)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0)

    def test_higher_value_gets_higher_prob(self):
        """较大的值应获得较高的概率。"""
        values = np.array([1000.0, 1200.0, 1400.0], dtype=np.float64)
        probs = logistic_softmax_probs(values, lam=1.0, s=100.0, D=400.0)
        # 1400 > 1200 > 1000
        self.assertGreater(probs[2], probs[1])
        self.assertGreater(probs[1], probs[0])

    def test_equal_values_yield_equal_probs(self):
        values = np.array([500.0, 500.0, 500.0], dtype=np.float64)
        probs = logistic_softmax_probs(values, lam=1.0, s=100.0, D=400.0)
        np.testing.assert_array_almost_equal(probs, np.full(3, 1.0 / 3.0))

    def test_larger_lam_sharper_distribution(self):
        """λ 越大，分布应越尖锐（高值概率更高）。"""
        values = np.array([1000.0, 1200.0, 1400.0], dtype=np.float64)
        probs_small_lam = logistic_softmax_probs(values, lam=0.1, s=100.0, D=400.0)
        probs_large_lam = logistic_softmax_probs(values, lam=5.0, s=100.0, D=400.0)
        # larger lam → highest value gets higher prob
        self.assertGreater(probs_large_lam[2], probs_small_lam[2])

    def test_empty_returns_empty(self):
        probs = logistic_softmax_probs(np.array([], dtype=np.float64), lam=1.0, s=100.0, D=400.0)
        self.assertEqual(len(probs), 0)

    def test_single_value_returns_one(self):
        values = np.array([42.0], dtype=np.float64)
        probs = logistic_softmax_probs(values, lam=1.0, s=100.0, D=400.0)
        self.assertAlmostEqual(float(probs[0]), 1.0)
        self.assertEqual(len(probs), 1)

    def test_median_invariant_to_shift_in_D(self):
        """D 变化时，高于 median 的 entry 仍然高于 median。"""
        values = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        probs_D400 = logistic_softmax_probs(values, lam=1.0, s=100.0, D=400.0)
        probs_D100 = logistic_softmax_probs(values, lam=1.0, s=100.0, D=100.0)
        # 300 (index 2) gets highest prob in both; 100 (index 0) gets lowest in both
        self.assertGreater(probs_D400[2], probs_D400[0])
        self.assertGreater(probs_D100[2], probs_D100[0])


if __name__ == "__main__":
    unittest.main()
