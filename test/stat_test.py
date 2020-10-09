# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring gate addition and functionality is working properly
in the library.
"""
import random
import unittest
import numpy as np
from quac_qiskit.stat import get_vec_angle, kl_dist_smoothing, discrete_one_samp_ks, choose_index


class StatTestCase(unittest.TestCase):
    """Tests QuaC noise model functionality by recovering model parameters with Qiskit fitters
    """

    def test_vec_angle(self):
        self.assertEqual(get_vec_angle([0, 1], [1, 0]), 90)
        self.assertEqual(get_vec_angle([1, 0], [1, 0]), 0)
        self.assertEqual(get_vec_angle([1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2)]), 45)
        self.assertEqual(get_vec_angle([0, 0], [0, 0]), None)

    def test_kl_dist_smoothing(self):
        p_dist = [0.36, 0.48, 0.16]
        q_dist = [1/3, 1/3, 1/3]

        self.assertEqual(round(kl_dist_smoothing(p_dist, q_dist, 1e-5), 7), 0.0852996)
        self.assertEqual(round(kl_dist_smoothing(q_dist, p_dist, 1e-5), 6), 0.097455)

    def test_discrete_one_samp_ks(self):
        dist1 = [0.2, 0.4, 0.1, 0.3]
        dist2 = [0.1, 0.3, 0.2, 0.4]
        cdf1 = [0.2, 0.6, 0.7, 1.0]
        cdf2 = [0.1, 0.4, 0.6, 1.0]

        ks_outcome = discrete_one_samp_ks(dist1, dist2, 1000)
        self.assertLess(abs(ks_outcome[0] - (np.array(cdf1) - np.array(cdf2)).max()), 1e-5)

    def test_choose_index(self):
        for _ in range(100):
            num_outcomes = random.randrange(4, 10)
            dist = []
            while len(dist) < num_outcomes:
                if len(dist) == num_outcomes - 1:
                    dist.append(1 - sum(dist))
                else:
                    next_prob = random.random()
                    if sum(dist) + next_prob <= 1:
                        dist.append(next_prob)

            experiments_accumulator = [0] * num_outcomes
            for _ in range(100000):
                exp_outcome = choose_index(dist)
                experiments_accumulator[exp_outcome] += 1 / 100000

            self.assertTrue(discrete_one_samp_ks(experiments_accumulator, dist, 100000)[1])


if __name__ == '__main__':
    unittest.main()
