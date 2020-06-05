# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring basic package utilities are working properly.
"""

import unittest
from qiskit.providers.quac.utils import choose_index


class UtilsTestCase(unittest.TestCase):
    """Tests various package utilities
    """

    def test_choose_index(self):
        """Tests the choose_index function from prob module
        """
        half_outcomes = [0, 0]
        for ind in range(1000000):
            outcome = choose_index([0.5, 0.5])
            half_outcomes[outcome] += 1
        self.assertLessEqual(half_outcomes[0] - half_outcomes[1], 1000)  # less than 0.1% error


if __name__ == '__main__':
    unittest.main()
