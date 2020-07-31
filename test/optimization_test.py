# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring gate addition and functionality is working properly
in the library.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.ignis.characterization.coherence import *
from qiskit.ignis.characterization.coherence import T1Fitter
from qiskit.providers.quac import Quac


class OptimizationTest(unittest.TestCase):
    """Tests optimization functionality
    """

    def setUp(self):
        """Set up Qiskit AER and QuaC simulators
        """
        self.qiskit_qasm_backend = Aer.get_backend('qasm_simulator')
        self.quac_counts_backend = Quac.get_backend('fake_yorktown_density_simulator')
        self.shots = 8000

    def t1_recovery_test(self):
        """Test T1 recovery (TODO)
        """
        pass


if __name__ == '__main__':
    unittest.main()
