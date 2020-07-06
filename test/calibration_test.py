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
from qiskit.providers.quac.utils.benchmarking import quac_t1_circuits


class GateTestCase(unittest.TestCase):
    """Tests gate functionality
    """

    def setUp(self):
        """Set up Qiskit AER and QuaC simulators for T1 calculations
        """
        self.qiskit_qasm_backend = Aer.get_backend('qasm_simulator')
        self.quac_counts_backend = Quac.get_backend('fake_yorktown_counts_simulator')
        self.shots = 8000
        self.num_samples = 10

    def test_quac_t1(self):
        """Test QuaC T1 approximations
        """
        hardware_props = self.quac_counts_backend.properties()
        num_gates = np.linspace(10, 3000, self.num_samples, dtype='int')
        qubits = list(range(5))

        actual_t1 = [hardware_props.t1(qubit) * 1e9 for qubit in qubits]
        quac_t1_circs, t1_delay = quac_t1_circuits(num_gates,
                                                   hardware_props.gate_length('id', [0]) * 1e9,
                                                   qubits)

        quac_t1_backend_result = execute(quac_t1_circs, self.quac_counts_backend,
                                         shots=self.shots,
                                         optimization_level=0).result()

        quac_t1_fit = T1Fitter(quac_t1_backend_result, t1_delay, qubits,
                               fit_p0=[1, 80000, 0],
                               fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                               time_unit="nano-seconds")

        for ind in range(5):
            quac_t1_fit.plot(ind)
            plt.show()

        quac_t1 = quac_t1_fit.time()

        print(f"Actual T1 times: {actual_t1} in nanoseconds.")
        print(f"Plugin T1 times: {quac_t1} in nanoseconds.")
        max_diff = abs((np.array(quac_t1) - np.array(actual_t1)).max())
        print(f"Maximum difference between T1 time calculations and actual times: {max_diff}")

        self.assertLess(max_diff, 1e4)  # of same magnitude?

    def test_quac_t2(self):
        """Test QuaC T2 approximations
        """
        hardware_props = self.quac_counts_backend.properties()
        num_gates = np.linspace(10, 1e4, self.num_samples, dtype='int')
        qubits = list(range(self.quac_counts_backend.configuration().num_qubits))

        t2_circs, t2_delay = t2_circuits(num_gates,
                                         hardware_props.gate_length('id', [0]) * 1e9,
                                         qubits)

        print("Running...")
        quac_t2_backend_result = execute(t2_circs, self.quac_counts_backend,
                                         shots=self.shots,
                                         optimization_level=0).result()

        quac_t2_fit = T2Fitter(quac_t2_backend_result, t2_delay, qubits,
                               fit_p0=[1, 1e4, 0],
                               fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                               time_unit="nano-seconds")

        quac_t2 = quac_t2_fit.time()
        actual_t2 = [hardware_props.t2(qubit) * 1e9 for qubit in qubits]

        print(f"Actual T2 times: {actual_t2} in nanoseconds.")
        print(f"QuaC-calculated T2 times: {quac_t2} in nanoseconds.")
        max_diff = abs((np.array(quac_t2) - np.array(actual_t2)).max())
        print(max_diff)

        self.assertLess(max_diff, 1e4)  # of same magnitude?


if __name__ == '__main__':
    unittest.main()
