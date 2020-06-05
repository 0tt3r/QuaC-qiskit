# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring gate addition and functionality is working properly
in the library.
"""

import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.characterization.coherence import *
from qiskit.ignis.characterization.coherence import T1Fitter
from qiskit.providers.quac import Quac


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

    def test_qiskit_t1(self):
        """Test Qiskit AER T1 approximations
        """
        hardware_props = self.quac_counts_backend.properties()

        num_gates = np.linspace(10, 1e4, self.num_samples, dtype='int')
        qubits = list(range(len(hardware_props.qubits)))

        id_gate_time = hardware_props.gate_length('id', 0) * 1e9
        hardware_t1 = [hardware_props.t1(qubit) * 1e9 for qubit in qubits]

        t1_circs, t1_delay = t1_circuits(num_gates, id_gate_time, qubits)

        # Set up simulated noise model for this qubit in Qiskit based on actual hardware
        qubit_noise_model = NoiseModel()
        for qubit in qubits:
            qubit_noise_model.add_quantum_error(thermal_relaxation_error(
                hardware_t1[qubit], 2 * hardware_t1[qubit], id_gate_time), 'id', [qubit])

        qiskit_start = time.perf_counter()
        qiskit_t1_backend_result = execute(t1_circs, self.qiskit_qasm_backend,
                                           shots=self.shots,
                                           noise_model=qubit_noise_model,
                                           optimization_level=0).result()
        print(f"Qiskit took {time.perf_counter() - qiskit_start} seconds.")

        qiskit_t1_fit = T1Fitter(qiskit_t1_backend_result, t1_delay, qubits,
                                 fit_p0=[1, 10000, 0],
                                 fit_bounds=([0, 0, -1], [2, 1e9, 1]),
                                 time_unit="nano-seconds")

        print(f"Actual T1 times: {hardware_t1} in nanoseconds.")
        print(f"Qiskit-calculated T1 times: {qiskit_t1_fit.time()} nanoseconds.")

        max_diff = abs((np.array(qiskit_t1_fit.time()) - np.array(hardware_t1)).max())
        self.assertLess(max_diff, 1e3)

    def test_quac_t1(self):
        """Test QuaC T1 approximations
        """
        hardware_props = self.quac_counts_backend.properties()

        num_gates = np.linspace(10, 1e4, self.num_samples, dtype='int')

        actual_t1 = list()
        quac_t1 = list()

        qubits = list(range(5))

        for qubit in qubits:
            id_gate_time = hardware_props.gate_length('id', qubit) * 1e9
            hardware_t1 = hardware_props.t1(qubit) * 1e9
            actual_t1.append(hardware_t1)

            t1_circs, t1_delay = t1_circuits(num_gates, id_gate_time, [qubit])

            quac_start = time.perf_counter()
            quac_t1_backend_result = execute(t1_circs, self.quac_counts_backend,
                                             shots=self.shots,
                                             optimization_level=0).result()
            print(f"QuaC took {time.perf_counter() - quac_start} seconds.")

            quac_t1_fit = T1Fitter(quac_t1_backend_result, t1_delay, [qubit],
                                   fit_p0=[1, hardware_t1 / 3, 0],
                                   fit_bounds=([0, 0, -1], [2, hardware_t1, 1]),
                                   time_unit="nano-seconds")

            quac_t1.append(quac_t1_fit.time()[0])

        print(f"Actual T1 times: {actual_t1} in nanoseconds.")
        print(f"QuaC-calculated T1 times: {quac_t1} in nanoseconds.")

        max_diff = abs((np.array(quac_t1) - np.array(actual_t1)).max())
        print(max_diff)
        self.assertLess(max_diff, 1e3)


if __name__ == '__main__':
    unittest.main()
