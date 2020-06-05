# -*- coding: utf-8 -*-

"""
This module contains test cases for ensuring gate addition and functionality is working properly
in the library.
"""

import unittest
import math
from qiskit import QuantumCircuit, execute
from qiskit.providers.quac import Quac


class GateTestCase(unittest.TestCase):
    """Tests gate functionality
    """

    def setUp(self):
        # Set up generic multinomial experiment QuaC simulator
        self.simulator = Quac.get_backend("generic_counts_simulator")
        # Define Lindblad emission and dephasing noise (in nanoseconds)
        self.lindblad_noise = {
            "0": {
                "T1": 60000,
                "T2": 50000
            },
            "1": {
                "T1": 60000,
                "T2": 50000
            }
        }

    def test_hadamard_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.h(0)
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        outcome_vals = list(outcome_dist.values())
        self.assertLessEqual(abs(outcome_vals[0] - outcome_vals[1]), 200)

    def test_id_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.id(0)
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        self.assertEqual(outcome_dist["0"], 1000)

    def test_x_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.x(0)
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        self.assertEqual(outcome_dist["1"], 1000)

    def test_y_gate(self):
        pass

    def test_z_gate(self):
        pass

    def test_rx_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.rx(math.pi, 0)
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        self.assertEqual(outcome_dist["1"], 1000)

        test_circuit = QuantumCircuit(1, 1)
        test_circuit.rx(math.pi / 2, 0)  # Hadamard gate equivalent
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        outcome_vals = list(outcome_dist.values())
        self.assertLessEqual(abs(outcome_vals[0] - outcome_vals[1]), 200)

    def test_ry_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.ry(math.pi, 0)
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        self.assertEqual(outcome_dist["1"], 1000)

        test_circuit = QuantumCircuit(1, 1)
        test_circuit.ry(math.pi/2, 0)  # Hadamard gate equivalent
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        outcome_vals = list(outcome_dist.values())
        self.assertLessEqual(abs(outcome_vals[0] - outcome_vals[1]), 200)

    def test_rz_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.rz(math.pi, 0)
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        self.assertEqual(outcome_dist["0"], 1000)  # should leave state alone

        test_circuit = QuantumCircuit(1, 1)
        test_circuit.h(0)
        test_circuit.rz(math.pi, 0)  # should not affect probabilities
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        outcome_vals = list(outcome_dist.values())
        self.assertLessEqual(abs(outcome_vals[0] - outcome_vals[1]), 200)

    def test_u1_gate(self):
        pass

    def test_u2_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.u2(0, math.pi, 0)  # behaves like a Hadamard gate
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        outcome_vals = list(outcome_dist.values())
        self.assertLessEqual(abs(outcome_vals[0] - outcome_vals[1]), 200)

    def test_u3_gate(self):
        test_circuit = QuantumCircuit(1, 1)
        test_circuit.u3(math.pi/2, 0, math.pi, 0)  # behaves like a Hadamard gate
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)
        outcome_vals = list(outcome_dist.values())
        self.assertLessEqual(abs(outcome_vals[0] - outcome_vals[1]), 200)

    def test_cnot_gate(self):
        test_circuit = QuantumCircuit(2, 2)
        test_circuit.h(0)
        test_circuit.cx(0, 1)
        outcome_dist = execute(test_circuit, self.simulator,
                               shots=1000, lindblad=self.lindblad_noise)

        self.assertEqual(outcome_dist["00"] + outcome_dist["11"], 1000)

    def test_cz_gate(self):
        pass


if __name__ == '__main__':
    unittest.main()
