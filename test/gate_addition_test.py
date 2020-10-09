# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring gate addition and functionality is working properly
in the library.
"""
import unittest
from qiskit import execute, Aer
from qiskit.circuit.random import random_circuit
from quac_qiskit import Quac
from quac_qiskit.format import *
from quac_qiskit.models import QuacNoiseModel
from quac_qiskit.stat import *


class GateAdditionTestCase(unittest.TestCase):
    """Tests gate functionality
    """

    def setUp(self):
        # Set up QuaC simulator and QASM simulator
        self.quac_simulator = Quac.get_backend("fake_yorktown_density_simulator")
        self.qasm_simulator = Aer.get_backend("statevector_simulator")

    def test_hadamard_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.h(0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0.5, 0.5]), 5)

    def test_id_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.id(0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [1, 0]), 5)

    def test_x_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.x(0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0, 1]), 5)

    def test_y_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.y(0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0, 1]), 5)

        test_circuit = QuantumCircuit(1)
        test_circuit.x(0)
        test_circuit.y(0)
        test_circuit.y(0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0, 1]), 5)

    def test_z_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.z(0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [1, 0]), 5)

    def test_rx_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.rx(np.pi, 0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0, 1]), 5)

        test_circuit = QuantumCircuit(1)
        test_circuit.rx(np.pi / 2, 0)  # Hadamard gate equivalent
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0.5, 0.5]), 5)

    def test_ry_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.ry(np.pi, 0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0, 1]), 5)

        test_circuit = QuantumCircuit(1)
        test_circuit.ry(np.pi / 2, 0)  # Hadamard gate equivalent
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [0.5, 0.5]), 5)

    def test_rz_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.rz(np.pi, 0)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLess(get_vec_angle(outcome_list, [1, 0]), 5)  # should leave state alone

        test_circuit = QuantumCircuit(1)
        test_circuit.h(0)
        test_circuit.rz(np.pi, 0)  # should not affect probabilities
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLess(get_vec_angle(outcome_list, [0.5, 0.5]), 5)

    def test_u1_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.u1(math.pi, 0)  # same as Z gate
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLessEqual(get_vec_angle(outcome_list, [1, 0]), 5)

    def test_u2_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.u2(0, np.pi, 0)  # behaves like a Hadamard gate
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLess(get_vec_angle(outcome_list, [0.5, 0.5]), 5)

    def test_u3_gate(self):
        test_circuit = QuantumCircuit(1)
        test_circuit.u3(np.pi / 2, 0, np.pi, 0)  # behaves like a Hadamard gate
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())
        self.assertLess(get_vec_angle(outcome_list, [0.5, 0.5]), 5)

    def test_cnot_gate(self):
        test_circuit = QuantumCircuit(2)
        test_circuit.h(0)
        test_circuit.cx(0, 1)
        test_circuit.measure_all()
        outcome_dist = execute(test_circuit, self.quac_simulator, shots=1000)
        outcome_list = counts_to_list(outcome_dist.result().get_counts())

        self.assertLess(get_vec_angle(outcome_list, [0.5, 0, 0, 0.5]), 5)

    def test_random_circuits(self):
        for circuit_index in range(1000):
            num_qubits = random.randrange(1, 6)

            # Generate random circuit and transpile it to run on specific hardware
            random_circ = transpile(random_circuit(num_qubits, 5, measure=False), self.quac_simulator)
            random_circ.measure_all()

            # Get QuaC-calculated probabilities in a list
            plugin_probs = counts_to_dist(
                execute(random_circ, self.quac_simulator, shots=1, optimization_level=0,
                        quac_noise_model=QuacNoiseModel.get_noiseless_model(5)).result().get_counts()
            )

            # Get Qiskit-calculated probabilities in a list
            random_circ.remove_final_measurements()  # avoid collapsing state vector
            qiskit_sv = execute(random_circ, self.qasm_simulator, shots=8000,
                                optimization_level=0,
                                ).result().get_statevector(random_circ)

            # qiskit_probs = qiskit_statevector_to_probabilities(qiskit_sv, 5)
            qiskit_probs = statevector_to_probabilities(qiskit_sv)

            # Calculate divergence of Qiskit and QuaC predictions
            difference_angle = get_vec_angle(qiskit_probs, plugin_probs)
            self.assertLess(difference_angle, 1e-5)


if __name__ == '__main__':
    unittest.main()
