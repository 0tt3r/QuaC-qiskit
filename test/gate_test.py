# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring gate addition and functionality is working properly
in the library.
"""

import unittest
import random
import csv
import os
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, transpile
from qiskit.test.mock import FakeYorktown
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.random import random_circuit
from qiskit.providers.quac import Quac
from qiskit.providers.quac.utils import *


class GateTestCase(unittest.TestCase):
    """Tests gate functionality
    """
    setup_done = False

    def setUp(self):
        # Remove existing files in output directory
        if not self.setup_done:
            # Set up is called for every test
            for filename in os.listdir('./output'):
                os.remove(f'./output/{filename}')
            GateTestCase.setup_done = True

        # Set up QuaC simulator and QASM simulator
        self.quac_simulator = Quac.get_backend("fake_yorktown_density_simulator", meas=True)
        self.qasm_simulator = Aer.get_backend("qasm_simulator")

        # Set up QASM noise model
        self.qasm_noise_model = NoiseModel.from_backend(FakeYorktown())

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
        pass

    def test_z_gate(self):
        pass

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
        pass

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

    # @unittest.skip("Skipping full random circuits...")
    def test_random_circuits(self):
        log = open('./output/log.txt', 'w')
        csvfile = open('./output/summary.csv', 'w')
        csvwriter = csv.DictWriter(csvfile, fieldnames=['circ_ind',
                                                        'qasm', 'plugin',
                                                        'quac', 'qasm_plugin_match'])
        csvwriter.writeheader()

        qasm_plugin_diffs = []

        for circuit_index in range(100):
            num_qubits = random.randrange(1, 6)

            # Generate random circuit and transpile it to run on specific hardware
            random_circ = random_circuit(num_qubits, 2, measure=False)
            transpiled_random_circ = transpile(random_circ,
                                               backend=self.quac_simulator,
                                               optimization_level=0)
            transpiled_random_circ.measure_all()

            # Record circuits as images and OpenQASM files
            with open(f"./output/{circuit_index}.qasm", "w") as qasm:
                qasm.write(quac_qasm_transpiler(transpiled_random_circ.qasm()))
            with open(f"./output/transpiled_{circuit_index}.qasm", "w") as qasm_original:
                qasm_original.write(transpiled_random_circ.qasm())
            with open(f"./output/untranspiled_{circuit_index}.qasm", "w") as qasm_original:
                qasm_original.write(random_circ.qasm())

            random_circ.draw(output="mpl", filename=f"./output/original_{circuit_index}.png")
            transpiled_random_circ.draw(output="mpl", filename=f"./output/{circuit_index}.png")

            # Get QuaC-calculated probabilities in a list
            plugin_probs = counts_to_list(
                execute(transpiled_random_circ, self.quac_simulator, shots=1,
                        optimization_level=0).result().get_counts()
            )

            # Get Qiskit-calculated probabilities in a list
            qiskit_counts = execute(transpiled_random_circ, self.qasm_simulator,
                                    shots=1000000,
                                    optimization_level=0,
                                    noise_model=self.qasm_noise_model
                                    ).result().get_counts()
            qiskit_counts_list = counts_to_list(qiskit_counts)
            qiskit_probs = [count / 1000000 for count in qiskit_counts_list]

            # Calculate divergence of Qiskit and QuaC predictions
            difference_angle = get_vec_angle(qiskit_probs, plugin_probs)
            qasm_plugin_diffs.append(difference_angle)

            # Record results in a CSV file
            log.write(f"Index: {circuit_index}\n")
            log.write(f"Number of qubits: {num_qubits}\n")
            log.write(f"Angle difference: {difference_angle}\n")
            log.write(f"QASM Probs: {qiskit_probs}\n")
            log.write(f"QuaC Probs: {plugin_probs}\n")
            log.write(f"======================================\n")

            csv_row = {
                'circ_ind': circuit_index,
                'qasm': qiskit_probs,
                'plugin': plugin_probs,
                'quac': [],
                'qasm_plugin_match': difference_angle < 7
            }
            csvwriter.writerow(csv_row)

        log.close()
        csvfile.close()

        avg_diff = sum(qasm_plugin_diffs) / len(qasm_plugin_diffs)
        self.assertLess(avg_diff, 10)


if __name__ == '__main__':
    unittest.main()
