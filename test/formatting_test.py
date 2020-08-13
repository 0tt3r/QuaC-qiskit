# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring formatting functionality is working properly
in the library.
"""
import unittest
from qiskit import execute, Aer
from qiskit.providers.quac import Quac
from qiskit.providers.quac.format import *


class FormattingTestCase(unittest.TestCase):
    """Tests gate functionality
    """

    def setUp(self):
        # Set up QuaC simulator and QASM simulator
        self.quac_simulator = Quac.get_backend("fake_yorktown_density_simulator")
        self.sv_simulator = Aer.get_backend("statevector_simulator")

    def test_counts_to_list(self):
        counts = {
            "00": 3,
            "01": 5,
            "10": 12,
            "11": 18
        }
        self.assertEqual(counts_to_list(counts), [3, 5, 12, 18])

        counts = {
            "01": 5,
            "10": 12,
            "11": 18
        }
        self.assertEqual(counts_to_list(counts), [0, 5, 12, 18])

        counts = {
            "00 00": 3,
            "10 01": 5,
            "01 10": 12,
            "11 11": 18
        }
        self.assertEqual(counts_to_list(counts), [3, 0, 0, 0, 0, 0, 12, 0, 0, 5, 0, 0, 0, 0, 0, 18])

    def test_counts_to_dist(self):
        counts = {
            "00": 3,
            "01": 5,
            "10": 12,
            "11": 18
        }
        self.assertEqual(counts_to_dist(counts), [3 / 38, 5 / 38, 12 / 38, 18 / 38])

        counts = {
            "01": 5,
            "10": 12,
            "11": 18
        }
        self.assertEqual(counts_to_dist(counts), [0, 5 / 35, 12 / 35, 18 / 35])

    def test_statevector_to_prob(self):
        for index in range(32):
            x_setup = list(bin(index)[2:].zfill(5))
            x_setup.reverse()
            x_qubits = [q_ind for q_ind in range(5) if x_setup[q_ind] == '1']

            qc = QuantumCircuit(5)
            for q_ind in x_qubits:
                qc.x(q_ind)
            qc.measure_all()

            qc_statevector = execute(qc, self.sv_simulator).result().get_statevector(qc)
            prob_list = statevector_to_probabilities(qc_statevector)

            self.assertEqual(prob_list, [int(element == index) for element in range(32)])

    def test_aggregate_counts(self):
        counts1 = {"00": 10, "10": 33}
        counts2 = {"01": 87, "00": 20}
        counts3 = {"00": 1, "11": 43}
        all_counts = [counts1, counts2, counts3]
        keys = ["00", "01", "10", "11"]

        agg_counts = aggregate_counts_results(all_counts, keys)
        self.assertEqual(agg_counts, {"00": 31, "01": 87, "10": 33, "11": 43})


if __name__ == '__main__':
    unittest.main()
