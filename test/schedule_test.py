# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring gate scheduling is working properly in the library.
"""
import unittest
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.test.mock import FakeBogota
from quac_qiskit import Quac
from quac_qiskit.simulators import list_schedule_experiment


class ScheduleTestCase(unittest.TestCase):
    """Tests QuaC noise model functionality by recovering model parameters with Qiskit fitters
    """

    def setUp(self):
        # Set up QuaC simulators
        self.quac_sim = Quac.get_backend("fake_yorktown_density_simulator", t1=True, t2=False, meas=False, zz=False)

    def test_list_schedule(self):
        example_circ = QuantumCircuit(5)
        example_circ.h(0)
        example_circ.x(2)
        example_circ.x(0)
        example_circ.cx(0, 2)
        example_circ.y(3)
        example_circ.y(2)
        example_circ.measure_all()

        qobj = assemble(transpile(example_circ, FakeBogota()), backend=FakeBogota())
        list_scheduled_circ = list_schedule_experiment(qobj.experiments[0], FakeBogota().properties())

        expected_gates = ['u3', 'u3', 'u2', 'cx', 'cx', 'cx', 'cx', 'u3', 'barrier', 'measure', 'measure', 'measure',
                          'measure', 'measure']
        expected_times = [1, 1, 1, 72.11111111111111, 513.0, 918.3333333333333, 1359.2222222222222,
                          1693.4444444444443, 1764.5555555555554, 1764.5555555555554, 1764.5555555555554,
                          1764.5555555555554, 1764.5555555555554, 1764.5555555555554]

        index = 0
        for gate, time in list_scheduled_circ:
            self.assertEqual(gate.name, expected_gates[index])
            self.assertEqual(time, expected_times[index])
            index += 1


if __name__ == '__main__':
    unittest.main()
