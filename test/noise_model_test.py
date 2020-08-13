# -*- coding: utf-8 -*-

"""This module contains test cases for ensuring gate addition and functionality is working properly
in the library.
"""
import random
import unittest
import numpy as np
from qiskit import execute, QuantumCircuit, Aer, transpile
from qiskit.circuit.random import random_circuit
from qiskit.ignis.characterization import t1_circuits, T1Fitter, t2_circuits, T2Fitter, zz_circuits, ZZFitter
from qiskit.ignis.mitigation import TensoredMeasFitter
from qiskit.test.mock import FakeYorktown
from qiskit.providers.quac import Quac
from qiskit.providers.quac.models import QuacNoiseModel


class NoiseModelTestCase(unittest.TestCase):
    """Tests QuaC noise model functionality by recovering model parameters with Qiskit fitters
    """

    def setUp(self):
        # Set up QuaC simulators
        self.quac_sim = Quac.get_backend("fake_yorktown_density_simulator", t1=True, t2=False, meas=False, zz=False)
        self.qasm_sim = Aer.get_backend("qasm_simulator")

    def test_t1_recovery(self):
        cal_circs, t1_delay = t1_circuits(np.linspace(10, 900, 10, dtype='int'),
                                          FakeYorktown().properties().gate_length('id', [0]) * 1e9,
                                          [0, 1, 2, 3, 4])

        true_t1 = [1000 * (1 + random.random()) for _ in range(5)]
        t1_noise_model = QuacNoiseModel(
            true_t1,
            [float('inf') for _ in range(5)]
        )

        t1_result = execute(cal_circs, self.quac_sim, quac_noise_model=t1_noise_model).result()

        t1_fit = T1Fitter(t1_result, t1_delay, [0, 1, 2, 3, 4],
                          fit_p0=[1, 1000, 0],
                          fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                          time_unit="nano-seconds")
        derived_t1 = t1_fit.time()

        max_diff = abs(np.array(derived_t1) - np.array(true_t1)).max()
        self.assertLess(max_diff, 1e-3)

    def test_t2_recovery(self):
        cal_circs, t2_delay = t2_circuits(np.floor(np.linspace(10, 900, 10, dtype='int') / 2).astype('int'),
                                          FakeYorktown().properties().gate_length('id', [0]) * 1e9,
                                          [0, 1, 2, 3, 4])

        true_t2 = [10000 * (1 + random.random()) for _ in range(5)]
        t2_noise_model = QuacNoiseModel(
            [float('inf') for _ in range(5)],
            true_t2
        )

        t2_result = execute(cal_circs, self.quac_sim, quac_noise_model=t2_noise_model).result()

        t2_fit = T2Fitter(t2_result, t2_delay, [0, 1, 2, 3, 4],
                          fit_p0=[1, 10000, 0],
                          fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                          time_unit="nano-seconds")
        derived_t2 = t2_fit.time()

        max_diff = abs(np.array(derived_t2) - np.array(true_t2)).max()
        self.assertLess(max_diff, 1e-3)

    def test_meas_recovery(self):
        qubits = list(range(5))  # we are working with a 5-qubit machine

        # Develop random measurement matrices for each qubit of the form
        # [ P(0|0)     P(0|1) ]
        # [ P(1|0)     P(1|1) ]
        meas_mats = []
        for _ in qubits:
            stay_zero = random.random()
            stay_one = random.random()
            meas_mats.append(
                np.array([
                    [stay_zero, 1 - stay_one],
                    [1 - stay_zero, stay_one]
                ])
            )

        # Encode measurement matrices in QuacNoiseModel object
        meas_noise_model = QuacNoiseModel(
            [float('inf') for _ in qubits],
            [float('inf') for _ in qubits],
            meas_mats
        )

        # Develop Quac-specfic measurement matrix calibration circuits
        # (cannot run circuits with 0 gates in Quac backends)
        empty_circ = QuantumCircuit(len(qubits))
        for i in qubits:
            empty_circ.id(i)
        empty_circ.measure_all()
        empty_circ.name = f"cal_{''.join(['0' for _ in qubits])}"

        full_circ = QuantumCircuit(len(qubits))
        for i in qubits:
            full_circ.x(i)
        full_circ.measure_all()
        full_circ.name = f"cal_{''.join(['1' for _ in qubits])}"

        meas_cal_circs = [empty_circ, full_circ]

        # Recover measurement matrices with Qiskit fitter
        meas_result = execute(meas_cal_circs, self.quac_sim, quac_noise_model=meas_noise_model).result()
        meas_fit = TensoredMeasFitter(meas_result, [[i] for i in qubits])
        meas_matrices = meas_fit.cal_matrices

        max_diff = max([(meas_matrices[ind] - meas_mats[ind]).max() for ind in qubits])
        self.assertLess(max_diff, 1e-10)

    def test_zz_recovery(self):
        qubits = list(range(5))
        num_of_gates = np.arange(0, 600, 30)
        gate_time = FakeYorktown().properties().gate_length('id', [0]) * 1e9

        # Develop zz-only noise model
        zz_dict = {}
        for qubit1 in qubits:
            for qubit2 in qubits:
                if qubit1 < qubit2:
                    zz_dict[(qubit1, qubit2)] = (1 + random.random() * 2) * 1e-5

        zz_noise_model = QuacNoiseModel(
            [float('inf') for _ in qubits],
            [float('inf') for _ in qubits],
            [np.eye(2) for _ in qubits],
            zz_dict
        )

        for qubit1 in qubits:
            for qubit2 in qubits:
                if qubit1 < qubit2:
                    print(qubit1)
                    print(qubit2)
                    zz_circs, xdata, osc_freq = zz_circuits(num_of_gates, gate_time, [qubit1], [qubit2], nosc=2)

                    zz_result = execute(zz_circs, self.quac_sim, quac_noise_model=zz_noise_model,
                                        shots=1).result()

                    fit = ZZFitter(zz_result, xdata, [qubit1], [qubit2],
                                   fit_p0=[1, osc_freq, -np.pi / 20, 0],
                                   fit_bounds=([-0.5, 0, -np.pi, -0.5],
                                               [1.5, 1e10, np.pi, 1.5]),
                                   time_unit="nano-seconds"
                                   )

                    self.assertLess(abs(abs(fit.ZZ_rate()[0]) - zz_dict[(qubit1, qubit2)]), 1e-10)

    def test_noise_model_from_backend(self):
        yorktown_quac_noise_model_t1t2 = QuacNoiseModel.from_backend(FakeYorktown(), t1=True, t2=True,
                                                                     meas=False, zz=False)
        yorktown_quac_noise_model_meas = QuacNoiseModel.from_backend(FakeYorktown(), t1=False, t2=False,
                                                                     meas=True, zz=False)

        qubits = list(range(5))

        cal_circs, t1_delay = t1_circuits(np.linspace(10, 900, 10, dtype='int'),
                                          FakeYorktown().properties().gate_length('id', [0]) * 1e9,
                                          qubits)

        true_t1 = [FakeYorktown().properties().t1(q) * 1e9 for q in qubits]

        t1_result = execute(cal_circs, self.quac_sim, quac_noise_model=yorktown_quac_noise_model_t1t2,
                            optimization_level=0).result()

        t1_fit = T1Fitter(t1_result, t1_delay, qubits,
                          fit_p0=[1, 10000, 0],
                          fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                          time_unit="nano-seconds")
        derived_t1 = t1_fit.time()
        max_diff = abs(np.array(derived_t1) - np.array(true_t1)).max()

        self.assertLess(max_diff, 1e-2)

        # Test T2 recovery
        cal_circs, t2_delay = t2_circuits(np.floor(np.linspace(10, 900, 10, dtype='int') / 2).astype('int'),
                                          FakeYorktown().properties().gate_length('id', [0]) * 1e9,
                                          qubits)

        true_t2 = [FakeYorktown().properties().t2(q) * 1e9 for q in qubits]

        t2_result = execute(cal_circs, self.quac_sim, quac_noise_model=yorktown_quac_noise_model_t1t2).result()

        t2_fit = T2Fitter(t2_result, t2_delay, qubits,
                          fit_p0=[1, 10000, 0],
                          fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                          time_unit="nano-seconds")
        derived_t2 = t2_fit.time()
        max_diff = abs(np.array(derived_t2) - np.array(true_t2)).max()
        self.assertLess(max_diff, 1e-2)

        # Recover measurement error
        meas_mats = []
        for q in qubits:
            flip_1_to_0 = FakeYorktown().properties().qubit_property(q, "prob_meas0_prep1")[0]
            flip_0_to_1 = FakeYorktown().properties().qubit_property(q, "prob_meas1_prep0")[0]
            meas_mats.append(
                np.array([
                    [1 - flip_0_to_1, flip_1_to_0],
                    [flip_0_to_1, 1 - flip_1_to_0]
                ])
            )

        # Develop Quac-specfic measurement matrix calibration circuits
        # (cannot run circuits with 0 gates in Quac backends)
        empty_circ = QuantumCircuit(len(qubits))
        for i in qubits:
            empty_circ.id(i)
        empty_circ.measure_all()
        empty_circ.name = f"cal_{''.join(['0' for _ in qubits])}"

        full_circ = QuantumCircuit(len(qubits))
        for i in qubits:
            full_circ.x(i)
        full_circ.measure_all()
        full_circ.name = f"cal_{''.join(['1' for _ in qubits])}"

        meas_cal_circs = [empty_circ, full_circ]

        # Recover measurement matrices with Qiskit fitter
        meas_result = execute(meas_cal_circs, self.quac_sim, quac_noise_model=yorktown_quac_noise_model_meas).result()
        meas_fit = TensoredMeasFitter(meas_result, [[i] for i in qubits])
        meas_matrices = meas_fit.cal_matrices

        max_diff = max([(meas_matrices[ind] - meas_mats[ind]).max() for ind in qubits])
        self.assertLess(max_diff, 1e-10)

    def test_noise_model_to_and_from_array(self):
        random_circs = [random_circuit(5, 5, measure=True) for _ in range(100)]
        true_t1 = [1000 * (1 + random.random()) for _ in range(5)]
        true_t2 = [10000 * (1 + random.random()) for _ in range(5)]

        # Develop random measurement matrices for each qubit of the form
        # [ P(0|0)     P(0|1) ]
        # [ P(1|0)     P(1|1) ]
        meas_mats = []
        for _ in range(5):
            stay_zero = random.random()
            stay_one = random.random()
            meas_mats.append(
                np.array([
                    [stay_zero, 1 - stay_one],
                    [1 - stay_zero, stay_one]
                ])
            )

        zz = {}
        for qubit1 in range(5):
            for qubit2 in range(5):
                if qubit1 != qubit2:
                    zz[(qubit1, qubit2)] = random.randrange(1, 10) * 1e-5

        random_quac_noise_model = QuacNoiseModel(
            true_t1,
            true_t2,
            [meas_mats[0] for _ in range(5)],
            None
        )

        recovered_quac_noise_model = QuacNoiseModel.from_array(random_quac_noise_model.to_array(), 5)

        for circ in random_circs:
            transpiled_circ = transpile(circ, self.quac_sim, optimization_level=0)
            dist_original = execute(transpiled_circ, self.quac_sim,
                                    quac_noise_model=random_quac_noise_model,
                                    optimization_level=0).result().get_counts()
            dist_recovered = execute(transpiled_circ, self.quac_sim,
                                     quac_noise_model=recovered_quac_noise_model,
                                     optimization_level=0).result().get_counts()
            self.assertEqual(dist_original, dist_recovered)


if __name__ == '__main__':
    unittest.main()
