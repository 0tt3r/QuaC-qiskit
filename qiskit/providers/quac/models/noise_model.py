# -*- coding: utf-8 -*-

"""This module contains functionality for defining noise models with QuaC.
"""
from typing import List, Dict, Tuple, Optional, Union
import warnings
import numpy as np
from scipy import sparse
from qiskit.ignis.characterization import T1Fitter, T2Fitter, ZZFitter
from qiskit.ignis.mitigation import TensoredMeasFitter
from qiskit.result import Result
from qiskit.providers import BaseBackend, BackendPropertyError


class QuacNoiseModel:
    """Defines noise to be applied to QuaC-based simulations
    """

    def __init__(self, t1_times: List[float], t2_times: List[float], meas_matrices: Optional[List[np.array]] = None,
                 zz: Optional[Dict[Tuple[int, int], float]] = None):
        """Constructor for QuaC noise model

        :param t1_times: a list of floats representing T1 relaxation times
        :param t2_times: a list of floats representing T2 (note: not T2*) decoherence times
        :param meas_matrices: a list of 2x2 numpy arrays representing measurement probabilities.
            Given the matrix [[A, B], [C, D]] , A is the probability of measuring a qubit in state 0 given
            it was prepped in state 0, and C is the probability of measuring the qubit in state 1 given it was
        prepped in state 0. Similar logic can be applied to entries B and D
        :param zz: a dictionary mapping ordered pairs of qubits to ZZ coupling frequency in GHz. Please note
        this is in regular frequency, not angular frequency
        """
        self._t1_times = t1_times
        self._t2_times = t2_times
        self._meas_matrices = meas_matrices
        self._full_meas_matrices = []
        self._zz = zz

    def __str__(self):
        string_representation = "Noise Model Description\n=============================="

        if self.has_t1():
            string_representation += "\nT1 times:"
            for qubit, time in enumerate(self._t1_times):
                string_representation += f"\n{qubit}: {time} ns"

        if self.has_t2():
            string_representation += "\nT2 times:"
            for qubit, time in enumerate(self._t2_times):
                string_representation += f"\n{qubit}: {time} ns"

        if self.has_meas():
            string_representation += "\nMeasurement error matrices:"
            for qubit, mat in enumerate(self._meas_matrices):
                string_representation += f"\n{qubit}: {mat}"

        if self.has_zz():
            string_representation += "\nZZ coupling terms:"
            for pair, value in self._zz.items():
                string_representation += f"\n{pair}: {value} GHz"

        return string_representation

    def has_t1(self) -> bool:
        """Check if T1 noise was defined

        :return: a boolean
        """
        return not (float('inf') in self._t1_times and len(set(self._t1_times)) == 1)

    def has_t2(self) -> bool:
        """Check if T2 noise was defined

        :return: a boolean
        """
        return not (float('inf') in self._t2_times and len(set(self._t2_times)) == 1)

    def has_meas(self) -> bool:
        """Check if measurement error was defined

        :return: a boolean
        """
        return self._meas_matrices is not None

    def has_zz(self) -> bool:
        """Check if ZZ coupling noise was defined

        :return: a boolean
        """
        return self._zz is not None

    def t1(self, qubit: int) -> float:
        """T1 getter method

        :param qubit: an integer
        :return: T1 time in nanoseconds
        """
        return self._t1_times[qubit]

    def t2(self, qubit: int) -> float:
        """T2 getter method

        :param qubit: an integer
        :return: T2 time in nanoseconds
        """
        return self._t2_times[qubit]

    def meas(self) -> List[sparse.csr_matrix]:
        """Measurement error matrix getter

        :return: a list of sparse matrices ready for application
        """
        if len(self._full_meas_matrices) == 0:
            self.build_full_measurement_matrices()
        return self._full_meas_matrices

    def flip_prob(self, qubit: int, prep: int, meas: int):
        """The probability a qubit is in state meas after being prepared
        in state prep

        :param qubit: integer
        :param prep: integer (0 or 1)
        :param meas: integer (0 or 1)
        :return: a float
        """
        return self._meas_matrices[qubit][prep][meas]

    def zz(self, qubit1: Optional[int] = None, qubit2: Optional[int] = None) -> Union[List[Tuple[int, int]], float]:
        """ZZ getter method

        :param qubit1: an integer
        :param qubit2: an integer
        :return: ZZ frequency in GHz (or a list of defined qubit pairs if either argument is None)
        """
        if qubit1 is None or qubit2 is None:
            return list(self._zz.keys())
        return self._zz[(qubit1, qubit2)]

    @staticmethod
    def get_noiseless_model(n_qubits: int):
        """Returns a QuacNoiseModel that is effectively noiseless

        :param n_qubits: number of qubits, an integer
        :return: a noiseless QuacNoiseModel
        """
        return QuacNoiseModel([float('inf')] * n_qubits, [float('inf')] * n_qubits)

    @classmethod
    def from_backend(cls, backend: BaseBackend, **kwargs):
        """Automatically loads a QuaC noise model given a backend of type BaseBackend. Primarily
        for speeding up definition of IBMQ hardware noise models in QuaC

        :param backend: an object of type BaseBackend
        :param kwargs: an optional dictionary mapping strings to booleans stating which types
            of noise to include (keys are t1, t2, meas, and zz)
        :return: a QuacNoiseModel object
        """
        n_qubits = len(backend.properties().qubits)
        qubits = list(range(n_qubits))

        # Set up defaults
        t1_times = [float('inf') for _ in range(n_qubits)]
        t2_times = [float('inf') for _ in range(n_qubits)]
        meas_matrices = None
        zz = None

        # Adjust defaults as appropriate
        if kwargs.get("t1"):
            t1_times = [backend.properties().t1(qubit) * 1e9 for qubit in qubits]
        if kwargs.get("t2"):
            t2_times = [backend.properties().t2(qubit) * 1e9 for qubit in qubits]
        if kwargs.get("meas"):
            meas_matrices = []
            # Construct probability matrix for measurement error adjustments
            for qubit in range(n_qubits):
                # Not all backends have measurement errors added
                try:
                    prob_meas0_prep1 = backend.properties().qubit_property(qubit, "prob_meas0_prep1")[0]
                    prob_meas1_prep0 = backend.properties().qubit_property(qubit, "prob_meas1_prep0")[0]
                except BackendPropertyError:
                    warnings.warn("Measurement error simulation not supported on this backend")
                    break

                qubit_measurement_error_matrix = np.array([
                    [1 - prob_meas1_prep0, prob_meas0_prep1],
                    [prob_meas1_prep0, 1 - prob_meas0_prep1]
                ])

                meas_matrices.append(qubit_measurement_error_matrix)
        if kwargs.get("zz"):
            warnings.warn("ZZ coupling not supported in automatic loading")

        return QuacNoiseModel(t1_times, t2_times, meas_matrices, zz)

    @classmethod
    def from_calibration_results(cls, backend: BaseBackend, t1_result: Tuple[np.array, Result],
                                 t2_result: Tuple[np.array, Result], meas_result: Result,
                                 zz_results: Dict[Tuple[int, int], Tuple[np.array, float, Result]]):
        """Takes results from running calibration circuits on hardware and constructs a
        QuacNoiseModel object

        :param backend: the backend on which the circuits were run (a BaseBackend object)
        :param t1_result: a tuple with a list of delay times (in ns) as the 0th element and the T1
            calibration Result object as the 1st element
        :param t2_result: a tuple with a list of delay times (in ns) as the 0th element and the T2
            calibration Result object as the 1st element
        :param meas_result: a Result object from running measurement calibration circuits
        :param zz_results: a dictionary mapping tuples of qubit indices to a ZZ coupling calibration circuit
            Result object
        :return: a QuacNoiseModel object
        """
        n_qubits = len(backend.properties().qubits)
        qubits = list(range(n_qubits))

        # Set up defaults
        t1_times = [float('inf') for _ in range(n_qubits)]
        t2_times = [float('inf') for _ in range(n_qubits)]
        meas_matrices = None
        zz = None

        # Adjust defaults as appropriate
        if t1_result:
            t1_fit = T1Fitter(t1_result[1], t1_result[0], qubits,
                              fit_p0=[1, 1e5, 0],
                              fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                              time_unit="nano-seconds")
            t1_times = t1_fit.time()
        if t2_result:
            t2_fit = T2Fitter(t2_result[1], t2_result[0], qubits,
                              fit_p0=[1, 1e4, 0],
                              fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                              time_unit="nano-seconds")
            t2_times = t2_fit.time()
        if meas_result:
            meas_fit = TensoredMeasFitter(meas_result, [[qubit] for qubit in qubits])
            meas_matrices = meas_fit.cal_matrices
        if zz_results:
            zz = {}
            for qubit1 in qubits:
                for qubit2 in qubits:
                    if qubit1 < qubit2:
                        zz_information = zz_results[(qubit1, qubit2)]
                        xdata, osc_freq, zz_result = zz_information
                        zz_fit = ZZFitter(zz_result, xdata, [qubit1], [qubit2],
                                          fit_p0=[1, osc_freq, -np.pi / 20, 0],
                                          fit_bounds=([-0.5, 0, -np.pi, -0.5],
                                                      [1.5, 1e10, np.pi, 1.5]),
                                          )
                        zz[(qubit1, qubit2)] = zz_fit.ZZ_rate()[0]

        return QuacNoiseModel(t1_times, t2_times, meas_matrices, zz)

    @classmethod
    def from_array(cls, array: np.array, n_qubits: int):
        """Convert an array to a QuacNoiseModel. Array must contain T1 and T2 times at a
        minimum

        :param array: a Numpy array
        :param n_qubits: the number of qubits simulated
        :return: a QuacNoiseModel object
        """
        list_array = list(100000 * array)

        # T1 and T2 times
        t1_times = list_array[:n_qubits]
        t2_times = list_array[n_qubits:2 * n_qubits]

        if len(array) == 2 * n_qubits:
            return QuacNoiseModel(t1_times, t2_times, None, None)

        # Measurement error
        meas_diagonals = list_array[2 * n_qubits:4 * n_qubits]
        meas_matrices = []
        for qubit in range(n_qubits):
            diagonal = meas_diagonals[2 * qubit:2 * qubit + 2]
            meas_matrices.append(np.array([
                [diagonal[0], 1 - diagonal[1]],
                [1 - diagonal[0], diagonal[1]]
            ]))

        if len(array) == 4 * n_qubits:
            return QuacNoiseModel(t1_times, t2_times, meas_matrices, None)

        # ZZ coupling error
        zz_compressed = list_array[4 * n_qubits:]
        zz = {}
        zz_ind = 0
        for qubit1 in range(n_qubits):
            for qubit2 in range(n_qubits):
                if qubit1 < qubit2:
                    zz[(qubit1, qubit2)] = zz_compressed[zz_ind]
                    zz_ind += 1

        return QuacNoiseModel(t1_times, t2_times, meas_matrices, zz)

    def to_array(self) -> np.array:
        """Converts a QuacNoiseModel object to an array. Especially useful for optimization

        :return: a Numpy array
        """
        n_qubits = len(self._t1_times)

        # Add T1 and T2 floats
        list_form = self._t1_times + self._t2_times

        # Add diagonal elements of 2x2 measurement matrices in qubit order
        if self.has_meas():
            for meas_matrix in self._meas_matrices:
                list_form += list(meas_matrix.diagonal())

        if self.has_zz():
            # Add zz coupling in order (0, 1), (0, 2) ...
            for qubit1 in range(n_qubits):
                for qubit2 in range(n_qubits):
                    if qubit1 < qubit2:
                        list_form.append(self._zz[(qubit1, qubit2)])

        return np.array(list_form) / 100000

    def build_full_measurement_matrices(self):
        """Uses Kronecker product on 2x2 measurement matrices to compose a list of matrices that,
        when all applied to a QuaC bitstring probability vector, result in a new bitstring
        probability vector adjusted for measurement noise
        """
        n_qubits = len(self._t1_times)
        full_meas_matrices = []

        if self._meas_matrices is None:
            return [sparse.eye(n_qubits) for _ in range(n_qubits)]

        for qubit in range(n_qubits):
            expanded_qubit_meas_mat = sparse.csr_matrix(np.array([1]))
            for ind in range(n_qubits):
                if qubit == ind:
                    expanded_qubit_meas_mat = sparse.kron(expanded_qubit_meas_mat,
                                                          self._meas_matrices[qubit],
                                                          format='csr')
                else:
                    expanded_qubit_meas_mat = sparse.kron(expanded_qubit_meas_mat,
                                                          sparse.eye(2),
                                                          format='csr')
            full_meas_matrices.append(expanded_qubit_meas_mat)

        self._full_meas_matrices = full_meas_matrices
