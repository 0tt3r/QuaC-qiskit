# -*- coding: utf-8 -*-

"""This module contains a QuaC backend for obtaining the frequency results of qobj experiments run
a specified number of times. This simulator is subject to stochastic noise. For comparisons and
benchmarking, the density backend is recommended.
"""

import time
import numpy as np
from collections import defaultdict
from qiskit.assembler import disassemble
from qiskit.result import Result
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.providers.models.backendproperties import BackendProperties
from .quac_simulator import QuacSimulator
from ..utils import choose_index


class QuacCountsSimulator(QuacSimulator):
    """Class for simulating a Qiskit-defined quantum experiment a number of times and returning
    a dictionary of the frequencies of each resulting pure state
    """

    def name(self) -> str:
        """Returns a name for identifying this specific density backend

        :return: a string used to identify this backend
        """
        return self.configuration().backend_name + "_counts_simulator"

    def properties(self) -> BackendProperties:
        """Returns backend properties that reflect the limitations of the hardware if it is
        specified or the QuaC simulator if not

        :return: a Qiskit BackendProperties object
        """
        return self._properties

    def _run_job(self, job_id: str, qobj: QasmQobj, **run_config) -> Result:
        """Specifies how to run a quantum object job on this backend. This is the method that
        changes between types of QuaC backends.

        :param job_id: a uuid4 string to uniquely identify this job
        :param qobj: an assembled quantum object of experiments
        :param run_config: injected parameters
        :return: a Qiskit Result object
        """
        qobj_start = time.perf_counter()
        results = list()

        for experiment in qobj.experiments:
            exp_start = time.perf_counter()
            final_quac_instance, qubit_measurements = super()._run_experiment(experiment, **run_config)

            # Create a frequency defaultdict for multinomial experiment tallying
            frequencies = defaultdict(lambda: 0)

            # Get probabilities of all states occurring and try to adjust them by measurement errors
            bitstring_probs = np.array(final_quac_instance.get_bitstring_probs())

            if self._meas:
                # If measurement error simulation is turned on, adjust probabilities accordingly
                bitstring_probs = np.dot(self._measurement_error_matrix, bitstring_probs)

            for _ in range(0, qobj.config.shots):
                # Run multinomial experiment and filter out unmeasured qubits from results
                outcome_state = bin(choose_index(bitstring_probs))[2:]
                padded_outcome_state = list(outcome_state.zfill(qobj.config.n_qubits))
                classical_register = ["0"] * qobj.config.memory_slots

                for qubit, outcome in enumerate(padded_outcome_state):
                    # Only measure specified qubits into the classical register
                    if qubit in qubit_measurements:
                        for register_slot in qubit_measurements[qubit]:
                            classical_register[register_slot] = outcome

                classical_register.reverse()  # convert to Qiskit MSB format
                classical_register_hex = hex(int(''.join(classical_register), 2))

                frequencies[classical_register_hex] += 1

            results.append({
                "name": experiment.header.name,
                "shots": qobj.config.shots,
                "data": {"counts": dict(frequencies)},
                "status": "DONE",
                "success": True,
                "time_taken": time.perf_counter() - exp_start,
                "header": experiment.header.to_dict()
            })

        job_result = {
            "backend_name": self.name(),
            "backend_version": self.configuration().backend_version,
            "qobj_id": qobj.qobj_id,
            "job_id": job_id,
            "results": results,
            "success": True,
            "time_taken": time.perf_counter() - qobj_start,
            "header": qobj.header.to_dict()
        }

        return Result.from_dict(job_result)
