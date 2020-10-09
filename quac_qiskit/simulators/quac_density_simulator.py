# -*- coding: utf-8 -*-

"""This module contains backend functionality for obtaining the density matrix diagonal from QuaC
simulations of a Qiskit-defined quantum circuit. Functionality is located in the
QuacDensitySimulator class.
"""
import time
import numpy as np
from scipy import sparse
from collections import defaultdict
from qiskit.result import Result
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.providers.models.backendproperties import BackendProperties
from quac_qiskit.simulators import QuacSimulator


class QuacDensitySimulator(QuacSimulator):
    """Class for simulating a Qiskit-defined quantum experiment and computing the diagonal of its
    density matrix
    """

    def name(self) -> str:
        """Returns a name for identifying this specific density backend

        :return: a string used to identify this backend
        """
        return self.configuration().backend_name + "_density_simulator"

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

        # Update noise model if injected
        job_noise_model = self._quac_noise_model
        if run_config.get("quac_noise_model"):
            job_noise_model = run_config.get("quac_noise_model")

        for experiment in qobj.experiments:
            exp_start = time.perf_counter()
            final_quac_instance, qubit_measurements = super()._run_experiment(experiment, **run_config)

            # Create a frequency defaultdict for multinomial experiment tallying
            frequencies = defaultdict(lambda: 0)

            # Get probabilities of all states occurring and try to adjust them by measurement errors
            bitstring_probs = sparse.csr_matrix(final_quac_instance.get_bitstring_probs()).transpose()
            if job_noise_model.has_meas():
                # If measurement error simulation is turned on, adjust probabilities accordingly
                for expanded_qubit_meas_mat in job_noise_model.meas():
                    bitstring_probs = np.dot(expanded_qubit_meas_mat, bitstring_probs)

            # Switch probability list least significant bit convention and add to dictionary
            for decimal_state in range(bitstring_probs.shape[0]):
                binary_state = bin(decimal_state)[2:]
                state_prob = bitstring_probs.toarray()[decimal_state][0]
                padded_outcome_state = list(binary_state.zfill(qobj.config.n_qubits))
                classical_register = ["0"] * qobj.config.memory_slots

                for qubit, outcome in enumerate(padded_outcome_state):
                    # Only measure specified qubits into the classical register
                    if qubit in qubit_measurements:
                        for register_slot in qubit_measurements[qubit]:
                            classical_register[register_slot] = outcome

                classical_register.reverse()  # convert to Qiskit MSB format
                classical_register_hex = hex(int(''.join(classical_register), 2))

                frequencies[classical_register_hex] += state_prob

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
