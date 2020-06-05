# -*- coding: utf-8 -*-

"""This module contains backend functionality for obtaining the density matrix from quac
simulations of a Qiskit-defined quantum circuit. Functionality is located in the
QuacDensitySimulator class. The configuration of this backend simulator is also found
in the QuacDensitySimulator in the class constructor.
"""

from typing import Optional
import uuid
import time
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.result import Result
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.quac.simulators import QuacSimulator
from qiskit.providers.quac.models import QuacJob
from qiskit.providers.quac.utils import DensityInterface


class QuacDensitySimulator(QuacSimulator):
    """Class for simulating a Qiskit-defined quantum experiment and printing its resulting
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

    def run(self, qobj: QasmQobj, **run_config) -> QuacJob:
        """Runs quantum experiments encoded in Qiskit Qasm quantum objects on either a thread
        executor pool or a process executor pool, depending on the platform

        :param qobj: an assembled QASM quantum object bundling all simulation information and
            experiments (NOTE: Pulse quantum objects not yet supported)
        :param run_config: a dictionary containing optional injected parameters, including the
            following keys:

            1. lindblad: a dictionary representing the Lindblad emission and dephasing time
            constants in units of nanoseconds. Here is an example for the lindblad dictionary
            for a two-qubit system:

                .. code-block:: python

                    lindblad = {
                        "0": {
                            "T1": 60000,
                            "T2": 50000
                         },
                        "1": {
                            "T1": 60000,
                            "T2": 50000
                        }
                    }

            2. gate_times: a list of integers specifying what time (in nanoseconds) to run each
            gate the user has added to their experiments corresponding to gates in the order they
            were added

            3. simulation_length: the total number of nanoseconds to run the simulator

            4. time_step: length between discrete time steps in QuaC simulation (nanoseconds)
        :return: a submitted QuacJob running the experiments in qobj
        """
        job = QuacJob(self, str(uuid.uuid4()), self._run_job, qobj, **run_config)
        job.submit()

        return job

    def _run_job(self, job_id: str, qobj: QasmQobj, **run_config) -> Result:
        """This method is run by the thread executor pool (or process executor pool) and is a
        helper method to the run method

        :param job_id: a uuid string to identify the job once submitted to a thread executor pool
        :param qobj: a QasmQobj containing quantum experiments and metadata
        :param run_config: a dictionary of optional injected parameters (see the documentation
        for run for a complete description of all possible parameters)
        :return: a Qiskit result object with DensityInterface objects as the returned data
        """
        qobj_start = time.perf_counter()

        results = list()
        for experiment in qobj.experiments:
            exp_start = time.perf_counter()
            final_quac_instance = super().run_experiment(experiment, **run_config)
            final_quac_instance.print_density_matrix()

            # TODO: create density interface object in QuaC Python bindings
            density_interface = DensityInterface()  # template interface for now

            results.append({
                "name": experiment.header.name,
                "shots": qobj.config.shots,
                "data": {"density": density_interface},
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
