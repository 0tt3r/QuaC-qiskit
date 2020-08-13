# -*- coding: utf-8 -*-

"""This module contains backend functionality for obtaining the density matrix from QuaC
simulations of a Qiskit-defined quantum circuit. Functionality is located in the QuacSimulator
class. The configuration of this backend simulator is also found in QuacSimulator in the class
constructor.
"""
from typing import Optional, Tuple, Dict, List, Union
from abc import abstractmethod
from collections import defaultdict
import math
import warnings
import uuid
import quac
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.qobj.qasm_qobj import QasmQobjExperiment
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.models.backendconfiguration import BackendConfiguration, QasmBackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.result import Result
from qiskit.providers.quac.models import QuacJob, QuacNoiseModel
from qiskit.providers.quac.exceptions import QuacOptionsError, QuacBackendError
from .schedule import list_schedule_experiment, no_schedule_experiment


class QuacSimulator(BaseBackend):
    """General class for simulating a Qiskit-defined quantum experiment in QuaC
    """

    def __init__(self, hardware_conf: Union[BackendConfiguration, QasmBackendConfiguration],
                 hardware_props: Optional[BackendProperties] = None,
                 quac_noise_model: Optional[QuacNoiseModel] = None):
        """Initialize QuaC backend simulator

        :param hardware_conf: desired hardware configuration
        :param hardware_props: desired hardware properties
        """
        self._configuration = hardware_conf
        self._properties = hardware_props
        self._quac_noise_model = quac_noise_model
        if not quac_noise_model:
            self._quac_noise_model = QuacNoiseModel.get_noiseless_model(hardware_conf.n_qubits)
        super().__init__(self._configuration, "QuacProvider")  # QuaC is the provider

    def properties(self) -> BackendProperties:
        """Allows access to hardware backend properties

        :return: Qiskit BackendProperties object
        """
        return self._properties

    def run(self, qobj: QasmQobj, **run_config) -> QuacJob:
        """Run method

        :param qobj: an assembled QASM quantum object bundling all simulation information and
            experiments (NOTE: Pulse quantum objects not yet supported)
        :param run_config: a dictionary containing optional injected parameters, including the
            following keys:
            1. quac_noise_model: a QuacNoiseModel object describing system noise
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

    @abstractmethod
    def _run_job(self, job_id: str, qobj: QasmQobj, **run_config) -> Result:
        """Specifies how to run a quantum object job on this backend. This is the method that
        changes between types of QuaC backends.

        :param job_id: a uuid4 string to uniquely identify this job
        :param qobj: an assembled quantum object of experiments
        :param run_config: injected parameters
        :return: a Qiskit Result object
        """
        pass

    def _run_experiment(self, qexp: QasmQobjExperiment, **run_config) -> Tuple[quac.Instance, Dict[int, List[int]]]:
        """Runs quantum experiments/circuits encoded in Qiskit QASM quantum objects
        Note: Pulse quantum objects not supported

        :param qexp: a Qasm quantum object experiment to run
        :param run_config: a dictionary containing all injected parameters, including a list of
            floating point gate times, a dictionary of relevant Lindblad noise parameters, and
            the duration of time to run the simulation
        :return: a QuaC instance that has run the experiment
        """
        # Gather parameters
        gate_times = run_config.get("gate_times")
        simulation_length = run_config.get("simulation_length")
        dt = run_config.get("dt")

        # Override noise model if necessary
        exp_noise_model = self._quac_noise_model
        if run_config.get("quac_noise_model"):
            exp_noise_model = run_config.get("quac_noise_model")

        # Schedule experiment
        instruction_time_order = list_schedule_experiment(qexp, self._properties)

        # Sanitize parameters
        if not dt:
            dt = 10  # default time step value (ns)

        if gate_times:
            if len(gate_times) < len(qexp.instructions):
                raise QuacOptionsError("Not enough gate times (did you set times after transpilation?)")
            elif len(gate_times) > len(qexp.instructions):
                raise QuacOptionsError("Too many gate times")

        if not simulation_length and not gate_times:
            simulation_length = instruction_time_order[-1][1]  # note that measurement is instantaneous
        elif not simulation_length and gate_times:
            simulation_length = gate_times[-1]

        if simulation_length < instruction_time_order[-1][1]:
            raise QuacOptionsError("Simulation length not long enough to accommodate circuit")

        # Create a new instance of the QuaC simulator
        quac_simulator = quac.Instance()

        # Build the circuit in QuaC
        quac_circuit = quac.Circuit()
        quac_circuit.initialize(len(qexp.instructions))

        # Keep track of when to schedule gates and which qubits are measured
        qubit_measurements = defaultdict(lambda: [])

        # Add instructions
        instruction_counter = 0
        for instruction, gate_application_time in instruction_time_order:
            # print(f"Applying gate {instruction.name} on {instruction.qubits} at time {gate_application_time} ns.")
            # Take care of custom timing
            if gate_times:
                gate_application_time = gate_times[instruction_counter]
                instruction_counter += 1

            # Take care of applying instruction
            if instruction.name == "measure":
                # Keep track of qubits to measure
                qubit_measurements[instruction.qubits[0]].append(instruction.memory[0])
                continue
            elif instruction.name == "barrier":
                # Ignore barrier construct when building QuaC circuit
                continue
            elif instruction.name == "cx":
                quac_circuit.add_gate(gate="cnot",
                                      qubit1=instruction.qubits[0],
                                      qubit2=instruction.qubits[1],
                                      time=gate_application_time)
            elif instruction.name == "cz":
                quac_circuit.add_gate(gate="cz",
                                      qubit1=instruction.qubits[0],
                                      qubit2=instruction.qubits[1],
                                      time=gate_application_time)
            elif instruction.name == "u1":
                quac_circuit.add_gate(gate="u1",
                                      qubit1=instruction.qubits[0],
                                      time=gate_application_time,
                                      lam=instruction.params[0])
            elif instruction.name == "u2":
                quac_circuit.add_gate(gate="u2",
                                      qubit1=instruction.qubits[0],
                                      time=gate_application_time,
                                      phi=instruction.params[0],
                                      lam=instruction.params[1])
            elif instruction.name == "u3":
                quac_circuit.add_gate(gate="u3",
                                      qubit1=instruction.qubits[0],
                                      time=gate_application_time,
                                      theta=instruction.params[0],
                                      phi=instruction.params[1],
                                      lam=instruction.params[2])
            elif instruction.name == "rx" or instruction.name == "ry" or instruction.name == "rz":
                quac_circuit.add_gate(gate=instruction.name,
                                      qubit1=instruction.qubits[0],
                                      time=gate_application_time,
                                      theta=instruction.params[0])
            elif instruction.name == "id":
                quac_circuit.add_gate(gate="i",
                                      qubit1=instruction.qubits[0],
                                      time=gate_application_time)
            else:
                # TODO: other two-qubit gates besides CNOT will throw an error here
                quac_circuit.add_gate(gate=instruction.name,
                                      qubit1=instruction.qubits[0],
                                      time=gate_application_time)

            # Just in case the user does not know to only measure at the end
            if instruction.name != "measure" and instruction.name != "barrier":
                for qubit in instruction.qubits:
                    if qubit in qubit_measurements:
                        warnings.warn(
                            """Only measurement at the end of the circuit is supported.
                            Your intermediate measurements will be ignored, and these qubits
                            will instead be measured at the end of the circuit."""
                        )

        # Check to make sure some qubits are measured
        if len(qubit_measurements) == 0:
            raise QuacBackendError("No qubits measured!")

        # Create qubits to simulate
        quac_simulator.num_qubits = qexp.config.n_qubits
        quac_simulator.create_qubits()

        # Add Lindblad emission and dephasing noise terms
        for qubit in range(quac_simulator.num_qubits):
            # NOTE: QuaC expects 1/T1 and 1/2T2* instead of T1 and T2
            gamma = 1 / exp_noise_model.t1(qubit)
            gamma2 = 2 / exp_noise_model.t2(qubit) - 1 / exp_noise_model.t1(qubit)
            quac_simulator.add_lindblad_emission(qubit, gamma)
            quac_simulator.add_lindblad_dephasing(qubit, gamma2)

        # Add ZZ coupling terms, if present
        # Note: zz coupling terms should be expressed in frequency, not angular frequency!
        if exp_noise_model.has_zz():
            for pair in exp_noise_model.zz():
                qubit1, qubit2 = pair
                zeta = exp_noise_model.zz(qubit1, qubit2)
                quac_simulator.add_ham_zz_coupling(qubit1=qubit1, qubit2=qubit2, zeta=zeta * 2 * math.pi)

        # Run the experiment
        quac_simulator.create_density_matrix()
        quac_simulator.start_circuit_at(quac_circuit)
        quac_simulator.run(max(simulation_length, dt), dt=dt)

        return quac_simulator, dict(qubit_measurements)
