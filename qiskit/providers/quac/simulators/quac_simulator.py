# -*- coding: utf-8 -*-

"""This module contains backend functionality for obtaining the density matrix from QuaC
simulations of a Qiskit-defined quantum circuit. Functionality is located in the QuacSimulator
class. The configuration of this backend simulator is also found in QuacSimulator in the class
constructor.
"""

from typing import Optional, Tuple, Dict, List
from abc import abstractmethod
from collections import defaultdict
import numpy as np
import warnings
import uuid
import quac
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit.instruction import Instruction
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.qobj.qasm_qobj import QasmQobjExperiment
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.quac.models import generic_quac_configuration, QuacJob
from qiskit.providers.quac.exceptions import QuacOptionsError, QuacBackendError


class QuacSimulator(BaseBackend):
    """General class for simulating a Qiskit-defined quantum experiment in QuaC
    """

    def __init__(self, hardware_conf: Optional[BackendConfiguration] = None,
                 hardware_props: Optional[BackendProperties] = None):
        """Initialize hardware and build measurement error matrix

        :param hardware_conf: desired hardware configuration
        :param hardware_props: desired hardware properties
        """
        self._hardware_specified = True
        self._meas_set = False

        if hardware_conf and hardware_props:
            self._configuration = hardware_conf
            self._properties = hardware_props
            self._measurement_error_matrices = []
        else:
            self._hardware_specified = False
            self._configuration = generic_quac_configuration
            self._properties = None

        super().__init__(self._configuration, "QuacProvider")  # QuaC is the provider

    @abstractmethod
    def setup_measurement_error(self):
        pass

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

    @abstractmethod
    def _run_job(self, job_id: str, qobj: QasmQobj, **run_config):
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
        # Create a new instance of the QuaC simulator
        quac_simulator = quac.Instance()

        # Get gate timing data and Lindblad noise parameters injected by provider via run_config
        lindblad = run_config.get("lindblad")
        no_noise = run_config.get("no_noise")
        if not self._hardware_specified and not lindblad and not no_noise:
            raise QuacOptionsError("No hardware specs and no user-defined noise model provided")

        # Attempt to retrieve gate times, otherwise calculate them
        gate_times = run_config.get("gate_times")

        # Build the circuit in QuaC
        quac_circuit = quac.Circuit()
        quac_circuit.initialize(len(qexp.instructions))

        # Keep track of when to schedule gates and which qubits are measured
        scheduling_times = [1] * qexp.config.n_qubits
        qubit_measurements = defaultdict(lambda: [])
        instruction_time_order = []

        # Schedule gate times
        for instruction in qexp.instructions:
            try:
                gate_length = self.properties().gate_property(gate=instruction.name,
                                                              qubits=instruction.qubits,
                                                              name="gate_length")[0]
                gate_length *= 1e9  # convert to natural time unit of ns
            except (BackendPropertyError, AttributeError):
                gate_length = 0  # TODO: should measure have a time?

            gate_application_time = max([scheduling_times[qubit] for qubit in instruction.qubits])

            # print(f"Applying gate {instruction.name} on {instruction.qubits} at time {gate_application_time} ns.")
            for qubit in instruction.qubits:
                scheduling_times[qubit] = gate_application_time
                scheduling_times[qubit] += gate_length

            instruction_time_order.append((instruction, gate_application_time))

        instruction_time_order.sort(key=lambda pair: pair[1])  # sort instructions by time

        # Add instructions
        for instruction, gate_application_time in instruction_time_order:
            if instruction.name == "measure":
                # Keep track of qubits to measure
                qubit_measurements[instruction.qubits[0]].append(instruction.memory[0])
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
                # TODO: other two-qubit gates besides CNOT will error here
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
        for qubit_index in range(0, quac_simulator.num_qubits):
            # Note: specifying no_noise overrides lindblad, lindblad overrides hardware
            if self._hardware_specified and not lindblad and not no_noise:
                # Grab T1 and T2 times and convert them to ns
                t1_time = self.properties().qubit_property(qubit_index, "T1")[0] * 1e9
                t2_time = self.properties().qubit_property(qubit_index, "T2")[0] * 1e9
            elif lindblad and not no_noise:
                # User override (injection of lindblad times)
                t1_time = lindblad[str(qubit_index)]["T1"]
                t2_time = lindblad[str(qubit_index)]["T2"]
            else:
                # No user override, no hardware specified, no_noise -> no noise added
                t1_time = float('inf')
                t2_time = float('inf')

            # NOTE: QuaC expects 1/T1 and 1/T2 instead of T1 and T2
            quac_simulator.add_lindblad_emission(qubit_index, 1 / t1_time)
            quac_simulator.add_lindblad_dephasing(qubit_index, 1 / t2_time)

        # Attempt to retrieve simulation length, otherwise calculate it
        simulation_length = run_config.get("simulation_length")
        if not simulation_length and not gate_times:
            simulation_length = max(scheduling_times)
        elif not simulation_length and gate_times:
            simulation_length = gate_times[-1] + 500

        # Attempt to retrieve simulation time step
        dt = run_config.get("time_step")
        if not dt:
            dt = 10  # default time step value (ns)

        # Run the experiment
        quac_simulator.create_density_matrix()
        quac_simulator.start_circuit_at(quac_circuit)
        quac_simulator.run(simulation_length, dt=dt)

        return quac_simulator, dict(qubit_measurements)

    def _run_experiment_dag(self, qexp: QasmQobjExperiment, **run_config) -> Tuple[quac.Instance, Dict[int, List[int]]]:
        """Runs quantum experiments/circuits encoded in Qiskit QASM quantum objects
        Note: Pulse quantum objects not supported

        :param qexp: a Qasm quantum object experiment to run
        :param run_config: a dictionary containing all injected parameters, including a list of
            floating point gate times, a dictionary of relevant Lindblad noise parameters, and
            the duration of time to run the simulation
        :return: a QuaC instance that has run the experiment
        """

        # Rebuild transpiled circuit in experiment
        experiment_circuit = QuantumCircuit(
            qexp.config.n_qubits,
            qexp.config.memory_slots
        )

        for instruction in qexp.instructions:
            instruction.qubits = [] if not hasattr(instruction, "qubits") else instruction.qubits
            instruction.memory = [] if not hasattr(instruction, "memory") else instruction.memory
            instruction.params = [] if not hasattr(instruction, "params") else instruction.params

            # Build generic instruction because only instructions of the type Instruction may be appended
            # to a QuantumCircuit type, not QasmQobjInstruction as are found in qexp.instructions. Also,
            # qubits on which the instruction is operating as well as classical memory associated with it
            # must be "injected" into the standard generic Instruction object.
            generic_instruction = Instruction(
                instruction.name,
                len(instruction.qubits),
                len(instruction.memory),
                instruction.params
            )

            generic_instruction.qubits = instruction.qubits
            generic_instruction.memory = instruction.memory

            # Add instruction to circuit
            experiment_circuit.append(generic_instruction, instruction.qubits, instruction.memory)

        # Use a DAG of the circuit to efficiently schedule parallel circuit elements
        experiment_dag = circuit_to_dag(experiment_circuit)  # build DAG from circuit
        parallel_layers = []

        for layer in experiment_dag.layers():
            parallel_layer = []
            for node in layer["graph"].topological_op_nodes():
                parallel_layer.append(node.op)

            parallel_layers.append(parallel_layer)

        # Uncomment to view scheduling layers
        # for layer in parallel_layers:
        #     for op in layer:
        #         print(op.name, end="(")
        #         print(op.params, end=") q")
        #         print(op.qubits, end="     ")
        #     print()

        # Create a new instance of the QuaC simulator
        quac_simulator = quac.Instance()

        # Get gate timing data and Lindblad noise parameters injected by provider via run_config
        lindblad = run_config.get("lindblad")
        no_noise = run_config.get("no_noise")
        if not self._hardware_specified and not lindblad and not no_noise:
            raise QuacOptionsError("No hardware specs and no user-defined noise model provided")

        # Attempt to retrieve gate times, otherwise calculate them
        gate_times = run_config.get("gate_times")

        # Build the circuit in QuaC
        quac_circuit = quac.Circuit()
        quac_circuit.initialize(len(qexp.instructions))

        circuit_time = 1
        qubit_measurements = defaultdict(lambda: [])

        for parallel_layer in parallel_layers:
            layer_time = 0

            for instruction in parallel_layer:
                # Determine layer time
                try:
                    gate_length = self.properties().gate_property(gate=instruction.name,
                                                                  qubits=instruction.qubits,
                                                                  name="gate_length")[0]
                    gate_length *= 1e9  # convert to natural time unit of ns
                except (BackendPropertyError, AttributeError):
                    gate_length = 100  # ns  # TODO: should measure have a time?
                layer_time = max(layer_time, gate_length)

                print(f"Applying gate {instruction.name} on {instruction.qubits} at time {circuit_time} ns.")

                # Add instruction
                if instruction.name == "measure":
                    # Keep track of qubits to measure
                    qubit_measurements[instruction.qubits[0]].append(instruction.memory[0])
                elif instruction.name == "barrier":
                    # Ignore barrier construct when building QuaC circuit
                    continue
                elif instruction.name == "cx":
                    quac_circuit.add_gate(gate="cnot",
                                          qubit1=instruction.qubits[0],
                                          qubit2=instruction.qubits[1],
                                          time=circuit_time)
                elif instruction.name == "cz":
                    quac_circuit.add_gate(gate="cz",
                                          qubit1=instruction.qubits[0],
                                          qubit2=instruction.qubits[1],
                                          time=circuit_time)
                elif instruction.name == "u1":
                    quac_circuit.add_gate(gate="u1",
                                          qubit1=instruction.qubits[0],
                                          time=circuit_time,
                                          lam=instruction.params[0])
                elif instruction.name == "u2":
                    quac_circuit.add_gate(gate="u2",
                                          qubit1=instruction.qubits[0],
                                          time=circuit_time,
                                          phi=instruction.params[0],
                                          lam=instruction.params[1])
                elif instruction.name == "u3":
                    quac_circuit.add_gate(gate="u3",
                                          qubit1=instruction.qubits[0],
                                          time=circuit_time,
                                          theta=instruction.params[0],
                                          phi=instruction.params[1],
                                          lam=instruction.params[2])
                elif instruction.name == "rx" or instruction.name == "ry" or instruction.name == "rz":
                    quac_circuit.add_gate(gate=instruction.name,
                                          qubit1=instruction.qubits[0],
                                          time=circuit_time,
                                          theta=instruction.params[0])
                elif instruction.name == "id":
                    quac_circuit.add_gate(gate="i",
                                          qubit1=instruction.qubits[0],
                                          time=circuit_time)
                else:
                    # TODO: other two-qubit gates besides CNOT will error here
                    quac_circuit.add_gate(gate=instruction.name,
                                          qubit1=instruction.qubits[0],
                                          time=circuit_time)

                # Just in case the user does not know to only measure at the end
                if instruction.name != "measure" and instruction.name != "barrier":
                    for qubit in instruction.qubits:
                        if qubit in qubit_measurements:
                            warnings.warn(
                                """Only measurement at the end of the circuit is supported.
                                Your intermediate measurements will be ignored, and these qubits
                                will instead be measured at the end of the circuit."""
                            )

            # Step time to next layer
            circuit_time += layer_time

        # Check to make sure some qubits are measured
        if len(qubit_measurements) == 0:
            raise QuacBackendError("No qubits measured!")

        # Create qubits to simulate
        quac_simulator.num_qubits = qexp.config.n_qubits
        quac_simulator.create_qubits()

        # Add Lindblad emission and dephasing noise terms
        for qubit_index in range(0, quac_simulator.num_qubits):
            # Note: specifying no_noise overrides lindblad, lindblad overrides hardware
            if self._hardware_specified and not lindblad and not no_noise:
                # Grab T1 and T2 times and convert them to ns
                t1_time = self.properties().qubit_property(qubit_index, "T1")[0] * 1e9
                t2_time = self.properties().qubit_property(qubit_index, "T2")[0] * 1e9
            elif lindblad and not no_noise:
                # User override (injection of lindblad times)
                t1_time = lindblad[str(qubit_index)]["T1"]
                t2_time = lindblad[str(qubit_index)]["T2"]
            else:
                # No user override, no hardware specified, no_noise -> no noise added
                t1_time = float('inf')
                t2_time = float('inf')

            # NOTE: QuaC expects 1/T1 and 1/T2 instead of T1 and T2
            quac_simulator.add_lindblad_emission(qubit_index, 1 / t1_time)
            quac_simulator.add_lindblad_dephasing(qubit_index, 1 / t2_time)

        # Attempt to retrieve simulation length, otherwise calculate it
        simulation_length = run_config.get("simulation_length")
        if not simulation_length and not gate_times:
            simulation_length = circuit_time
        elif not simulation_length and gate_times:
            simulation_length = gate_times[-1] + 1  # TODO: add buffer at the end of experiment

        # Attempt to retrieve simulation time step
        dt = run_config.get("time_step")
        if not dt:
            dt = 10  # default time step value (ns)

        # Run the experiment
        quac_simulator.create_density_matrix()
        quac_simulator.start_circuit_at(quac_circuit)
        quac_simulator.run(simulation_length, dt=dt)

        return quac_simulator, dict(qubit_measurements)

    def _run_experiment_no_scheduling(self, qexp: QasmQobjExperiment,
                                      **run_config) -> Tuple[quac.Instance, Dict[int, List[int]]]:
        """Runs quantum experiments/circuits encoded in Qiskit QASM quantum objects
        Note: Pulse quantum objects not supported

        :param qexp: a Qasm quantum object experiment to run
        :param run_config: a dictionary containing all injected parameters, including a list of
            floating point gate times, a dictionary of relevant Lindblad noise parameters, and
            the duration of time to run the simulation
        :return: a QuaC instance that has run the experiment
        """
        # Create a new instance of the QuaC simulator
        quac_simulator = quac.Instance()

        # Get gate timing data and Lindblad noise parameters injected by provider via run_config
        lindblad = run_config.get("lindblad")
        no_noise = run_config.get("no_noise")
        if not self._hardware_specified and not lindblad and not no_noise:
            raise QuacOptionsError("No hardware specs and no user-defined noise model provided")

        # Attempt to retrieve gate times, otherwise calculate them
        gate_times = run_config.get("gate_times")

        # Build the circuit in QuaC
        quac_circuit = quac.Circuit()
        quac_circuit.initialize(len(qexp.instructions))

        total_circuit_time = 1
        qubit_measurements = defaultdict(lambda: [])

        for (time_index, instruction) in enumerate(qexp.instructions):
            # Figure out gate timing information
            if gate_times:
                try:
                    gate_time = gate_times[time_index]
                except IndexError:
                    raise QuacBackendError("Not enough gate times specified!")
            else:
                try:
                    gate_length = self.properties().gate_property(gate=instruction.name,
                                                                  qubits=instruction.qubits,
                                                                  name="gate_length")[0]
                    gate_length *= 1e9  # convert to natural time unit of ns
                except (AttributeError, BackendPropertyError):
                    gate_length = 100  # default gate length when hardware is not specified

                total_circuit_time += gate_length
                gate_time = 1 if time_index == 0 else gate_time + gate_length  # must not start @0

            # Add gate

            if instruction.name == "measure":
                # Keep track of qubits to measure
                qubit_measurements[instruction.qubits[0]].append(instruction.memory[0])
            elif instruction.name == "barrier":
                # Ignore barrier construct when building QuaC circuit
                time_index -= 1
                continue  # barriers are only useful in transpilation
            elif instruction.name == "cx":
                quac_circuit.add_gate(gate="cnot",
                                      qubit1=instruction.qubits[0],
                                      qubit2=instruction.qubits[1],
                                      time=gate_time)
            elif instruction.name == "cz":
                quac_circuit.add_gate(gate="cz",
                                      qubit1=instruction.qubits[0],
                                      qubit2=instruction.qubits[1],
                                      time=gate_time)
            elif instruction.name == "u1":
                quac_circuit.add_gate(gate="u1",
                                      qubit1=instruction.qubits[0],
                                      time=gate_time,
                                      lam=instruction.params[0])
            elif instruction.name == "u2":
                quac_circuit.add_gate(gate="u2",
                                      qubit1=instruction.qubits[0],
                                      time=gate_time,
                                      phi=instruction.params[0],
                                      lam=instruction.params[1])
            elif instruction.name == "u3":
                quac_circuit.add_gate(gate="u3",
                                      qubit1=instruction.qubits[0],
                                      time=gate_time,
                                      theta=instruction.params[0],
                                      phi=instruction.params[1],
                                      lam=instruction.params[2])
            elif instruction.name == "rx" or instruction.name == "ry" or instruction.name == "rz":
                quac_circuit.add_gate(gate=instruction.name,
                                      qubit1=instruction.qubits[0],
                                      time=gate_time,
                                      theta=instruction.params[0])
            elif instruction.name == "id":
                quac_circuit.add_gate(gate="i",
                                      qubit1=instruction.qubits[0],
                                      time=gate_time)
            else:
                # TODO: other two-qubit gates besides CNOT will error here
                quac_circuit.add_gate(gate=instruction.name,
                                      qubit1=instruction.qubits[0],
                                      time=gate_time)

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
        for qubit_index in range(0, quac_simulator.num_qubits):
            # Note: specifying no_noise overrides lindblad, lindblad overrides hardware
            if self._hardware_specified and not lindblad and not no_noise:
                # Grab T1 and T2 times and convert them to ns
                t1_time = self.properties().qubit_property(qubit_index, "T1")[0] * 1e9
                t2_time = self.properties().qubit_property(qubit_index, "T2")[0] * 1e9
            elif lindblad and not no_noise:
                # User override (injection of lindblad times)
                t1_time = lindblad[str(qubit_index)]["T1"]
                t2_time = lindblad[str(qubit_index)]["T2"]
            else:
                # No user override, no hardware specified, no_noise -> no noise added
                t1_time = float('inf')
                t2_time = float('inf')

            # NOTE: QuaC expects 1/T1 and 1/T2 instead of T1 and T2
            quac_simulator.add_lindblad_emission(qubit_index, 1 / t1_time)
            quac_simulator.add_lindblad_dephasing(qubit_index, 1 / t2_time)

        # Attempt to retrieve simulation length, otherwise calculate it
        simulation_length = run_config.get("simulation_length")
        if not simulation_length and not gate_times:
            simulation_length = total_circuit_time
        elif not simulation_length and gate_times:
            simulation_length = gate_times[-1] + 1  # TODO: add buffer at the end of experiment

        # Attempt to retrieve simulation time step
        dt = run_config.get("time_step")
        if not dt:
            dt = 10  # default time step value (ns)

        # Run the experiment
        quac_simulator.create_density_matrix()
        quac_simulator.start_circuit_at(quac_circuit)
        quac_simulator.run(simulation_length, dt=dt)

        return quac_simulator, dict(qubit_measurements)
