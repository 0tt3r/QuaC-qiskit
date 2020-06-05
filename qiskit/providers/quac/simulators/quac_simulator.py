# -*- coding: utf-8 -*-

"""This module contains backend functionality for obtaining the density matrix from QuaC
simulations of a Qiskit-defined quantum circuit. Functionality is located in the QuacSimulator
class. The configuration of this backend simulator is also found in QuacSimulator in the class
constructor.
"""

from typing import Optional
from abc import abstractmethod
import concurrent.futures
import quac
from qiskit.providers.basebackend import BaseBackend
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.qobj.qasm_qobj import QasmQobjExperiment
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.quac.models import generic_configuration
from qiskit.providers.quac.exceptions import QuacOptionsError, QuacBackendError


class QuacSimulator(BaseBackend):
    """General class for simulating a Qiskit-defined quantum experiment in QuaC
    """

    def __init__(self, hardware_conf: Optional[BackendConfiguration] = None,
                 hardware_props: Optional[BackendProperties] = None):
        """Initialize a QuacSimulator object

        :param hardware_conf: desired hardware configuration
        :param hardware_props: desired hardware properties
        """
        self._hardware_specified = True

        if hardware_conf and hardware_props:
            self._configuration = hardware_conf
            self._properties = hardware_props
        else:
            self._hardware_specified = False
            self._configuration = generic_configuration
            self._properties = None

        super().__init__(self._configuration, "QuacProvider")  # QuaC is the provider

    def properties(self) -> BackendProperties:
        """Allows access to hardware backend properties

        :return: Qiskit BackendProperties object
        """
        return self._properties

    @abstractmethod
    def run(self, qobj: QasmQobj, **run_config) -> concurrent.futures.Future:
        """Abstract run method

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
        pass

    def run_experiment(self, qexp: QasmQobjExperiment, **run_config) -> quac.Instance:
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
        if not self._hardware_specified and not lindblad:
            raise QuacOptionsError("No hardware specs and no user-defined noise model provided")

        # Attempt to retrieve gate times, otherwise calculate them
        gate_times = run_config.get("gate_times")

        # Build the circuit in QuaC
        quac_circuit = quac.Circuit()
        quac_circuit.initialize(len(qexp.instructions))

        total_circuit_time = 0
        for (time_index, instruction) in enumerate(qexp.instructions):
            # Ignore certain constructs when building QuaC circuit
            if instruction.name == "measure" or instruction.name == "barrier":
                time_index -= 1
                continue  # barriers are only useful in transpilation

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
                except AttributeError:
                    gate_length = 100  # default gate length when hardware is not specified

                total_circuit_time += gate_length
                gate_time = 1 if time_index == 0 else gate_time + gate_length

            # Add gate
            if instruction.name == "cx":
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
                quac_circuit.add_gate(gate=instruction.name,
                                      qubit1=instruction.qubits[0],
                                      time=gate_time)

        # Create qubits to simulate
        quac_simulator.num_qubits = qexp.config.n_qubits
        quac_simulator.create_qubits()

        # Add Lindblad emission and dephasing noise terms
        for qubit_index in range(0, quac_simulator.num_qubits):
            if self._hardware_specified and not lindblad:
                # Grab T1 and T2 times and convert them to ns
                t1_time = self.properties().qubit_property(qubit_index, "T1")[0] * 1e9
                t2_time = self.properties().qubit_property(qubit_index, "T2")[0] * 1e9
            else:
                t1_time = lindblad[str(qubit_index)]["T1"]
                t2_time = lindblad[str(qubit_index)]["T2"]

            # NOTE: QuaC expects 1/T1 and 1/T2 instead of T1 and T2
            quac_simulator.add_lindblad_emission(qubit_index, 1/t1_time)
            quac_simulator.add_lindblad_dephasing(qubit_index, 1/t2_time)

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

        return quac_simulator
