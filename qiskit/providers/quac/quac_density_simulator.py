# -*- coding: utf-8 -*-

"""
This module contains backend functionality for obtaining the density matrix from quac simulations
of a qiskit-defined quantum circuit. Functionality is located in the QuacDensitySimulator class.
The configuration of this backend simulator is also found in the QuacDensitySimulator in the class
constructor.
"""

from qiskit.providers.basebackend import BaseBackend
from qiskit.qobj.qasm_qobj import QasmQobj
import quac
from .exceptions import QuacOptionsError
from .quac_backend_configuration import configuration


class QuacDensitySimulator(BaseBackend):
    """
    Class for simulating a qiskit-defined quantum experiment and printing its resulting
    density matrix
    """

    def __init__(self):
        """Initialize a quac Density Simulation backend"""
        quac.initialize()
        self._quac_simulator = quac.Instance()
        super().__init__(configuration, "QuacProvider")  # quac is the provider

    def run(self, qobj: QasmQobj, **run_config) -> str:
        """
        Runs quantum experiments/circuits encoded in qiskit QASM quantum objects
        Note: Pulse quantum objects not supported
        :param qobj: an assembled quantum object bundling all simulation information and
        experiments
        :param run_config: a dictionary containing all injected parameters, including a
        list of floating point gate
        times, a dictionary of relevant Lindblad noise parameters, and the duration of
        time to run the simulation
        :return: a string representing whether the simulation succeeded ("Success")
        """
        # Get gate timing data and Lindblad noise parameters injected by provider via run_config
        # NOTE: for now, only allowing one shot and one experiment
        # TODO: add support for multiple experiments in one quantum object
        if ("gate_times" not in run_config) or ("lindblad" not in run_config):
            raise QuacOptionsError("gate times and Lindblad noise parameters not specified")
        gate_times = run_config["gate_times"]
        lindblad = run_config["lindblad"]

        try:
            simulation_length = run_config["simulation_length"]
        except KeyError:
            simulation_length = gate_times[-1] + 1

        # Build the circuit in QuaC
        experiment = qobj.experiments[0]
        quac_circuit = quac.Circuit()
        quac_circuit.initialize(25)

        for (time_index, instruction) in enumerate(experiment.instructions):
            if instruction.name == "measure":
                pass  # TODO: once QuaC diagonalize branch is merged, add QuaC measurements here
            elif instruction.name == "cx":
                quac_circuit.add_gate(gate="cnot",
                                      qubit1=instruction.qubits[0],
                                      qubit2=instruction.qubits[1],
                                      time=gate_times[time_index])
            else:
                quac_circuit.add_gate(gate=instruction.name,
                                      qubit1=instruction.qubits[0],
                                      time=gate_times[time_index])

        self._quac_simulator.num_qubits = experiment.config.n_qubits
        self._quac_simulator.create_qubits()

        # Add Lindblad noise terms
        for qubit_index in range(0, self._quac_simulator.num_qubits):
            self._quac_simulator.add_lindblad_emission(qubit_index, lindblad["emission"])
            self._quac_simulator.add_lindblad_dephasing(qubit_index, lindblad["dephasing"])
            self._quac_simulator.add_lindblad_thermal_coupling(qubit_index, lindblad["thermal"])
            for qubit_index_2 in range(0, self._quac_simulator.num_qubits):
                if qubit_index != qubit_index_2:
                    self._quac_simulator.add_lindblad_cross_coupling(qubit_index,
                                                                     qubit_index_2,
                                                                     lindblad["coupling"])

        # Run the experiment
        self._quac_simulator.create_density_matrix()
        self._quac_simulator.start_circuit_at(quac_circuit)
        self._quac_simulator.run(simulation_length, dt=0.1)

        self._quac_simulator.print_density_matrix()

        return "Success"
