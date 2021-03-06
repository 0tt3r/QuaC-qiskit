# -*- coding: utf-8 -*-

"""Module that contains scheduling schemes for Quantum Circuits.
"""
from typing import Tuple, List
from qiskit.providers.models.backendproperties import BackendProperties, BackendPropertyError
from qiskit.circuit.instruction import QasmQobjInstruction
from qiskit.qobj.qasm_qobj import QasmQobjExperiment


def list_schedule_experiment(qexp: QasmQobjExperiment,
                             hardware_props: BackendProperties) -> List[Tuple[QasmQobjInstruction, float]]:
    """List schedule experiment to minimize run time

    :param qexp: quantum experiment to schedule
    :param hardware_props: hardware properties (T1, T2, etc.)
    :return: a list of tuples with an instruction and its corresponding execution time
    """
    # Keep track of when to schedule gates and which qubits are measured
    scheduling_times = [1] * qexp.config.n_qubits
    instruction_time_order = []

    # Schedule gate times
    for index, instruction in enumerate(qexp.instructions):
        instruction.id = index
        try:
            gate_length = hardware_props.gate_length(gate=instruction.name, qubits=instruction.qubits) * 1e9
        except (BackendPropertyError, AttributeError):
            gate_length = 0  # TODO: should measure have a time?

        gate_application_time = max([scheduling_times[qubit] for qubit in instruction.qubits])

        if instruction.name == 'u1':
            gate_length = 10

        for qubit in instruction.qubits:
            scheduling_times[qubit] = gate_application_time
            scheduling_times[qubit] += gate_length

        instruction_time_order.append((instruction, gate_application_time))

    instruction_time_order.sort(key=lambda pair: pair[1])  # sort instructions by time

    # for instruction, gate_application_time in instruction_time_order:
    #     print(f"Applying gate {instruction.name} on {instruction.qubits} at time {gate_application_time} ns.")

    return instruction_time_order


def no_schedule_experiment(qexp: QasmQobjExperiment,
                           hardware_props: BackendProperties) -> List[Tuple[QasmQobjInstruction, float]]:
    """A very basic scheduler for debugging

    :param qexp: quantum experiment to schedule
    :param hardware_props: hardware properties (T1, T2, etc.)
    :return: a list of tuples with an instruction and its corresponding execution time
    """
    gate_application_time = 1
    instruction_time_order = []

    # Schedule gate times
    for index, instruction in enumerate(qexp.instructions):
        # print(f"Applying gate {instruction.name} on {instruction.qubits} at time {gate_application_time} ns.")
        instruction_time_order.append((instruction, gate_application_time))
        gate_application_time += 500

    return instruction_time_order
