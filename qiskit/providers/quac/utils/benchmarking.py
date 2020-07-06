# -*- coding: utf-8 -*-

"""This module contains relevant functions for calculating T1, T2, and measurement errors on QuaC, as
well as functions useful for benchmarking QuaC against Qiskit.
"""

from typing import Union, Tuple, List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag


def quac_t1_circuits(num_gates: Union[List[int], np.array],
                     gate_time: float,
                     qubits: List[int]) -> Tuple[List[QuantumCircuit], np.array]:
    """Generates T1 calibration circuits for QuaC. These calibration circuits are identical to the
    Qiskit calibration circuits, just added differently so that QuaC qubit calibrations can be
    done in parallel.
    :param num_gates: list of numbers of identity gates to add to each circuit
    :param gate_time: length of an identity gate (in nanoseconds)
    :param qubits: qubits on which to add calibration circuits
    :return: T1 calibration times, gate execution times
    """
    gate_times = gate_time * np.array(num_gates)

    t1_circs = list()
    quantum_register = QuantumRegister(max(qubits) + 1)
    classical_register = ClassicalRegister(len(qubits))

    for circ_ind, circ_length in enumerate(num_gates):
        # Generate a new T1 circuit with circ_length identity gates
        t1_circ = QuantumCircuit(quantum_register, classical_register)
        t1_circ.name = f"t1circuit_{circ_ind}_0"

        for qubit in qubits:
            t1_circ.x(qubit)

        for ind in range(circ_length):
            for qubit in qubits:
                t1_circ.id(quantum_register[qubit])

        for qubit in qubits:
            t1_circ.measure(quantum_register[qubit], classical_register[qubit])

        t1_circ.barrier()
        t1_circs.append(t1_circ)

    return t1_circs, gate_times


def add_parallel_id(circuit: QuantumCircuit) -> QuantumCircuit:
    """Adds identity gates to all other qubits not occupied by each instruction such that
    time implicitly passes for each qubit. Necessary for comparing QuaC and Qiskit simulator
    results with noise.
    :param circuit: a QuantumCircuit object
    :return: a QuantumCircuit object
    """
    identity_padded_circuit = QuantumCircuit()  # keeps track of new padded circuit

    # Add registers to new circuit
    for qreg in circuit.qregs:
        identity_padded_circuit.add_register(qreg)
    for creg in circuit.cregs:
        identity_padded_circuit.add_register(creg)

    # Rebuild circuit with parallel identity gate padding
    dag_circuit = circuit_to_dag(circuit)
    for node in dag_circuit.topological_op_nodes():
        # Add original instruction and do not worry about measurements or barriers
        identity_padded_circuit.append(node.op, node.qargs, node.cargs)
        if node.name in ["measure", "barrier"]:
            continue

        # Add parallel identity gates
        for qubit in set(dag_circuit.qubits()) - set(node.qargs):
            identity_padded_circuit.id(qubit)

    return identity_padded_circuit
