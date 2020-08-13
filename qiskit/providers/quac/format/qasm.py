# -*- coding: utf-8 -*-

"""This module contains functions for generating QASM with QuaC
"""
import re
import numpy as np
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers import BaseBackend
from qiskit.providers.quac.simulators import list_schedule_experiment


def quac_time_qasm_transpiler(circuit: QuantumCircuit, backend: BaseBackend) -> str:
    """Converts a circuit of type QuantumCircuit to a string of TIMEQASM specification

    :param circuit: a QuantumCircuit (need not be transpiled)
    :param backend: a specific backend to generate the QASM for (for tranpsilation)
    :return: a string containing necessary QASM with times for each gate
    """
    # Get original QASM
    transpiled_circuit = transpile(circuit, backend)
    original_qasm = transpiled_circuit.qasm()

    # Get body of original QASM
    start = 2 + len(circuit.qregs) + len(circuit.cregs)
    original_qasm_body = original_qasm.splitlines()[start:]

    # Formulate header
    qasm_modified = "TIMEQASM 1.0;\n"
    qasm_modified += "\n".join(original_qasm.splitlines()[1:start])
    qasm_modified += "\n"
    # Schedule circuit
    qobj = assemble(transpiled_circuit, backend)
    qexp = qobj.experiments[0]
    qschedule = list_schedule_experiment(qexp, backend.properties())

    # Formulate body
    for instruction, time in qschedule:
        # Note: ID was custom injected by the scheduler in this plugin
        qasm_modified += f"{original_qasm_body[instruction.id][:-1]} @{time};\n"

    return qasm_modified


def quac_qasm_transpiler(qiskit_qasm: str) -> str:
    """Converts Qiskit-generated QASM instructions into QuaC-supported instructions

    :param qiskit_qasm: a string with a QASM program
    :return: a string with a modified QuaC-supported QASM program
    """
    quac_qasm = ""
    for line in qiskit_qasm.splitlines():
        # Remove certain types of unsupported operations
        if any(instruction in line for instruction in ["measure", "creg", "barrier", "id"]):
            continue

        # Reformat implicit multiplication by pi
        for pi_mult in re.findall("[0-9]pi", line):
            line = line.replace(pi_mult, pi_mult[0] + '*pi')

        # Evaluate and inject new parameters
        instruction_params = re.findall("\(.+\)", line)  # only one parameter set per line
        if len(instruction_params) == 1:
            instruction_params = instruction_params[0]
        else:
            quac_qasm += line + "\n"
            continue

        # Evaluate pi-based parameter expressions
        evaluated_instruction_params = "("
        for parameter in instruction_params[1:-1].split(","):
            parameter = parameter.replace("pi", "np.pi")
            evaluated_instruction_params += str(eval(parameter)) + ","
        evaluated_instruction_params = evaluated_instruction_params[:-1] + ")"
        line = line.replace(instruction_params, evaluated_instruction_params)

        # Add formatted QASM line to final result
        quac_qasm += line + "\n"

    return quac_qasm
