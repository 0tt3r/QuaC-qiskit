# -*- coding: utf-8 -*-

"""This module contains functions useful for QuaC
"""

from typing import Dict, List
import re
import math
import numpy as np


def quac_qasm_transpiler(qiskit_qasm: str) -> str:
    """Converts Qiskit-generated QASM instructions into QuaC supported instructions
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


def counts_to_list(counts: Dict[str, int]) -> List[int]:
    """Converts counts to a list representation.
    :param counts: a Qiskit-style counts dictionary
    :return: a list of integers
    """
    num_qubits = len(list(counts.keys())[0].replace(' ', ''))
    counts_list = [0] * 2 ** num_qubits
    for state in counts:
        f_state = state.replace(' ', '')
        counts_list[int(f_state, 2)] = counts[state]
    return counts_list


def qiskit_statevector_to_probabilities(statevector: np.array, non_ancilla: int,
                                        meas_mappings: Dict[int, int]) -> List[float]:
    """A simple utility to convert Qiskit statevectors to probability lists. Warning: assumes
    all ancilla bits are the most significant bits.
    :param statevector: an np array
    :param non_ancilla: an integer holding the number of non-ancilla bits
    :param meas_mappings: a list of integer mappings from qubit index to classical register index
    :return: a list of probabilities parallel to input "statevector"
    """
    filtered_probs = [0] * 2 ** non_ancilla
    probs = [np.absolute(entry) ** 2 for entry in statevector]
    num_qubits = int(math.log2(len(statevector)))

    for state_ind, prob in enumerate(probs):
        filtered_state = list(bin(state_ind)[2:].zfill(num_qubits)[-non_ancilla:])
        filtered_state.reverse()
        mapped_state = [0] * non_ancilla
        for qubit_ind in range(non_ancilla):
            mapped_state[meas_mappings[qubit_ind]] = filtered_state[qubit_ind]
        filtered_state_combined = ''.join(mapped_state)
        filtered_state_ind = int(filtered_state_combined, 2)
        filtered_probs[filtered_state_ind] += prob

    return filtered_probs
