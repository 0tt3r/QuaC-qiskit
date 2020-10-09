# -*- coding: utf-8 -*-

"""Module that contains data manipulation functions for simulator and hardware output
"""
from typing import Dict, List
import math
from deprecation import deprecated
import numpy as np
from qiskit.quantum_info import Statevector


def counts_to_list(counts: Dict[str, int]) -> List[int]:
    """Converts counts to a list representation

    :param counts: a Qiskit-style counts dictionary
    :return: a list of integers
    """
    num_bits = len(list(counts.keys())[0].replace(' ', ''))
    counts_list = [0] * 2 ** num_bits
    for state in counts:
        f_state = state.replace(' ', '')
        counts_list[int(f_state, 2)] = counts[state]
    return counts_list


def counts_to_dist(counts: Dict[str, int]) -> List[int]:
    """Converts counts to a distribution

    :param counts: a Qiskit-style counts dictionary
    :return: a probability distribution
    """
    num_qubits = len(list(counts.keys())[0].replace(' ', ''))
    counts_list = [0] * 2 ** num_qubits
    for state in counts:
        f_state = state.replace(' ', '')
        counts_list[int(f_state, 2)] = counts[state]
    dist = [element / sum(counts_list) for element in counts_list]
    return dist


@deprecated("0.0.1. Please use statevector_to_probablities")
def qiskit_statevector_to_probabilities(statevector: np.array, non_ancilla: int) -> List[float]:
    """A simple utility to convert Qiskit statevectors to probability lists. Warning: assumes
    all ancilla bits are the most significant bits. Warning: assumes ancilla bits come last and
    that measurement was applied via measure_all

    :param statevector: an np array
    :param non_ancilla: an integer holding the number of non-ancilla bits
    :return: a list of probabilities parallel to input "statevector"
    """
    filtered_probs = [0] * 2 ** non_ancilla
    probs = [np.absolute(entry) ** 2 for entry in statevector]
    num_qubits = int(math.log2(len(statevector)))

    for state_ind, prob in enumerate(probs):
        filtered_state = list(bin(state_ind)[2:].zfill(num_qubits)[-non_ancilla:])
        mapped_state = ['0'] * non_ancilla
        for qubit_ind in range(non_ancilla):
            mapped_state[qubit_ind] = filtered_state[qubit_ind]
        filtered_state_combined = ''.join(mapped_state)
        filtered_state_ind = int(filtered_state_combined, 2)
        filtered_probs[filtered_state_ind] += prob

    return filtered_probs


def statevector_to_probabilities(statevector: np.array) -> List[float]:
    """A simple utility to convert Qiskit statevectors to probability lists. Warning: assumes
    all ancilla bits are the most significant bits. Warning: assumes ancilla bits come last and
    that measurement was applied via measure_all

    :param statevector: an np array
    :return: a list of probabilities
    """
    return counts_to_list(Statevector(statevector).probabilities_dict())


def aggregate_counts_results(counts: List[Dict[str, int]], keys: List[str]) -> Dict[str, int]:
    """Aggregates hardware experiment results so experiments with higher count number can be run

    :param counts: a list of counts dicts
    :param keys: a list of statevectors in counts dicts
    :return: a counts dict summarizing all input counts dicts
    """
    aggregate_counts = {}
    for key in keys:
        aggregate_counts[key] = 0

    for counts_list in counts:
        for key in counts_list:
            aggregate_counts[key] += counts_list[key]

    return aggregate_counts
