# -*- coding: utf-8 -*-

"""
This module contains the generic configuration for QuaC backends and can be changed as additional
gates and functionality become available.
"""

from typing import Optional, List
from qiskit.providers.models.backendconfiguration import QasmBackendConfiguration


def get_generic_configuration(n_qubits: int, max_shots: Optional[int] = 8000, max_exp: Optional[int] = 1,
                              basis_gates: Optional[List[str]] = None) -> QasmBackendConfiguration:
    """
    Returns a generic backend configuration for users wishing to define their own hardware
    Note: defaults are max_shots=8000,
    :param n_qubits: the number of qubits in the hardware
    :param max_shots: the maximum number of shots per experiment that can be run
    :param max_exp: the maximum number of experiments (or circuits) that can be run at once
    :param basis_gates: a list of strings representing the available basis gates
    :return: a QasmBackendConfiguration object
    """
    if basis_gates is None:
        # Handle basis_gates default outside of signature due to mutability
        basis_gates = ['id', 'cx', 'h', 'x', 'y', 'z',
                       'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
                       'cxz', 'czx', 'cmz', 'cz']

    return QasmBackendConfiguration(
            backend_name="generic_quac",
            backend_version="0.0.1",
            n_qubits=n_qubits,
            basis_gates=basis_gates,
            gates=[],
            coupling_map=[],
            local=True,
            simulator=True,
            conditional=True,
            max_experiments=max_exp,
            max_shots=max_shots,
            memory=True,
            open_pulse=False
        )  # configuration for QuaC backends
