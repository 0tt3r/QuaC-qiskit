# -*- coding: utf-8 -*-

"""
This module contains the configuration for all quac backends and can be changed as additional
gates and functionalities become available.
"""

from qiskit.providers.models.backendconfiguration import QasmBackendConfiguration

configuration = QasmBackendConfiguration(
    backend_name="density_simulator",
    backend_version="0.0.1",
    n_qubits=5,
    basis_gates=['id', 'cx', 'h', 'x', 'y', 'z', 'rx', 'ry', 'rz'],
    gates=[],
    coupling_map=[],
    local=True,
    simulator=True,
    conditional=True,
    max_experiments=1,
    max_shots=1,
    memory=True,
    open_pulse=False
)  # configuration for quac backends
