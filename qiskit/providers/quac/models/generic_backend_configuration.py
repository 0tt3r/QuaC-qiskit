# -*- coding: utf-8 -*-

"""
This module contains the generic configuration for QuaC backends and can be changed as additional
gates and functionality become available.
"""

from qiskit.providers.models.backendconfiguration import QasmBackendConfiguration

generic_quac_configuration = QasmBackendConfiguration(
    backend_name="generic_quac",
    backend_version="0.0.1",
    n_qubits=2,
    basis_gates=['id', 'cx', 'h', 'x', 'y', 'z',
                 'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
                 'cxz', 'czx', 'cmz', 'cz'],
    gates=[],
    coupling_map=[],
    local=True,
    simulator=True,
    conditional=True,
    max_experiments=1,
    max_shots=1024,
    memory=True,
    open_pulse=False
)  # configuration for QuaC backends
