# -*- coding: utf-8 -*-

import numpy as np
from qiskit import IBMQ, execute
from qiskit.providers.quac import Quac
from qiskit.test.mock import FakeBurlington
from qiskit.ignis.characterization.coherence import t1_circuits, t2_circuits
from qiskit.providers.quac.optimization import optimize_lindblad


def main():
    # Get backends
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="chm168")
    backend = provider.get_backend("ibmq_burlington")
    plugin_backend = Quac.get_backend("fake_burlington_density_simulator", meas=True)

    # Get calibration circuits
    hardware_properties = FakeBurlington().properties()
    num_gates = np.linspace(1, 11, 10, dtype='int')
    qubits = list(range(len(backend.properties().qubits)))

    t1_circs, t1_delay = t1_circuits(num_gates,
                                     hardware_properties.gate_length('id', [0]) * 1e9,
                                     qubits)

    t2_circs, t2_delay = t2_circuits(num_gates,
                                     hardware_properties.gate_length('id', [0]) * 1e9,
                                     qubits)

    # Formulate initial guess lindblad
    lindblad = {}
    for qubit_index in range(len(qubits)):
        print(hardware_properties.qubit_property(qubit_index, "T1")[0] * 1e9)
        print(hardware_properties.qubit_property(qubit_index, "T2")[0] * 1e9)
        print("------")
        lindblad[str(qubit_index)] = {}
        lindblad[str(qubit_index)]["T1"] = hardware_properties.qubit_property(qubit_index, "T1")[0] * 1e9 - 1000
        lindblad[str(qubit_index)]["T2"] = hardware_properties.qubit_property(qubit_index, "T2")[0] * 1e9 + 5000

    # Calculate calibration circuit reference results
    reference_results = execute(t1_circs + t2_circs, plugin_backend).result()

    # Calculate optimized lindblad
    new_lindblad = optimize_lindblad(
        lindblad,
        t1_circs + t2_circs,
        plugin_backend,
        reference_results
    )

    # Show original lindblad and optimized lindblad
    print(f"Original lindblad: {lindblad}")
    print(f"New lindblad: {new_lindblad}")


if __name__ == '__main__':
    main()
