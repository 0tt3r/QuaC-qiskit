# -*- coding: utf-8 -*-

"""In this example, two-qubit Bell states are generated using the plugin. Additionally,
a Quac noise model is applied.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
from quac_qiskit import Quac
from quac_qiskit.models import QuacNoiseModel


def main():
    circuit = QuantumCircuit(2, 2)
    circuit.u2(0, math.pi, 0)
    circuit.cx(0, 1)
    circuit.measure_all()

    circuit2 = QuantumCircuit(2, 2)
    circuit2.h(0)
    circuit2.cx(0, 1)
    circuit2.measure_all()

    print("Available QuaC backends:")
    print(Quac.backends())
    simulator = Quac.get_backend('generic_counts_simulator',
                                 n_qubits=2,
                                 max_shots=10000,
                                 max_exp=75,
                                 basis_gates=['u1', 'u2', 'u3', 'cx'])  # generic backends require extra parameters

    # Noise model with T1, T2, and measurement error terms
    noise_model = QuacNoiseModel(
        [1000, 1000],
        [50000, 50000],
        [np.eye(2), np.array([[0.95, 0.1], [0.05, 0.9]])],
        None
    )

    # Execute the circuit on the QuaC simulator
    job = execute([circuit, circuit2], simulator, shots=1000, quac_noise_model=noise_model)
    print(job.result())
    print(job.result().get_counts())

    plot_histogram(job.result().get_counts(), title="Bell States Example")
    plt.show()


if __name__ == '__main__':
    main()
