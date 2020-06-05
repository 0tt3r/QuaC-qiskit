# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.quac import Quac


def main():
    circuit = QuantumCircuit(2, 2)
    circuit.u2(0, math.pi, 0)
    circuit.cx(0, 1)

    circuit2 = QuantumCircuit(2, 2)
    circuit2.h(0)
    circuit2.cx(0, 1)

    print("Available QuaC backends:")
    print(Quac.backends())
    simulator = Quac.get_backend('generic_counts_simulator')

    # Define Lindblad emission and dephasing terms
    lindblad = {
        "0": {
            "T1": 1000,
            "T2": 50000
        },
        "1": {
            "T1": 1000,
            "T2": 50000
        }
    }

    # Execute the circuit on the QuaC simulator
    job = execute([circuit, circuit2], simulator, shots=1000, lindblad=lindblad)
    print(job.result())
    print(job.result().get_counts())

    plot_histogram(job.result().get_counts(), title="Bell States Example")
    plt.show()


if __name__ == '__main__':
    main()
