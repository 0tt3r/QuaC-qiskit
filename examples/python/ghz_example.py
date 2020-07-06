# -*- coding: utf-8 -*-

# This example demonstrates the applying measurement error to a GHZ circuit in the
# plugin. Additionally, the use of multiple classical registers is demonstrated.

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.quac import Quac
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Note: measurement errors are included by setting the "meas" option to true
    # There is a larger frequency of state |00100> and |11011> because the third
    # qubit of this hardware is particularly noisy compared to the others:
    # Qubit  |  P(0 | 1)  |  P(1 | 0)
    #  0         0.012        0.021
    #  1         0.027        0.007
    #  2         0.376        0.225
    #  3         0.025        0.031
    #  4         0.053        0.021
    simulator = Quac.get_backend('fake_yorktown_counts_simulator', meas=True)

    # Build the GhZ circuit over five qubits
    quantum_register = QuantumRegister(4, "qreg")
    classical_register1 = ClassicalRegister(2, "creg1")
    classical_register2 = ClassicalRegister(2, "creg2")

    ghz_circ = QuantumCircuit()
    ghz_circ.add_register(quantum_register)
    ghz_circ.add_register(classical_register1)
    ghz_circ.add_register(classical_register2)
    ghz_circ.h(0)

    for qubit in range(1, 4):
        ghz_circ.cx(qubit - 1, qubit)

    ghz_circ.measure(quantum_register[0], classical_register1[0])
    ghz_circ.measure(quantum_register[1], classical_register1[1])
    ghz_circ.measure(quantum_register[2], classical_register2[0])
    ghz_circ.measure(quantum_register[3], classical_register2[1])

    quac_job = execute(ghz_circ, simulator, shots=1000)

    # print(quac_job.status())
    print(f"Frequency results: {quac_job.result().get_counts()}")
    plot_histogram(quac_job.result().get_counts())
    plt.title("GHZ States")
    plt.show()
