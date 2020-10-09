# -*- coding: utf-8 -*-

"""A brief example showing the workflow of simulation with the plugin.
"""
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute
from qiskit.tools.visualization import plot_histogram
from quac_qiskit import Quac


def main():
    circuit1 = QuantumCircuit(5, 5)
    circuit2 = QuantumCircuit(5, 5)
    circuit3 = QuantumCircuit(1, 1)

    circuit1.h(0)
    circuit2.u2(0, math.pi, 0)
    circuit3.u3(math.pi/2, math.pi, 0, 0)

    circuit1.measure(0, 0)
    circuit2.measure(0, 0)
    circuit3.measure(0, 0)

    print("Available QuaC backends:")
    print(Quac.backends())
    simulator = Quac.get_backend('fake_vigo_density_simulator', meas=True)

    # Execute the circuit on the QuaC simulator
    job1 = execute(circuit1, simulator)
    job2 = execute(circuit2, simulator)
    job3 = execute(circuit3, simulator)

    print(f"Hadamard counts: {job1.result().get_counts()}")
    print(f"U2 counts: {job2.result().get_counts()}")
    print(f"U3 counts: {job3.result().get_counts()}")

    plot_histogram(job1.result().get_counts())
    plot_histogram((job2.result().get_counts()))
    plot_histogram(job3.result().get_counts())
    plt.show()


if __name__ == '__main__':
    main()
