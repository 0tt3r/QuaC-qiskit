# -*- coding: utf-8 -*-

import math
from qiskit import QuantumCircuit, execute
from qiskit.providers.quac import Quac


def main():
    circuit1 = QuantumCircuit(1, 1)
    circuit2 = QuantumCircuit(1, 1)
    circuit3 = QuantumCircuit(1, 1)

    circuit1.h(0)
    circuit2.u2(0, math.pi, 0)
    circuit3.u3(math.pi/2, 0, math.pi, 0)

    print("Available QuaC backends:")
    print(Quac.backends())
    simulator = Quac.get_backend('fake_vigo_counts_simulator')

    # Execute the circuit on the QuaC simulator
    job1 = execute(circuit1, simulator, shots=1000)
    job2 = execute(circuit2, simulator, shots=1000)
    job3 = execute(circuit3, simulator, shots=1000)

    print(f"Hadamard counts: {job1.result().get_counts()}")
    print(f"U2 counts: {job2.result().get_counts()}")
    print(f"U3 counts: {job3.result().get_counts()}")


if __name__ == '__main__':
    main()
