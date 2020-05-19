# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit, execute
from qiskit.providers.quac import Quac


def main():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)

    print("Available Quac backends:")
    print(Quac.backends())
    simulator = Quac.get_backend('density_simulator')

    # Define Lindblad noise terms
    lindblad = {
        "emission": 1e-4,
        "thermal": 1e-5,
        "coupling": 1e-5,
        "dephasing": 1e-6
    }

    # Execute the circuit on the quac simulator
    execute(circuit, simulator, shots=1, gate_times=[1, 2], lindblad=lindblad)


if __name__ == '__main__':
    main()
