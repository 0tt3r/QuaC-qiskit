# -*- coding: utf-8 -*-

"""This example demonstrates the applying measurement error to a GHZ circuit in the
plugin. Additionally, the use of multiple classical registers is demonstrated.
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram
from quac_qiskit import Quac
from quac_qiskit.format import quac_time_qasm_transpiler
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Note: measurement errors are included by setting the "meas" option to True
    # Qubit         P(0 | 1)              P(1 | 0)
    #  0	    0.10799999999999998	       0.024
    #  1	    0.039000000000000035	   0.004
    #  2	    0.026	                   0.009000000000000008
    #  3	    0.03400000000000003	       0.005
    #  4	    0.135             	       0.019
    print(Quac.backends())
    simulator = Quac.get_backend('fake_bogota_counts_simulator', t1=False, t2=False, meas=True, zz=True)

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

    print("TIMEQASM of GHZ Circuit:")
    print(quac_time_qasm_transpiler(ghz_circ, simulator))
