from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.quac import Quac
import matplotlib.pyplot as plt


if __name__ == '__main__':
    simulator = Quac.get_backend('fake_yorktown_counts_simulator')

    # Build the GhZ circuit over five qubits
    ghz_circ = QuantumCircuit(5, 5)
    ghz_circ.h(0)

    for qubit in range(1, 5):
        ghz_circ.cx(qubit - 1, qubit)

    quac_job = execute(ghz_circ, simulator, shots=1000)

    print(quac_job.status())
    print(quac_job.result().get_counts())
    plot_histogram(quac_job.result().get_counts())
    plt.title("GHZ States")
    plt.show()
