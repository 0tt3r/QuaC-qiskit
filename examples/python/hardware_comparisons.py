# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.providers.quac import Quac
from qiskit.providers.ibmq.ibmqbackend import JobStatus
import time
from qiskit.providers.quac.utils import counts_to_list, get_vec_angle


def main():
    # Build a circuit
    circ = QuantumCircuit(3)
    circ.y(0)
    circ.x(1)
    circ.y(2)
    circ.cx(0, 1)
    circ.x(2)
    circ.y(0)
    circ.y(2)
    circ.cx(0, 2)
    circ.measure_all()

    # Recent T1 and T2 times from the Ourense backend
    lindblad = {
        "0": {
            "T1": 152.083899529714e3,
            "T2": 82.9734836989614e3
        },
        "1": {
            "T1": 92.0450753975342e3,
            "T2": 30.54670590822364e3
        },
        "2": {
            "T1": 112.555615194585e3,
            "T2": 155.800917673813e3
        },
        "3": {
            "T1": 106.954896165079e3,
            "T2": 95.6895178797113e3
        },
        "4": {
            "T1": 50.48821212593e3,
            "T2": 26.0491447427924e3
        }
    }

    plugin_sim = Quac.get_backend('fake_ourense_density_simulator', meas=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="chm168")
    backend = provider.get_backend("ibmq_ourense")

    plugin_experiment_result = execute(circ, plugin_sim, optimization_level=0, lindblad=lindblad).result()
    backend_job = execute(circ, backend, optimization_level=0, shots=8000)

    print("Running job", end="")
    while backend_job.status() != JobStatus.DONE:
        print(".", end="")
        time.sleep(0.5)

    real_results = counts_to_list(backend_job.result().get_counts())
    normalized_real_results = [element / sum(real_results) for element in real_results]
    print(f"Real results: {normalized_real_results}")
    print(plugin_experiment_result.get_counts())
    print(f"Plugin results: {counts_to_list(plugin_experiment_result.get_counts())}")
    print(f"Difference: {get_vec_angle(normalized_real_results, counts_to_list(plugin_experiment_result.get_counts()))}")


if __name__ == '__main__':
    main()
