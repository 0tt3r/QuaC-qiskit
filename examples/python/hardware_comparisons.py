# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.stats import ks_2samp
from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.providers.quac import Quac
from qiskit.ignis.characterization.coherence import *
from qiskit.ignis.characterization.coherence import T1Fitter, T2Fitter
from qiskit.providers.ibmq.ibmqbackend import JobStatus
from qiskit.providers.quac.utils import counts_to_list, get_vec_angle


def main():
    # Get backends
    plugin_sim = Quac.get_backend('fake_vigo_density_simulator', meas=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="chm168")
    backend = provider.get_backend("ibmq_vigo")

    # Get calibration circuits
    hardware_props = backend.properties()
    num_gates = np.linspace(10, 300, 10, dtype='int')
    qubits = list(range(5))

    t1_circs, t1_delay = t1_circuits(num_gates,
                                     hardware_props.gate_length('id', [0]) * 1e9,
                                     qubits)

    t2_circs, t2_delay = t2_circuits(np.floor(num_gates / 2).astype('int'),
                                     hardware_props.gate_length('id', [0]) * 1e9,
                                     qubits)

    # Build actual circuit
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

    # Get experimental results

    # hardware_experiment_job = execute(t1_circs + t2_circs + [circ], backend,
    #                                   optimization_level=0)
    #
    # print("Running job", end="")
    # while hardware_experiment_job.status() != JobStatus.DONE:
    #     print(".", end="")
    #     time.sleep(0.5)
    hardware_experiment_job = backend.retrieve_job("5f036eca256e510012518f3e")

    t1_fit = T1Fitter(hardware_experiment_job.result(), t1_delay, qubits,
                      fit_p0=[1, 80000, 0],
                      fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                      time_unit="nano-seconds")

    t2_fit = T2Fitter(hardware_experiment_job.result(), t2_delay, qubits,
                      fit_p0=[1, 1e4, 0],
                      fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                      time_unit="nano-seconds")

    lindblad = {}

    for i in range(5):
        print(f"Qubit {i} T1 is {t1_fit.time(i)} and T2 is {t2_fit.time(i)}")
        lindblad[str(i)] = {}
        lindblad[str(i)]["T1"] = t1_fit.time(i)
        lindblad[str(i)]["T2"] = t2_fit.time(i)

    # Get plugin simulation results
    plugin_experiment_result = execute(circ, plugin_sim,
                                       optimization_level=0, lindblad=lindblad).result()

    # Perform distribution comparisons
    real_results = counts_to_list(hardware_experiment_job.result().get_counts(circ))
    normalized_real_results = [element / sum(real_results) for element in real_results]
    print(f"Real results: {normalized_real_results}")
    print(f"Plugin results: {counts_to_list(plugin_experiment_result.get_counts())}")
    print(
        f"Difference: {get_vec_angle(normalized_real_results, counts_to_list(plugin_experiment_result.get_counts()))}"
    )

    real_distribution = normalized_real_results
    plugin_distribution = counts_to_list(plugin_experiment_result.get_counts())
    ks_pvalue = ks_2samp(real_distribution, plugin_distribution)[1]

    if ks_pvalue > 0.05:
        print(f"Distributions are identical (p={ks_pvalue}).")


if __name__ == '__main__':
    main()
