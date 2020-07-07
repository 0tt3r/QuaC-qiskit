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
    plugin_sim = Quac.get_backend('fake_yorktown_density_simulator', meas=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="chm168")
    backend = provider.get_backend("ibmq_5_yorktown")

    # Get calibration circuits
    hardware_props = backend.properties()
    num_gates = np.linspace(10, 300, 10, dtype='int')
    qubits = list(range(len(backend.properties().qubits)))

    t1_circs, t1_delay = t1_circuits(num_gates,
                                     hardware_props.gate_length('id', [0]) * 1e9,
                                     qubits)

    t2_circs, t2_delay = t2_circuits(np.floor(num_gates / 2).astype('int'),
                                     hardware_props.gate_length('id', [0]) * 1e9,
                                     qubits)

    # Build actual circuit
    # circ = QuantumCircuit(3)
    # circ.y(0)
    # circ.x(1)
    # circ.y(2)
    # circ.cx(0, 1)
    # circ.x(2)
    # circ.y(0)
    # circ.y(2)
    # circ.cx(0, 2)
    # circ.measure_all()
    circ_ind = [32, 35, 47, 67, 79, 84, 93, 96, 99]
    circs = []
    for ind in circ_ind:
        circ = QuantumCircuit.from_qasm_file(f"../../test/output/untranspiled_{ind}.qasm")
        circs.append(circ)

    # Get experimental results
    # hardware_experiment_job = execute(t1_circs + t2_circs + circs, backend, optimization_level=0)
    #
    # print("Running job", end="")
    # while hardware_experiment_job.status() != JobStatus.DONE:
    #     print(".", end="")
    #     time.sleep(5)
    hardware_experiment_job = backend.retrieve_job("5f04c329790ba000127184e9")

    t1_fit = T1Fitter(hardware_experiment_job.result(), t1_delay, qubits,
                      fit_p0=[1, 80000, 0],
                      fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                      time_unit="nano-seconds")

    t2_fit = T2Fitter(hardware_experiment_job.result(), t2_delay, qubits,
                      fit_p0=[1, 1e4, 0],
                      fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                      time_unit="nano-seconds")

    lindblad = {}

    for i in range(len(qubits)):
        print(f"Qubit {i} T1 is {t1_fit.time(i)} ns and T2 is {t2_fit.time(i)} ns")
        lindblad[str(i)] = {}
        lindblad[str(i)]["T1"] = t1_fit.time(i)
        lindblad[str(i)]["T2"] = t2_fit.time(i)

    # Get plugin simulation results
    plugin_experiment_job = execute(circs, plugin_sim, optimization_level=0, lindblad=lindblad)

    # Perform distribution comparisons
    for circ in circs:
        real_counts = counts_to_list(hardware_experiment_job.result().get_counts(circ))
        real_distribution = [count / sum(real_counts) for count in real_counts]
        plugin_distribution = counts_to_list(plugin_experiment_job.result().get_counts(circ))

        print(f"Counts Comparison for Circuit {circ.name}\n------------------------")
        for cind in range(len(plugin_distribution)):
            # if real_distribution[ind] < 0.01:
            #     real_distribution[ind] = 0
            # if plugin_distribution[ind] < 0.01:
            #     plugin_distribution[ind] = 0
            print(f"{real_distribution[cind]}\t{plugin_distribution[cind]}")

        print(f"Angle difference: {get_vec_angle(real_distribution, plugin_distribution)}")

        ks_pvalue = ks_2samp(real_distribution, plugin_distribution)[1]
        if ks_pvalue > 0.05:
            print(f"Distributions are identical (p={ks_pvalue}).")
        else:
            print(f"Distributions are NOT identical (p={ks_pvalue}).")


if __name__ == '__main__':
    main()
