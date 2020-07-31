# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, IBMQ, Aer
from qiskit.providers.quac import Quac
from qiskit.ignis.characterization.coherence import *
from qiskit.ignis.characterization.coherence import T1Fitter, T2Fitter
from qiskit.ignis.mitigation.measurement import tensored_meas_cal, TensoredMeasFitter
from qiskit.providers.ibmq.ibmqbackend import JobStatus
from qiskit.providers.quac.utils import *


def main():
    # Get backends
    plugin_sim = Quac.get_backend('fake_burlington_density_simulator', meas=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="chm168")
    backend = provider.get_backend("ibmq_burlington")

    sv_sim = Aer.get_backend("statevector_simulator")

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

    meas_cal_circs, state_labels = tensored_meas_cal([[i] for i in qubits])

    # Build actual circuit
    circ_ind = [28, 82, 27, 85, 23, 83, 14, 2, 63, 81, 87, 69, 52, 75, 3, 35, 36, 11, 31, 56, 50, 6,
                73, 1, 43, 44, 96, 79, 68, 76, 37, 0, 58, 19, 89, 71, 59, 26, 7, 13, 21, 66, 20, 97,
                17, 53, 78, 18, 24, 61, 30, 16, 92]
    # while len(circ_ind) < 75 - 22:
    #     ind = random.randrange(100)
    #     if ind not in circ_ind:
    #         circ_ind.append(ind)
    # print(f"Using circuits: {circ_ind}")

    circs = {}
    for ind in circ_ind:
        circ = QuantumCircuit.from_qasm_file(f"../../test/output/untranspiled_{ind}.qasm")
        circ.name = f"circuit_{ind}"
        circs[ind] = circ

    # Get experimental calibration results
    # hardware_experiment_job = execute(t1_circs + t2_circs + meas_cal_circs + list(circs.values()), backend,
    #                                   optimization_level=0, shots=8000)
    #
    # print("Running job", end="")
    # while hardware_experiment_job.status() != JobStatus.DONE:
    #     print(".", end="")
    #     time.sleep(5)
    hardware_experiment_job = backend.retrieve_job("5f14bf0ebed433001971c4e3")

    t1_fit = T1Fitter(hardware_experiment_job.result(), t1_delay, qubits,
                      fit_p0=[1, 80000, 0],
                      fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                      time_unit="nano-seconds")

    t2_fit = T2Fitter(hardware_experiment_job.result(), t2_delay, qubits,
                      fit_p0=[1, 1e4, 0],
                      fit_bounds=([0, 0, -1], [2, 1e10, 1]),
                      time_unit="nano-seconds")

    meas_fit = TensoredMeasFitter(hardware_experiment_job.result(), [[i] for i in qubits])
    meas_matrices = meas_fit.cal_matrices

    zz = {(0, 1): 3.143236682931741e-05, (0, 2): 4.3798945172376926e-07, (0, 3): 5.710467343009174e-07,
          (0, 4): 5.255829128220091e-07, (1, 0): 2.9605797262289958e-05, (1, 2): -2.2201862048211673e-05,
          (1, 3): 1.1509998231940323e-05, (1, 4): -1.1352178756706967e-06, (2, 0): -4.654590305690487e-07,
          (2, 1): -3.062206923970773e-05, (2, 3): -3.7459810377513116e-07, (2, 4): 3.3650800003023896e-07,
          (3, 0): -6.2849536852988e-07, (3, 1): 9.268968720548915e-06, (3, 2): -9.067960754656768e-07,
          (3, 4): -4.1651763103070784e-06, (4, 0): 9.158131021284599e-07, (4, 1): 1.0867636225178612e-07,
          (4, 2): 7.950744417123402e-08, (4, 3): -5.827315040873638e-06}  # ZZ values calibrated externally

    lindblad = {}

    for i in range(len(qubits)):
        print(f"Qubit {i} T1 is {t1_fit.time(i)} ns and T2 is {t2_fit.time(i)} ns")
        lindblad[str(i)] = {}
        lindblad[str(i)]["T1"] = t1_fit.time(i)
        lindblad[str(i)]["T2"] = t2_fit.time(i)

    # Get actual hardware circuit results
    hardware_dist = {}
    for ind, circ in circs.items():
        counts = hardware_experiment_job.result().get_counts(circ)
        counts = counts_to_list(counts)
        norm = sum(counts)
        hardware_dist[ind] = [count / norm for count in counts]

    # Get plugin simulation results
    plugin_dist = {}
    for ind, circ in circs.items():
        dist = execute(circ, plugin_sim, shots=8000,
                       lindblad=lindblad, meas_mats=meas_matrices, zz=zz,
                       optimization_level=0).result().get_counts()
        plugin_dist[ind] = counts_to_list(dist)

    # Get theoretical, noiseless results
    sv_dist = {}
    for ind, circ in circs.items():
        circ.remove_final_measurements()
        dist = execute(circ, sv_sim, shots=8000,
                       optimization_level=0)
        sv_dist[ind] = qiskit_statevector_to_probabilities(dist.result().get_statevector(circ),
                                                           circ.num_qubits)

    # Perform comparisons
    j = 7
    print(f"Hardware\tPlugin\tSV")
    for i in range(len(hardware_dist[j])):
        print(f"{hardware_dist[j][i]}\t{plugin_dist[j][i]}\t{sv_dist[j][i]}")

    angles = []
    kls = []
    kss = []
    for ind in circs:
        angles.append(get_vec_angle(hardware_dist[ind], plugin_dist[ind]))
        kls.append(kl_dist_smoothing(hardware_dist[ind], plugin_dist[ind], 1e-5))
        kss.append(discrete_one_samp_ks(hardware_dist[ind], plugin_dist[ind], 8000))
        print(f"""
            Circuit: {ind}
            KL divergence: {kl_dist_smoothing(hardware_dist[ind], plugin_dist[ind], 1e-5)}
            Angle divergence: {get_vec_angle(hardware_dist[ind], plugin_dist[ind])}
            KS D value: {discrete_one_samp_ks(hardware_dist[ind], plugin_dist[ind], 8000)}
        """)

    print(f"""
        ------
        Angle max: {max(angles)}
        KL Divergence max: {max(kls)}
        KS Divergence max: {max(kss)}
    """)

    print(f"""
        ------
        Angle avg: {sum(angles) / len(angles)}
        KL Divergence avg: {sum(kls) / len(kls)}
        KS Divergence avg: {sum([e[0] for e in kss]) / len(kss)}
    """)

    print(f"""
        ------
        Angle min: {min(angles)}
        KL Divergence min: {min(kls)}
        KS Divergence min: {min(kss)}
    """)


if __name__ == '__main__':
    main()
