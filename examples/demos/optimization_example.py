# -*- coding: utf-8 -*-

"""Demonstrates plugin noise model optimization abilities by recovering T1 noise from an ideal
user-defined noise model.
"""
from qiskit import IBMQ
from quac_qiskit import Quac
from qiskit.test.mock import FakeBurlington
from qiskit.ignis.characterization.coherence import t1_circuits, t2_circuits
from quac_qiskit.optimization import *
from quac_qiskit.models import QuacNoiseModel


def main():
    # Get backends
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="chm168")
    backend = provider.get_backend("ibmq_burlington")
    plugin_backend = Quac.get_backend("fake_burlington_density_simulator", t1=True, t2=True, meas=False, zz=False)

    # Get calibration circuits
    hardware_properties = FakeBurlington().properties()
    num_gates = np.linspace(10, 500, 30, dtype='int')
    qubits = list(range(len(backend.properties().qubits)))

    t1_circs, t1_delay = t1_circuits(num_gates,
                                     hardware_properties.gate_length('id', [0]) * 1e9,
                                     qubits)

    t2_circs, t2_delay = t2_circuits((num_gates / 2).astype('int'),
                                     hardware_properties.gate_length('id', [0]) * 1e9,
                                     qubits)

    # Formulate real noise model
    real_noise_model = QuacNoiseModel(
        t1_times=[1234, 2431, 2323, 2222, 3454],
        t2_times=[12000, 14000, 14353, 20323, 30232]
    )

    # Formulate initial guess noise model (only same order of magnitude)
    guess_noise_model = QuacNoiseModel(
        t1_times=[1000, 1000, 1000, 1000, 1000],
        t2_times=[10000, 10000, 10000, 10000, 10000]
    )

    # Calculate calibration circuit reference results
    reference_job = execute(t1_circs + t2_circs, plugin_backend,
                            quac_noise_model=real_noise_model,
                            optimization_level=0)
    reference_result = reference_job.result()

    # Calculate optimized noise model
    new_noise_model = optimize_noise_model_ng(
        guess_noise_model=guess_noise_model,
        circuits=t1_circs + t2_circs,
        backend=plugin_backend,
        reference_result=reference_result,
        loss_function=angle_objective_function
    )

    # Show original noise model and optimized noise model
    print(f"Original noise model: {real_noise_model}")
    print(f"New noise model: {new_noise_model}")


if __name__ == '__main__':
    main()
