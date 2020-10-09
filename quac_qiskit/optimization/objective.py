# -*- coding: utf-8 -*-

"""This module contains objective functions for optimizing noise models.
"""
from typing import List
import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.result import Result
from quac_qiskit.stat import kl_dist_smoothing, discrete_one_samp_ks, get_vec_angle
from quac_qiskit.format import counts_to_list
from quac_qiskit.models import QuacNoiseModel


def kl_div_sum(circuits: List[QuantumCircuit], simulation_result: Result, reference_result: Result) -> float:
    """Given a set of test circuits and a Qiskit Result object for a simulation and hardware, the sum of
    K-L divergence between circuit result distributions for each circuit is computed

    :param circuits: a list of QuantumCircuit objects
    :param simulation_result: a Qiskit Result object
    :param reference_result: a Qiskit Result object
    :return: a float representing the total K-L divergence between distributions of all circuits
    """
    total_kl_div = 0
    for circuit in circuits:
        simulation_counts = np.array(counts_to_list(simulation_result.get_counts(circuit)))
        simulation_dist = simulation_counts / simulation_counts.sum()  # normalize if using counts simulator
        reference_counts = np.array(counts_to_list(reference_result.get_counts(circuit)))
        reference_dist = reference_counts / reference_counts.sum()  # normalize if using counts simulator

        total_kl_div += kl_dist_smoothing(reference_dist, simulation_dist, 1e-5)

    return total_kl_div


def ks_div_sum(circuits: List[QuantumCircuit], simulation_result: Result, reference_result: Result) -> float:
    """Given a set of test circuits and a Qiskit Result object for a simulation and hardware, the sum of
    K-S distance between circuit result distributions for each circuit is computed

    :param circuits: a list of QuantumCircuit objects
    :param simulation_result: a Qiskit Result object
    :param reference_result: a Qiskit Result object
    :return: a float representing the total K-S distance between distributions of all circuits
    """
    total_ks_div = 0
    for circuit in circuits:
        simulation_counts = np.array(counts_to_list(simulation_result.get_counts(circuit)))
        simulation_dist = simulation_counts / simulation_counts.sum()  # normalize if using counts simulator
        reference_counts = np.array(counts_to_list(reference_result.get_counts(circuit)))
        reference_dist = reference_counts / reference_counts.sum()  # normalize if using counts simulator

        total_ks_div += discrete_one_samp_ks(reference_dist, simulation_dist, 8000)[0]

    return total_ks_div


def angle_div_sum(circuits: List[QuantumCircuit], simulation_result: Result, reference_result: Result) -> float:
    """Given a set of test circuits and a Qiskit Result object for a simulation and hardware, the sum of
    angle distance between circuit result distributions for each circuit is computed

    :param circuits: a list of QuantumCircuit objects
    :param simulation_result: a Qiskit Result object
    :param reference_result: a Qiskit Result object
    :return: a float representing the total angle distance between distributions of all circuits
    """
    total_angle_div = 0
    for ind, circuit in enumerate(circuits):
        simulation_counts = np.array(counts_to_list(simulation_result.get_counts(circuit)))
        simulation_dist = simulation_counts / simulation_counts.sum()  # normalize if using counts simulator
        reference_counts = np.array(counts_to_list(reference_result.get_counts(circuit)))
        reference_dist = reference_counts / reference_counts.sum()  # normalize if using counts simulator

        total_angle_div += get_vec_angle(reference_dist, simulation_dist)

    return total_angle_div


def kl_objective_function(noise_model_array: np.array, *args):
    """An objective function to be minimized based on K-L divergence

    :param noise_model_array: a Numpy array generated via QuacNoiseModel.to_array()
    :param args: QuantumCircuit objects run, the simulator to run on, and the hardware results
    :return: a float representing the "loss" over the set of circuits
    """
    circuits, backend, reference_result = args
    noise_model = QuacNoiseModel.from_array(noise_model_array, backend.configuration().n_qubits)
    simulation_result = execute(circuits, backend, shots=1, quac_noise_model=noise_model).result()

    return kl_div_sum(circuits, simulation_result, reference_result)


def ks_objective_function(noise_model_array: np.array, *args):
    """An objective function to be minimized based on K-S distance

    :param noise_model_array: a Numpy array generated via QuacNoiseModel.to_array()
    :param args: QuantumCircuit objects run, the simulator to run on, and the hardware results
    :return: a float representing the "loss" over the set of circuits
    """
    circuits, backend, reference_result = args
    noise_model = QuacNoiseModel.from_array(noise_model_array, backend.configuration().n_qubits)
    simulation_result = execute(circuits, backend, shots=1, quac_noise_model=noise_model).result()

    return ks_div_sum(circuits, simulation_result, reference_result)


def angle_objective_function(noise_model_array: np.array, *args):
    """An objective function to be minimized based on angle divergence

    :param noise_model_array: a Numpy array generated via QuacNoiseModel.to_array()
    :param args: QuantumCircuit objects run, the simulator to run on, and the hardware results
    :return: a float representing the "loss" over the set of circuits
    """
    circuits, backend, reference_result = args
    noise_model = QuacNoiseModel.from_array(noise_model_array, backend.configuration().n_qubits)
    simulation_result = execute(circuits, backend, shots=1, quac_noise_model=noise_model).result()

    return angle_div_sum(circuits, simulation_result, reference_result)
