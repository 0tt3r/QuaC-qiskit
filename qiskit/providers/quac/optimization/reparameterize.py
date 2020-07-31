"""This module contains functions for reparameterizing noise models.
"""

from typing import List, Dict
import numpy as np
from scipy import optimize
import nevergrad as ng
from qiskit import QuantumCircuit, execute
from qiskit.providers import BaseBackend
from qiskit.result import Result
from qiskit.providers.quac.utils import kl_dist_smoothing, counts_to_list, discrete_one_samp_ks


def kl_div_sum(circuits: List[QuantumCircuit], simulation_result: Result, reference_result: Result) -> float:
    """Given a set of test circuits and a Qiskit Result object for a simulation and hardware, the sum of
    K-L divergence between circuit result distributions for each circuit is computed.
    :param circuits: a list of QuantumCircuit objects
    :param simulation_result: a Qiskit Result object
    :param reference_result: a Qiskit Result object
    :return: a float representing the total K-L divergence between distributions of all circuits
    """
    total_kl_div = 0
    for circuit in circuits:
        simulation_counts = np.array(counts_to_list(simulation_result.get_counts(circuit)))
        simulation_dist = simulation_counts / np.linalg.norm(simulation_counts)  # normalize if using counts simulator
        reference_counts = np.array(counts_to_list(reference_result.get_counts(circuit)))
        reference_dist = reference_counts / np.linalg.norm(reference_counts)  # normalize if using counts simulator

        total_kl_div += kl_dist_smoothing(reference_dist, simulation_dist, 1e-5)

    return total_kl_div


def ks_div_sum(circuits: List[QuantumCircuit], simulation_result: Result, reference_result: Result) -> float:
    """Given a set of test circuits and a Qiskit Result object for a simulation and hardware, the sum of
    K-S distance between circuit result distributions for each circuit is computed.
    :param circuits: a list of QuantumCircuit objects
    :param simulation_result: a Qiskit Result object
    :param reference_result: a Qiskit Result object
    :return: a float representing the total K-S distance between distributions of all circuits
    """
    total_ks_div = 0
    for circuit in circuits:
        simulation_counts = np.array(counts_to_list(simulation_result.get_counts(circuit)))
        simulation_dist = simulation_counts / np.linalg.norm(simulation_counts)  # normalize if using counts simulator
        reference_counts = np.array(counts_to_list(reference_result.get_counts(circuit)))
        reference_dist = reference_counts / np.linalg.norm(reference_counts)  # normalize if using counts simulator

        total_ks_div += discrete_one_samp_ks(reference_dist, simulation_dist, 8000)[0]

    return total_ks_div


def lindblad_objective(lindblad: np.array, circuits: List[QuantumCircuit],
                       backend: BaseBackend, reference_result: Result) -> float:
    """An objective function with T1 and T2 values as variables. Designed to fit the NeverGrad
    optimizers.
    :param lindblad: an array of form [T1_0, T2_0, T1_1, ...]
    :param circuits: a list of QuantumCircuit objects
    :param backend: a BaseBackend object
    :param reference_result: a Qiskit Result object
    :return: a float indicating the performance of input lindblad parameters
    """
    # Unpack lindblad terms from array
    formatted_lindblad = {}
    for qubit in range(backend.configuration().n_qubits):
        formatted_lindblad[str(qubit)] = {}
        formatted_lindblad[str(qubit)]["T1"] = lindblad[2 * qubit]
        # print(lindblad[2 * qubit] * l_norm)
        formatted_lindblad[str(qubit)]["T2"] = lindblad[2 * qubit + 1]

    # Return K-S divergence of simulation from reference
    simulation_result = execute(circuits, backend, lindblad=formatted_lindblad).result()
    # print(kl_div_sum(circuits, simulation_result, reference_result))
    return ks_div_sum(circuits, simulation_result, reference_result)


def lindblad_objective_scipy(lindblad: np.array, *args) -> float:
    """An objective function with T1 and T2 values as variables. Designed to fit the SciPy
    optimize module minimizer.
    :param lindblad: an array of form [T1_0, T2_0, T1_1, ...]
    :param args: a tuple of form (a list of QuantumCircuit object, a backend, a Result object)
    :return: a float indicating the performance of input lindblad parameters
    """
    # Unpack lindblad terms from array
    circuits, backend, reference_result = args
    formatted_lindblad = {}
    for qubit in range(backend.configuration().n_qubits):
        formatted_lindblad[str(qubit)] = {}
        formatted_lindblad[str(qubit)]["T1"] = lindblad[2 * qubit]
        formatted_lindblad[str(qubit)]["T2"] = lindblad[2 * qubit + 1]

    # Return K-S divergence of simulation from reference
    print(formatted_lindblad)
    simulation_result = execute(circuits, backend, lindblad=formatted_lindblad).result()
    print(f"KS: {ks_div_sum(circuits, simulation_result, reference_result)}")

    return ks_div_sum(circuits, simulation_result, reference_result)


def optimize_lindblad(lindblad_guess: Dict[str, Dict[str, float]], circuits: List[QuantumCircuit],
                      backend: BaseBackend, reference_result: Result) -> Dict[str, Dict[str, float]]:
    """Refines T1 and T2 noise parameters until the divergence between the probability distributions
    of a provided list of circuits simulated with the plugin and run with the hardware is minimized.
    :param lindblad_guess: a T1/T2 dictionary near the actual T1/T2 dictionary
    :param circuits: a list of QuantumCircuit objects
    :param backend: the backend on which the circuits were run
    :param reference_result: the Qiskit Result object from running circuits on the hardware backend
    :return: a T1/T2 dictionary of optimized parameters
    """
    # Formulate a 1D array of T1 and T2 values for the optimizer
    lindblad_guess_array = []
    for qubit in lindblad_guess:
        lindblad_guess_array.append(lindblad_guess[qubit]["T1"])
        lindblad_guess_array.append(lindblad_guess[qubit]["T2"])
    lindblad_guess_array = np.array(lindblad_guess_array)

    # Define objective function schema
    lindblad_array_data_type = ng.p.Array(init=lindblad_guess_array)
    lindblad_lower = np.zeros(2 * backend.configuration().n_qubits)
    lindblad_upper = np.array([float('inf') for _ in range(2 * backend.configuration().n_qubits)])
    lindblad_array_data_type.set_bounds(lindblad_lower, lindblad_upper)

    # param = ng.p.Instrumentation(
    #     lindblad_array_data_type,
    #     circuits,
    #     backend,
    #     reference_result
    # )

    # Define and run differential evolution optimizer
    # optimizer = ng.optimizers.TwoPointsDE(parametrization=param, budget=10000, num_workers=20)
    #
    # with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    #     recommendation = optimizer.minimize(lindblad_objective, executor=executor, batch_mode=False)
    #     print(f"Recommendation loss: {recommendation.loss}")
    #
    # optimized_lindblad_array = recommendation.value[0][0]

    optimized_lindblad = {}
    result = optimize.minimize(lindblad_objective_scipy, lindblad_guess_array,
                               args=(circuits, backend, reference_result), method="Powell")

    optimized_lindblad_array = result.x

    for qubit in range(backend.configuration().n_qubits):
        optimized_lindblad[str(qubit)] = {}
        optimized_lindblad[str(qubit)]["T1"] = optimized_lindblad_array[qubit * 2]
        optimized_lindblad[str(qubit)]["T2"] = optimized_lindblad_array[qubit * 2 + 1]

    return optimized_lindblad
