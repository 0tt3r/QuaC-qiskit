# -*- coding: utf-8 -*-

"""This module contains objective functions for optimizing noise models.
"""
from concurrent import futures
from typing import List
import numpy as np
from typing import Callable
from scipy import optimize
import nevergrad as ng
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.result import Result
from quac_qiskit.models import QuacNoiseModel


def optimize_noise_model_ng(guess_noise_model: QuacNoiseModel, circuits: List[QuantumCircuit],
                            backend: BaseBackend, reference_result: Result,
                            loss_function: Callable[[np.array], float]) -> QuacNoiseModel:
    """Refines T1 and T2 noise parameters until the divergence between the probability distributions
    of a provided list of circuits simulated with the plugin and run with the hardware is minimized
    using the Nevergrad TwoPointsDE optimizer

    :param guess_noise_model: a QuacNoiseModel object as an initial ideal guess
    :param circuits: a list of QuantumCircuit objects for loss computation
    :param backend: the backend simulator on which circuits should be run
    :param reference_result: the Qiskit Result object from running circuits on real hardware
    :param loss_function: the loss function that should be used for optimization (i.e., kl_objective_function)
    :return: an optimized QuacNoiseModel object
    """
    arr = ng.p.Array(init=guess_noise_model.to_array())
    arr.set_bounds(0, float('inf'))
    param = ng.p.Instrumentation(arr, circuits, backend, reference_result)

    optimizer = ng.optimizers.TwoPointsDE(parametrization=param, budget=100, num_workers=5)
    print(optimizer.num_workers)
    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        result = optimizer.minimize(loss_function, executor=executor, batch_mode=True, verbosity=2)

    print(result[0][0].value)
    return QuacNoiseModel.from_array(result[0][0].value, backend.configuration().n_qubits)


def optimize_noise_model(guess_noise_model: QuacNoiseModel, circuits: List[QuantumCircuit],
                         backend: BaseBackend, reference_result: Result,
                         loss_function: Callable[[np.array], float]) -> QuacNoiseModel:
    """Refines T1 and T2 noise parameters until the divergence between the probability distributions
    of a provided list of circuits simulated with the plugin and run with the hardware is minimized

    :param guess_noise_model: a QuacNoiseModel object as an initial ideal guess
    :param circuits: a list of QuantumCircuit objects for loss computation
    :param backend: the backend simulator on which circuits should be run
    :param reference_result: the Qiskit Result object from running circuits on real hardware
    :param loss_function: the loss function that should be used for optimization (i.e., kl_objective_function)
    :return: an optimized QuacNoiseModel object
    """
    result = optimize.minimize(loss_function, guess_noise_model.to_array(),
                               args=(circuits, backend, reference_result), method="Nelder-Mead",
                               options={"disp": True, "adaptive": True})

    return result.x
