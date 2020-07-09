# -*- coding: utf-8 -*-

"""This module contains math-related utility functions for the QuaC-Qiskit plugin
"""

from typing import List, Union
import random
import numpy as np
from scipy.special import kl_div


def choose_index(prob_dist: Union[List[float], np.array]) -> int:
    """Chooses an index i from a list l with probability l[i]

    :param prob_dist: a list of floating point probabilities
    :type prob_dist: List[float]
    :return: an integer representing the chosen index
    """

    chooser = random.random()
    upper_limit = 0
    lower_limit = 0

    for i, prob in enumerate(prob_dist):
        upper_limit += prob
        if lower_limit <= chooser < upper_limit:
            return i
        lower_limit += prob

    # return -1  #  TODO: update when bitstring bug is fixed
    return 0


def get_vec_angle(vec1: List, vec2: List) -> float:
    """Calculates the degree angle between to vectors
    :param vec1: a list of vector entries
    :param vec2: a list of vector entries
    :return: a float
    """
    diff_degree = np.dot(np.array(vec1), np.array(vec2))
    diff_degree /= np.linalg.norm(np.array(vec1))
    diff_degree /= np.linalg.norm(np.array(vec2))
    diff_degree = np.clip(diff_degree, -1, 1)
    diff_degree = np.arccos(diff_degree) * 180 / np.pi
    return diff_degree


def kl_dist_smoothing(distribution1: np.array, distribution2: np.array, epsilon: float) -> float:
    """Calculates the Kullback-Leibler Divergence of distribution2 from distribution1.
    :param distribution1: a numpy array representing a probability distribution
    :param distribution2: a numpy array representing a probability distribution
    :param epsilon: the smoothing parameter
    :return: the Kullback-Leibler divergence
    """
    # Performs smoothing
    distributions = [distribution1, distribution2]
    smoothed_distributions = []
    for distribution in distributions:
        nonzeros = np.count_nonzero(distribution)
        zeros = len(distribution) - nonzeros
        smoothed_distributions.append([epsilon if prob == 0 else prob - zeros * epsilon / nonzeros
                                       for prob in distribution])

    return sum(kl_div(smoothed_distributions[0], smoothed_distributions[1]))
