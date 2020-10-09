# -*- coding: utf-8 -*-

"""This module contains comparison-related statistics (or metric) functions for the QuaC-Qiskit plugin
"""
from typing import List, Tuple, Union
import warnings
import math
import numpy as np
from scipy.special import kl_div


def get_vec_angle(vec1: List, vec2: List) -> Union[float, None]:
    """Calculates the degree angle between two vectors

    :param vec1: a list of vector entries
    :param vec2: a list of vector entries
    :return: a float
    """
    if np.linalg.norm(np.array(vec1)) == 0 or np.linalg.norm(np.array(vec2)) == 0:
        warnings.warn("Do not input 0 vector")
        return

    diff_degree = np.dot(np.array(vec1), np.array(vec2))
    diff_degree /= np.linalg.norm(np.array(vec1))
    diff_degree /= np.linalg.norm(np.array(vec2))
    diff_degree = np.clip(diff_degree, -1, 1)
    diff_degree = np.arccos(diff_degree) * 180 / np.pi
    return diff_degree


def kl_dist_smoothing(distribution1: np.array, distribution2: np.array, epsilon: float) -> float:
    """Calculates the Kullback-Leibler Divergence of distribution2 from distribution1

    :param distribution1: a numpy array representing a probability distribution
    :param distribution2: a numpy array representing a probability distribution
    :param epsilon: a smoothing parameter
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


def discrete_one_samp_ks(distribution1: np.array, distribution2: np.array, num_samples: int) -> Tuple[float, bool]:
    """Uses the one-sample Kolmogorov-Smirnov test to determine if the empirical results in
    distribution1 come from the distribution represented in distribution2

    :param distribution1: empirical distribution (numpy array)
    :param distribution2: reference distribution (numpy array)
    :param num_samples: number of samples used to generate distribution1
    :return: a tuple (D, D<D_{alpha})
    """
    cutoff = 1.36 / math.sqrt(num_samples)
    ecdf1 = np.array([sum(distribution1[:i + 1]) for i in range(len(distribution1))])
    ecdf2 = np.array([sum(distribution2[:i + 1]) for i in range(len(distribution2))])
    max_diff = np.absolute(ecdf1 - ecdf2).max()
    return max_diff, max_diff < cutoff
