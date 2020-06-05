# -*- coding: utf-8 -*-

"""This module contains probability-related utility functions for the QuaC-Qiskit plugin
"""

from typing import List
import random


def choose_index(prob_dist: List[float]) -> int:
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

    return -1
