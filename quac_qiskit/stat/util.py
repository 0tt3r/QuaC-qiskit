# -*- coding: utf-8 -*-

"""This module contains probability-related utility functions for the QuaC-Qiskit plugin.
"""
from typing import List, Union
import random
import numpy as np


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
