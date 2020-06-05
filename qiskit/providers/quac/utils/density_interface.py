# -*- coding: utf-8 -*-

"""This module provides a utility class for interacting with the density matrix calculated
by QuaC, which can be too large to load on a single machine
"""


class DensityInterface:
    """QuaC Density Interface (essentially just a template for now)
    """

    def __init_(self):
        """Initialize a density matrix interface
        """
        self._c_inf = None

    def access(self, row: int, col: int):
        """Access a specific row and column of the density matrix

        :param row: an integer
        :param col: an integer
        :return: a complex number
        """
        return self._c_inf.access(row, col)
