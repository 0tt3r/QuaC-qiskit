# -*- coding: utf-8 -*-

"""This module that contains a class that expands on the native Qiskit gate set
"""
from qiskit.quantum_info import Operator


class SpecialQuacGates:
    """Contains special gates supported in QuaC"""

    @staticmethod
    def get_gate_unitary(name: str) -> Operator:
        """Returns a Qiskit operator for various types of QuaC-supported (but not Qiskit-supported)
        gates that can be used to construct and add a unitary representing said gate to a Qiskit
        circuit.

        :param name: the name of the special QuaC-supported gate (czx, cmz, or cxz)
        :return: an operator corresponding to the gate name provided. If the name is unrecognized,
            a 4x4 identity operator will be returned
        """

        if name.lower() == "czx":
            return Operator([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1],
                [0, 0, 1, 0]
            ])
        elif name.lower() == "cmz":
            return Operator([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
        elif name.lower() == "cxz":
            return Operator([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, -1, 0]
            ])
        return Operator([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
