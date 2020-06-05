# -*- coding: utf-8 -*-

"""This module contains various kinds of QuaC-related exceptions a user may encounter"""


class QuacBackendError(Exception):
    """
    Exception to throw if a nonexistent QuaC backend is found
    """

    def __init__(self, message):
        """
        Initialize backend not found exception

        :param message: exception message
        """
        self.message = message
        super().__init__(message)


class QuacOptionsError(Exception):
    """
    Exception to throw if bad options are given to a backend simulator
    """

    def __init__(self, message):
        """
        Initialize faulty execution options exception

        :param message: exception message
        """
        self.message = message
        super().__init__(message)
