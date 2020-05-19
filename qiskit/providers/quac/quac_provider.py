# -*- coding: utf-8 -*-

"""
This module defines a quac provider class through which all quac backend simulators can be
interfaced with.
"""

from typing import List, Optional
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.baseprovider import BaseProvider
from .quac_density_simulator import QuacDensitySimulator
from .exceptions import QuacBackendError


class QuacProvider(BaseProvider):
    """
    QuaC Provider to serve all QuaC backends
    """

    def __init__(self):
        """Initialize a QuaC provider"""
        self._backends = [QuacDensitySimulator()]
        super().__init__()

    def backends(self, name: Optional[str] = None, **kwargs) -> List[BaseBackend]:
        """
        Returns a list of all backends associated with the quac provider.
        :param name: optional name to refine backend search
        :param kwargs: optional additional params
        :return: a list of supported QuaC backends of type BaseBackend
        """
        if not name:
            return self._backends
        return [backend for backend in self._backends if name in backend.name()]

    def get_backend(self, name: Optional[str] = None, **kwargs) -> BaseBackend:
        """
        Selects a specific backend to perform quantum simulations on.
        :param name: the name of the desired backend
        :param kwargs: optional additional params
        :return: the selected backend with associated name "name"
        """
        if not name:
            return self._backends[0]  # if no name is provided, serve the first backend listed

        backends = list(filter(lambda backend: backend.name() == name, self._backends))

        if len(backends) == 0:
            raise QuacBackendError("nonexistent backend")

        return backends[0]
