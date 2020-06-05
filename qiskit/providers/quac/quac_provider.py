# -*- coding: utf-8 -*-

"""
This module defines a QuaC provider class through which all QuaC backend simulators can be
interfaced with
"""

from typing import List, Optional
import quac
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.baseprovider import BaseProvider
from qiskit.test.mock.fake_provider import FakeProvider
from qiskit.providers.quac.simulators import QuacDensitySimulator
from qiskit.providers.quac.simulators import QuacCountsSimulator
from .exceptions import QuacBackendError


class QuacProvider(BaseProvider):
    """
    QuaC Provider to serve all QuaC backends
    """
    provider_instantiated = False

    def __init__(self, user_def_backends: Optional[List[BaseBackend]] = None):
        """Initialize a QuaC provider"""
        if not QuacProvider.provider_instantiated:
            quac.initialize()  # QuaC must only be initialized once

        QuacProvider.provider_instantiated = True

        ibmq_provider = FakeProvider()

        self._backends = [QuacDensitySimulator(), QuacCountsSimulator()]

        # Add IBMQ backends
        for hardware_backend in ibmq_provider.backends():
            if "qasm" not in hardware_backend.name() and "pulse" not in hardware_backend.name():
                self._backends.append(QuacDensitySimulator(hardware_backend.configuration(),
                                                           hardware_backend.properties()))
                self._backends.append(QuacCountsSimulator(hardware_backend.configuration(),
                                                          hardware_backend.properties()))

        # Add user-defined hardware backends
        if user_def_backends:
            for hardware_backend in user_def_backends:
                if hardware_backend.name() in [backend.name()
                                               for backend in ibmq_provider.backends()]:
                    raise QuacBackendError("User backend name collides with IBMQ backend name")
                self._backends.append(QuacDensitySimulator(hardware_backend.configuration(),
                                                           hardware_backend.properties()))
                self._backends.append(QuacCountsSimulator(hardware_backend.configuration(),
                                                          hardware_backend.properties()))

        super().__init__()

    def backends(self, name: Optional[str] = None, **kwargs) -> List[BaseBackend]:
        """
        Returns a list of all backends associated with the QuaC provider

        :param name: optional name to refine backend search
        :param kwargs: optional additional params
        :return: a list of supported QuaC backends of type BaseBackend
        """
        if not name:
            return self._backends
        return [backend for backend in self._backends if name in backend.name()]

    def get_backend(self, name: Optional[str] = None, **kwargs) -> BaseBackend:
        """
        Selects a specific backend on which to perform quantum simulations

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
