# -*- coding: utf-8 -*-

"""This module defines a QuaC provider class through which all QuaC backend simulators can be
interfaced with
"""

from typing import List, Optional
import quac
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.baseprovider import BaseProvider
from qiskit.test.mock.fake_provider import FakeProvider
from qiskit.providers.quac.simulators import QuacCountsSimulator, QuacDensitySimulator
from qiskit.providers.quac.models import get_generic_configuration, QuacNoiseModel
from .exceptions import QuacBackendError


class QuacProvider(BaseProvider):
    """QuaC Provider to serve all QuaC backends
    """
    provider_instantiated = False

    def __init__(self, user_def_backends: Optional[List[BaseBackend]] = None):
        """Initialize a QuaC provider
        """
        if not QuacProvider.provider_instantiated:
            quac.initialize()  # QuaC must only be initialized once

        QuacProvider.provider_instantiated = True

        self._sim_types = ["density", "counts"]
        self._ibmq_provider = FakeProvider()
        self._backend_options = []
        self._user_def_backends = user_def_backends

        for sim_type in self._sim_types:
            self._backend_options.append(f"generic_{sim_type}_simulator")
            for hardware_backend in self._ibmq_provider.backends():
                if "qasm" not in hardware_backend.name() and "pulse" not in hardware_backend.name():
                    # statement to filter out simulators from list
                    self._backend_options.append(f"{hardware_backend.name()}_{sim_type}_simulator")

        # Add user-defined hardware backends
        if user_def_backends:
            for sim_type in self._sim_types:
                for hardware_backend in user_def_backends:
                    if hardware_backend.name() in [backend.name()
                                                   for backend in self._ibmq_provider.backends()]:
                        raise QuacBackendError("User backend name collides with IBMQ backend name")
                    self._backend_options.append(f"{hardware_backend.name()}_{sim_type}_simulator")

        super().__init__()

    def backends(self, name: Optional[str] = None, **kwargs) -> List[str]:
        """Returns a list of all backend names associated with the QuaC provider

        :param name: optional name to refine backend search
        :param kwargs: optional additional params
        :return: a list of names of supported QuaC backends
        """
        if not name:
            return self._backend_options
        return [backend_name for backend_name in self._backend_options if name in backend_name]

    def get_backend(self, name: Optional[str] = None, **kwargs) -> BaseBackend:
        """Selects a specific backend on which to perform quantum simulations

        :param name: the name of the desired backend
        :param kwargs: optional additional params. If the user is retrieving a generic backend,
            then n_qubits (int), max_shots (int), max_exp (int), and basis_gates (List[str]) are expected.
            If the user is retrieving an existing backend, then t1 (bool), t2 (bool), meas (bool), and
            zz (Dict[Tuple[int, int], float]) are expected
        :return: the selected backend with associated name "name"
        """
        if not name or name is "generic_density_simulator":
            return QuacDensitySimulator(
                get_generic_configuration(
                    n_qubits=kwargs.get("n_qubits"),
                    max_shots=kwargs.get("max_shots"),
                    max_exp=kwargs.get("max_exp"),
                    basis_gates=kwargs.get("basis_gates")
                )
            )
        elif name is "generic_counts_simulator":
            return QuacCountsSimulator(
                get_generic_configuration(
                    n_qubits=kwargs.get("n_qubits"),
                    max_shots=kwargs.get("max_shots"),
                    max_exp=kwargs.get("max_exp"),
                    basis_gates=kwargs.get("basis_gates")
                )
            )

        backend_names = list(filter(lambda backend_name: backend_name == name, self._backend_options))

        if len(backend_names) == 0:
            raise QuacBackendError("nonexistent backend")

        chosen_name = backend_names[0]
        chosen_backend = None
        for backend in self._ibmq_provider.backends():
            # We need to convert the user's choice to just the hardware name
            filtered_name = chosen_name
            for sim_type in self._sim_types:
                filtered_name = filtered_name.replace(f'_{sim_type}_simulator', '')

            # Now, we can select the appropriate backend by name
            if filtered_name == backend.name():
                chosen_backend = backend

        for backend in self._user_def_backends or []:
            if backend.name() == backend_names[0]:
                chosen_backend = backend

        quac_noise_model = QuacNoiseModel.from_backend(chosen_backend, **kwargs)

        if "density" in backend_names[0]:
            return QuacDensitySimulator(hardware_conf=chosen_backend.configuration(),
                                        hardware_props=chosen_backend.properties(),
                                        quac_noise_model=quac_noise_model)
        else:
            return QuacCountsSimulator(hardware_conf=chosen_backend.configuration(),
                                       hardware_props=chosen_backend.properties(),
                                       quac_noise_model=quac_noise_model)
