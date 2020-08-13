# -*- coding: utf-8 -*-

"""This module contains a QuaC job class whose objects are to be submitted to a process executor
pool. This extends the Qiskit BasicAer job class
"""
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.basicaer import BasicAerJob
from qiskit.providers.exceptions import JobError


class QuacJob(BasicAerJob):
    """A QuaC Job to be executed on a Thread Pool or Process Pool
    """
    def __init__(self, backend, job_id, func, qobj, **run_config):
        super().__init__(backend, job_id, func, qobj)
        self._injected_params = run_config

    def submit(self) -> None:
        """Submits a job to the executor
        :return: none (void)
        """
        if self._future is not None:
            raise JobError("QuaC job already submitted.")

        self._future = self._executor.submit(self._fn, self._job_id,
                                             self._qobj, **self._injected_params)

    def status(self) -> JobStatus:
        """Status of the job submitted
        :return: job status
        """
        status = super().status()
        print(f"Exception: {self._future.exception()}")
        return status
