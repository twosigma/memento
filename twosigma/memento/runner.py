# Copyright (c) 2023 Two Sigma Investments, LP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Provides a mechanism for Memento to have pluggable runner backend
implementations.

New runner backends should register using :py:method:`RunnerBackend.register`.

To create a new runner backend instance, use :py:method:`RunnerBackend.create`.

"""
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Optional, NamedTuple

from .exception import MementoException
from .logging import log
from .metadata import Memento
from .storage import StorageBackend
from .context import InvocationContext
from .reference import FunctionReferenceWithArguments

# The set of known runners. Register new runners using RunnerBackend.register.
_registered_runner_backends = {}

ExistingMementoResult = NamedTuple(
    "ExistingMementoResult", [("result", Any), ("valid_result", bool)]
)


def process_existing_memento(
    storage_backend: StorageBackend, existing_memento: Memento, ignore_result: bool
) -> ExistingMementoResult:
    """
    This logic is reused several times by runners when processing an existing
    memento for a function invocation.

    If the provided memento is an exception, and exceptions_valid is False,
    logs that the result will be recomputed. The previously memoized result
    is forgotten and (None, False) is returned.

    Otherwise, if ignore_result is True, log that the result was memoized but
    ignored. (None, True) is returned.

    Otherwise, the results of the invocation are retrieved and returned. If
    there was an IO error retrieving results, the error is logged and None is
    returned. (value, True) is returned

    """
    assert existing_memento

    fn_reference_with_args = existing_memento.invocation_metadata.fn_reference_with_args

    try:
        # If result already exists, deserialize and return
        if ignore_result:
            log.debug(
                "Result of {} was already memoized and is ignored".format(
                    str(fn_reference_with_args)
                )
            )
            return ExistingMementoResult(result=None, valid_result=True)

        # Unwrap exception from MementoException
        result = storage_backend.read_result(existing_memento)
        if isinstance(result, MementoException):
            e = result  # type: MementoException
            log.warning("Processing a MemoizedException")
            result = e.to_exception()

        log.info(
            "Previous result for {} was memoized and is of type {}.".format(
                str(fn_reference_with_args),
                existing_memento.invocation_metadata.result_type.name,
            )
        )
        return ExistingMementoResult(result=result, valid_result=True)
    except IOError:
        log.warning(
            "IO Error while reading memoized result. Recomputing.", exc_info=True
        )
        return ExistingMementoResult(result=None, valid_result=False)


class RunnerBackend(ABC):
    """
    This is the abstract base class for a Memento runner backend. Different
    runner backend providers will extend this class to offer different ways
    of running functions (e.g. locally, or remotely in a compute cluster).

    """

    runner_type = None  # type: str
    config = None  # type: dict

    def __init__(self, runner_type: str, config: dict = None):
        """
        Initializes this runner backend with the provided configuration.

        The details of what is expected in the configuration vary by provider.

        :param config: The configuration for this runner backend.

        """
        config = config if config is not None else {}
        self.runner_type = runner_type
        self.config = config

    @staticmethod
    def ensure_correlation_id(
        context: InvocationContext,
        fn_reference_with_args: List[FunctionReferenceWithArguments],
    ) -> InvocationContext:
        """
        Utility method for subclasses to generate a new correlation id, if one is not already
        present.

        """
        if not context.recursive.correlation_id:
            correlation_id = "cid_" + uuid.uuid4().hex[0:12]
            log.debug(
                "{}: Generating new correlation id for {}".format(
                    correlation_id, fn_reference_with_args
                )
            )
            return context.update_recursive("correlation_id", correlation_id)
        return context

    @abstractmethod
    def batch_run(
        self,
        context: InvocationContext,
        storage_backend: StorageBackend,
        fn_reference_with_args: List[FunctionReferenceWithArguments],
        log_runner_backend: "RunnerBackend",
        caller_memento: Optional[Memento],
    ) -> List[Any]:
        """
        Run a series of memento functions using this runner. If the runner is capable, these
        may be run in parallel.

        The runner must use the correlation id provided in the recursive context, or create a new
        correlation id if this is the roof of an invocation tree.

        :param context:                 Context for this invocation
        :param storage_backend:         The storage backend containing the memoized data.
                                        This is useful in case runners would like to
                                        execute remotely and read back the memoized result.
        :param fn_reference_with_args:  An iterator over function references and arguments
                                        to the functions being called.
        :param log_runner_backend:      Use this runner when writing the log. This is used
                                        when the function is run in a cluster - initial call
                                        will be a cluster runner and then the function is run
                                        again on the local machine. We want to log the original
                                        runner.
        :param caller_memento:          The memento of the calling function, so that its
                                        record of invocations can be updated after the batch
                                        run is complete.
        :return: A list of results of executing the function, in the same order as
                 fn_reference_with_args. If a function raises an Exception, the exception
                 instance is returned in the list.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Return a dict representation of the configuration for this runner

        """
        pass

    @classmethod
    def create(cls, runner_type, config) -> "RunnerBackend":
        """
        Static factory method that creates a new runner backend based on
        the type and config provided.

        :param runner_type: The type of backend configuration. Must be one of the
            known registered runner backends
        :param config: The configuration used to initialize this backend.
        :return: An instance of the appropriate runner backend.
        """

        global _registered_runner_backends
        if runner_type not in _registered_runner_backends:
            raise ValueError("Unrecognized runner backend type {}".format(runner_type))
        return _registered_runner_backends.get(runner_type)(config)

    @classmethod
    def register(cls, runner_type, clazz):
        """
        Static method to register a new type of runner backend. This is
        used by runner backend providers.

        Note that if two providers register a backend with the same name,
        the last one to register wins.

        :param runner_type: The name of the runner backend
        :param clazz: The class used to instantiate this runner backend.
            It is assumed the class takes a single parameter in its
            constructor, which is the configuration for the backend.

        """
        global _registered_runner_backends
        _registered_runner_backends[runner_type] = clazz
