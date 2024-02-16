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
Provides a mechanism for Memento to have pluggable storage backend
implementations.

New storage backends should register using :py:method:`StorageBackend.register`.

To create a new storage backend instance, use :py:method:`StorageBackend.create`.

"""

from abc import ABC, abstractmethod
from typing import List, Iterable, Optional

from .reference import (
    FunctionReference,
    FunctionReferenceWithArguments,
    FunctionReferenceWithArgHash,
)
from .metadata import Memento

# The set of known backends. Register new backends using StorageBackend.register.
from .types import VersionedDataSourceKey

_registered_storage_backends = {}


class StorageBackend(ABC):
    """
    This is the abstract base class for a Memento storage backend. Different
    storage backend providers will extend this class to offer different ways
    of serializing Memento metadata and function results.

    """

    storage_type = None  # type: str
    config = None  # type: dict
    read_only = None  # type: bool

    def __init__(self, storage_type: str, config: dict = None, read_only: bool = None):
        """
        Initializes this backend with the provided configuration.

        The details of what is expected in the configuration vary by provider.
        However, the "readonly" parameter is always parsed and set in the
        `readonly` attribute. If not specified, this defaults to `False`.

        :param config:      The configuration for this backend.
        :param read_only:   Whether to make this storage back-end read-only.
                            If specified, overrides `readonly` in `config`

        """

        config = config if config is not None else {}
        self.storage_type = storage_type
        self.config = config
        self.read_only = config.get("readonly", False)
        if read_only is not None:
            self.read_only = read_only

    def get_memento(self, fn: FunctionReferenceWithArgHash) -> Memento:
        """
        Convenience method to call :meth:`get_mementos` with one argument. Returns None if
        no memento exists for the given parameters.

        """
        return self.get_mementos([fn])[0]

    @abstractmethod
    def get_mementos(
        self, fns: List[FunctionReferenceWithArgHash]
    ) -> List[Optional[Memento]]:
        """
        For each invocation in the list, returns the call's Memento (memoization metadata)
        if the function is memoized for the given arg hash.

        The resulting list will be the same size as `fns`, with each output element corresponding
        to the element in the input list. If a memento does not exist, then `None` is returned
        in the corresponding element.

        """
        pass

    @abstractmethod
    def read_result(self, memento: Memento) -> object:
        """
        Read the memoized result from the last time this function was called
        with arguments that resulted in the given hash

        :param memento:         The memento (memoization metadata) from the last call

        """
        pass

    @abstractmethod
    def make_url_for_result(self, memento: Memento) -> Optional[str]:
        """
        Create a URL that points to the data for the memoized result

        :param memento: The memento containing the metadata for the result

        """
        pass

    @abstractmethod
    def read_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        retry_on_none=False,
    ) -> Optional[bytes]:
        """
        Read custom metadata for the given arguments from the given
        key. This is useful, for example, for reading logs
        associated with a given memoized result. Note that some
        metadata stores are eventually consistent.

        :param fn_with_arg_hash:  The function / arg hash with which the metadata is associated.
        :param key:         The metadata key
        :param retry_on_none:  If `True`, retries this operation several times if the result is
                               `None`. This is useful, as some metadata stores are eventually
                               consistent. Defaults to `False`.
        :return:  The metadata, or `None` if not found
        """
        pass

    @abstractmethod
    def write_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        value: bytes,
        store_with_content_key: Optional[VersionedDataSourceKey] = None,
    ):
        """
        Write custom metadata for the given arguments for the given
        key. This is useful, for example, for writing logs
        associated with a given memoized result.

        :param fn_with_arg_hash:   The function / arg hash with which the metadata is associated.
        :param key:         The metadata key
        :param value:       The value to write
        :param store_with_content_key:  If not `None`, the metadata value is stored with the
                            data aside the given versioned content key, instead of in the metadata
                            store directly. In this case, value is ignored and the metadata store
                            is updated to reflect the location of the metadata.

        """
        pass

    @abstractmethod
    def is_memoized(self, fn_reference: FunctionReference, arg_hash: str) -> bool:
        """
        Returns True if there is memoized data available for the given function
        with arguments that resulted in the given hash, or False otherwise.
        This does not return the actual result, but rather whether there is data.

        :param fn_reference:    The function for which to query the previous result
        :param arg_hash:        The argument hash for the call to query

        """
        pass

    @abstractmethod
    def is_all_memoized(self, fns: Iterable[FunctionReferenceWithArguments]) -> bool:
        """
        Returns True if there is memoized data available for all of the given function
        with arguments, or False otherwise.

        This does not return the actual result, but rather whether there is data.

        Underlying storage implementations should perform this as a batch query,
        if possible.

        :param fns:             The functions with arguments for which to query

        """
        pass

    @abstractmethod
    def list_functions(self) -> List[FunctionReference]:
        """
        Return a list of the functions that have memoized data stored.

        """
        pass

    @abstractmethod
    def list_mementos(self, fn: FunctionReference, limit: int = None) -> List[Memento]:
        """
        Return a list of the mementos recorded for the given function

        :param fn:      The function reference for which to list mementos
        :param limit:   Maximum number of mementos to return in this query

        """
        pass

    @abstractmethod
    def memoize(
        self, key_override: Optional[str], memento: Memento, result: object
    ) -> None:
        """
        Remember the result. As a side effect of this call, the content_hash is computed and
        set in the provided memento object.

        If `key_override` is specified, it will be used in place of any content-addressable
        key that would have been generated.

        """
        pass

    @abstractmethod
    def forget_call(self, fn_with_arg_hash: FunctionReferenceWithArgHash) -> None:
        """
        Remove any memoized data for a specific call to this function

        :param fn_with_arg_hash:  The function for which to forget memoized data, along
            with argument hash

        """
        pass

    @abstractmethod
    def forget_everything(self) -> None:
        """
        Remove any memoized data for all calls to all functions

        """
        pass

    @abstractmethod
    def forget_function(self, fn_reference: FunctionReference) -> None:
        """
        Remove any memoized data for all calls to an entire function

        :param fn_reference:    The function for which to forget memoized data

        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Return a dict representation of this storage backend configuration

        """
        pass

    @classmethod
    def create(cls, storage_type, config):
        """
        Static factory method that creates a new storage backend based on
        the type and config provided.

        :param storage_type: The type of backend configuration. Must be one of the
            known registered storage backends
        :param config: The configuration used to initialize this backend.
        :return: An instance of the appropriate storage backend.
        """

        global _registered_storage_backends
        if storage_type not in _registered_storage_backends:
            raise ValueError(
                "Unrecognized storage backend type {}".format(storage_type)
            )
        return _registered_storage_backends.get(storage_type)(config)

    @classmethod
    def register(cls, storage_type, clazz):
        """
        Static method to register a new type of storage backend. This is
        used by storage backend providers.

        Note that if two providers register a backend with the same name,
        the last one to register wins.

        :param storage_type: The name of the storage backend
        :param clazz: The class used to instantiate this storage backend.
            It is assumed the class takes a single parameter in its
            constructor, which is the configuration for the backend.

        """
        global _registered_storage_backends
        _registered_storage_backends[storage_type] = clazz
