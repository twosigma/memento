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
Backend that uses memory to store data and metadata.

This backend is designed to remove the overhead of IO as well as
the serialization cost for cases where persistence is not
required.

The following configuration options apply to this backend store:
* readonly - If true, data will never be written, only read.

"""

from collections import defaultdict
from typing import Iterable, List, Optional, Dict  # noqa: F401

from .metadata import Memento
from .reference import (
    FunctionReference,
    FunctionReferenceWithArguments,
    FunctionReferenceWithArgHash,
)
from .types import FunctionNotFoundError, VersionedDataSourceKey
from .storage import StorageBackend


class MemoryStorageBackend(StorageBackend):
    metadata = None  # type: Dict[str, Dict[str, bytes]]
    """Map from memento key to map of key to value"""

    result = None  # type: Dict[str, object]
    """Map from memento key to result object for that invocation"""

    mementos = None  # type: Dict[str, Dict[str, Memento]]
    """Map from qualified name to map of arg hash to memento"""

    def __init__(self, config: dict = None, read_only: bool = None):
        """
        Create a storage backend that reads from memory.
        See module documentation for parameters. Parameters that follow
        `config` override its values.

        """
        config = config if config is not None else {}
        super().__init__("memory", config=config, read_only=read_only)
        self.metadata = defaultdict(dict)
        self.result = dict()
        self.mementos = defaultdict(dict)

    @staticmethod
    def _get_memento_key(fn_with_arg_hash: FunctionReferenceWithArgHash) -> str:
        return (
            fn_with_arg_hash.fn_reference.qualified_name
            + "/"
            + fn_with_arg_hash.arg_hash
        )

    def get_mementos(
        self, fns: List[FunctionReferenceWithArgHash]
    ) -> List[Optional[Memento]]:
        results = []
        for fn_ref_with_arg_hash in fns:
            try:
                memento_dict = self.mementos[
                    fn_ref_with_arg_hash.fn_reference.qualified_name
                ]
                memento = memento_dict.get(fn_ref_with_arg_hash.arg_hash)
            except FunctionNotFoundError:
                memento = None
            results.append(memento)
        return results

    def read_result(self, memento: Memento) -> object:
        return self.result[
            self._get_memento_key(
                memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash()
            )
        ]

    def make_url_for_result(self, memento: Memento) -> Optional[str]:
        return memento.content_key

    def read_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        retry_on_none=False,
    ) -> bytes:
        # Ignore retry_on_none since the in-memory metadata store is consistent.
        memento_key = self._get_memento_key(fn_with_arg_hash)
        metadata_dict = self.metadata[memento_key]  # type: Dict[str, bytes]
        return metadata_dict.get(key)

    def write_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        value: bytes,
        store_with_content_key: Optional[VersionedDataSourceKey] = None,
    ):
        if self.read_only:
            raise ValueError("Cannot write metadata to a read-only storage backend")
        memento_key = self._get_memento_key(fn_with_arg_hash)
        metadata_dict = self.metadata[memento_key]  # type: Dict[str, bytes]
        metadata_dict[key] = value

    def is_memoized(self, fn_reference: FunctionReference, arg_hash: str) -> bool:
        memento_dict = self.mementos.get(fn_reference.qualified_name)
        return memento_dict and arg_hash in memento_dict

    def is_all_memoized(self, fns: Iterable[FunctionReferenceWithArguments]) -> bool:
        return all(
            [
                self.is_memoized(fn_ref_with_arg.fn_reference, fn_ref_with_arg.arg_hash)
                for fn_ref_with_arg in fns
            ]
        )

    def list_functions(self) -> List[FunctionReference]:
        return [
            FunctionReference.from_qualified_name(key) for key in self.mementos.keys()
        ]

    def list_mementos(self, fn: FunctionReference, limit: int = None) -> List[Memento]:
        return list(self.mementos[fn.qualified_name].values())[0:limit]

    def memoize(self, key_override: str, memento: Memento, result: object) -> None:
        if self.read_only:
            return
        memento_key = self._get_memento_key(
            memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash()
        )
        memento.content_key = (
            VersionedDataSourceKey(key_override, "") if key_override else None
        )
        self.result[memento_key] = result
        qualified_name = (
            memento.invocation_metadata.fn_reference_with_args.fn_reference.qualified_name
        )
        arg_hash = memento.invocation_metadata.fn_reference_with_args.arg_hash
        self.mementos[qualified_name][arg_hash] = memento

    def forget_call(self, fn_with_arg_hash: FunctionReferenceWithArgHash) -> None:
        if self.read_only:
            raise ValueError("Cannot forget with a storage backend that is read-only")
        qualified_name = fn_with_arg_hash.fn_reference.qualified_name
        arg_hash = fn_with_arg_hash.arg_hash
        memento_key = qualified_name + "/" + arg_hash
        memento_dict = self.mementos[qualified_name]
        if arg_hash in memento_dict:
            del memento_dict[arg_hash]
        if memento_key in self.result:
            del self.result[memento_key]
        if memento_key in self.metadata:
            del self.metadata[memento_key]

    def forget_everything(self) -> None:
        if self.read_only:
            raise ValueError("Cannot forget with a storage backend that is read-only")
        self.metadata.clear()
        self.result.clear()
        self.mementos.clear()

    def forget_function(self, fn_reference: FunctionReference) -> None:
        if self.read_only:
            raise ValueError("Cannot forget with a storage backend that is read-only")
        qualified_name = fn_reference.qualified_name
        for memento in self.list_mementos(fn_reference):
            self.forget_call(
                memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash()
            )
        self.mementos[qualified_name].clear()

    def to_dict(self):
        config = {"type": "memory"}
        if self.read_only is not None:
            config["readonly"] = self.read_only
        return config


StorageBackend.register("memory", MemoryStorageBackend)
