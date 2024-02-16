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
Implements a storage backend that always returns null or None.

"""
from typing import List, Iterable, Optional

from .reference import (
    FunctionReference,
    FunctionReferenceWithArguments,
    FunctionReferenceWithArgHash,
)
from .metadata import Memento
from .storage import StorageBackend
from .types import VersionedDataSourceKey


class NullStorageBackend(StorageBackend):
    def __init__(self, config: dict = None):
        super().__init__("null", config=config)

    def get_mementos(
        self, fns: List[FunctionReferenceWithArgHash]
    ) -> List[Optional[Memento]]:
        return [None] * len(fns)

    def read_result(self, memento: Memento) -> object:
        raise ValueError("Null backend has no memoized results for any invocations")

    def make_url_for_result(self, memento: Memento) -> Optional[str]:
        raise ValueError("Null backend has no memoized results for any invocations")

    def read_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        retry_on_none=False,
    ) -> object:
        return None

    def write_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        value: bytes,
        store_with_content_key: Optional[VersionedDataSourceKey] = None,
    ):
        pass

    def is_memoized(self, fn_reference: FunctionReference, arg_hash: str) -> bool:
        return False

    def is_all_memoized(self, fns: Iterable[FunctionReferenceWithArguments]) -> bool:
        return False

    def list_functions(self, cluster_name: str = None) -> List[FunctionReference]:
        return []

    def list_mementos(self, fn: FunctionReference, limit: int = None) -> List[Memento]:
        pass

    def memoize(self, key_override: str, memento: Memento, result: object) -> None:
        memento.content_key = None
        pass

    def forget_call(self, fn_with_arg_hash: FunctionReferenceWithArgHash) -> None:
        pass

    def forget_everything(self) -> None:
        pass

    def forget_function(self, fn_reference: FunctionReference) -> None:
        pass

    def to_dict(self):
        config = {"type": "null"}
        if self.read_only is not None:
            config["readonly"] = self.read_only
        return config


StorageBackend.register("null", NullStorageBackend)
