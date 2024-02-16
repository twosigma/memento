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

import hashlib
import io
import pickle
from io import BytesIO, TextIOWrapper
import json
import os
import sys
import traceback
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from time import sleep
from typing import List, Iterable, Dict, Type, Optional, Union, cast
from weakref import WeakValueDictionary

import pandas as pd
import numpy as np

from .storage import StorageBackend
from .exception import MementoException
from .logging import log
from .metadata import Memento, ResultType
from .partition import Partition, InMemoryPartition
from .reference import (
    FunctionReference,
    FunctionReferenceWithArgHash,
    FunctionReferenceWithArguments,
)
from .serialization import MementoCodec
from .types import (
    FunctionNotFoundError,
    VersionedDataSourceKey,
    DataSourceKey,
    ContentAddressableHash,
)

# The set of known codecs. Register new codecs using Codec.register.
_registered_codecs = {}

# Standard file extensions based on result type
_extension_for_result_type = {
    ResultType.exception: "json",
    ResultType.null: "json",
    ResultType.boolean: "pickle",
    ResultType.string: "pickle",
    ResultType.binary: "pickle",
    ResultType.number: "pickle",
    ResultType.date: "pickle",
    ResultType.timestamp: "pickle",
    ResultType.list_result: "pickle",
    ResultType.dictionary: "pickle",
    ResultType.array_boolean: "pickle",
    ResultType.array_int8: "pickle",
    ResultType.array_int16: "pickle",
    ResultType.array_int32: "pickle",
    ResultType.array_int64: "pickle",
    ResultType.array_float32: "pickle",
    ResultType.array_float64: "pickle",
    ResultType.index: "pickle",
    ResultType.series: "pickle",
    ResultType.data_frame: "pickle",
    ResultType.partition: "partition",
}  # type: Dict[ResultType, str]


_ResultTypeAndContentKey = namedtuple(
    "_ResultTypeAndContentKey", ["result_type", "content_key", "from_parent"]
)


class DataSource(ABC):
    """
    Abstracts the source of data, allowing codec logic to be reused regardless
    of the storage backend. An instance of this class is provided by each
    storage backend and is consumed by a `Codec`.

    The `input` and `output` methods provide the ability to read and write
    data for a given `memento` and a `key`, which is needed for some
    codecs in order to write multiple files for a memento (e.g.
    for columnar storage).

    `DataSource`s implement versioned object storage, meaning that a single key
    can have multiple versions. Keys are either bare or versioned. When a value
    is written, the bare key is used as input and a versioned key is returned.
    When a value is read, the versioned key must be provided as input. If the underlying
    storage system supports versioned objects, the feature is leveraged. Otherwise,
    versioned storage is emulated by the subclass.

    """

    def __init__(self):
        pass

    @abstractmethod
    def input_nonversioned(self, key: DataSourceKey) -> BytesIO:
        """
        Return a binary file-like object that reads the data for the given
        key, relative to a base path.

        This function is for non-versioned keys.

        :raises IOError:    If the key does not exist.

        """
        pass

    @abstractmethod
    def input_versioned(self, key: VersionedDataSourceKey) -> BytesIO:
        """
        Return a binary file-like object that reads the data for the given
        key, relative to a base path.

        This function is for versioned keys.

        :raises IOError:    If the key does not exist.

        """
        pass

    @abstractmethod
    def input_metadata(
        self, content_key: VersionedDataSourceKey, metadata_key: str
    ) -> BytesIO:
        """
        Return a binary file-like object that reads data for the given metadata key
        for a given content key. This is for metadata that is stored in the object store
        alongside the object.
        """
        pass

    @abstractmethod
    def make_url_for_key(self, key: Optional[VersionedDataSourceKey]) -> Optional[str]:
        """
        Create a URL that will return the data for the given key.
        The URL may be transient.

        """
        pass

    @abstractmethod
    def output(self, key: DataSourceKey, data: BytesIO) -> VersionedDataSourceKey:
        """
        Store data from the given binary file-like object to the given
        key, relative to a base path.

        """
        pass

    @abstractmethod
    def reference(
        self,
        src_data_source: "DataSource",
        src_key: VersionedDataSourceKey,
        target_key: VersionedDataSourceKey,
    ) -> VersionedDataSourceKey:
        """
        Mark a reference to the given data source key coming from the given data source.
        This is a NOP for data sources that do not perform reference counting for garbage
        collection.
        """
        pass

    @abstractmethod
    def output_metadata(
        self, content_key: VersionedDataSourceKey, metadata_key: str, value: bytes
    ):
        """
        Store data for the given metadata key for a given content key. This is for metadata
        that is stored in the object store alongside the object.
        """
        pass

    @abstractmethod
    def delete_nonversioned_key(self, key: DataSourceKey):
        """
        Remove the nonversioned key while leaving all the versions intact. After this change,
        the data source should claim the nonversioned key does not exist, but it should
        still be possible to retrieve any previous version by its versioned key.

        If doesn't exist, this operation is a NOP (no exceptions thrown)
        """
        pass

    @abstractmethod
    def delete_all_versions(self, key: DataSourceKey, recursive: bool):
        """
        Delete data for the given key.
        If recursive is set to True, deletes everything underneath this key.

        This method deletes all versions of a given key.

        If doesn't exist, this operation is a NOP (no exceptions thrown)
        """
        pass

    @abstractmethod
    def exists_versioned(self, key: VersionedDataSourceKey) -> bool:
        """
        Returns True iff the given versioned key exists.

        """
        pass

    @abstractmethod
    def exists_nonversioned(self, key: DataSourceKey) -> bool:
        """
        Returns True iff the given non-versioned key exists.

        """
        pass

    @abstractmethod
    def all_exist_versioned(self, keys: List[VersionedDataSourceKey]) -> List[bool]:
        """
        Batch query version of `exists` - will return a list of the same length as keys containing
        True in each slot where the key exists, or False where the key does not exist.
        If possible, the underlying data store should perform this query in bulk.

        This method is for versioned keys.

        """
        pass

    @abstractmethod
    def all_exist_nonversioned(self, keys: List[DataSourceKey]) -> List[bool]:
        """
        Batch query version of `exists` - will return a list of the same length as keys containing
        True in each slot where the key exists, or False where the key does not exist.
        If possible, the underlying data store should perform this query in bulk.

        This method is for non-versioned keys.

        """
        pass

    @abstractmethod
    def list_keys_nonversioned(
        self,
        directory: DataSourceKey,
        file_prefix: str = "",
        recursive: bool = False,
        limit: int = None,
        endswith: str = None,
    ) -> Iterable[DataSourceKey]:
        """
        List the keys in the path provided by the 'prefix' parameter.

        The non-versioned keys are listed.

        :param directory:       The part of the key that is the directory under
                                which to list sub-keys.
        :param file_prefix:     Filename must begin with this string to be
                                included in the result.
        :param recursive:       If True, will search directories recursively,
                                else just the provided directory.
        :param limit:           Maximum limit on how many keys are returned.
                                Defaults to None, meaning no limit.
        :param endswith:        Limit results to those that end with this
                                string, if specified.

        """
        pass

    @abstractmethod
    def get_versioned_key(self, key: DataSourceKey) -> VersionedDataSourceKey:
        """
        Return the most recent versioned key for the given non-versioned key
        """
        pass


class Codec(ABC):
    """
    Encapsulates logic of encoding and decoding function results

    """

    config = None  # type: dict

    class Strategy(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def load(self, data_source: DataSource, key: VersionedDataSourceKey) -> object:
            """Load the object from the store and return the resulting object"""
            pass

        @staticmethod
        def make_url_for_key(
            data_source: DataSource, key: VersionedDataSourceKey
        ) -> str:
            return data_source.make_url_for_key(key)

        @abstractmethod
        def store(
            self, data_source: DataSource, key_override: str, obj: object
        ) -> VersionedDataSourceKey:
            """Store the object to the store and return the key under which the data was stored"""
            pass

        @staticmethod
        def output_key_for_content_key(
            content_key: ContentAddressableHash,
        ) -> DataSourceKey:
            """Converts a content key to the key at which it should be stored"""
            return DataSourceKey("c/{}".format(content_key.key))

        @staticmethod
        def output_key_for_override_key(override_key: str) -> Optional[DataSourceKey]:
            """Converts an override key to the key at which it should be stored"""
            return DataSourceKey(override_key) if override_key else None

    class NullStrategy(Strategy):
        """
        Strategy that writes nothing, as the result type sufficiently describes the result:
        `Null` (or `None`, in Python).

        If a key override is specified, the null strategy takes an additional step and removes
        the symlink to the previous version, if it exists. This keeps the non-versioned namespace
        clean.

        """

        def __init__(self):
            super().__init__()

        def store(
            self, data_source: DataSource, key_override: str, obj: object
        ) -> Optional[VersionedDataSourceKey]:
            if key_override:
                key = self.output_key_for_override_key(key_override)
                data_source.delete_nonversioned_key(key)
            return None

        def load(self, data_source: DataSource, key: VersionedDataSourceKey) -> object:
            return None

    class BlobStrategy(Strategy):
        """
        Strategy that writes a single blob of data (a hash is computed over the bytes and that
        becomes the key for the object).

        Strategies that derive from this class will override `encode` and provide a way to
        translate content into bytes over which a hash can be computed and then the result
        can be stored.

        """

        def __init__(self):
            super(Codec.BlobStrategy, self).__init__()

        def store(
            self, data_source: DataSource, key_override: str, obj: object
        ) -> VersionedDataSourceKey:
            data = self.encode(obj)
            content_hash = hashlib.sha256(data).hexdigest()
            key = self.output_key_for_content_key(ContentAddressableHash(content_hash))
            if key_override:
                key = self.output_key_for_override_key(key_override)
            else:
                if data_source.exists_nonversioned(key):
                    # A key already exists at the content addressable hash location.
                    # Do not create a new object version - it can be reused.
                    return data_source.get_versioned_key(key)
            return data_source.output(key, BytesIO(data))

        @abstractmethod
        def encode(self, obj: object) -> bytes:
            """Encode the object into a binary representation"""
            pass

    _strategy = dict()  # type: Dict[ResultType, Strategy]

    def __init__(self, config: dict, strategy: Dict[ResultType, Type[Strategy]]):
        self.config = config
        self._strategy = strategy

    def load(
        self,
        result_type: ResultType,
        data_source: DataSource,
        key: VersionedDataSourceKey,
    ) -> object:
        """
        Loads a memento function result from the given data source
        and returns the result as a file-like object.

        """
        return self._strategy[result_type].load(data_source, key)

    def make_url_for_result(
        self,
        result_type: ResultType,
        data_source: DataSource,
        key: VersionedDataSourceKey,
    ) -> str:
        """
        Returns the url from which the memento function result from the given data source
        can be retrieved.

        """
        return self._strategy[result_type].make_url_for_key(data_source, key)

    def store(
        self,
        result_type: ResultType,
        data_source: DataSource,
        key_override: Optional[str],
        obj: object,
    ) -> VersionedDataSourceKey:
        """
        Encode and store the memento function result in the provided object to the datastore,
        returning the key under which the data was stored.

        """
        return self._strategy[result_type].store(data_source, key_override, obj)

    @classmethod
    def create(cls, codec_type, config):
        """
        Static factory method that creates a new codec based on
        the type and config provided.

        :param codec_type: The type of codec. Must be one of the
            known registered codecs.
        :param config: The configuration used to initialize this codec.
        :return: An instance of the appropriate codec.
        """

        global _registered_codecs
        if codec_type not in _registered_codecs:
            raise ValueError("Unrecognized codec type {}".format(codec_type))
        return _registered_codecs.get(codec_type)(config)

    @classmethod
    def register(cls, codec_type, clazz):
        """
        Static method to register a new type of codec. This is
        used by codec providers.

        Note that if two providers register a codec with the same name,
        the last one to register wins.

        :param codec_type: The name of the codec
        :param clazz: The class used to instantiate this codec.
            It is assumed the class takes a single parameter in its
            constructor, which is the configuration for the codec.

        """
        _registered_codecs[codec_type] = clazz


class DefaultCodec(Codec):
    """
    Default logic for loading and storing.

    """

    def __init__(self, config):
        super().__init__(
            config,
            {
                ResultType.exception: self.JsonExceptionStrategy(),
                ResultType.null: self.NullStrategy(),
                ResultType.boolean: self.ValuePickleStrategy(),
                ResultType.string: self.ValuePickleStrategy(),
                ResultType.binary: self.ValuePickleStrategy(),
                ResultType.number: self.ValuePickleStrategy(),
                ResultType.date: self.ValuePickleStrategy(),
                ResultType.timestamp: self.ValuePickleStrategy(),
                ResultType.list_result: self.ValuePickleStrategy(),
                ResultType.dictionary: self.ValuePickleStrategy(),
                ResultType.array_boolean: self.ValuePickleStrategy(),
                ResultType.array_int8: self.ValuePickleStrategy(),
                ResultType.array_int16: self.ValuePickleStrategy(),
                ResultType.array_int32: self.ValuePickleStrategy(),
                ResultType.array_int64: self.ValuePickleStrategy(),
                ResultType.array_float32: self.ValuePickleStrategy(),
                ResultType.array_float64: self.ValuePickleStrategy(),
                ResultType.index: self.ValuePickleStrategy(),
                ResultType.series: self.ValuePickleStrategy(),
                ResultType.data_frame: self.ValuePickleStrategy(),
                ResultType.partition: self.PicklePartitionStrategy(self),
            },
        )

    class JsonExceptionStrategy(Codec.BlobStrategy):
        """JSON encoding, returning as a MementoException"""

        def __init__(self):
            super().__init__()

        def load(
            self, data_source: DataSource, key: VersionedDataSourceKey
        ) -> MementoException:
            with data_source.input_versioned(key) as f:
                with io.TextIOWrapper(f, encoding="utf-8") as t:
                    result_dict = json.load(t)
            return MementoException(
                result_dict["exception_name"],
                result_dict["message"],
                result_dict["stack_trace"],
            )

        def encode(self, obj: MementoException) -> bytes:
            store_dict = {
                "exception_name": obj.exception_name,
                "message": obj.message,
                "stack_trace": "".join(
                    traceback.format_exception(type(obj), obj, obj.__traceback__)
                ),
            }
            return json.dumps(store_dict).encode("utf-8")

    class PicklePartition(Partition):
        """
        Implements Partition by storing a separate file for
        each key, under a base key. The base key stores a map
        from key to result type and content key so the appropriate strategy can
        be used to load the partition.

        """

        _codec = None  # type: Codec
        _data_source = None  # type: DataSource
        _base_key = None  # type: VersionedDataSourceKey
        _index = None  # type: Dict[str, _ResultTypeAndContentKey]
        """"""

        def __init__(
            self,
            codec: Codec,
            data_source: DataSource,
            base_key: VersionedDataSourceKey,
        ):
            super().__init__()
            self._codec = codec
            self._data_source = data_source
            self._base_key = base_key
            with data_source.input_versioned(self._base_key) as f:
                self._index = self._deserialize_index(f.read())

        @staticmethod
        def _deserialize_index(data: bytes) -> Dict[str, _ResultTypeAndContentKey]:
            d = json.loads(data.decode("utf-8"))  # type: Dict[str, Dict[str, str]]
            return {
                k: _ResultTypeAndContentKey(
                    result_type=ResultType[v["result_type"]],
                    content_key=MementoCodec.decode_versioned_data_source_key(
                        v["content_key"]
                    ),
                    from_parent=v.get("from_parent", False),
                )
                for (k, v) in d.items()
            }

        @staticmethod
        def _serialize_index(index: Dict[str, _ResultTypeAndContentKey]) -> bytes:
            d = {
                k: {
                    "result_type": v.result_type.name,
                    "content_key": MementoCodec.encode_versioned_data_source_key(
                        v.content_key
                    ),
                    "from_parent": v.from_parent,
                }
                for (k, v) in index.items()
            }
            return json.dumps(d, sort_keys=True).encode("utf-8")

        def get(self, key: str) -> object:
            if key not in self._index:
                raise ValueError(
                    "Key '{}' is not in key list for partition".format(key)
                )
            index_entry = self._index[key]
            return self._codec.load(
                index_entry.result_type, self._data_source, index_entry.content_key
            )

        def list_keys(self, _include_merge_parent: bool = True) -> Iterable[str]:
            if _include_merge_parent:
                keys = self._index.keys()
            else:
                keys = [
                    key
                    for key in self._index.keys()
                    if not self._index.get(key).from_parent
                ]
            return sorted(keys)

    class PicklePartitionStrategy(Codec.BlobStrategy):
        """
        Pickle encoding, partitioned by key into separate files.
        The primary file encodes a list of available keys.
        The content hash for the partition is the hash of each of the partition_key + content_key
        of the value in partition_key order.

        """

        _codec = None  # type: Codec

        def __init__(self, codec: Codec):
            super().__init__()
            self._codec = codec

        def load(
            self, data_source: DataSource, key: VersionedDataSourceKey
        ) -> Partition:
            return DefaultCodec.PicklePartition(self._codec, data_source, key)

        def store(
            self, data_source: DataSource, key_override: str, obj: Partition
        ) -> VersionedDataSourceKey:
            # build a dict of key to result type
            index = dict()  # type: Dict[str, _ResultTypeAndContentKey]

            # If a merge parent is set, merge index with parent:
            # noinspection PyProtectedMember
            merge_parent = obj._merge_parent
            if merge_parent:
                if isinstance(merge_parent, DefaultCodec.PicklePartition):
                    pickle_partition_parent = cast(
                        DefaultCodec.PicklePartition, merge_parent
                    )
                    # noinspection PyProtectedMember
                    parent_index = pickle_partition_parent._index
                    # noinspection PyProtectedMember
                    parent_data_source = pickle_partition_parent._data_source
                elif hasattr(merge_parent, "_output_keys") and hasattr(
                    merge_parent, "_data_source"
                ):
                    # noinspection PyProtectedMember
                    parent_index = merge_parent._output_keys
                    # noinspection PyProtectedMember
                    parent_data_source = merge_parent._data_source
                else:
                    raise IOError(
                        "Could not merge partitions: parent is not "
                        "a PicklePartition or has never been serialized"
                    )
                for k, v in parent_index.items():
                    # Mark a reference to all values that come from parents. This is for storage
                    # backends that do reference counting.
                    data_source.reference(
                        parent_data_source, v.content_key, v.content_key
                    )

                    index[k] = _ResultTypeAndContentKey(
                        result_type=v.result_type,
                        content_key=v.content_key,
                        from_parent=True,
                    )

            # Layer current keys on top of parent's keys
            output_keys = dict()
            keys = obj.list_keys(_include_merge_parent=False)
            for k in keys:
                result = obj.get(k)
                result_type = ResultType.from_object(result)

                # Store
                content_key_override = (
                    "{}/{}".format(key_override, k)
                    if key_override is not None
                    else None
                )
                partition_content_key = self._codec.store(
                    result_type, data_source, content_key_override, result
                )

                # Update map of key to result info
                index_entry = _ResultTypeAndContentKey(
                    result_type=result_type,
                    content_key=partition_content_key,
                    from_parent=False,
                )
                output_keys[k] = index_entry
                index[k] = index_entry

            # If this is an InMemoryPartition, remember the output keys so they can be
            # referred to when merging partitions in the future
            if hasattr(obj, "_output_keys") and hasattr(obj, "_data_source"):
                obj._output_keys = output_keys
                obj._data_source = data_source

            # noinspection PyProtectedMember
            obj._index_bytes = DefaultCodec.PicklePartition._serialize_index(index)

            index_key_override = (
                "{}/index.json".format(key_override)
                if key_override is not None
                else None
            )
            return super().store(
                data_source=data_source, key_override=index_key_override, obj=obj
            )

        def encode(self, obj: Partition) -> bytes:
            # noinspection PyProtectedMember
            return obj._index_bytes

    class ValuePickleStrategy(Codec.BlobStrategy):
        """Pickle encoded value"""

        def __init__(self):
            super().__init__()

        def load(self, data_source: DataSource, key: VersionedDataSourceKey) -> object:
            with data_source.input_versioned(key) as f:
                return pickle.loads(f.read())

        def decode(self, obj: bytes) -> object:
            return pickle.loads(obj)

        def encode(self, obj: object) -> bytes:
            return pickle.dumps(obj, protocol=5)


class ResultIsWithData:
    """
    Marker class to indicate that metadata is stored with the data.
    """

    pass


class MetadataSource(ABC):
    """
    Abstracts the source of metadata for the efficient storage and retrieval
    of mementos.

    """

    def __init__(self):
        pass

    @abstractmethod
    def get_mementos(
        self, fns: List[FunctionReferenceWithArgHash]
    ) -> List[Optional[Memento]]:
        """
        See StorageBackend.get_mementos()

        """
        pass

    @abstractmethod
    def all_mementos_exist(self, fns: List[FunctionReferenceWithArgHash]) -> bool:
        """
        Returns `True` if all mementos exist for the given set of function references
        with argument hashes.

        """
        pass

    @abstractmethod
    def list_functions(self) -> List[FunctionReference]:
        """
        Returns a list of all functions known in this metadata store

        """
        pass

    @abstractmethod
    def list_mementos(
        self, fn: FunctionReference, limit: Optional[int]
    ) -> List[Memento]:
        """
        List all mementos for the given function

        :param fn:      The function to list mementos for
        :param limit:   Maximum number of results to return, or None for no limit.

        """
        pass

    @abstractmethod
    def put_memento(self, memento: Memento):
        """
        Store the given memento to the metadata store

        """
        pass

    @abstractmethod
    def read_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        retry_on_none=False,
    ) -> Optional[Union[bytes, ResultIsWithData]]:
        """
        Read metadata (e.g. logs) associated with the Memento for the provided `fn_with_arg_hash`.

        Returns None if no metadata present for key. Note that some metadata stores are
        eventually consistent.

        Returns :class:`ResultIsWithData` if the result is stored with the data instead of in
        the metadata store.

        :param fn_with_arg_hash:  The function / arg hash for which to read metadata
        :param key:  The metadata key to be read.
        :param retry_on_none:  If `True`, retry several times if the result is `None`. This is
                               useful for metadata stores that are eventually consistent.

        """
        pass

    @abstractmethod
    def write_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        value: bytes,
        stored_with_data: bool,
    ):
        """
        Write metadata (e.g. logs) associated with the Memento for the given `fn_with_arg_hash`

        """
        pass

    @abstractmethod
    def forget_call(self, fn_with_arg_hash: FunctionReferenceWithArgHash):
        """
        Forget all mementos and metadata associated with this function reference with arguments

        """
        pass

    @abstractmethod
    def forget_everything(self):
        """
        Forget everything in this cluster.

        """
        pass

    @abstractmethod
    def forget_function(self, fn_reference: FunctionReference):
        """
        Forget everything we know about this function

        """
        pass


class DataSourceMetadataSource(MetadataSource):
    """
    Implementation of a MetadataSource using a standard
    `DataSource` for reading and writing. This is useful for implementing
    a metadata source on a standard filesystem-like backend.

    """

    data_source = None  # type: DataSource
    _function_path_prefix = DataSourceKey("m")

    def __init__(self, data_source: DataSource):
        super().__init__()
        self.data_source = data_source

    @staticmethod
    def _get_function_path(fn_reference: FunctionReference) -> DataSourceKey:
        return DataSourceKey(
            "{}/{}".format(
                DataSourceMetadataSource._function_path_prefix.key,
                fn_reference.qualified_name,
            )
        )

    @staticmethod
    def _get_path(fn_reference: FunctionReference, arg_hash: str) -> str:
        return "{}/{}".format(
            DataSourceMetadataSource._get_function_path(fn_reference).key, arg_hash
        )

    @staticmethod
    def _get_metadata_path(
        fn_with_arg_hash: FunctionReferenceWithArgHash,
    ) -> DataSourceKey:
        return DataSourceKey(
            "{}.memento.json".format(
                DataSourceMetadataSource._get_path(
                    fn_with_arg_hash.fn_reference, fn_with_arg_hash.arg_hash
                )
            )
        )

    @staticmethod
    def _get_metadata_key(
        fn_with_arg_hash: FunctionReferenceWithArgHash, key: str, stored_with_data: bool
    ) -> DataSourceKey:
        return DataSourceKey(
            "{}.metadata.{}{}".format(
                DataSourceMetadataSource._get_path(
                    fn_with_arg_hash.fn_reference, fn_with_arg_hash.arg_hash
                ),
                key,
                ".with_data" if stored_with_data else "",
            )
        )

    def _read_memento(self, path: DataSourceKey) -> Memento:
        with self.data_source.input_nonversioned(path) as f:
            with TextIOWrapper(f, encoding="utf-8") as t:
                return MementoCodec.decode_memento(json.load(t))

    def get_mementos(
        self, fns: List[FunctionReferenceWithArgHash]
    ) -> List[Optional[Memento]]:
        results = []
        for fn_ref_with_arg_hash in fns:
            metadata_path = DataSourceMetadataSource._get_metadata_path(
                fn_ref_with_arg_hash
            )
            try:
                memento = self._read_memento(metadata_path)
            except FunctionNotFoundError as e:
                log.debug(
                    "Ignoring memoized result: while decoding Memento for {}, "
                    "could not find function: {}".format(fn_ref_with_arg_hash, e)
                )
                memento = None
            except IOError:
                memento = None
            results.append(memento)
        return results

    def all_mementos_exist(self, fns: List[FunctionReferenceWithArgHash]) -> bool:
        keys = [DataSourceMetadataSource._get_metadata_path(fn) for fn in fns]
        return all(self.data_source.all_exist_nonversioned(keys))

    def list_functions(self) -> List[FunctionReference]:
        fpp = DataSourceMetadataSource._function_path_prefix
        return [
            FunctionReference.from_qualified_name(
                x.key[len(DataSourceMetadataSource._function_path_prefix) + 1 :]
            )
            for x in self.data_source.list_keys_nonversioned(
                directory=fpp, file_prefix="", recursive=False
            )
        ]

    def list_mementos(
        self, fn: FunctionReference, limit: Optional[int]
    ) -> List[Memento]:
        path = DataSourceMetadataSource._get_function_path(fn)
        result = []
        for key in self.data_source.list_keys_nonversioned(
            directory=path,
            file_prefix="",
            recursive=False,
            limit=limit,
            endswith=".memento.json",
        ):
            result.append(self._read_memento(key))
        return result

    def put_memento(self, memento: Memento):
        metadata_path = DataSourceMetadataSource._get_metadata_path(
            memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash()
        )
        log.debug("Writing metadata to {}...".format(metadata_path))
        memento_json = json.dumps(MementoCodec.encode_memento(memento))
        self.data_source.output(metadata_path, io.BytesIO(memento_json.encode("utf-8")))

    def read_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        retry_on_none=False,
    ) -> Optional[Union[bytes, ResultIsWithData]]:
        metadata_key = DataSourceMetadataSource._get_metadata_key(
            fn_with_arg_hash, key, False
        )
        metadata_key_with_data = DataSourceMetadataSource._get_metadata_key(
            fn_with_arg_hash, key, True
        )

        def data_source_exists():
            retries = 3 if retry_on_none else 1
            for retry in range(0, retries):
                if self.data_source.exists_nonversioned(metadata_key):
                    return True
                if self.data_source.exists_nonversioned(metadata_key_with_data):
                    return ResultIsWithData()
                if retry < (retries - 1):
                    sleep(1)
            return False

        exists = data_source_exists()
        if exists:
            if isinstance(exists, ResultIsWithData):
                return exists

            with self.data_source.input_nonversioned(metadata_key) as f:
                return f.read()
        else:
            return None

    def write_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        value: bytes,
        stored_with_data: bool,
    ):
        metadata_key = DataSourceMetadataSource._get_metadata_key(
            fn_with_arg_hash, key, stored_with_data
        )
        log.debug("Writing metadata to key {}...".format(metadata_key))
        self.data_source.output(
            metadata_key, io.BytesIO(bytes() if stored_with_data else value)
        )

    def forget_call(self, fn_with_arg_hash: FunctionReferenceWithArgHash):
        call_path_prefix = DataSourceMetadataSource._get_path(
            fn_with_arg_hash.fn_reference, fn_with_arg_hash.arg_hash
        )
        for key in self.data_source.list_keys_nonversioned(
            directory=DataSourceKey(os.path.dirname(call_path_prefix)),
            file_prefix=os.path.basename(call_path_prefix),
            recursive=False,
        ):
            self.data_source.delete_all_versions(key, False)

    def forget_everything(self):
        self.data_source.delete_all_versions(DataSourceKey(""), True)

    def forget_function(self, fn_reference: FunctionReference):
        function_path = DataSourceMetadataSource._get_function_path(fn_reference)
        self.data_source.delete_all_versions(function_path, True)


class _CacheEntry:
    """
    Each entry stores a memento and, optionally, a value. Note that `None` is a valid value.
    """

    obj_size = None  # type: int
    memento = None  # type: Memento
    value = None  # type: object
    has_value = None  # type: bool

    def __init__(self, obj_size: int, memento: Memento, value: object, has_value: bool):
        self.obj_size = obj_size
        self.memento = memento
        self.value = value
        self.has_value = has_value


class MemoryCache:
    """
    Write-through memory cache for memoized data

    """

    memory_cache_bytes = None  # type: int
    memory_usage = None  # type: int
    lru_deque = None  # type: deque
    cache = None  # type: Dict[str, _CacheEntry]
    refs = None  # type: WeakValueDictionary

    def __init__(self, memory_cache_mb: int = None):
        self.memory_cache_bytes = memory_cache_mb * 1024 * 1024
        self.memory_usage = 0
        self.lru_deque = deque()
        self.cache = dict()
        self.refs = WeakValueDictionary()

    @staticmethod
    def _pd_mem_usage(obj: Union[pd.DataFrame, pd.Series]) -> int:
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, pd.Series):
            return obj.memory_usage(deep=True)
        raise TypeError("Can't compute the memory usage for type {}".format(type(obj)))

    @staticmethod
    def _pd_linreg_mem_usage(
        obj: Union[pd.DataFrame, pd.Series], sample_size: int = 100
    ) -> int:
        """
        A good estimate of memory usage for DataFrames and Series without taking too much time.
        Methodology splits the sample unevenly, deeply computes the memory usage of each split,
        and uses the results of a linear regression to estimate the deep memory usage of the full
        data.
        """

        # handle too little data to make sampling worth it
        # maybe the threshold should be a multiple of the sample size?
        if len(obj) <= sample_size:
            return MemoryCache._pd_mem_usage(obj)

        sample = obj.sample(sample_size)
        split = sample_size // 3
        small = sample.iloc[:split]
        big = sample.iloc[split:]
        small_mem = MemoryCache._pd_mem_usage(small)
        big_mem = MemoryCache._pd_mem_usage(big)
        m = (big_mem - small_mem) / (len(big) - len(small))
        b = big_mem - m * len(big)
        return int(m * len(obj) + b)

    @staticmethod
    def _estimate_object_size(obj: object) -> int:
        if obj is None:
            return sys.getsizeof(None)
        elif isinstance(obj, Memento):
            return sys.getsizeof(obj)
        elif isinstance(obj, pd.Index):
            return obj.memory_usage()
        elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            return MemoryCache._pd_linreg_mem_usage(obj)
        elif isinstance(obj, np.ndarray):
            return sys.getsizeof(obj)
        elif isinstance(obj, InMemoryPartition):
            result = 0
            for key in obj.list_keys():
                result += MemoryCache._estimate_object_size(obj.get(key))
                if not isinstance(result, int):
                    pass
            return sys.getsizeof(obj) + result
        elif isinstance(obj, dict):
            # Assume keys are small - estimate size of values
            return sys.getsizeof(obj) + MemoryCache._estimate_object_size(obj.values())
        elif isinstance(obj, list):
            # To avoid iterating over the whole list, estimate the size by multiplying the size
            # of the first element
            result = 0
            if hasattr(obj, "__len__"):
                length = len(obj)
                result += (
                    length * MemoryCache._estimate_object_size(next(iter(obj)))
                    if length > 0
                    else 0
                )
            else:
                i = obj  # type: Iterable
                result += sum([MemoryCache._estimate_object_size(x) for x in i])
            return sys.getsizeof(obj) + result
        return sys.getsizeof(obj)

    @staticmethod
    def _cache_key_for_memento(memento: Memento) -> str:
        return MemoryCache._cache_key_for_fn(
            memento.invocation_metadata.fn_reference_with_args.fn_reference,
            memento.invocation_metadata.fn_reference_with_args.arg_hash,
        )

    @staticmethod
    def _cache_key_for_fn(fn_ref: FunctionReference, arg_hash: str) -> str:
        return fn_ref.qualified_name + "/" + arg_hash

    def _mark_used(self, cache_key: str):
        try:
            self.lru_deque.remove(cache_key)
        except ValueError:
            pass
        self.lru_deque.append(cache_key)

    def _evict(self, cache_key: str):
        """Evict cache_key from cache."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            self.memory_usage -= entry.obj_size
            del self.cache[cache_key]
        if cache_key in self.lru_deque:
            self.lru_deque.remove(cache_key)

    def get_mementos(
        self, fns: List[FunctionReferenceWithArgHash]
    ) -> List[Optional[Memento]]:
        result = []
        for fn in fns:
            cache_key = self._cache_key_for_fn(fn.fn_reference, fn.arg_hash)
            entry = self.cache.get(cache_key)
            if entry is None:
                result.append(None)
            else:
                memento = entry.memento
                result.append(memento)
        return result

    def read_result(self, memento: Memento) -> object:
        """Return the memento if it exists in the cache, else raise KeyError"""
        cache_key = self._cache_key_for_memento(memento)
        if cache_key in self.cache:
            entry = self.cache[cache_key]  # May raise KeyError
            if not entry.has_value:
                raise KeyError()
            self._mark_used(cache_key)
            return entry.value
        else:
            # return a cached ref if it's still in memory
            return self.refs[cache_key]  # May raise KeyError

    def is_memoized(self, fn_reference: FunctionReference, arg_hash: str) -> bool:
        cache_key = self._cache_key_for_fn(fn_reference, arg_hash)
        if cache_key in self.cache:
            self._mark_used(cache_key)
            return True
        return cache_key in self.refs

    def is_all_memoized(self, fns: Iterable[FunctionReferenceWithArguments]) -> bool:
        return all([self.is_memoized(x.fn_reference, x.arg_hash) for x in fns])

    def _put_ref(self, cache_key, result):
        try:
            self.refs[cache_key] = result
        except TypeError:
            # primitives like ints, strs, and dicts can't be weakrefed
            pass

    def put(self, memento: Memento, result: object, has_result: bool):
        cache_key = self._cache_key_for_memento(memento)
        if has_result:
            self._put_ref(cache_key, result)

        # If the object is too big to fit in the cache, return immediately
        obj_size = self._estimate_object_size(result)
        if obj_size > self.memory_cache_bytes:
            return

        # "view busting"
        # until we have reliable view detection code, we'll just copy everything to prevent
        # memory leaks
        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            result = result.copy()
            self._put_ref(cache_key, result)

        # Remove any existing cached items for this memento
        self._evict(cache_key)

        # Free up memory in the cache (if needed) by discarding LRU
        while (
            len(self.lru_deque) > 0
            and self.memory_usage + obj_size > self.memory_cache_bytes
        ):
            self._evict(self.lru_deque.popleft())

        # Add to cache
        entry = _CacheEntry(obj_size, memento, result, has_value=has_result)
        self.cache[cache_key] = entry
        self.lru_deque.append(cache_key)
        self.memory_usage += obj_size

    def forget_call(self, fn_with_arg_hash: FunctionReferenceWithArgHash):
        cache_key = self._cache_key_for_fn(
            fn_with_arg_hash.fn_reference, fn_with_arg_hash.arg_hash
        )
        self.refs.pop(cache_key, None)
        self._evict(cache_key)

    def forget_everything(self):
        self.memory_usage = 0
        self.cache.clear()
        self.lru_deque.clear()
        self.refs.clear()

    def forget_function(self, fn_reference: FunctionReference):
        qualified_name = fn_reference.qualified_name
        qualified_name_slash = qualified_name + "/"
        # This is O(n) but should be a rare operation and spares us the complexity of maintaining
        # a second map
        ref_list = [
            key for key in self.refs.keys() if key.startswith(qualified_name_slash)
        ]
        for key in ref_list:
            del self.refs[key]
        evict_list = [
            key for key in self.cache.keys() if key.startswith(qualified_name_slash)
        ]
        for key in evict_list:
            self._evict(key)


class StorageBackendBase(StorageBackend, ABC):
    """
    A standard implementation of StorageBackend that delegates to a
    data store and a metadata store.

    All storage backends that derive from this backend can include an optional
    memory_cache_mb config option that enables an in-memory write-through cache.

    """

    _data_source = None  # type: DataSource
    _metadata_source = None  # type: MetadataSource
    _memory_cache = None  # type: MemoryCache
    codec = None  # type: Codec

    def __init__(
        self,
        storage_type: str,
        data_source: DataSource,
        metadata_source: MetadataSource,
        memory_cache_mb: int = None,
        config: dict = None,
        read_only: bool = None,
    ):
        super().__init__(storage_type, config=config, read_only=read_only)
        self._data_source = data_source
        self._metadata_source = metadata_source
        if memory_cache_mb:
            self._memory_cache = MemoryCache(memory_cache_mb)
        codec_config = config.get("codecConfig", {})
        codec_type = config.get("codec", "default")
        self.codec = Codec.create(codec_type, codec_config)

    @staticmethod
    def _get_function_path(fn_reference: FunctionReference) -> str:
        return fn_reference.qualified_name

    @staticmethod
    def _get_path(fn_reference: FunctionReference, arg_hash: str) -> DataSourceKey:
        return DataSourceKey(
            "{}/{}".format(
                StorageBackendBase._get_function_path(fn_reference), arg_hash
            )
        )

    def get_mementos(
        self, fns: List[FunctionReferenceWithArgHash]
    ) -> List[Optional[Memento]]:
        # First, consult cache
        if self._memory_cache:
            cache_result = self._memory_cache.get_mementos(fns)
        else:
            cache_result = [None] * len(fns)

        # For each case where we do not have a cache hit, retrieve result:
        query_fns = [fns[i] for i in range(0, len(fns)) if cache_result[i] is None]
        query_result = self._metadata_source.get_mementos(query_fns)

        # Merge cache_result and query_result into the final result list
        results = []
        query_index = 0
        for i in range(0, len(fns)):
            cr = cache_result[i]
            if cr is None:
                qr = query_result[query_index]
                results.append(qr)
                # Store only the memento in the cache
                if self._memory_cache and qr is not None:
                    self._memory_cache.put(qr, None, has_result=False)
                query_index += 1
            else:
                results.append(cr)

        return results

    def read_result(self, memento: Memento) -> object:
        if self._memory_cache:
            try:
                return self._memory_cache.read_result(memento)
            except KeyError:
                # Fall back to storage
                pass

        result = self.codec.load(
            memento.invocation_metadata.result_type,
            self._data_source,
            memento.content_key,
        )
        if self._memory_cache:
            self._memory_cache.put(memento, result, has_result=True)

        return result

    def make_url_for_result(self, memento: Memento) -> str:
        return self.codec.make_url_for_result(
            memento.invocation_metadata.result_type,
            self._data_source,
            memento.content_key,
        )

    def is_memoized(self, fn_reference: FunctionReference, arg_hash: str) -> bool:
        if self._memory_cache:
            if self._memory_cache.is_memoized(fn_reference, arg_hash):
                return True
            # if not in memory cache, fall back to storage
        return self._metadata_source.all_mementos_exist(
            [FunctionReferenceWithArgHash(fn_reference, arg_hash)]
        )

    def is_all_memoized(self, fns: Iterable[FunctionReferenceWithArguments]) -> bool:
        if self._memory_cache:
            if self._memory_cache.is_all_memoized(fns):
                return True
            # if not in memory cache, fall back to storage
        return self._metadata_source.all_mementos_exist(
            [fn.fn_reference_with_arg_hash() for fn in fns]
        )

    def list_functions(self) -> List[FunctionReference]:
        # Do not consult cache since it would not give us the full picture
        return self._metadata_source.list_functions()

    def list_mementos(self, fn: FunctionReference, limit: int = None) -> List[Memento]:
        # Do not consult cache since it would not give us the full picture
        return self._metadata_source.list_mementos(fn, limit)

    def memoize(self, key_override: str, memento: Memento, result: object) -> None:
        if self.read_only:
            return

        if self._memory_cache:
            # Write through to memory cache
            self._memory_cache.put(memento, result, has_result=True)

        # Write data
        result_type = memento.invocation_metadata.result_type
        content_key = self.codec.store(
            result_type, self._data_source, key_override, result
        )
        log.debug("Wrote data to {}".format(content_key))
        assert (result_type == ResultType.null) or (content_key is not None)
        memento.content_key = content_key

        # Write metadata
        self._metadata_source.put_memento(memento)

    def read_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        retry_on_none=False,
    ) -> bytes:
        # Metadata is not currently cached
        result = self._metadata_source.read_metadata(
            fn_with_arg_hash, key, retry_on_none=retry_on_none
        )
        if isinstance(result, ResultIsWithData):
            # Metadata is stored alongside the data object
            memento = self.get_mementos([fn_with_arg_hash])[0]
            if memento is None:
                raise IOError(
                    "Metadata shows metadata should exist with data, but could "
                    "not retrieve Memento: {}".format(fn_with_arg_hash)
                )
            result = self._data_source.input_metadata(memento.content_key, key)

        return result

    def write_metadata(
        self,
        fn_with_arg_hash: FunctionReferenceWithArgHash,
        key: str,
        value: bytes,
        store_with_content_key: Optional[VersionedDataSourceKey] = None,
    ):
        assert fn_with_arg_hash is not None
        if not self.read_only:
            if store_with_content_key:
                self._metadata_source.write_metadata(
                    fn_with_arg_hash, key, bytes(), stored_with_data=True
                )
                self._data_source.output_metadata(store_with_content_key, key, value)
            else:
                self._metadata_source.write_metadata(
                    fn_with_arg_hash, key, value, stored_with_data=False
                )
        else:
            raise ValueError("Cannot write metadata to a read-only storage backend")

    def forget_call(self, fn_with_arg_hash: FunctionReferenceWithArgHash) -> None:
        if self.read_only:
            raise ValueError("Cannot forget with a storage backend that is read-only")
        if self._memory_cache:
            self._memory_cache.forget_call(fn_with_arg_hash)
        self._metadata_source.forget_call(fn_with_arg_hash)
        # Note that we do not remove the storage associated with the call as it could
        # be shared by other mementos.

    def forget_everything(self) -> None:
        if self.read_only:
            raise ValueError("Cannot forget with a storage backend that is read-only")
        if self._memory_cache:
            self._memory_cache.forget_everything()
        self._metadata_source.forget_everything()
        # Note that we do not remove the storage associated with the call as it could
        # be shared by other mementos.

    def forget_function(self, fn_reference: FunctionReference) -> None:
        if self.read_only:
            raise ValueError("Cannot forget with a storage backend that is read-only")
        if self._memory_cache:
            self._memory_cache.forget_function(fn_reference)
        self._metadata_source.forget_function(fn_reference)
        # Note that we do not remove the storage associated with the call as it could
        # be shared by other mementos.


Codec.register("default", DefaultCodec)
