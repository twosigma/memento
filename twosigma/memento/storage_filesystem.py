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
Backend that uses the filesystem to store data and metadata.

Metadata is stored in json files. Data is stored in pickle files.

The following configuration options apply to this backend store:
* path - (Optional) The root path under which the metadata and data are stored.
* metadata_path - (Optional) The root path under which the metadata
  is stored, if different from the data path.
* memory_cache_mb - (Optional) Number of megabytes of memory to
  reserve for a write-through memory cache
* readonly - If true, data will never be written, only read.

"""

import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import IO, Iterable, List, cast, Optional, Dict  # noqa: F401
from urllib.parse import unquote
from uuid import uuid4

from .logging import log
from .metadata import ResultType
from .partition import Partition
from .storage import StorageBackend
from .storage_base import (
    DataSource,
    DataSourceMetadataSource,
    StorageBackendBase,
    DefaultCodec,
    Codec,
)
from .types import DataSourceKey, VersionedDataSourceKey


class _FilesystemDataSource(DataSource):
    """
    Data source for reading / writing on the filesystem

    Versioned objects are emulated as follows:
    * Actual data for key `a/b` is written in `a/.versions/uuid/b`
    * A file `a/b.link` is created containing a path to `a/.versions/uuid/b`

    """

    base_path = None  # type: Path

    def __init__(self, base_path):
        super().__init__()
        self.base_path = Path(base_path)

    def _escape_key(self, key: str) -> str:
        # Windows does not support ":" in path names, so escape with %3A
        return key.replace(":", "%3A")

    def _get_non_versioned_path(self, key: DataSourceKey) -> Path:
        return self.base_path.joinpath(self._escape_key(key.key))

    def _get_non_versioned_link_path(self, key: str) -> Path:
        return self.base_path.joinpath(self._escape_key(key) + ".link")

    def _write_non_versioned_link(self, versioned_key: VersionedDataSourceKey):
        non_versioned_path = self._get_non_versioned_link_path(versioned_key.key)
        versioned_path = self._get_path_versioned(versioned_key)
        with open(str(non_versioned_path), "w") as f:
            f.write(str(versioned_path))

    def _delete_non_versioned_link(self, key: DataSourceKey):
        non_versioned_path = self._get_non_versioned_link_path(
            self._escape_key(key.key)
        )
        if os.path.isfile(non_versioned_path):
            os.unlink(non_versioned_path)

    def _read_non_versioned_link(self, key: DataSourceKey) -> Path:
        non_versioned_path = self._get_non_versioned_link_path(
            self._escape_key(key.key)
        )
        with open(str(non_versioned_path), "r") as f:
            versioned_path = Path(f.read())
        return versioned_path

    def _get_versions_directory(self, key: DataSourceKey) -> Path:
        path = self._get_non_versioned_link_path(self._escape_key(key.key))
        dirname = path.parent
        return dirname.joinpath(".versions")

    def _get_path_versioned(
        self, key: VersionedDataSourceKey, metadata_key: Optional[str] = None
    ) -> Path:
        escaped_key = self._escape_key(key.key)
        dirname = os.path.dirname(escaped_key)
        basename = os.path.basename(escaped_key)
        if metadata_key:
            metafile = "{}.meta.{}".format(basename, metadata_key)
            return self.base_path.joinpath(dirname, ".versions", key.version, metafile)
        else:
            return self.base_path.joinpath(dirname, ".versions", key.version, basename)

    @staticmethod
    def _do_input(path: Path):
        result = io.FileIO(Path(path))
        log.debug("Reading {}".format(path))
        return cast(IO, result)

    def input_nonversioned(self, key: DataSourceKey) -> IO:
        return self._do_input(self._read_non_versioned_link(key))

    def input_versioned(self, key: VersionedDataSourceKey) -> IO:
        return self._do_input(self._get_path_versioned(key))

    def input_metadata(
        self, content_key: VersionedDataSourceKey, metadata_key: str
    ) -> bytes:
        path = self._get_path_versioned(content_key, metadata_key=metadata_key)
        with self._do_input(path) as f:
            return f.read()

    def make_url_for_key(self, key: Optional[VersionedDataSourceKey]) -> Optional[str]:
        return self._get_path_versioned(key).as_uri() if key is not None else None

    def exists_versioned(self, key: VersionedDataSourceKey) -> bool:
        path = self._get_path_versioned(key)
        result = path.exists()
        log.debug("Exists {}? {}".format(key, result))
        return result

    def exists_nonversioned(self, key: DataSourceKey) -> bool:
        non_versioned_path = self._get_non_versioned_link_path(
            self._escape_key(key.key)
        )
        if not os.path.exists(non_versioned_path):
            result = False
        else:
            path = self._read_non_versioned_link(key)
            result = path.exists()
        log.debug("Exists {}? {}".format(key, result))
        return result

    def all_exist_versioned(self, keys: List[VersionedDataSourceKey]) -> List[bool]:
        return [self.exists_versioned(key) for key in keys]

    def all_exist_nonversioned(self, keys: List[DataSourceKey]) -> List[bool]:
        return [self.exists_nonversioned(key) for key in keys]

    def output(self, key: DataSourceKey, data: IO) -> VersionedDataSourceKey:
        uuid = str(uuid4())
        versioned_key = VersionedDataSourceKey(key=key.key, version=uuid)
        versioned_path = self._get_path_versioned(versioned_key)
        os.makedirs(str(versioned_path.parent), exist_ok=True)
        log.debug("Writing {} -> {}".format(key, versioned_key))
        with versioned_path.open(mode="wb") as f:
            shutil.copyfileobj(data, f)
        self._write_non_versioned_link(versioned_key)
        return versioned_key

    def reference(
        self,
        src_data_source: DataSource,
        src_key: VersionedDataSourceKey,
        target_key: VersionedDataSourceKey,
    ):
        # This data source does not perform reference counting
        pass

    def output_metadata(
        self, content_key: VersionedDataSourceKey, metadata_key: str, value: bytes
    ):
        versioned_path = self._get_path_versioned(
            content_key, metadata_key=metadata_key
        )
        log.debug("Writing {}".format(versioned_path))
        with versioned_path.open(mode="wb") as f:
            f.write(value)

    def delete_nonversioned_key(self, key: DataSourceKey):
        self._delete_non_versioned_link(key)

    def _delete_all_versions_for_key(self, key: DataSourceKey):
        # Remove the symlink
        self._delete_non_versioned_link(key)

        # Remove all files that match the pattern dirname/.versions/uuid/basename
        versions_dir = self._get_versions_directory(key)
        basename = os.path.basename(self._escape_key(key.key))
        # Remove metadata stored with objects:
        for match in versions_dir.glob("*/{}.meta.*".format(basename)):
            match.unlink()
        # Remove objects
        for match in versions_dir.glob("*/{}".format(basename)):
            match.unlink()
            match.parent.rmdir()  # remove uuid dir as well
        if len(list(versions_dir.iterdir())) == 0:
            versions_dir.rmdir()

    def delete_all_versions(self, key: DataSourceKey, recursive: bool):
        link_path = self._get_non_versioned_link_path(self._escape_key(key.key))
        path = self._get_non_versioned_path(key)
        if link_path.exists() or path.exists():
            log.debug("Removing {}{}".format(key, " recursively" if recursive else ""))
            self._delete_non_versioned_link(key)

            if recursive:
                # this will take care of the .versions/ folder as well
                if os.path.isdir(str(path)):
                    shutil.rmtree(str(path))
            else:
                self._delete_all_versions_for_key(key)

            # Scan all parent paths, and, if empty, remove those dirs
            while path != self.base_path:
                if path.is_dir() and len(list(path.iterdir())) == 0:
                    log.debug("Removing empty directory {}".format(str(path)))
                    path.rmdir()
                path = path.parent

    def list_keys_nonversioned(
        self,
        directory: DataSourceKey,
        file_prefix: str = "",
        recursive: bool = False,
        limit: int = None,
        endswith: str = None,
    ) -> Iterable[DataSourceKey]:
        dir_path = self._get_non_versioned_path(directory)
        if not dir_path.is_dir():
            return []
        escaped_key = self._escape_key(directory.key)
        dir_prefix = (
            (escaped_key + "/") if escaped_key and not escaped_key.endswith("/") else ""
        )
        if recursive:

            def walk_path_recursive():
                count = 0
                dir_path_str = str(dir_path)
                for dirpath, dirname, filenames in os.walk(dir_path_str):
                    if (
                        "{}.versions{}".format(os.sep, os.sep) in dirpath
                        or "{}.tmp{}".format(os.sep, os.sep) in dirpath
                    ):
                        continue
                    for filename in filenames:
                        entry = (
                            dir_prefix
                            + os.path.join(dirpath, filename)[len(dir_path_str) + 1 :]
                        )
                        if entry.endswith(".link"):
                            entry = entry[0:-5]  # strip .link off end of string
                        # Filter down to files that begin with file_prefix
                        if os.path.basename(entry).startswith(file_prefix):
                            if endswith is not None and not os.path.basename(
                                entry
                            ).endswith(endswith):
                                continue
                            count += 1
                            yield DataSourceKey(entry.replace(os.sep, "/"))
                            if count == limit:
                                return

            entries = list(walk_path_recursive())
        else:

            def walk_path():
                count = 0
                for entry in dir_path.iterdir():
                    if entry.name == ".versions" or entry.name == ".tmp":
                        continue
                    # Filter down to files that begin with file_prefix
                    if entry.name.startswith(file_prefix):
                        entry_name = unquote(entry.name)
                        if entry_name.endswith(".link"):
                            entry_name = entry_name[
                                0:-5
                            ]  # strip .link off end of string
                        if endswith is not None and not entry_name.endswith(endswith):
                            continue
                        count += 1
                        yield DataSourceKey(dir_prefix + entry_name)
                        if count == limit:
                            return

            entries = list(walk_path())
        return sorted(entries, key=lambda k: k.key)

    def get_versioned_key(self, key: DataSourceKey) -> VersionedDataSourceKey:
        versioned_path = self._read_non_versioned_link(key)
        uuid = versioned_path.parent.name
        return VersionedDataSourceKey(key=key.key, version=uuid)


class FilesystemStorageBackend(StorageBackendBase):
    config_path = None  # type: str
    metadata_config_path = None  # type: str

    def __init__(
        self,
        config: dict = None,
        path: str = None,
        metadata_path: str = None,
        memory_cache_mb: int = None,
        read_only: bool = None,
    ):
        """
        Create a storage backend that reads from the filesystem.
        See module documentation for parameters. Parameters that follow
        `config` override its values.

        """
        config = config if config is not None else {}
        config_path = config.get("path", None)
        if path is not None:
            config_path = path
        if config_path is None:
            config_path = str(Path("~").expanduser().joinpath(".memento", "data"))
        self.config_path = config_path

        metadata_config_path = config.get("metadata_path", None)
        if metadata_path is not None:
            metadata_config_path = metadata_path
        if metadata_config_path is None:
            metadata_config_path = config_path
        self.metadata_config_path = metadata_config_path

        data_source = _FilesystemDataSource(self.config_path)
        metadata_source = DataSourceMetadataSource(
            _FilesystemDataSource(self.metadata_config_path)
            if self.metadata_config_path != self.config_path
            else data_source
        )

        super().__init__(
            "filesystem",
            data_source=data_source,
            metadata_source=metadata_source,
            memory_cache_mb=memory_cache_mb,
            config=config,
            read_only=read_only,
        )

    def to_dict(self):
        config = {"type": "filesystem"}
        if self.read_only is not None:
            config["readonly"] = self.read_only
        if self.config_path is not None:
            config["path"] = self.config_path
        if self._memory_cache is not None:
            config["memory_cache_mb"] = (
                self._memory_cache.memory_cache_bytes / 1024 / 1024
            )
        return config


class OnDiskPartition(Partition):
    """
    Partition that stores all results in temporary files on disk.

    This is typically the partition used by functions when they wish to stage and return
    large amounts of data that won't all fit in memory at once. Note that the index is still
    stored in memory. Note also there must be sufficient local disk to stage the data.

    To use this class, construct it and add each key, one at a time. The data will be
    staged to the local disk as it is written, and then upon return the partition will be
    re-read and re-serialized to its final storage.

    Example:
    .. code-block:: python
       result = OnDiskPartition()
       result["a"] = data_frame_a
       result["b"] = data_frame_b
       return result

    Finally, when the object is garbage collected, the temporary files are removed.
    """

    _tempdir = None  # type: Path
    _data_source = None  # type: DataSource
    _result_keys = None  # type: Dict[str, VersionedDataSourceKey]
    _result_types = None  # type: Dict[str, ResultType]
    _codec = None  # type: Codec

    _output_keys = None  # type: Optional[Dict]
    """
    If set, tracks where the output keys were written when this partition was last serialized.
    This allows the partition to be stored in the in-memory cache and still serve as a parent
    for a merged partition.

    These are not to be confused with the keys for the staged data on disk, which is stored
    in `_result_keys`.
    """

    _parent_data_source = None
    """
    If set, tracks the data source where output keys were written when this partition was last
    serialized.
    """

    def __init__(self):
        """
        Creates an on-disk partition, initially empty.
        """
        super().__init__()
        self._tempdir = Path(tempfile.mkdtemp(prefix="memento_partition_"))
        self._data_source = _FilesystemDataSource(str(self._tempdir))
        self._codec = DefaultCodec(config={})
        self._result_keys = dict()
        self._result_types = dict()
        self._output_keys = None
        self._parent_data_source = None

    def __del__(self):
        """
        Remove temporary files when object is destructed
        """
        shutil.rmtree(self._tempdir)

    def __setitem__(self, key: str, value: object):
        """
        Add the given value to the partition for the given key.
        This writes the data to disk for later retrieval.
        """
        result_type = ResultType.from_object(value)
        self._result_types[key] = result_type
        self._result_keys[key] = self._codec.store(
            result_type, self._data_source, None, value
        )

    def __getitem__(self, item: str) -> object:
        """
        Convenience syntax for get(key)
        """
        return self.get(item)

    def get(self, key: str) -> object:
        if key not in self._result_types.keys():
            if self._merge_parent:
                return self._merge_parent.get(key)
            raise ValueError("Key '{}' not in key list for partition".format(key))
        return self._codec.load(
            self._result_types[key], self._data_source, self._result_keys[key]
        )

    def list_keys(self, _include_merge_parent: bool = True) -> Iterable[str]:
        if _include_merge_parent and self._merge_parent:
            result = set(self._merge_parent.list_keys())
            result.update(self._result_types.keys())
            return sorted(list(result))

        return sorted(list(self._result_types.keys()))


StorageBackend.register("filesystem", FilesystemStorageBackend)
