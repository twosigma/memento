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
import os
from typing import cast

import pandas as pd
import shutil

import twosigma.memento as m
import tempfile

from twosigma.memento import Environment, ConfigurationRepository, FunctionCluster
from twosigma.memento.reference import FunctionReferenceWithArguments
from twosigma.memento.storage_filesystem import FilesystemStorageBackend
from tests.test_storage_backend import (
    StorageBackendTester,
    DataSourceTester,
    MetadataSourceTester,
    fn1,
)


class TestStorageFilesystem(StorageBackendTester):
    """Class to test filesystem backend."""

    def setup_method(self):
        super().setup_method()
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_storage_filesystem_test")
        self.data_path = "{}/data".format(self.base_path)
        m.Environment.set(
            Environment(
                name="test1",
                base_dir=self.base_path,
                repos=[
                    ConfigurationRepository(
                        name="repo1",
                        clusters={
                            "cluster1": FunctionCluster(
                                name="cluster1",
                                storage=FilesystemStorageBackend(
                                    path="{}/data".format(self.base_path),
                                    # test with a different metadata path from data path
                                    metadata_path="{}/metadata".format(self.base_path),
                                ),
                            )
                        },
                    )
                ],
            )
        )
        self.cluster = m.Environment.get().get_cluster("cluster1")
        self.backend = self.cluster.storage

    def teardown_method(self):
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)
        super().teardown_method()

    def test_dir_created_lazily(self):
        """Test that the base path is created lazily"""
        assert not os.path.isdir(self.data_path)

    # The rest of the test methods come from StorageBackendTester


class TestStorageFilesystemWithMemoryCache(StorageBackendTester):
    """Class to test filesystem backend with an in-memory cache."""

    def setup_method(self):
        super().setup_method()
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(
            prefix="memento_storage_filesystem_with_memory_cache_test"
        )
        self.data_path = "{}/data".format(self.base_path)
        m.Environment.set(
            Environment(
                name="test1",
                base_dir=self.base_path,
                repos=[
                    ConfigurationRepository(
                        name="repo1",
                        clusters={
                            "cluster1": FunctionCluster(
                                name="cluster1",
                                storage=FilesystemStorageBackend(
                                    path="{}/data".format(self.base_path),
                                    # test with a different metadata path from data path
                                    metadata_path="{}/metadata".format(self.base_path),
                                    memory_cache_mb=16,
                                ),
                            )
                        },
                    )
                ],
            )
        )
        self.cluster = m.Environment.get().get_cluster("cluster1")
        self.backend = self.cluster.storage

    def teardown_method(self):
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)
        super().teardown_method()

    def test_cache_eviction(self):
        cache = cast(FilesystemStorageBackend, self.backend)._memory_cache
        for i in range(0, 32):
            fn_ref = fn1.fn_reference().with_args(
                i
            )  # type: FunctionReferenceWithArguments
            mm = self.get_dummy_memento(fn_ref)
            self.backend.memoize(None, mm, "." * 1024000)
            assert self.backend.is_memoized(fn_ref.fn_reference, fn_ref.arg_hash)
            s = cast(pd.Series, self.backend.read_result(mm))
            assert 1024000 == len(s)
            assert cache.memory_usage < 16 * 1024 * 1024

        assert 16 == len(cache.cache.keys())

    # The rest of the test methods come from StorageBackendTester


class TestFilesystemDataSourceTest(DataSourceTester):
    """Class to test filesystem backend."""

    def setup_method(self):
        super().setup_method()
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_storage_filesystem_test")
        self.data_path = "{}/data".format(self.base_path)
        m.Environment.set(
            Environment(
                name="test1",
                base_dir=self.base_path,
                repos=[
                    ConfigurationRepository(
                        name="repo1",
                        clusters={
                            "cluster1": FunctionCluster(
                                name="cluster1",
                                storage=FilesystemStorageBackend(
                                    path="{}/data".format(self.base_path)
                                ),
                            )
                        },
                    )
                ],
            )
        )
        self.cluster = m.Environment.get().get_cluster("cluster1")
        # noinspection PyUnresolvedReferences
        self.data_source = self.cluster.storage._data_source

    def teardown_method(self):
        shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)
        super().teardown_method()

    # The rest of the test methods come from DataSourceTester


class TestFilesystemMetadataSource(MetadataSourceTester):
    """Class to test filesystem backend as a metadata source."""

    def setup_method(self):
        super().setup_method()
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_storage_filesystem_test")
        self.data_path = "{}/metadata".format(self.base_path)
        m.Environment.set(
            Environment(
                name="test1",
                base_dir=self.base_path,
                repos=[
                    ConfigurationRepository(
                        name="repo1",
                        clusters={
                            "cluster1": FunctionCluster(
                                name="cluster1",
                                storage=FilesystemStorageBackend(
                                    path="{}/data".format(self.base_path)
                                ),
                            )
                        },
                    )
                ],
            )
        )
        self.cluster = m.Environment.get().get_cluster("cluster1")
        # noinspection PyUnresolvedReferences
        self.metadata_source = self.cluster.storage._metadata_source

    def teardown_method(self):
        shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)
        super().teardown_method()

    # The rest of the test methods come from DataSourceTester
