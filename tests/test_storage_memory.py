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
import shutil

import twosigma.memento as m
import tempfile

from twosigma.memento import Environment, ConfigurationRepository, FunctionCluster
from twosigma.memento.storage_memory import MemoryStorageBackend
from tests.test_storage_backend import StorageBackendTester


class TestStorageMemory(StorageBackendTester):
    """Class to test memory backend."""

    def setup_method(self):
        super().setup_method()
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_storage_memory_test")
        m.Environment.set(
            Environment(
                name="test1",
                base_dir=self.base_path,
                repos=[
                    ConfigurationRepository(
                        name="repo1",
                        clusters={
                            "cluster1": FunctionCluster(
                                name="cluster1", storage=MemoryStorageBackend()
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

    # The rest of the test methods come from StorageBackendTester

    def test_make_url_for_result(self):
        # This test is not applicable for the memory store
        pass
