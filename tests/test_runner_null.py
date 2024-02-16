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

import shutil
import tempfile

import pytest

import twosigma.memento as m
from twosigma.memento import (
    RunnerBackend,
    Environment,
    ConfigurationRepository,
    FunctionCluster,
)  # noqa: F401
from twosigma.memento.runner_null import NullRunnerBackend
from twosigma.memento.storage_null import NullStorageBackend


@m.memento_function(cluster="cluster1")
def fn1():
    return 1


class TestRunnerNull:
    """
    Test the null runner backend. This does not extend the
    base runner test because this runner intentionally
    does not run anything.

    """

    backend = None  # type: RunnerBackend

    def setup_method(self):
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_runner_null_test")
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
                                storage=NullStorageBackend(),
                                runner=NullRunnerBackend(),
                            )
                        },
                    )
                ],
            )
        )
        self.cluster = m.Environment.get().get_cluster("cluster1")
        self.backend = self.cluster.runner

    def teardown_method(self):
        shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)

    @staticmethod
    def test_null_runner():
        with pytest.raises(RuntimeError):
            fn1()
