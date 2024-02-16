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

import datetime
import shutil
import tempfile

import twosigma.memento as m

from twosigma.memento import (
    StorageBackend,
    FunctionCluster,
    ConfigurationRepository,
    Environment,
)  # noqa: F401
from twosigma.memento.metadata import ResultType, InvocationMetadata, Memento
from twosigma.memento.storage_null import NullStorageBackend
from twosigma.memento.types import VersionedDataSourceKey


@m.memento_function(cluster="cluster1")
def fn_return_none_1():
    return None


@m.memento_function(cluster="cluster1")
def fn_return_none_2():
    return None


class TestStorageNull:
    """
    Test the null storage backend. This does not extend the
    base storage test because this filesystem intentionally
    does not store anything.

    """

    backend = None  # type: StorageBackend

    def setup_method(self):
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_storage_null_test")
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
                                name="cluster1", storage=NullStorageBackend()
                            )
                        },
                    )
                ],
            )
        )
        self.cluster = m.Environment.get().get_cluster("cluster1")
        self.backend = self.cluster.storage

    def teardown_method(self):
        shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)

    def test_list_functions(self):
        assert 0 == len(self.backend.list_functions())

    def test_memoize(self):
        fn1_reference = fn_return_none_1.fn_reference().with_args()
        fn2_reference = fn_return_none_2.fn_reference().with_args()
        now = datetime.datetime.now(datetime.timezone.utc)
        test_runtime = 123.0
        memento = Memento(
            time=now,
            invocation_metadata=InvocationMetadata(
                fn_reference_with_args=fn1_reference,
                runtime=datetime.timedelta(test_runtime),
                result_type=ResultType.string,
                invocations=[fn2_reference],
                resources=[],
            ),
            function_dependencies={
                fn1_reference.fn_reference,
                fn2_reference.fn_reference,
            },
            runner={},
            correlation_id="abc123",
            content_key=VersionedDataSourceKey("key", "def456"),
        )
        result = fn_return_none_1()
        self.backend.memoize(None, memento, result)

        # The null storage should not waste compute cycles computing the content hash
        assert memento.content_key is None

        assert (
            self.backend.get_memento(fn1_reference.fn_reference_with_arg_hash()) is None
        )

        assert not self.backend.is_memoized(
            fn1_reference.fn_reference, fn1_reference.arg_hash
        )
        self.backend.forget_call(fn1_reference.fn_reference_with_arg_hash())
        assert not self.backend.is_memoized(
            fn1_reference.fn_reference, fn1_reference.arg_hash
        )
