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
    Environment,
    ConfigurationRepository,
    FunctionCluster,
    memento_function,
    file_resource,
    Memento,
)  # noqa: F401
from twosigma.memento.context import RecursiveContext
from twosigma.memento.reference import FunctionReferenceWithArguments
from twosigma.memento.runner_null import NullRunnerBackend
from twosigma.memento.serialization import MementoCodec
from twosigma.memento.storage_null import NullStorageBackend


@memento_function
def fn_test(x):
    return x * 2


class TestSerialization:
    def setup_method(self):
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_serialization_test")
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

    def teardown_method(self):
        shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)

    def test_date(self):
        obj = datetime.date(2019, 9, 24)
        result = MementoCodec.decode_datetime(MementoCodec.encode_datetime(obj))
        assert obj == result

    def test_memento(self):
        fn_test(1)
        obj = fn_test.memento(1)  # type: Memento
        result = MementoCodec.decode_memento(MementoCodec.encode_memento(obj))
        assert repr(obj) == repr(result)

    def test_invocation_metadata(self):
        fn_test(1)
        obj = fn_test.memento(1).invocation_metadata
        result = MementoCodec.decode_invocation_metadata(
            MementoCodec.encode_invocation_metadata(obj)
        )
        assert repr(obj) == repr(result)

    def test_fn_reference_with_args(self):
        obj = FunctionReferenceWithArguments(
            fn_reference=fn_test.fn_reference(), args=(1,), kwargs={}
        )
        result = MementoCodec.decode_fn_reference_with_args(
            MementoCodec.encode_fn_reference_with_args(obj)
        )
        assert repr(obj) == repr(result)

    def test_fn_reference_with_arg_hash(self):
        obj = FunctionReferenceWithArguments(
            fn_reference=fn_test.fn_reference(), args=(1,), kwargs={}
        ).fn_reference_with_arg_hash()
        result = MementoCodec.decode_fn_reference_with_arg_hash(
            MementoCodec.encode_fn_reference_with_arg_hash(obj)
        )
        assert repr(obj) == repr(result)

    def test_resource_handle(self):
        obj = file_resource("/dev/null")
        result = MementoCodec.decode_resource_handle(
            MementoCodec.encode_resource_handle(obj)
        )
        assert repr(obj) == repr(result)

    def test_fn_reference(self):
        obj = fn_test.fn_reference()
        result = MementoCodec.decode_fn_reference(MementoCodec.encode_fn_reference(obj))
        assert repr(obj) == repr(result)

    def test_recursive_context(self):
        obj = RecursiveContext(correlation_id="123", retry_on_remote_call=True)
        result = MementoCodec.decode_recursive_context(
            MementoCodec.encode_recursive_context(obj)
        )
        assert obj.__dict__ == result.__dict__

    def test_arg(self):
        obj = fn_test
        result = MementoCodec.decode_arg(MementoCodec.encode_arg(fn_test))
        assert obj == result
