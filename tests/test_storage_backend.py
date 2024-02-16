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
import io
import time
from abc import ABC
from io import BytesIO
from typing import cast, Optional
from unittest import TestCase  # noqa: F401

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

# noinspection PyUnresolvedReferences
from pandas import Timestamp
import numpy as np
import twosigma.memento as m
from twosigma.memento.types import VersionedDataSourceKey
from twosigma.memento import StorageBackend, Environment  # noqa: F401
from twosigma.memento.memento import FunctionReference
from twosigma.memento.metadata import ResultType, InvocationMetadata, Memento
from twosigma.memento.partition import InMemoryPartition
from twosigma.memento.reference import FunctionReferenceWithArguments
from twosigma.memento.storage_base import (
    DataSource,
    MetadataSource,
    DataSourceKey,
    MemoryCache,
    ResultIsWithData,
)

now_date = Timestamp.today(None).date()
now_time = Timestamp.today(None)


@m.memento_function(cluster="cluster1")
def fn1(a):
    return a


@m.memento_function(cluster="cluster1")
def fn2(a):
    return a


@m.memento_function(cluster="cluster1")
def fn_return_none_1():
    return None


@m.memento_function(cluster="cluster1")
def fn_return_none_2():
    return None


@m.memento_function(cluster="cluster1")
def fn_raise_value_error():
    raise ValueError("test message")


@m.memento_function(cluster="cluster1")
def fn_return_str():
    return "abc123"


@m.memento_function(cluster="cluster1")
def fn_return_int():
    return 123


@m.memento_function(cluster="cluster1")
def fn_return_float():
    return 123.456


@m.memento_function(cluster="cluster1")
def fn_return_bool():
    return True


@m.memento_function(cluster="cluster1")
def fn_return_date():
    global now_date
    return now_date


@m.memento_function(cluster="cluster1")
def fn_return_time():
    global now_time
    return now_time


@m.memento_function(cluster="cluster1")
def fn_return_dictionary():
    return {
        "a": 2,
        "b": {"c": "d"},
        "e": {"f": [1.0, 2.1, 3.2]},
        "g": pd.Series([1, 2, 3]),
    }


@m.memento_function(cluster="cluster1")
def fn_return_list():
    return [1, 2, 3, "a", "b", "c", pd.Series([1, 2, 3])]


@m.memento_function(cluster="cluster1")
def fn_return_2d_array():
    return [[1, 2, 3], [4, 5, 6]]


@m.memento_function(cluster="cluster1")
def fn_return_numpy_array_bool():
    return np.array([True, False])


@m.memento_function(cluster="cluster1")
def fn_return_numpy_array_int8():
    return np.array([1, 2, 3], dtype="int8")


@m.memento_function(cluster="cluster1")
def fn_return_numpy_array_int16():
    return np.array([1, 2, 3], dtype="int16")


@m.memento_function(cluster="cluster1")
def fn_return_numpy_array_int32():
    return np.array([1, 2, 3], dtype="int32")


@m.memento_function(cluster="cluster1")
def fn_return_numpy_array_int64():
    return np.array([1, 2, 3], dtype="int64")


@m.memento_function(cluster="cluster1")
def fn_return_numpy_array_float32():
    return np.array([1.0, 2.0, 3.0], dtype="float32")


@m.memento_function(cluster="cluster1")
def fn_return_numpy_array_float64():
    return np.array([1.0, 2.0, 3.0], dtype="float64")


@m.memento_function(cluster="cluster1")
def fn_return_index():
    return pd.DatetimeIndex(
        [Timestamp.now() - datetime.timedelta(days=1), Timestamp.now()]
    )


@m.memento_function(cluster="cluster1")
def fn_return_series():
    return pd.Series([1, 2, 3])


@m.memento_function(cluster="cluster1")
def fn_return_series_with_index():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    return pd.Series([1, 2, 3], index=idx, name="foo")


@m.memento_function(cluster="cluster1")
def fn_return_series_with_multiindex():
    idx = pd.DatetimeIndex(
        [
            Timestamp.now() - datetime.timedelta(days=2),
            Timestamp.now() - datetime.timedelta(days=1),
            Timestamp.now(),
        ]
    )
    idx2 = pd.Index([11, 22, 33])
    idx3 = pd.date_range("2020-01-01", periods=3, freq="D")
    return pd.Series([1, 2, 3], index=[idx, idx2, idx3], name="foo")


@m.memento_function(cluster="cluster1")
def fn_return_dataframe():
    return pd.DataFrame(
        [
            {"name": "a", "val": "1"},
            {"name": "b", "val": "2"},
            {"name": "c", "val": "3"},
        ]
    )


@m.memento_function(cluster="cluster1")
def fn_return_dataframe_with_index():
    return pd.DataFrame(
        [
            {"name": "a", "val": "1"},
            {"name": "b", "val": "2"},
            {"name": "c", "val": "3"},
        ],
        index=pd.Index(["x", "y", "z"]),
    )


@m.memento_function(cluster="cluster1")
def fn_return_partition():
    return InMemoryPartition(
        {
            "a": None,
            "b": True,
            "c": 1,
            "d": 2.0,
            "e": [1, 2, 3],
            "f": {"a": "b"},
            "g": np.array([1, 2, 3], dtype="int8"),
            "h": pd.DataFrame([{"name": "a", "val": "1"}]),
            "i": InMemoryPartition({"j": "k"}),
        }
    )


@m.memento_function(cluster="cluster1")
def fn_return_partition_with_parent_a():
    return InMemoryPartition({"a": 1, "b": 2})


@m.memento_function(cluster="cluster1")
def fn_return_partition_with_parent_b():
    partition_a = fn_return_partition_with_parent_a()
    partition_b = InMemoryPartition({"b": 3, "c": 4})
    partition_b._merge_parent = partition_a
    return partition_b


@m.memento_function(cluster="cluster1")
def fn_return_partition_with_parent_c():
    partition_b = fn_return_partition_with_parent_b()
    partition_c = InMemoryPartition({"c": 5, "d": 6})
    partition_c._merge_parent = partition_b
    return partition_c


@m.memento_function(cluster="cluster1")
def fn_add_one(a):
    return a + 1


def retry_until(f, criteria):
    """Since some operations are eventually consistent, this function allows for retries."""
    result = None
    for i in range(0, 3):
        result = f()
        if criteria(result):
            break
        time.sleep(1)
    return result


class StorageBackendTester(ABC):
    """
    Abstract base class to test storage backends.

    Subclasses should set `self.backend` to the storage backend being tested.

    """

    backend = None  # type: StorageBackend
    test = None  # type: TestCase

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @staticmethod
    def get_dummy_memento(
        fn_reference_with_args: FunctionReferenceWithArguments,
    ) -> Memento:
        now = datetime.datetime.now(datetime.timezone.utc)
        return Memento(
            time=now,
            invocation_metadata=InvocationMetadata(
                runtime=datetime.timedelta(seconds=123.0),
                fn_reference_with_args=fn_reference_with_args,
                result_type=ResultType.string,
                invocations=[],
                resources=[],
            ),
            function_dependencies={fn_reference_with_args.fn_reference},
            runner={},
            correlation_id="abc123",
            content_key=VersionedDataSourceKey("def456", "0"),
        )

    def test_list_functions(self):
        assert 0 == len(self.backend.list_functions())
        fn1(1)
        retry_until(lambda: self.backend.list_functions(), lambda x: len(x) == 1)
        assert {FunctionReference(fn1).qualified_name} == set(
            [x.qualified_name for x in self.backend.list_functions()]
        )
        fn1(1)
        fn1(2)
        fn_return_none_1()
        retry_until(lambda: self.backend.list_functions(), lambda x: len(x) == 2)
        assert {
            FunctionReference(fn1).qualified_name,
            FunctionReference(fn_return_none_1).qualified_name,
        } == set([x.qualified_name for x in self.backend.list_functions()])

    def test_list_mementos(self):
        assert 0 == len(self.backend.list_mementos(FunctionReference(fn1)))
        assert 0 == len(self.backend.list_mementos(FunctionReference(fn2)))
        fn1(1)
        retry_until(
            lambda: self.backend.list_mementos(FunctionReference(fn1)),
            lambda x: len(x) == 1,
        )
        assert {1} == set(
            [
                x.invocation_metadata.fn_reference_with_args.args[0]
                for x in self.backend.list_mementos(FunctionReference(fn1))
            ]
        )
        fn1(1)
        fn1(2)
        retry_until(
            lambda: self.backend.list_mementos(FunctionReference(fn1)),
            lambda x: len(x) == 2,
        )
        assert {1, 2} == set(
            [
                x.invocation_metadata.fn_reference_with_args.args[0]
                for x in self.backend.list_mementos(FunctionReference(fn1))
            ]
        )
        fn2(3)
        retry_until(
            lambda: self.backend.list_mementos(FunctionReference(fn2)),
            lambda x: len(x) == 1,
        )
        assert {3} == set(
            [
                x.invocation_metadata.fn_reference_with_args.args[0]
                for x in self.backend.list_mementos(FunctionReference(fn2))
            ]
        )

    def test_memoize(self):
        fn1_reference = fn_return_none_1.fn_reference().with_args()
        fn2_reference = fn_return_none_2.fn_reference().with_args()
        now = datetime.datetime.now(datetime.timezone.utc)
        test_runtime = 123.0
        memento = Memento(
            time=now,
            invocation_metadata=InvocationMetadata(
                runtime=datetime.timedelta(test_runtime),
                fn_reference_with_args=fn1_reference,
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
            content_key=VersionedDataSourceKey("def456", "0"),
        )
        result = None
        self.backend.memoize(None, memento, result)

        # Validate the content hash was changed during the call to memoize
        assert VersionedDataSourceKey("def456", "0") != memento.content_key

        actual_memento = self.backend.get_memento(
            fn1_reference.fn_reference_with_arg_hash()
        )
        actual_result = self.backend.read_result(actual_memento)

        assert memento.time == actual_memento.time, "now={}".format(now)
        assert (
            memento.invocation_metadata.runtime
            == actual_memento.invocation_metadata.runtime
        )
        assert (
            memento.invocation_metadata.fn_reference_with_args.fn_reference.qualified_name
            == actual_memento.invocation_metadata.fn_reference_with_args.fn_reference.qualified_name
        )
        assert (
            memento.invocation_metadata.fn_reference_with_args.args
            == actual_memento.invocation_metadata.fn_reference_with_args.args
        )
        assert (
            memento.invocation_metadata.fn_reference_with_args.kwargs
            == actual_memento.invocation_metadata.fn_reference_with_args.kwargs
        )
        assert (
            memento.invocation_metadata.fn_reference_with_args.arg_hash
            == actual_memento.invocation_metadata.fn_reference_with_args.arg_hash
        )
        assert (
            memento.invocation_metadata.result_type
            == actual_memento.invocation_metadata.result_type
        )
        assert (
            memento.invocation_metadata.invocations[0].fn_reference.qualified_name
            == actual_memento.invocation_metadata.invocations[
                0
            ].fn_reference.qualified_name
        )
        assert (
            memento.invocation_metadata.invocations[0].arg_hash
            == actual_memento.invocation_metadata.invocations[0].arg_hash
        )
        assert result == actual_result

        assert self.backend.is_memoized(
            fn1_reference.fn_reference, fn1_reference.arg_hash
        )
        self.backend.forget_call(fn1_reference.fn_reference_with_arg_hash())
        assert not self.backend.is_memoized(
            fn1_reference.fn_reference, fn1_reference.arg_hash
        )

    @staticmethod
    def test_memoize_exception():
        assert fn_raise_value_error.memento() is None
        with pytest.raises(ValueError):
            fn_raise_value_error()
        with pytest.raises(ValueError):
            fn_raise_value_error()

    @staticmethod
    def test_memoize_null():
        assert fn_return_none_1.memento() is None
        first = fn_return_none_1()
        assert fn_return_none_1.memento() is not None
        second = fn_return_none_1()
        assert first == second

    @staticmethod
    def test_memoize_str():
        assert fn_return_str.memento() is None
        first = fn_return_str()
        assert fn_return_str.memento() is not None
        second = fn_return_str()
        assert first == second

    @staticmethod
    def test_memoize_int():
        assert fn_return_int.memento() is None
        first = fn_return_int()
        assert fn_return_int.memento() is not None
        second = fn_return_int()
        assert first == second

    @staticmethod
    def test_memoize_float():
        assert fn_return_float.memento() is None
        first = fn_return_float()
        assert fn_return_float.memento() is not None
        second = fn_return_float()
        assert first == second

    @staticmethod
    def test_memoize_bool():
        assert fn_return_bool.memento() is None
        first = fn_return_bool()
        assert fn_return_bool.memento() is not None
        second = fn_return_bool()
        assert first == second

    @staticmethod
    def test_memoize_date():
        assert fn_return_date.memento() is None
        first = fn_return_date()
        assert fn_return_date.memento() is not None
        second = fn_return_date()
        assert first == second

    @staticmethod
    def test_memoize_time():
        assert fn_return_time.memento() is None
        first = fn_return_time()
        assert fn_return_time.memento() is not None
        second = fn_return_time()
        assert first == second

    @staticmethod
    def test_memoize_dictionary():
        assert fn_return_dictionary.memento() is None
        first = fn_return_dictionary()
        assert fn_return_dictionary.memento() is not None
        second = fn_return_dictionary()
        for key in ["a", "b", "e"]:
            assert first[key] == second[key]
        assert first["g"].equals(second["g"])

    @staticmethod
    def test_memoize_list():
        assert fn_return_list.memento() is None
        first = fn_return_list()
        assert fn_return_list.memento() is not None
        second = fn_return_list()
        for x in range(0, 6):
            assert first[x] == second[x]
        assert first[6].equals(second[6])

    @staticmethod
    def test_memoize_2d_array():
        assert fn_return_2d_array.memento() is None
        first = fn_return_2d_array()
        assert fn_return_2d_array.memento() is not None
        second = fn_return_2d_array()
        assert first == second

    @staticmethod
    def test_numpy_array_bool():
        assert fn_return_numpy_array_bool.memento() is None
        first = fn_return_numpy_array_bool()
        assert fn_return_numpy_array_bool.memento() is not None
        second = fn_return_numpy_array_bool()
        assert np.array_equal(first, second)

    @staticmethod
    def test_numpy_array_int8():
        assert fn_return_numpy_array_int8.memento() is None
        first = fn_return_numpy_array_int8()
        assert fn_return_numpy_array_int8.memento() is not None
        second = fn_return_numpy_array_int8()
        assert np.array_equal(first, second)

    @staticmethod
    def test_numpy_array_int16():
        assert fn_return_numpy_array_int16.memento() is None
        first = fn_return_numpy_array_int16()
        assert fn_return_numpy_array_int16.memento() is not None
        second = fn_return_numpy_array_int16()
        assert np.array_equal(first, second)

    @staticmethod
    def test_numpy_array_int32():
        assert fn_return_numpy_array_int32.memento() is None
        first = fn_return_numpy_array_int32()
        assert fn_return_numpy_array_int32.memento() is not None
        second = fn_return_numpy_array_int32()
        assert np.array_equal(first, second)

    @staticmethod
    def test_numpy_array_int64():
        assert fn_return_numpy_array_int64.memento() is None
        first = fn_return_numpy_array_int64()
        assert fn_return_numpy_array_int64.memento() is not None
        second = fn_return_numpy_array_int64()
        assert np.array_equal(first, second)

    @staticmethod
    def test_numpy_array_float32():
        assert fn_return_numpy_array_float32.memento() is None
        first = fn_return_numpy_array_float32()
        assert fn_return_numpy_array_float32.memento() is not None
        second = fn_return_numpy_array_float32()
        assert np.array_equal(first, second)

    @staticmethod
    def test_numpy_array_float64():
        assert fn_return_numpy_array_float64.memento() is None
        first = fn_return_numpy_array_float64()
        assert fn_return_numpy_array_float64.memento() is not None
        second = fn_return_numpy_array_float64()
        assert np.array_equal(first, second)

    @staticmethod
    def test_memoize_index():
        assert fn_return_index.memento() is None
        first = fn_return_index()
        assert fn_return_index.memento() is not None
        second = fn_return_index()
        assert first.equals(second)

    @staticmethod
    def test_memoize_series():
        assert fn_return_series.memento() is None
        first = fn_return_series()
        assert fn_return_series.memento() is not None
        second = fn_return_series()
        assert_series_equal(first, second)

    @staticmethod
    def test_memoize_series_with_index():
        assert fn_return_series_with_index.memento() is None
        first = fn_return_series_with_index()
        assert fn_return_series_with_index.memento() is not None
        second = fn_return_series_with_index()
        assert_series_equal(first, second)

    @staticmethod
    def test_memoize_series_with_multiindex():
        assert fn_return_series_with_multiindex.memento() is None
        first = fn_return_series_with_multiindex()
        assert fn_return_series_with_multiindex.memento() is not None
        second = fn_return_series_with_multiindex()
        assert_series_equal(first, second)

    @staticmethod
    def test_memoize_dataframe():
        assert fn_return_dataframe.memento() is None
        first = fn_return_dataframe()
        assert fn_return_dataframe.memento() is not None
        second = fn_return_dataframe()
        assert first.equals(second)

    @staticmethod
    def test_memoize_dataframe_with_index():
        assert fn_return_dataframe_with_index.memento() is None
        first = fn_return_dataframe_with_index()
        assert fn_return_dataframe_with_index.memento() is not None
        second = fn_return_dataframe_with_index()
        assert first.equals(second)

    @staticmethod
    def test_memoize_partition():
        assert fn_return_partition.memento() is None
        first = fn_return_partition()
        assert fn_return_partition.memento() is not None
        second = fn_return_partition()
        assert second.get("a") is None
        assert second.get("b")
        assert 1 == second.get("c")
        assert 2.0 == second.get("d")
        assert [1, 2, 3] == second.get("e")
        assert {"a": "b"} == second.get("f")
        assert np.array_equal(
            np.array([1, 2, 3], dtype="int8"), cast(np.array, second.get("g"))
        )
        # noinspection PyUnresolvedReferences
        assert first.get("h").equals(second.get("h"))
        # noinspection PyUnresolvedReferences
        assert "k" == second.get("i").get("j")

    def test_memoize_partition_with_parent(self):
        memory_cache = (
            self.backend._memory_cache
            if hasattr(self.backend, "_memory_cache")
            else None
        )  # type: Optional[MemoryCache]

        # Ensure a is memoized to disk
        assert fn_return_partition_with_parent_a.memento() is None
        fn_return_partition_with_parent_a()
        assert fn_return_partition_with_parent_a.memento() is not None
        # Purge memory cache so that we get an PicklePartition if available:
        if memory_cache:
            memory_cache.forget_everything()
        a = fn_return_partition_with_parent_a()
        assert {"a", "b"} == set(a.list_keys())
        assert {"a", "b"} == set(a.list_keys(_include_merge_parent=False))
        assert 1 == a.get("a")
        assert 2 == a.get("b")

        # Now, b will return a partition whose parent is a.
        # We load it twice to ensure it comes from disk
        assert fn_return_partition_with_parent_b.memento() is None
        fn_return_partition_with_parent_b()
        assert fn_return_partition_with_parent_b.memento() is not None
        # Purge memory cache so that we get an PicklePartition if available:
        if memory_cache:
            memory_cache.forget_everything()
        b = fn_return_partition_with_parent_b()
        assert {"a", "b", "c"} == set(b.list_keys())
        assert {"b", "c"} == set(b.list_keys(_include_merge_parent=False))
        assert 1 == b.get("a")
        assert 3 == b.get("b")
        assert 4 == b.get("c")

        # Now, c will return a partition whose parent is b.
        # Again, we load it twice to ensure it comes from disk
        assert fn_return_partition_with_parent_c.memento() is None
        fn_return_partition_with_parent_c()
        assert fn_return_partition_with_parent_c.memento() is not None
        # Purge memory cache so that we get an PicklePartition if available:
        if memory_cache:
            memory_cache.forget_everything()
        c = fn_return_partition_with_parent_c()
        assert {"a", "b", "c", "d"} == set(c.list_keys())
        assert {"c", "d"} == set(c.list_keys(_include_merge_parent=False))
        assert 1 == c.get("a")
        assert 3 == c.get("b")
        assert 5 == c.get("c")
        assert 6 == c.get("d")

    def test_memoize_with_key_override(self):
        fn_ref_1 = fn1.fn_reference().with_args(1)
        self.backend.memoize("custom_key", self.get_dummy_memento(fn_ref_1), 1)
        assert "custom_key" == fn1.memento(1).content_key.key

    def test_make_url_for_result(self):
        fn_add_one(1)
        memento = fn_add_one.memento(1)
        url = self.backend.make_url_for_result(memento)
        assert url is not None
        # The URL should probably have the content key in it somewhere (though this is not technically a requirement)
        ck = memento.content_key.key[memento.content_key.key.rfind("/") + 1 :]
        assert url.find(ck) != -1, url

    def test_forget_call(self):
        result = 1
        fn_ref_1 = fn1.fn_reference().with_args(1)
        fn_ref_2 = fn1.fn_reference().with_args(2)
        memento_1 = self.get_dummy_memento(fn_ref_1)
        self.backend.memoize(None, memento_1, result)
        memento_2 = self.get_dummy_memento(fn_ref_2)
        self.backend.memoize(None, memento_2, result)

        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        assert self.backend.is_memoized(fn_ref_2.fn_reference, fn_ref_2.arg_hash)
        self.backend.forget_call(fn_ref_1.fn_reference_with_arg_hash())
        assert not self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        assert self.backend.is_memoized(fn_ref_2.fn_reference, fn_ref_2.arg_hash)

    def test_forget_function(self):
        fn_ref_1 = fn1.fn_reference().with_args(1)
        fn_ref_2 = fn1.fn_reference().with_args(2)
        self.backend.memoize(None, self.get_dummy_memento(fn_ref_1), 1)
        self.backend.memoize(None, self.get_dummy_memento(fn_ref_2), 2)
        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        self.backend.forget_function(fn1.fn_reference())
        assert not self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        assert not self.backend.is_memoized(fn_ref_2.fn_reference, fn_ref_2.arg_hash)

    def test_forget_everything(self):
        fn_ref_1 = fn1.fn_reference().with_args(1)
        fn_ref_2 = fn2.fn_reference().with_args(2)
        self.backend.memoize(None, self.get_dummy_memento(fn_ref_1), 1)
        self.backend.memoize(None, self.get_dummy_memento(fn_ref_2), 2)
        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        assert self.backend.is_memoized(fn_ref_2.fn_reference, fn_ref_2.arg_hash)
        self.backend.forget_everything()
        assert not self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        assert not self.backend.is_memoized(fn_ref_2.fn_reference, fn_ref_2.arg_hash)

    def test_readonly(self):
        repo0 = Environment.get().repos[0]
        cluster1 = repo0.clusters.get("cluster1")
        cluster1.storage.read_only = False

        fn_ref_1 = fn_add_one.fn_reference().with_args(1)
        memento = self.get_dummy_memento(fn_ref_1)
        result = fn_add_one(1)

        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        self.backend.forget_everything()
        assert not self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)

        # Make the cluster read-only and then ensure nothing is written
        cluster1.storage.read_only = True
        self.backend.memoize(None, memento, result)
        assert not self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        # including metadata
        with pytest.raises(ValueError):
            self.backend.write_metadata(
                memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
                "key1",
                "value1".encode("utf-8"),
            )

        # Make the cluster writable and then ensure memoization occurs
        cluster1.storage.read_only = False
        self.backend.memoize(None, memento, result)
        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)

        # Make the cluster read-only and ensure forgetting does not work
        cluster1.storage.read_only = True
        with pytest.raises(ValueError):
            self.backend.forget_everything()
        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        with pytest.raises(ValueError):
            self.backend.forget_function(fn_add_one.fn_reference())
        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)
        with pytest.raises(ValueError):
            self.backend.forget_call(fn_ref_1.fn_reference_with_arg_hash())
        assert self.backend.is_memoized(fn_ref_1.fn_reference, fn_ref_1.arg_hash)

    @staticmethod
    def test_metadata_rw():
        fn_add_one(1)
        memento = fn_add_one.memento(1)

        storage = m.Environment.get().get_cluster("cluster1").storage
        storage.write_metadata(
            memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            "value1".encode("utf-8"),
        )
        # noinspection PyUnresolvedReferences
        assert "value1" == storage.read_metadata(
            memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            retry_on_none=True,
        ).decode("utf-8")

        # test that deleting the memento deletes the metadata
        fn_add_one.forget(1)
        assert (
            storage.read_metadata(
                memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
                "key1",
                retry_on_none=True,
            )
            is None
        )

    @staticmethod
    def test_metadata_rw_store_with_data():
        fn_add_one(1)
        memento = fn_add_one.memento(1)

        storage = m.Environment.get().get_cluster("cluster1").storage
        storage.write_metadata(
            memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            "value1".encode("utf-8"),
        )
        # noinspection PyUnresolvedReferences
        assert "value1" == storage.read_metadata(
            memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            retry_on_none=True,
        ).decode("utf-8")

        # test that deleting the memento deletes the metadata
        fn_add_one.forget(1)
        assert (
            storage.read_metadata(
                memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
                "key1",
                retry_on_none=True,
            )
            is None
        )

    @staticmethod
    def test_content_addressable_storage():
        r1 = fn1(100)
        r2 = fn2(100)
        assert r1 == r2
        m1 = fn1.memento(100)
        m2 = fn2.memento(100)
        assert (m1.content_key is None and m2.content_key is None) or (
            m1.content_key.key == m2.content_key.key
        )


class DataSourceTester(ABC):
    """
    Abstract base class to test DataSource concrete classes.

    Subclasses should set `self.data_source` to the data source being tested.

    """

    data_source = None  # type: DataSource
    test = None  # type: TestCase
    data = None  # type: BytesIO
    data2 = None  # type: BytesIO
    data3 = None  # type: BytesIO
    data4 = None  # type: BytesIO

    def setup_method(self):
        self.data = BytesIO("abc".encode("utf-8"))
        self.data2 = BytesIO("def".encode("utf-8"))
        self.data3 = BytesIO("ghi".encode("utf-8"))
        self.data4 = BytesIO("jkl".encode("utf-8"))

    def teardown_method(self):
        pass

    @staticmethod
    def read_result(f: BytesIO) -> bytes:
        try:
            return f.read()
        finally:
            f.close()

    def test_output_input_nonversioned(self):
        data = self.data
        data2 = self.data2
        a = DataSourceKey("a")
        bc = DataSourceKey("b/c")

        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(a))

        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(bc))

        self.data_source.output(a, data)
        assert b"abc" == self.read_result(self.data_source.input_nonversioned(a))
        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(bc))

        self.data_source.output(bc, data2)
        assert b"abc" == self.read_result(self.data_source.input_nonversioned(a))
        assert b"def" == self.read_result(self.data_source.input_nonversioned(bc))

    def test_output_input_versioned(self):
        data = self.data
        data2 = self.data2
        data3 = self.data3
        a = DataSourceKey("a")
        bc = DataSourceKey("b/c")

        key = self.data_source.output(a, data)
        assert b"abc" == self.read_result(self.data_source.input_versioned(key))

        key2 = self.data_source.output(bc, data2)
        assert b"abc" == self.read_result(self.data_source.input_versioned(key))
        assert b"def" == self.read_result(self.data_source.input_versioned(key2))

        key3 = self.data_source.output(a, data3)
        assert b"abc" == self.read_result(self.data_source.input_versioned(key))
        assert b"ghi" == self.read_result(self.data_source.input_versioned(key3))

    def test_zero_byte_value(self):
        # Ensure a 0-byte value still results in creating the object
        data = b""
        a = DataSourceKey("a")

        key = self.data_source.output(a, io.BytesIO(data))
        assert data == self.read_result(self.data_source.input_nonversioned(a))
        assert data == self.read_result(self.data_source.input_versioned(key))

    def test_large_value(self):
        # ~ 5 MB of data:
        data = b"0123456789" * (1024 * 512)
        a = DataSourceKey("a")

        key = self.data_source.output(a, io.BytesIO(data))
        assert data == self.read_result(self.data_source.input_nonversioned(a))
        assert data == self.read_result(self.data_source.input_versioned(key))

    def test_store_with_object_metadata(self):
        data = self.data
        a = DataSourceKey("a")

        # Write an object, write metadata and check that it is equal
        key = self.data_source.output(a, data)
        with pytest.raises(IOError):
            self.data_source.input_metadata(key, "meta1")

        self.data_source.output_metadata(key, "meta1", b"value1")
        assert b"value1" == self.data_source.input_metadata(key, "meta1")

        # Check second metadata property:
        self.data_source.output_metadata(key, "meta2", b"value2")
        assert b"value1" == self.data_source.input_metadata(key, "meta1")
        assert b"value2" == self.data_source.input_metadata(key, "meta2")

        # Check metadata is deleted when object is deleted
        self.data_source.delete_all_versions(DataSourceKey(key.key), False)
        with pytest.raises(IOError):
            self.data_source.input_metadata(key, "meta1")

        with pytest.raises(IOError):
            self.data_source.input_metadata(key, "meta2")

    def test_delete_nonversioned_key(self):
        data = self.data
        ab = DataSourceKey("a/b")

        # Removing a nonversioned key that does not exist is a nop:
        assert not self.data_source.exists_nonversioned(ab)
        self.data_source.delete_nonversioned_key(ab)
        assert not self.data_source.exists_nonversioned(ab)

        key_ab = self.data_source.output(ab, data)
        assert self.data_source.exists_nonversioned(ab)
        assert self.data_source.exists_versioned(key_ab)

        self.data_source.delete_nonversioned_key(ab)
        assert not self.data_source.exists_nonversioned(ab)
        assert self.data_source.exists_versioned(key_ab)

    def test_delete(self):
        data = self.data
        data2 = self.data2
        ab = DataSourceKey("a/b")
        ac = DataSourceKey("a/c")

        with pytest.raises(IOError):
            self.data_source.input_nonversioned(ab)

        with pytest.raises(IOError):
            self.data_source.input_nonversioned(ac)

        key_ab = self.data_source.output(ab, data)
        key_ac = self.data_source.output(ac, data2)
        assert b"abc" == self.read_result(self.data_source.input_nonversioned(ab))
        assert b"abc" == self.read_result(self.data_source.input_versioned(key_ab))
        assert b"def" == self.read_result(self.data_source.input_nonversioned(ac))
        assert b"def" == self.read_result(self.data_source.input_versioned(key_ac))

        self.data_source.delete_all_versions(ab, recursive=False)
        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(ab))

        with pytest.raises(IOError):
            self.read_result(self.data_source.input_versioned(key_ab))

        assert b"def" == self.read_result(self.data_source.input_nonversioned(ac))
        assert b"def" == self.read_result(self.data_source.input_versioned(key_ac))

    def test_delete_recursive(self):
        data = self.data
        data2 = self.data2
        data3 = self.data3
        ab = DataSourceKey("a/b")
        ac = DataSourceKey("a/c")
        d = DataSourceKey("b")

        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(ab))
        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(ac))
        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(d))

        key_ab = self.data_source.output(ab, data)
        key_ac = self.data_source.output(ac, data2)
        key_d = self.data_source.output(d, data3)
        assert b"abc" == self.read_result(self.data_source.input_nonversioned(ab))
        assert b"abc" == self.read_result(self.data_source.input_versioned(key_ab))
        assert b"def" == self.read_result(self.data_source.input_nonversioned(ac))
        assert b"def" == self.read_result(self.data_source.input_versioned(key_ac))
        assert b"ghi" == self.read_result(self.data_source.input_nonversioned(d))
        assert b"ghi" == self.read_result(self.data_source.input_versioned(key_d))

        self.data_source.delete_all_versions(DataSourceKey("a"), recursive=True)
        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(ab))

        with pytest.raises(IOError):
            self.read_result(self.data_source.input_versioned(key_ab))

        with pytest.raises(IOError):
            self.read_result(self.data_source.input_nonversioned(ac))

        with pytest.raises(IOError):
            self.read_result(self.data_source.input_versioned(key_ac))

        assert b"ghi" == self.read_result(self.data_source.input_nonversioned(d))
        assert b"ghi" == self.read_result(self.data_source.input_versioned(key_d))

    def test_exists(self):
        data = self.data
        a = DataSourceKey("a")

        assert not self.data_source.exists_nonversioned(a)
        self.data_source.output(a, data)
        assert self.data_source.exists_nonversioned(a)

    def test_list_keys(self):
        data = self.data
        data2 = self.data2
        data3 = self.data3
        data4 = self.data4
        root = DataSourceKey("")
        a = DataSourceKey("a")
        a_b = DataSourceKey("a/b")
        a_c = DataSourceKey("a/c")
        a_d = DataSourceKey("a/d")
        a_d_ef = DataSourceKey("a/d/ef")
        f = DataSourceKey("f")

        # Some storage implementations need to set up an initial key with which to associate
        # the data source. This will be added to the expected results in the following tests.
        baseline_set = set(self.data_source.list_keys_nonversioned(root))
        baseline_set_recursive = set(
            self.data_source.list_keys_nonversioned(root, recursive=True)
        )

        self.data_source.output(a_b, data)
        self.data_source.output(a_c, data2)
        self.data_source.output(a_d_ef, data3)
        self.data_source.output(f, data4)

        # Test recursive=False and limit
        assert baseline_set.union({a, f}) == set(
            self.data_source.list_keys_nonversioned(
                directory=root, file_prefix="", recursive=False
            )
        )
        assert 2 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=root, file_prefix="", recursive=False, limit=2
                )
            )
        )

        assert {a_b, a_c, a_d} == set(
            self.data_source.list_keys_nonversioned(
                directory=a, file_prefix="", recursive=False
            )
        )
        assert 2 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a, file_prefix="", recursive=False, limit=2
                )
            )
        )

        assert {a_d_ef} == set(
            self.data_source.list_keys_nonversioned(
                directory=a_d, file_prefix="e", recursive=False
            )
        )

        assert 1 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a_d, file_prefix="e", recursive=False, limit=2
                )
            )
        )

        assert set() == set(
            self.data_source.list_keys_nonversioned(
                directory=a, file_prefix="g", recursive=False
            )
        )
        assert 0 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a, file_prefix="g", recursive=False, limit=2
                )
            )
        )

        assert {a_d_ef} == set(
            self.data_source.list_keys_nonversioned(
                directory=a_d, file_prefix="e", recursive=False
            )
        )

        assert 1 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a_d, file_prefix="e", recursive=False, limit=2
                )
            )
        )

        # Test recursive=True and limit
        assert baseline_set_recursive.union({a_b, a_c, a_d_ef, f}) == set(
            self.data_source.list_keys_nonversioned(
                directory=root, file_prefix="", recursive=True
            )
        )

        assert min(2, len(baseline_set) + 2) == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=root, file_prefix="", recursive=True, limit=2
                )
            )
        )
        assert {a_b, a_c, a_d_ef} == set(
            self.data_source.list_keys_nonversioned(
                directory=a, file_prefix="", recursive=True
            )
        )

        assert 2 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a, file_prefix="", recursive=True, limit=2
                )
            )
        )
        assert {a_d_ef} == set(
            self.data_source.list_keys_nonversioned(
                directory=a_d, file_prefix="e", recursive=True
            )
        )
        assert 1 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a_d, file_prefix="e", recursive=True, limit=2
                )
            )
        )
        assert set() == set(
            self.data_source.list_keys_nonversioned(
                directory=a, file_prefix="g", recursive=True
            )
        )
        assert 0 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a, file_prefix="g", recursive=True, limit=2
                )
            )
        )
        assert {a_d_ef} == set(
            self.data_source.list_keys_nonversioned(
                directory=a_d, file_prefix="e", recursive=True
            )
        )
        assert 1 == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=a_d, file_prefix="e", recursive=True, limit=2
                )
            )
        )

        # Remove some files and test
        self.data_source.delete_all_versions(a_b, recursive=False)
        self.data_source.delete_all_versions(a_c, recursive=False)
        self.data_source.delete_all_versions(a_d_ef, recursive=False)
        assert baseline_set.union({f}) == set(
            self.data_source.list_keys_nonversioned(
                directory=root, file_prefix="", recursive=False
            )
        )
        assert baseline_set_recursive.union({f}) == set(
            self.data_source.list_keys_nonversioned(
                directory=root, file_prefix="", recursive=True
            )
        )
        assert min(2, len(baseline_set) + 1) == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=root, file_prefix="", recursive=False, limit=2
                )
            )
        )
        assert min(2, len(baseline_set) + 1) == len(
            list(
                self.data_source.list_keys_nonversioned(
                    directory=root, file_prefix="", recursive=True, limit=2
                )
            )
        )

    def test_get_versioned_key(self):
        data = self.data
        data2 = self.data2
        a = DataSourceKey("a")
        key_a = self.data_source.output(a, data)
        assert key_a == self.data_source.get_versioned_key(a)
        key_a2 = self.data_source.output(a, data2)
        assert key_a != key_a2
        assert key_a2 == self.data_source.get_versioned_key(a)


class MetadataSourceTester(ABC):
    """
    Abstract base class to test MetadataDataSource concrete classes.

    Subclasses should set `self.metadata_source` to the metadata source being tested.

    """

    metadata_source = None  # type: MetadataSource
    test = None  # type: TestCase
    test_memento_1 = None  # type: Memento
    test_memento_1_0 = None  # type: Memento
    test_memento_2 = None  # type: Memento
    test_memento_date = None  # type: Memento
    fn_ref_1 = None  # type: FunctionReferenceWithArguments
    fn_ref_1_0 = None  # type: FunctionReferenceWithArguments
    fn_ref_2 = None  # type: FunctionReferenceWithArguments
    fn_ref_date = None  # type: FunctionReferenceWithArguments

    def setup_method(self):
        self.fn_ref_1 = fn1.fn_reference().with_args(1)
        self.fn_ref_1_0 = fn1.fn_reference().with_args(0)
        self.test_memento_1 = StorageBackendTester.get_dummy_memento(self.fn_ref_1)
        self.test_memento_1_0 = StorageBackendTester.get_dummy_memento(self.fn_ref_1_0)
        self.fn_ref_2 = fn2.fn_reference().with_args(2)
        self.test_memento_2 = StorageBackendTester.get_dummy_memento(self.fn_ref_2)
        self.fn_ref_date = fn1.fn_reference().with_args(
            datetime.datetime(2019, 1, 1).date()
        )
        self.test_memento_date = StorageBackendTester.get_dummy_memento(
            self.fn_ref_date
        )

    def teardown_method(self):
        pass

    def test_put_and_get_memento(self):
        self.metadata_source.put_memento(self.test_memento_1)
        result = self.metadata_source.get_mementos(
            [self.fn_ref_1.fn_reference_with_arg_hash()]
        )[0]
        assert (
            self.test_memento_1.invocation_metadata.fn_reference_with_args.args[0]
            == result.invocation_metadata.fn_reference_with_args.args[0]
        )

    def test_put_and_get_memento_with_date(self):
        self.metadata_source.put_memento(self.test_memento_date)
        result = self.metadata_source.get_mementos(
            [self.fn_ref_date.fn_reference_with_arg_hash()]
        )[0]
        assert (
            self.test_memento_date.invocation_metadata.fn_reference_with_args.args[0]
            == result.invocation_metadata.fn_reference_with_args.args[0]
        )

    def test_get_mementos(self):
        all_three = [
            self.fn_ref_1.fn_reference_with_arg_hash(),
            self.fn_ref_1_0.fn_reference_with_arg_hash(),
            self.fn_ref_2.fn_reference_with_arg_hash(),
        ]
        assert [None, None, None] == self.metadata_source.get_mementos(all_three)

        self.metadata_source.put_memento(self.test_memento_1)
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0] if mem else None
            for mem in self.metadata_source.get_mementos(all_three)
        ]
        assert [1, None, None] == listed_args

        self.metadata_source.put_memento(self.test_memento_1_0)
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0] if mem else None
            for mem in self.metadata_source.get_mementos(all_three)
        ]
        assert [1, 0, None] == listed_args

        self.metadata_source.put_memento(self.test_memento_2)
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0] if mem else None
            for mem in self.metadata_source.get_mementos(all_three)
        ]
        assert [1, 0, 2] == listed_args

    def test_all_mementos_exist(self):
        full_list = [
            self.fn_ref_1.fn_reference_with_arg_hash(),
            self.fn_ref_2.fn_reference_with_arg_hash(),
        ]
        assert not self.metadata_source.all_mementos_exist(full_list)
        self.metadata_source.put_memento(self.test_memento_1)
        assert not self.metadata_source.all_mementos_exist(full_list)
        self.metadata_source.put_memento(self.test_memento_2)
        assert self.metadata_source.all_mementos_exist(full_list)

    def test_list_functions(self):
        assert [] == self.metadata_source.list_functions()

        self.metadata_source.put_memento(self.test_memento_1)

        fn_list = retry_until(
            lambda: self.metadata_source.list_functions(), lambda x: len(x) > 0
        )
        assert [self.fn_ref_1.fn_reference.qualified_name] == [
            f.qualified_name for f in fn_list
        ]

        self.metadata_source.put_memento(self.test_memento_2)
        fn_list = retry_until(
            lambda: self.metadata_source.list_functions(), lambda x: len(x) > 0
        )
        assert [
            self.fn_ref_1.fn_reference.qualified_name,
            self.fn_ref_2.fn_reference.qualified_name,
        ] == [f.qualified_name for f in fn_list]

    def test_list_mementos(self):
        assert [] == self.metadata_source.list_mementos(
            self.fn_ref_1.fn_reference, None
        )
        self.metadata_source.put_memento(self.test_memento_1)
        retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            ),
            lambda x: len(x) == 1,
        )
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            )
        ]
        assert {1} == set(listed_args)
        self.metadata_source.put_memento(self.test_memento_1_0)
        retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            ),
            lambda x: len(x) == 2,
        )
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            )
        ]
        assert {0, 1} == set(listed_args)
        # Test limit
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(self.fn_ref_1.fn_reference, 1)
        ]
        assert 1 == len(listed_args)

    def test_write_and_read_metadata(self):
        self.metadata_source.put_memento(self.test_memento_1)
        self.metadata_source.write_metadata(
            self.test_memento_1.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            "value1".encode("utf-8"),
            stored_with_data=False,
        )
        assert "value1" == self.metadata_source.read_metadata(
            self.test_memento_1.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            retry_on_none=True,
        ).decode("utf-8")

    def test_write_and_read_large_metadata(self):
        # ~ 5 MB of data:
        data = "0123456789" * (1024 * 512)
        self.metadata_source.put_memento(self.test_memento_1)
        self.metadata_source.write_metadata(
            self.test_memento_1.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            data.encode("utf-8"),
            stored_with_data=False,
        )
        assert data == self.metadata_source.read_metadata(
            self.test_memento_1.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            retry_on_none=True,
        ).decode("utf-8")

    def test_write_and_read_metadata_store_with_data(self):
        self.metadata_source.put_memento(self.test_memento_1)
        self.metadata_source.write_metadata(
            self.test_memento_1.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            "key1",
            "value1".encode("utf-8"),
            stored_with_data=True,
        )
        assert isinstance(
            self.metadata_source.read_metadata(
                self.test_memento_1.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
                "key1",
                retry_on_none=True,
            ),
            ResultIsWithData,
        )

    def test_forget_call(self):
        self.metadata_source.put_memento(self.test_memento_1)
        self.metadata_source.put_memento(self.test_memento_1_0)
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            )
        ]
        assert {0, 1} == set(listed_args)
        self.metadata_source.forget_call(self.fn_ref_1_0.fn_reference_with_arg_hash())
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            )
        ]
        assert {1} == set(listed_args)
        self.metadata_source.forget_call(self.fn_ref_1.fn_reference_with_arg_hash())
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            )
        ]
        assert set() == set(listed_args)

    def test_forget_everything(self):
        self.metadata_source.put_memento(self.test_memento_1)
        self.metadata_source.put_memento(self.test_memento_1_0)
        self.metadata_source.put_memento(self.test_memento_2)
        retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            ),
            lambda x: set(
                mem.invocation_metadata.fn_reference_with_args.args[0] for mem in x
            )
            == {0, 1},
        )
        retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_2.fn_reference, None
            ),
            lambda x: set(
                mem.invocation_metadata.fn_reference_with_args.args[0] for mem in x
            )
            == {2},
        )

        self.metadata_source.forget_everything()
        retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            ),
            lambda x: set(
                mem.invocation_metadata.fn_reference_with_args.args[0] for mem in x
            )
            == set(),
        )
        retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_2.fn_reference, None
            ),
            lambda x: set(
                mem.invocation_metadata.fn_reference_with_args.args[0] for mem in x
            )
            == set(),
        )

    def test_forget_function(self):
        self.metadata_source.put_memento(self.test_memento_1)
        self.metadata_source.put_memento(self.test_memento_1_0)
        self.metadata_source.put_memento(self.test_memento_2)
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            )
        ]
        assert {0, 1} == set(listed_args)
        list_result = retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_2.fn_reference, None
            ),
            lambda x: len(x) == 1,
        )
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in list_result
        ]
        assert {2} == set(listed_args)

        self.metadata_source.forget_function(self.fn_ref_1.fn_reference)
        listed_result = retry_until(
            lambda: self.metadata_source.list_mementos(
                self.fn_ref_1.fn_reference, None
            ),
            lambda x: len(x) == 0,
        )
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in listed_result
        ]
        assert set() == set(listed_args)
        listed_args = [
            mem.invocation_metadata.fn_reference_with_args.args[0]
            for mem in self.metadata_source.list_mementos(
                self.fn_ref_2.fn_reference, None
            )
        ]
        assert {2} == set(listed_args)
