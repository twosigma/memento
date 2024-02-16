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

import pandas as pd
import numpy as np

import pytest

# There is no public equivalent, so we import the protected function.
# If it is ever removed, we can replace it in the unit test.
# noinspection PyProtectedMember
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
from numpy.testing import assert_array_equal

from twosigma.memento import Memento, InvocationMetadata, memento_function
from twosigma.memento.metadata import ResultType
from twosigma.memento.reference import FunctionReferenceWithArguments
from twosigma.memento.storage_base import MemoryCache, DefaultCodec
from twosigma.memento.types import VersionedDataSourceKey


@memento_function
def fn_test1():
    return 1


@memento_function
def fn_test_long():
    return bytes([1]) * (2**31 - 12)


@memento_function
def fn_test_long_str():
    return "1" * (2**31 - 12)


class TestMemoryCache:

    @staticmethod
    def get_dummy_memento() -> Memento:
        now = datetime.datetime.now(datetime.timezone.utc)
        return Memento(
            time=now,
            invocation_metadata=InvocationMetadata(
                runtime=datetime.timedelta(seconds=123.0),
                fn_reference_with_args=FunctionReferenceWithArguments(
                    fn_test1.fn_reference(), (), {}
                ),
                result_type=ResultType.number,
                invocations=[],
                resources=[],
            ),
            function_dependencies={fn_test1.fn_reference()},
            runner={},
            correlation_id="abc123",
            content_key=VersionedDataSourceKey("key", "def456"),
        )

    @pytest.mark.skip(reason="test needs further investigation")
    def test_pd_linreg_mem_usage(self):
        empty_s = pd.Series(dtype=int)
        assert 0 == MemoryCache._pd_linreg_mem_usage(empty_s, sample_size=10)

        small_s = pd.Series([1, 2, 3, 4, 5])
        # pandas 0.24: 120      pandas 0.25: 168
        assert 120 <= MemoryCache._pd_linreg_mem_usage(small_s, sample_size=10) <= 168

        big_s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        assert 176 == MemoryCache._pd_linreg_mem_usage(big_s, sample_size=10)

        empty_df = pd.DataFrame()
        assert 0 == MemoryCache._pd_linreg_mem_usage(empty_df, sample_size=10)

        small_df = pd.DataFrame({"Small Series": small_s})
        # pandas 0.24: 120      pandas 0.25: 168
        assert 120 <= MemoryCache._pd_linreg_mem_usage(small_df, sample_size=10) <= 168

        big_df = pd.DataFrame({"Big Series": big_s})
        assert 176 == MemoryCache._pd_linreg_mem_usage(big_df, sample_size=10)

    @pytest.mark.needs_canonical_version
    def test_cache_key_for_memento(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        arg_hash = memento.invocation_metadata.fn_reference_with_args.arg_hash
        # Note: This also tests the stability of the function code hash
        assert "tests.test_storage_base:fn_test1#de769e9c8c9b500e/{}".format(
            arg_hash
        ) == cache._cache_key_for_memento(memento)

    @pytest.mark.needs_canonical_version
    def test_cache_key_for_fn(self):
        cache = MemoryCache(1)
        arg_hash = FunctionReferenceWithArguments(
            fn_test1.fn_reference(), (), {}
        ).arg_hash
        # Note: This also tests the stability of the function code hash
        assert "tests.test_storage_base:fn_test1#de769e9c8c9b500e/{}".format(
            arg_hash
        ) == cache._cache_key_for_fn(fn_test1.fn_reference(), arg_hash)

    def test_put_read_result(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        with pytest.raises(KeyError):
            cache.read_result(memento)
        cache.put(memento, 1, has_result=True)
        assert 1 == cache.read_result(memento)

    def test_get_mementos(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        fn_ref = FunctionReferenceWithArguments(fn_test1.fn_reference(), (), {})

        with pytest.raises(KeyError):
            cache.read_result(memento)

        assert [None] == cache.get_mementos([fn_ref.fn_reference_with_arg_hash()])
        cache.put(memento, 1, has_result=False)

        with pytest.raises(KeyError):
            cache.read_result(memento)

        mementos = cache.get_mementos([fn_ref.fn_reference_with_arg_hash()])
        assert 1 == len(mementos)
        assert ResultType.number == mementos[0].invocation_metadata.result_type
        cache.forget_everything()

        with pytest.raises(KeyError):
            cache.read_result(memento)

        assert [None] == cache.get_mementos([fn_ref.fn_reference_with_arg_hash()])

    def test_is_memoized(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        arg_hash = FunctionReferenceWithArguments(
            fn_test1.fn_reference(), (), {}
        ).arg_hash
        assert not cache.is_memoized(fn_test1.fn_reference(), arg_hash)
        cache.put(memento, 1, has_result=True)
        assert cache.is_memoized(fn_test1.fn_reference(), arg_hash)

    def test_is_all_memoized(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        fn_reference_with_args = FunctionReferenceWithArguments(
            fn_test1.fn_reference(), (), {}
        )
        assert not cache.is_all_memoized([fn_reference_with_args])
        cache.put(memento, 1, has_result=True)
        assert cache.is_all_memoized([fn_reference_with_args])

    def test_forget_call(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        cache.put(memento, 1, has_result=True)
        fn_reference_with_args = FunctionReferenceWithArguments(
            fn_test1.fn_reference(), (), {}
        )
        assert cache.is_memoized(
            fn_test1.fn_reference(), fn_reference_with_args.arg_hash
        )
        cache.forget_call(fn_reference_with_args.fn_reference_with_arg_hash())
        assert not cache.is_memoized(
            fn_test1.fn_reference(), fn_reference_with_args.arg_hash
        )

    def test_forget_everything(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        cache.put(memento, 1, has_result=True)
        fn_reference_with_args = FunctionReferenceWithArguments(
            fn_test1.fn_reference(), (), {}
        )
        assert cache.is_memoized(
            fn_test1.fn_reference(), fn_reference_with_args.arg_hash
        )
        cache.forget_everything()
        assert not cache.is_memoized(
            fn_test1.fn_reference(), fn_reference_with_args.arg_hash
        )

    def test_forget_function(self):
        cache = MemoryCache(1)
        memento = self.get_dummy_memento()
        cache.put(memento, 1, has_result=True)
        fn_reference_with_args = FunctionReferenceWithArguments(
            fn_test1.fn_reference(), (), {}
        )
        assert cache.is_memoized(
            fn_test1.fn_reference(), fn_reference_with_args.arg_hash
        )
        cache.forget_function(fn_test1.fn_reference())
        assert not cache.is_memoized(
            fn_test1.fn_reference(), fn_reference_with_args.arg_hash
        )


class TestSerializationStrategy:

    @pytest.mark.slow
    def test_super_large_dataframe(self):
        # This test takes a long time to run because it constructs a > 2 GB
        # DataFrame to ensure it can be encoded.

        arr = np.random.randint(0, 1000000, size=250) * 10
        df = pd.DataFrame({"test": arr})
        total_mem = df.memory_usage(deep=True).sum()
        assert total_mem >= 1024 * 1024 * 1024 * 2, "Memory used: {}".format(
            total_mem
        )  # 2 GB

        # IPC strategy should work
        strategy = DefaultCodec.ValuePickleStrategy()
        strategy.encode(df)

    def assert_equality(self, first, second, message):
        """Deep equality assert that handles Series/Index/DataFrame/ndarray"""

        assert type(first) == type(second), message
        if isinstance(first, pd.DataFrame):
            assert_frame_equal(first, second, message)
        elif isinstance(first, pd.Series):
            assert_series_equal(first, second, message)
        elif isinstance(first, pd.Index):
            assert_index_equal(first, second, message)
        elif isinstance(first, np.ndarray):
            assert_array_equal(first, second, message)
        elif isinstance(first, list):
            assert len(first) == len(
                second
            ), f"Different lengths: {len(first)} vs {len(second)}: {message}"
            for i in range(0, len(first)):
                self.assert_equality(
                    first[i],
                    second[i],
                    f"Difference at index {i}: {first[i]} vs {second[i]}: {message}",
                )
        elif isinstance(first, dict):
            assert (
                first.keys() == second.keys()
            ), f"Different keys: {first.keys()} vs {second.keys()}: {message}"
            for key in first:
                self.assert_equality(
                    first[key],
                    second[key],
                    f"Difference at key {key}: {first[key]} vs {second[key]}: {message}",
                )
        else:
            assert first == second, message

    def test_encode_values(self):
        test_values = [
            False,
            True,
            "",
            "abc",
            b"\x80\x02\x03",
            "42",
            "-9.2718",
            datetime.date.today(),
            datetime.datetime.now(),
            datetime.timedelta(days=3, minutes=7, milliseconds=200),
            {},
            {"a": [4, "foo", None]},
            [],
            [5, "wat", True, [{}, []]],
            pd.DataFrame({"a": np.random.rand(5)}),
            np.random.rand(5),
            pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]), name="foo"),
            pd.Index([1, 2, 3], name="foo"),
        ]
        strategy = DefaultCodec.ValuePickleStrategy()
        for val in test_values:
            encoded = strategy.encode(val)
            decoded = strategy.decode(encoded)
            self.assert_equality(val, decoded, f"Failed on: {val}, decoded={decoded}")

    def test_encode_list(self):
        test_values = [
            [],
            [7],
            [[42]],
            ["hello", [42], [[[datetime.datetime.now()], -8.6], [["abc"], "xyz"]]],
            [True, [], None, {}, "z", 42, 3.14159, b"\x80\x02\x03"],
            [pd.DataFrame([1, 2]), [pd.DataFrame([3, 4])]],
        ]
        strategy = DefaultCodec.ValuePickleStrategy()
        for val in test_values:
            assert isinstance(val, list)
            encoded = strategy.encode(val)
            decoded = strategy.decode(encoded)
            assert isinstance(decoded, list)
            self.assert_equality(val, decoded, f"Failed on: {val}, decoded={decoded}")

    def test_encode_dict(self):
        test_values = [
            {},
            {"a": None},
            {"a": "foo"},
            {"a": "foo", "b": "bar", "c": 42, "d": False},
            {"a": -6.1, "b": None, "c": [], "d": {"e": {}, "f": ["nested", "things"]}},
            {"a": pd.DataFrame([1, 2]), "b": {"c": [pd.DataFrame([3, 4])]}},
        ]
        strategy = DefaultCodec.ValuePickleStrategy()
        for val in test_values:
            assert isinstance(val, dict)
            encoded = strategy.encode(val)
            decoded = strategy.decode(encoded)
            assert isinstance(decoded, dict)
            self.assert_equality(val, decoded, f"Failed on: {val}, decoded={decoded}")

    def test_encode_series(self):
        test_values = [
            pd.Series([], dtype=int),
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3], name="foo"),
            pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"])),
            pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]), name="foo"),
        ]
        strategy = DefaultCodec.ValuePickleStrategy()
        for val in test_values:
            assert isinstance(val, pd.Series)
            encoded = strategy.encode(val)
            decoded = strategy.decode(encoded)
            assert isinstance(decoded, pd.Series)
            self.assert_equality(val, decoded, f"Failed on: {val}, decoded={decoded}")

    def test_encode_index(self):
        test_values = [
            pd.Index([]),
            pd.Index([1, 2, 3]),
            pd.Index([1, 2, 3], name="foo"),
            pd.date_range("2020-01-01", periods=10, freq="D"),
            pd.MultiIndex.from_arrays(
                [[7, 8, 9], ["red", "green", "blue"]], names=("n", "c")
            ),
        ]
        strategy = DefaultCodec.ValuePickleStrategy()
        for val in test_values:
            assert isinstance(val, pd.Index)
            encoded = strategy.encode(val)
            decoded = strategy.decode(encoded)
            assert isinstance(decoded, pd.Index)
            self.assert_equality(val, decoded, f"Failed on: {val}, decoded={decoded}")

    def test_encode_numpy_array(self):
        test_values = [
            np.array([]),
            np.array([0]),
            np.array([1, 2, 3, 4, 5]),
            np.random.rand(10),
        ]
        strategy = DefaultCodec.ValuePickleStrategy()
        for val in test_values:
            assert isinstance(val, np.ndarray)
            encoded = strategy.encode(val)
            decoded = strategy.decode(encoded)
            assert isinstance(decoded, np.ndarray)
            self.assert_equality(val, decoded, f"Failed on: {val}, decoded={decoded}")

    @pytest.mark.slow
    def test_serialize_longbytes(self):
        strategy = DefaultCodec.ValuePickleStrategy()
        val = b"12345678" * (2**28)
        encoded = strategy.encode(val)
        decoded = strategy.decode(encoded)

        assert val == decoded, f"Failed test on: {val}, decoded={decoded}"
        self.assert_equality(val, decoded, f"Failed test on: {val}, decoded={decoded}")

    @pytest.mark.slow
    def test_serialize_longstring(self):
        strategy = DefaultCodec.ValuePickleStrategy()
        val = "\u0009\u000A\u0026\u2022\u25E6\u2219\u2023\u2043" * (2**27)
        encoded = strategy.encode(val)
        decoded = strategy.decode(encoded)
        assert val == decoded, f"Failed test on: {val}, decoded={decoded}"

    @pytest.mark.slow
    def test_long_bytes(self):
        # Check that a memento function that returns a huge byte stream
        # does not throw an error (result can be serialized)
        fn_test_long()

    @pytest.mark.slow
    def test_long_string(self):
        # Check that a memento function that returns a huge string > 2Gb
        # does not throw an error (result can be serialized)
        fn_test_long_str()
