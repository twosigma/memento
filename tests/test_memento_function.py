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
import functools
import logging
import shutil
import sys
import tempfile
import unittest
from time import sleep

import pandas as pd
import pytest
from pandas import DataFrame

import twosigma.memento as m
from pandas.testing import assert_frame_equal
from twosigma.memento import (
    Environment,
    ConfigurationRepository,
    FunctionCluster,
    Memento,
    FunctionReference,
    MementoFunction,
)
from twosigma.memento.metadata import ResultType
from twosigma.memento.partition import InMemoryPartition
from twosigma.memento.code_hash import fn_code_hash
from twosigma.memento.storage_filesystem import FilesystemStorageBackend

_called = False
_today = datetime.date.today()
_now = datetime.datetime.now(datetime.timezone.utc)


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def sample_a(x):
    global _called
    _called = True
    return x


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def sample_b(x):
    global _called
    _called = True
    return x


@m.memento_function
def fn_test_memoize_0a():
    pass


@m.memento_function(cluster="a")
def fn_test_memoize_0b():
    pass


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_1():
    global _called
    _called = True
    return 7


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_2(arg1):
    global _called
    _called = True
    return arg1 * 2


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_3(arg1=10):
    global _called
    _called = True
    return arg1 * 3


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_4(arg1=10, arg2=20):
    global _called
    _called = True
    return arg1 * 2 + arg2


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_5():
    global _called
    _called = True
    return 42


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_6(a):
    global _called
    _called = True
    return a * 2


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_7(a):
    global _called
    _called = _called + 1
    return a * 7


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_df():
    global _called
    _called = True
    return pd.DataFrame([{"name": "a", "value": 1}, {"name": "b", "value": 2}])


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_null():
    global _called
    _called = True
    return None


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_boolean():
    global _called
    _called = True
    return True


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_str():
    global _called
    _called = True
    return "foo"


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_bin():
    global _called
    _called = True
    return bytes([0x01, 0x02, 0x03])


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_int():
    global _called
    _called = True
    return 7


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_float():
    global _called
    _called = True
    return 4.2


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_date():
    global _called, _today
    _called = True
    return _today


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_datetime():
    global _called, _now
    _called = True
    return _now


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_dict():
    global _called
    _called = True
    return {"a": "b"}


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_list():
    global _called
    _called = True
    return [1, 2, 3]


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_series():
    global _called
    _called = True
    return pd.Series([1, 2, 3])


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_data_frame():
    global _called
    _called = True
    return pd.DataFrame([{"a": 1}, {"a": 2}])


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_memoize_partition():
    global _called
    _called = True
    return InMemoryPartition({"a": True, "b": 2})


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False, cluster="bad_cluster")
def fn_test_memoize_string():
    global _called
    _called = True
    return "memoize me"


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_test_raises_value_error():
    global _called
    _called = True
    raise ValueError("test message")


@m.memento_function
def add(x, y):
    return x + y


@m.memento_function
def apply_and_double(fn, x):
    return 2 * fn(x)


# noinspection PyUnusedLocal
@m.memento_function
def fn_test_fn_with_arg_noreturn(a):
    pass


@m.memento_function
def fn_double(a):
    return a * 2


@m.memento_function
def fn_triple_double(a):
    return fn_double(a) * 3


@m.memento_function(version="1")
def fn_current_time():
    return datetime.datetime.now(datetime.timezone.utc)


@m.memento_function
def fn_calls_current_time():
    return fn_current_time()


@m.memento_function
def fn_calls_calls_current_time():
    return fn_calls_current_time()


@m.memento_function()
def fn_raise_before(time: datetime.datetime):
    if time.now() < time:
        raise RuntimeError("Time before provided time")
    return True


@m.memento_function()
def fn_calls_raise_before(time: datetime.datetime):
    return fn_raise_before(time)


@m.memento_function()
def fn_calls_sample_a(x):
    return sample_a(x)


# Turn off auto-dependencies, else global variable _called will introduce version change
@m.memento_function(auto_dependencies=False)
def fn_access_resource(file):
    global _called
    _called = True
    m.file_resource(file)


@m.memento_function()
def fn_calls_access_resource(file):
    fn_access_resource(file)


# This function is not yet bound to a memento function.
def fn_unbound_1(x):
    return x + 1


# This function is not yet bound to a memento function.
def fn_unbound_2(x):
    return x + 2


def my_decorator(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)

    return decorated


@m.memento_function()
@my_decorator
def fn_decorated(x):
    return fn_double(x)


fn_decorated_v1 = fn_decorated.version()


@m.memento_function()
@my_decorator
def fn_decorated(x):
    return fn_double(x) + 1


@m.memento_function
def fn_raise_on_odd(x):
    if x % 2 == 1:
        raise ValueError("x is odd")
    return x


@m.memento_function
def fn_calls_raise_on_odd(x):
    return fn_raise_on_odd(x)


@m.memento_function
def fn_okay():
    pass


@m.memento_function
def fn_raises_exception():
    raise ValueError("test exception")


@m.memento_function
def fn_okay_and_exception():
    fn_okay()
    fn_raises_exception()


@m.memento_function
def fn_c():
    pass


@m.memento_function
def fn_b():
    fn_c()


@m.memento_function
def fn_a():
    fn_b()


@m.memento_function
def fn_fib(x):
    return fn_fib(x - 1) + fn_fib(x - 2) if x > 2 else 1


fn_decorated_v2 = fn_decorated.version()


class TestMemoize:
    """Class to test memoize and forget."""

    def setup_method(self):
        global _called
        self.env_before = m.Environment.get()
        self.env_dir = tempfile.mkdtemp(prefix="memoizeTest")
        env_file = "{}/env.json".format(self.env_dir)
        with open(env_file, "w") as f:
            print("""{"name": "test"}""", file=f)
        m.Environment.set(env_file)
        _called = False

    def teardown_method(self):
        shutil.rmtree(self.env_dir)
        m.Environment.set(self.env_before)

    def test_fn_reference(self):
        ref0a = m.FunctionReference(fn_test_memoize_0a)
        qual_name_0a = "tests.test_memento_function:fn_test_memoize_0a"
        assert qual_name_0a == ref0a.qualified_name[0 : ref0a.qualified_name.find("#")]
        assert ref0a.cluster_name is None

        ref0a = m.FunctionReference.from_qualified_name(qual_name_0a)
        assert qual_name_0a == ref0a.qualified_name[0 : ref0a.qualified_name.find("#")]
        assert ref0a.cluster_name is None

        ref0b = m.FunctionReference(fn_test_memoize_0b)
        qual_name_0b = "tests.test_memento_function:fn_test_memoize_0b"
        assert (
            "a::" + qual_name_0b
            == ref0b.qualified_name[0 : ref0b.qualified_name.find("#")]
        )
        assert "a" == ref0b.cluster_name

        ref0b = m.FunctionReference.from_qualified_name("a::" + qual_name_0b)
        assert ref0b.memento_fn.version() is not None
        assert (
            "a::" + qual_name_0b
            == ref0b.qualified_name[0 : ref0b.qualified_name.find("#")]
        )
        assert "a" == ref0b.cluster_name

    def test_memoize(self):
        global _called
        _called = False

        fn_test_memoize_1.forget_all()
        assert not _called

        assert 7 == fn_test_memoize_1()
        assert _called

        _called = False
        assert 7 == fn_test_memoize_1()
        assert not _called

    def test_memoize_with_arg(self):
        global _called
        _called = False

        fn_test_memoize_2.forget_all()
        assert not _called

        assert 14 == fn_test_memoize_2(7)
        assert _called

        _called = False
        assert 14 == fn_test_memoize_2(7)
        assert not _called

        assert 16 == fn_test_memoize_2(8)
        assert _called

        _called = False
        assert 16 == fn_test_memoize_2(8)
        assert not _called

    def test_memoize_with_kwarg(self):
        global _called
        _called = False

        fn_test_memoize_3.forget_all()

        assert not _called

        assert 30 == fn_test_memoize_3()
        assert _called

        _called = False
        assert 30 == fn_test_memoize_3()
        assert not _called

        assert 300 == fn_test_memoize_3(arg1=100)
        assert _called

        _called = False
        assert 300 == fn_test_memoize_3(arg1=100)
        assert not _called

        assert 600 == fn_test_memoize_3(arg1=200)
        assert _called

        _called = False
        assert 600 == fn_test_memoize_3(arg1=200)
        assert not _called

    def test_memoize_kwarg_order_doesnt_matter(self):
        global _called
        _called = False

        fn_test_memoize_4.forget_all()

        assert not _called

        assert 4 == fn_test_memoize_4(arg1=1, arg2=2)
        assert _called

        _called = False
        assert 4 == fn_test_memoize_4(arg1=1, arg2=2)
        assert not _called

        _called = False
        assert 4 == fn_test_memoize_4(arg2=2, arg1=1)
        assert not _called

    def test_forget(self):
        global _called
        _called = False

        fn_test_memoize_5.forget_all()

        assert not _called

        assert 42 == fn_test_memoize_5()
        assert _called

        _called = False
        assert 42 == fn_test_memoize_5()
        assert not _called

        fn_test_memoize_5.forget_all()

        assert 42 == fn_test_memoize_5()
        assert _called

    def test_forget_with_arg(self):
        global _called
        _called = False

        fn_test_memoize_6.forget_all()

        assert not _called

        assert 10 == fn_test_memoize_6(5)
        assert _called
        assert 12 == fn_test_memoize_6(6)
        assert _called

        _called = False
        assert 10 == fn_test_memoize_6(5)
        assert not _called
        _called = False
        assert 12 == fn_test_memoize_6(6)
        assert not _called

        fn_test_memoize_6.forget(5)
        _called = False
        assert 10 == fn_test_memoize_6(5)
        assert _called
        _called = False
        assert 12 == fn_test_memoize_6(6)
        assert not _called

    def test_dataframe(self):
        global _called
        _called = False

        fn_test_memoize_df.forget_all()

        expected = pd.DataFrame([{"name": "a", "value": 1}, {"name": "b", "value": 2}])

        assert not _called

        assert_frame_equal(expected, fn_test_memoize_df())
        assert _called

        _called = False
        assert_frame_equal(expected, fn_test_memoize_df())
        assert not _called

    def test_return_type_recorded(self):
        global _called
        _called = False

        fn_test_memoize_null.forget_all()
        fn_test_memoize_boolean.forget_all()
        fn_test_memoize_str.forget_all()
        fn_test_memoize_bin.forget_all()
        fn_test_memoize_int.forget_all()
        fn_test_memoize_float.forget_all()
        fn_test_memoize_date.forget_all()
        fn_test_memoize_datetime.forget_all()
        fn_test_memoize_dict.forget_all()
        fn_test_memoize_list.forget_all()
        fn_test_memoize_series.forget_all()
        fn_test_memoize_data_frame.forget_all()
        fn_test_memoize_partition.forget_all()
        assert not _called

        assert fn_test_memoize_null() is None
        assert fn_test_memoize_boolean()
        assert "foo" == fn_test_memoize_str()
        assert bytes([0x01, 0x02, 0x03]) == fn_test_memoize_bin()
        assert 7 == fn_test_memoize_int()
        assert 4.2 == fn_test_memoize_float()
        assert _today == fn_test_memoize_date()
        assert _now == fn_test_memoize_datetime()
        assert {"a": "b"} == fn_test_memoize_dict()
        assert [1, 2, 3] == fn_test_memoize_list()
        assert pd.Series([1, 2, 3]).equals(fn_test_memoize_series())
        assert pd.DataFrame([{"a": 1}, {"a": 2}]).equals(fn_test_memoize_data_frame())
        partition = fn_test_memoize_partition()
        assert partition.get("a")
        assert 2 == partition.get("b")
        assert _called

        assert (
            ResultType.null
            == fn_test_memoize_null.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.boolean
            == fn_test_memoize_boolean.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.string
            == fn_test_memoize_str.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.binary
            == fn_test_memoize_bin.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.number
            == fn_test_memoize_int.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.number
            == fn_test_memoize_float.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.date
            == fn_test_memoize_date.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.timestamp
            == fn_test_memoize_datetime.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.dictionary
            == fn_test_memoize_dict.memento().invocation_metadata.result_type
        )
        # note for future test maintainers: array would also be acceptable
        assert (
            ResultType.list_result
            == fn_test_memoize_list.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.series
            == fn_test_memoize_series.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.data_frame
            == fn_test_memoize_data_frame.memento().invocation_metadata.result_type
        )
        assert (
            ResultType.partition
            == fn_test_memoize_partition.memento().invocation_metadata.result_type
        )

    def test_robust_handling_of_io_error_during_memoization(self):
        global _called
        env_before = m.Environment.get()
        logger = logging.getLogger("memento")
        prev_log_level = logger.level
        # Suppress warnings during test
        logger.setLevel(logging.CRITICAL)

        try:
            # Use a path that is illegal on both Linux and Windows
            m.Environment.set(
                Environment(
                    name="bad_env",
                    repos=[
                        ConfigurationRepository(
                            name="bad_repo",
                            clusters={
                                "bad_cluster": FunctionCluster(
                                    name="bad_cluster",
                                    storage=FilesystemStorageBackend(
                                        path="/proc/test<"
                                    ),
                                )
                            },
                        )
                    ],
                )
            )

            _called = False
            assert "memoize me" == fn_test_memoize_string()
            assert _called

            _called = False
            assert "memoize me" == fn_test_memoize_string()
            assert _called

        finally:
            m.Environment.set(env_before)
            logger.setLevel(prev_log_level)

    def test_validate_args(self):
        # These should all be valid arguments
        sample_a(None)
        sample_a(True)
        sample_a("a")
        sample_a(1)
        sample_a(2.0)
        sample_a(datetime.date.today())
        sample_a(datetime.datetime.now(datetime.timezone.utc))
        sample_a([1, 2, 3])
        multi_list = [
            None,
            True,
            "a",
            1,
            2.0,
            datetime.date.today(),
            datetime.datetime.now(datetime.timezone.utc),
            [1, 2, 3],
            {"a": "b", "c": "d"},
        ]
        sample_a(multi_list)
        sample_a({"a": "b", "c": "d"})
        sample_a({"a": "b", "c": multi_list})
        # Memento cannot serialize a function as a return value
        fn_test_fn_with_arg_noreturn(fn_test_memoize_0a)
        fn_test_fn_with_arg_noreturn(
            [1, 2, fn_test_memoize_0a, {"a": fn_test_memoize_0a}]
        )

        # These should not be accepted
        class CustomClass:
            def f(self):
                pass

        with pytest.raises(ValueError):
            sample_a((1, 2))

        with pytest.raises(ValueError):
            sample_a(CustomClass())

        with pytest.raises(ValueError):
            sample_a([CustomClass()])

        with pytest.raises(ValueError):
            sample_a({"a": CustomClass()})

    def test_force_local(self):
        global _called
        _called = False

        sample_a.forget_all()
        assert not _called

        assert 7 == sample_a.force_local().call(7)
        assert _called

        _called = False
        assert 7 == sample_a.force_local().call(7)
        assert not _called

    def test_with_context_args(self):
        fn_with_context_args = fn_test_memoize_1.with_context_args({"a": 1})
        assert 1 == fn_with_context_args.context.recursive.context_args["a"]

    def test_ignore_result(self):
        global _called
        _called = False

        fn_test_memoize_1.forget_all()
        assert not _called

        assert sample_a.ignore_result().call(7) is None
        assert _called

        _called = False
        assert sample_a.ignore_result().call(7) is None
        assert not _called

    def test_ignore_result_local(self):
        global _called
        _called = False

        sample_a.forget_all()
        assert not _called

        assert sample_a.ignore_result().force_local().call(7) is None
        assert _called

        _called = False
        assert sample_a.ignore_result().force_local().call(7) is None
        assert not _called

    def test_handle_exception(self):
        global _called
        _called = False

        fn_test_raises_value_error.forget_all()
        assert not _called

        with pytest.raises(ValueError):
            fn_test_raises_value_error()

        assert _called

        _called = False
        with pytest.raises(ValueError, match="test message"):
            fn_test_raises_value_error()

    def test_partial(self):
        """
        Check behavior of partial functions

        """

        assert 5 == add(3, 2)
        add3 = add.partial(3)
        assert 9 == add3(6)

        assert 7 == add(x=4, y=3)
        add_kw_5 = add.partial(x=5)
        assert 11 == add_kw_5(y=6)

        # Test chaining two partials together
        add_kw_6 = add_kw_5.partial(x=6)
        assert 12 == add_kw_6(y=6)

        # Ensure arg hash is properly computed for partial functions
        add4 = add.partial(4)
        assert 10 == add4(6)

    def test_partial_as_argument(self):
        """
        Check behavior of partial functions when passed as an argument to another function.
        The partial arguments should affect the hash so that memoized results are not
        returned.

        """

        add3 = add.partial(3)
        assert 14 == apply_and_double(add3, 4)

        add4 = add.partial(4)
        assert 16 == apply_and_double(add4, 4)

    def test_memento(self):
        assert sample_a.memento(1) is None
        assert 1 == sample_a(1)
        memento = sample_a.memento(1)  # type: Memento
        assert memento is not None
        assert ResultType.number == memento.invocation_metadata.result_type

    def test_list_memoized_functions(self):
        assert 0 == len(m.list_memoized_functions())
        sample_a(1)
        sample_a(2)
        fns = m.list_memoized_functions()
        assert 1 == len(fns)
        assert FunctionReference(sample_a).qualified_name == fns[0].qualified_name
        sample_b(1)
        sample_b(2)
        fns = m.list_memoized_functions()
        assert 2 == len(fns)
        expected_names = {
            FunctionReference(sample_a).qualified_name,
            FunctionReference(sample_b).qualified_name,
        }
        actual_names = set([x.qualified_name for x in fns])
        assert expected_names == actual_names

    def test_list_mementos(self):
        assert 0 == len(sample_a.list_mementos())
        sample_a(1)
        mementos = sample_a.list_mementos()
        assert 1 == len(mementos)
        assert ResultType.number == mementos[0].invocation_metadata.result_type
        sample_a(2)
        mementos = sample_a.list_mementos()
        assert 2 == len(mementos)
        assert ResultType.number == mementos[0].invocation_metadata.result_type
        assert ResultType.number == mementos[1].invocation_metadata.result_type
        # Test limit
        mementos = sample_a.list_mementos(limit=1)
        assert 1 == len(mementos)

    def test_trace(self):
        assert 12 == fn_triple_double(2)

        output = str(fn_triple_double.memento(2).trace())
        assert ":fn_triple_double#" in output and "(a=2)" in output == output
        assert ":fn_double#" in output and "(a=2)" in output == output

    def test_trace_max_depth(self):
        fn_a()

        output = str(fn_a.memento().trace(max_depth=1))
        assert ":fn_a#" in output and "()" in output, output
        assert ":fn_b#" in output
        assert ":fn_c#" not in output

    def test_trace_only_exceptions(self):
        with pytest.raises(ValueError):
            fn_okay_and_exception()

        output = str(fn_okay_and_exception.memento().trace(only_exceptions=True))
        assert ":fn_raises_exception#" in output, output
        assert ":fn_okay#" not in output, output

    def test_graph(self):
        assert 12 == fn_triple_double(2)

        output = str(fn_triple_double.memento(2).graph())
        assert "fn_triple_double(a=2)" in output, output
        assert "fn_double(a=2)" in output, output

    def test_graph_max_depth(self):
        fn_a()

        output = str(fn_a.memento().graph(max_depth=1))
        assert "fn_a()" in output and "()" in output, output
        assert "fn_b()" in output
        assert "fn_c()" not in output

    def test_graph_only_exceptions(self):
        with pytest.raises(ValueError):
            fn_okay_and_exception()

        output = str(fn_okay_and_exception.memento().graph(only_exceptions=True))
        assert "fn_raises_exception()" in output, output
        assert "fn_okay#" not in output, output

    def test_memento_forget(self):
        assert fn_c.memento() is None
        fn_c()
        assert fn_c.memento() is not None
        fn_c.memento().forget()
        assert fn_c.memento() is None

    def test_memento_forget_exceptions_recursively(self):
        assert fn_calls_raise_on_odd.memento(0) is None
        assert fn_calls_raise_on_odd.memento(1) is None
        assert fn_raise_on_odd.memento(0) is None
        assert fn_raise_on_odd.memento(1) is None
        fn_calls_raise_on_odd(0)
        with pytest.raises(ValueError):
            fn_calls_raise_on_odd(1)
        assert fn_calls_raise_on_odd.memento(0) is not None
        assert fn_calls_raise_on_odd.memento(1) is not None
        assert fn_raise_on_odd.memento(0) is not None
        assert fn_raise_on_odd.memento(1) is not None

        # Nothing should happen if the result is not exceptional
        fn_calls_raise_on_odd.memento(0).forget_exceptions_recursively()
        assert fn_calls_raise_on_odd.memento(0) is not None
        assert fn_calls_raise_on_odd.memento(1) is not None
        assert fn_raise_on_odd.memento(0) is not None
        assert fn_raise_on_odd.memento(1) is not None

        # Nothing should happen in dry-run mode
        fn_calls_raise_on_odd.memento(1).forget_exceptions_recursively(dry_run=True)
        assert fn_calls_raise_on_odd.memento(0) is not None
        assert fn_calls_raise_on_odd.memento(1) is not None
        assert fn_raise_on_odd.memento(0) is not None
        assert fn_raise_on_odd.memento(1) is not None

        # Should forget all exceptional results recursively
        fn_calls_raise_on_odd.memento(1).forget_exceptions_recursively()
        assert fn_calls_raise_on_odd.memento(0) is not None
        assert fn_calls_raise_on_odd.memento(1) is None
        assert fn_raise_on_odd.memento(0) is not None
        assert fn_raise_on_odd.memento(1) is None

    def test_with_args(self):
        ref_a = fn_double.fn_reference().with_args(3)
        assert (
            fn_double.fn_reference().qualified_name == ref_a.fn_reference.qualified_name
        )
        assert (3,) == ref_a.args
        assert {} == ref_a.kwargs

    def test_fib(self):
        # Test a self-recursive function, like fibonacci
        assert fn_fib(3) == 2
        assert fn_fib(4) == 3

    def test_map_over_range(self):
        result = add.partial(x=2).map_over_range(y=range(1, 4))

        assert 3 == len(result)
        assert 3 == result[1]
        assert 4 == result[2]
        assert 5 == result[3]

    def test_call_batch(self):
        # Make sure call_batch raises an error if kwarg keys are not strings
        kwarg_list = [{"x": 1, "y": 2}, {1: 2, "b": 4}]
        with pytest.raises(TypeError):
            add.call_batch(kwarg_list)

        # Try a successful call:
        kwarg_list = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        results = add.call_batch(kwarg_list)
        expected = [3, 7]
        assert expected == results

    def test_call_batch_raise_first_exception(self):
        kwarg_list = [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}]
        with pytest.raises(ValueError):
            fn_raise_on_odd.call_batch(kwarg_list)

        result = fn_raise_on_odd.call_batch(kwarg_list, raise_first_exception=False)
        assert 4 == len(result)
        assert not isinstance(result[0], ValueError)
        assert isinstance(result[1], ValueError)
        assert not isinstance(result[2], ValueError)
        assert isinstance(result[3], ValueError)

    def test_monitor_progress(self):
        # Try a successful call:
        kwarg_list = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        results = add.monitor_progress().call_batch(kwarg_list)
        expected = [3, 7]
        assert expected == results

    def test_does_not_accept_local_functions(self):
        try:

            @m.memento_function
            def local_fn():
                pass

            pytest.fail(
                "Should not have allowed declaration of a local memento function"
            )
        except ValueError:
            pass

    @staticmethod
    @m.memento_function
    def fn_static_method():
        return 42

    def test_allows_static_methods(self):
        assert 42 == self.fn_static_method()
        assert 42 == self.fn_static_method()

    def test_new_version_invalidates_memoized_data(self):
        t1 = fn_current_time()
        sleep(0.01)
        t2 = fn_current_time()
        assert t1 == t2

        try:
            # Modify the version at runtime (generally dangerous, but this is for the test)
            self.redefine_version(fn_current_time, "2")
            sleep(0.01)
            t3 = fn_current_time()
            assert t1 != t3
            sleep(0.01)
            t4 = fn_current_time()
            assert t3 == t4
        finally:
            self.redefine_version(fn_current_time, "1")

    def test_new_version_invalidates_dependent_memoized_data(self):
        t1 = fn_calls_calls_current_time()
        sleep(0.01)
        t2 = fn_calls_calls_current_time()
        assert t1 == t2

        try:
            # Modify the version at runtime (generally dangerous, but this is for the test)
            self.redefine_version(fn_current_time, "2")
            sleep(0.01)
            t3 = fn_calls_calls_current_time()
            assert t1 != t3
            sleep(0.01)
            t4 = fn_calls_calls_current_time()
            assert t3 == t4
        finally:
            self.redefine_version(fn_current_time, "1")

    @unittest.skipUnless(
        (sys.version_info.major, sys.version_info.minor) == (3, 11),
        "Code hash test requires Python is 3.11",
    )
    def test_code_hash(self):
        """
        Test the stability of code hashing

        """
        assert "32eac03cc26e644b" == fn_code_hash(sample_a)

    def test_version_code_hash_attribute(self):
        """
        Test that version_code_hash overrides the automatically-computed code hash, but
        if a dependency changes, the version still changes.

        """
        f1 = m.memento_function(fn_unbound_1)
        f2 = m.memento_function(fn_unbound_1)
        assert f1.version() == f2.version()

        orig_fn_unbound_2_name = fn_unbound_2.__name__
        orig_fn_unbound_2_qualname = fn_unbound_2.__qualname__
        try:
            # Change the code of f2 - should get different version
            fn_unbound_2.__name__ = fn_unbound_1.__name__
            fn_unbound_2.__qualname__ = fn_unbound_1.__qualname__
            f2 = m.memento_function(fn_unbound_2)
            assert f1.version() != f2.version()

            # Change the code of f2 but set the code hash the same as f1
            f2 = m.memento_function(fn_unbound_2, version_code_hash=f1.code_hash)
            assert f1.version() == f2.version()

            # Change the dependency of f2 and make sure version does change
            f2 = m.memento_function(
                fn_unbound_2, version_code_hash=f1.code_hash, dependencies=[sample_a]
            )
            assert f1.version() != f2.version()
        finally:
            fn_unbound_2.__name__ = orig_fn_unbound_2_name
            fn_unbound_2.__qualname__ = orig_fn_unbound_2_qualname

    def test_version_salt_attribute(self):
        """
        Test that version_salt adds entropy to the version hash in a consistent way.

        """
        f1 = m.memento_function(fn_unbound_1)
        f2 = m.memento_function(fn_unbound_1, version_salt="a")
        f2a = m.memento_function(fn_unbound_1, version_salt="a")
        f3 = m.memento_function(fn_unbound_1, version_salt="b")

        assert f1.version() != f2.version()
        assert f2.version() == f2a.version()
        assert f2.version() != f3.version()

    def test_list(self):
        """
        Test fn.list()
        """
        assert add.list() is None

        add(1, 1)
        df = add.list()
        expected = DataFrame(data=[{"x": 1, "y": 1, "result_type": "number"}])
        assert_frame_equal(expected, df)

        add(1, 2)
        df = add.list()
        expected = DataFrame(
            data=[
                {"x": 1, "y": 1, "result_type": "number"},
                {"x": 1, "y": 2, "result_type": "number"},
            ]
        )
        assert_frame_equal(expected, df)

        df = add.list(y=2)
        expected = DataFrame(data=[{"x": 1, "y": 2, "result_type": "number"}])
        assert_frame_equal(expected, df)

        assert add.list(y=3) is None

    @staticmethod
    def redefine_version(memento_fn: MementoFunction, new_version: str):
        """Modify the version at runtime (generally dangerous, but this is for the test)"""
        memento_fn.explicit_version = new_version
        memento_fn._fn_reference = None
        MementoFunction.increment_global_fn_generation()

    def test_decorated_method_versions(self):
        assert fn_decorated_v1 != fn_decorated_v2
