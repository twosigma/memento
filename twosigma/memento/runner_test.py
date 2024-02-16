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
This is a test function used for unit testing. It appears in the
runner_test module because it must be available in the container's
installed package during the unit testing process. The unit tests
are not distributed in the installed package.

"""
from time import sleep

import twosigma.memento as m
from .result import KeyOverrideResult
from twosigma.memento import MementoFunction

runner_fn_test_1_called = False


def set_runner_fn_test_1_called(value: bool):
    global runner_fn_test_1_called
    runner_fn_test_1_called = value


def get_runner_fn_test_1_called():
    global runner_fn_test_1_called
    return runner_fn_test_1_called


# version=2 is arbitrary and to help test version functionality
@m.memento_function(cluster="memento.unit_test", version="2")
def runner_fn_test_1(a, b, c=None, d=None, e=None, f=None):
    global runner_fn_test_1_called
    runner_fn_test_1_called = True
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e == runner_fn_test_1,
        "f": (f[0]["a"] == runner_fn_test_1) if f is not None else None,
    }


@m.memento_function(cluster="memento.unit_test")
def runner_fn_f(x) -> int:
    return x


@m.memento_function(cluster="memento.unit_test")
def runner_fn_calls_runner_fn_f(x) -> int:
    try:
        return runner_fn_f(x)
    except RuntimeError as e:
        if "Further Memento calls are prevented in this context" in str(e):
            raise RuntimeError("Caught further calls error")
        raise e


@m.memento_function(cluster="memento.unit_test")
def runner_fn_test_add(x: int, y: int) -> int:
    """
    Computes x + y

    """
    return x + y


# version=3 is arbitrary and to help test version functionality
@m.memento_function(cluster="memento.unit_test", version="3")
def runner_fn_test_apply_and_double(fn: MementoFunction, y: int) -> int:
    """
    Computes 2 * fn(y)

    """
    return 2 * fn(y)


@m.memento_function(cluster="memento.unit_test")
def runner_fn_test_sum_double_batch(fn: MementoFunction, y_from: int, y_to: int) -> int:
    """
    Computes sum(2 * fn(y)) for y in range y_from to y_to

    """
    kwargs_list = [{"y": y} for y in range(y_from, y_to)]
    return sum([2 * result for result in fn.call_batch(kwargs_list)])


@m.memento_function(cluster="memento.unit_test")
def fn_calls_undeclared_dependency():
    """
    Function that calls a function that is not auto-detected or declared as a dependency

    """
    return globals()["fn_undeclared_dependency"]()


@m.memento_function(cluster="memento.unit_test", version="1")
def fn_with_explicit_version_calls_undeclared_dependency():
    """
    Function that calls a function that is not auto-detected or declared as a dependency

    """
    return globals()["fn_undeclared_dependency"]()


@m.memento_function(cluster="memento.unit_test")
def fn_undeclared_dependency():
    return 1


@m.memento_function(cluster="memento.unit_test")
def fn_returns_key_override_result():
    return KeyOverrideResult(result="abc123", key_override="custom_key")


@m.memento_function(cluster="memento.unit_test")
def fn_recursive_c(x) -> int:
    return x * 10


@m.memento_function(cluster="memento.unit_test")
def fn_recursive_b(x) -> int:
    return fn_recursive_c(x)


@m.memento_function(cluster="memento.unit_test")
def fn_recursive_a() -> int:
    return sum(fn_recursive_b.call_batch([{"x": 1}, {"x": 2}]))


@m.memento_function(cluster="cluster1")
def fn_wait_seconds(seconds):
    sleep(seconds)


# These help test dependencies are accurately represented in the graph
fn_G = 42


@m.memento_function(cluster="memento.unit_test")
def fn_a():
    fn_e()


@m.memento_function(cluster="memento.unit_test")
def fn_c():
    fn_b()


@m.memento_function(cluster="memento.unit_test")
def fn_d():
    fn_a()


@m.memento_function(cluster="memento.unit_test")
def fn_e():
    fn_c()
    fn_d()
    _ = fn_G


@m.memento_function(cluster="memento.unit_test")
def fn_f():
    pass


def fn_b():
    fn_a()
    fn_f()
    fn_e()


@m.memento_function(cluster="memento.unit_test")
def fn_B():
    return 1


@m.memento_function(cluster="memento.unit_test")
def fn_A():
    return fn_B() + 1
