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

from abc import ABC

import pytest

from twosigma.memento import Memento
from twosigma.memento.exception import UndeclaredDependencyError
from twosigma.memento.reference import FunctionReferenceWithArguments
from typing import List  # noqa: F401
from unittest import TestCase  # noqa: F401

import twosigma.memento as m  # noqa: F401
from twosigma.memento.runner_test import (
    runner_fn_test_1,
    runner_fn_test_apply_and_double,
    runner_fn_test_add,
    runner_fn_test_sum_double_batch,
    fn_calls_undeclared_dependency,
    fn_with_explicit_version_calls_undeclared_dependency,
    fn_returns_key_override_result,
    runner_fn_calls_runner_fn_f,
    fn_recursive_a,
    fn_recursive_b,
    fn_recursive_c,
)


class RunnerBackendTester(ABC):
    """
    Abstract base class to test runner backends.

    Subclasses should set `self.backend` to the runner backend being tested.

    """

    backend = None  # type: m.RunnerBackend

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @staticmethod
    def test_memoize():
        # This also tests serializing function references
        assert {
            "a": 1,
            "b": 2,
            "c": [{"three": 3}],
            "d": None,
            "e": True,
            "f": True,
        } == runner_fn_test_1(
            1, c=[{"three": 3}], b=2, e=runner_fn_test_1, f=[{"a": runner_fn_test_1}]
        )

    @staticmethod
    def test_call_stack_invocations_tracked():
        # runner_fn_test_apply_and_double -->
        # runner_fn_test_apply_and_double -->
        # runner_fn_test_add
        add2 = runner_fn_test_add.partial(2)
        add_2_and_double = runner_fn_test_apply_and_double.partial(add2)
        runner_fn_test_apply_and_double(add_2_and_double, 2)

        memento = runner_fn_test_apply_and_double.memento(add_2_and_double, 2)
        invocations = (
            memento.invocation_metadata.invocations
        )  # type: List[FunctionReferenceWithArguments]
        assert 1 == len(invocations)
        assert (
            "runner_fn_test_apply_and_double"
            == invocations[0].fn_reference.function_name
        )

        # Try with a batch run - make sure all invocations from the batch are recorded in invocations
        # runner_fn_test_sum_double_batch --> runner_fn_test_apply_and_double --> runner_fn_test_add
        runner_fn_test_sum_double_batch(add_2_and_double, 10, 12)
        memento = runner_fn_test_sum_double_batch.memento(add_2_and_double, 10, 12)
        invocations = memento.invocation_metadata.invocations
        assert 2 == len(invocations)
        assert (
            "runner_fn_test_apply_and_double"
            == invocations[0].fn_reference.function_name
        )
        assert (
            "runner_fn_test_apply_and_double"
            == invocations[1].fn_reference.function_name
        )

    @staticmethod
    def test_correlation_id():
        """
        Test that a correlation id is generated and properly propagated

        """
        add2 = runner_fn_test_add.partial(2)
        runner_fn_test_apply_and_double(add2, 2)

        memento_1 = runner_fn_test_add.memento(2, 2)
        corr_id_1 = memento_1.correlation_id
        assert corr_id_1 is not None

        memento_2 = runner_fn_test_apply_and_double.memento(add2, 2)
        corr_id_2 = memento_2.correlation_id
        assert corr_id_1 == corr_id_2

    @staticmethod
    def test_prevent_further_calls():
        """
        Test that prevent_further_calls in recursive context prevents additional
        Memento calls.

        """
        assert 1 == runner_fn_calls_runner_fn_f(1)
        try:
            runner_fn_calls_runner_fn_f.with_prevent_further_calls(True).call(2)
            pytest.fail("Should not have allowed call")
        except RuntimeError as e:
            # The function changes the message of the error. If this is not in the text of
            # the error, that means the RuntimeError happened on the wrong call (too early).
            assert "Caught further calls error" in str(e)

    @staticmethod
    def test_refuse_execution_of_non_dependencies():
        """
        Test that Memento refuses to execute a down-stream function that is not declared
        or detected as a dependency in the parent chain.

        """
        with pytest.raises(UndeclaredDependencyError):
            fn_calls_undeclared_dependency()

        # But allowed if explicit version is set:
        assert 1 == fn_with_explicit_version_calls_undeclared_dependency()

    @staticmethod
    def test_key_override_result():
        """
        Test that, if a function returns `KeyOverrideResult`, the result is stored at
        the overridden key instead of the standard location.
        """
        result = fn_returns_key_override_result()
        assert "abc123" == result
        mem = fn_returns_key_override_result.memento()
        assert "custom_key" == mem.content_key.key

    @staticmethod
    def test_memento_dependencies():
        """
        Test that, given a -> b -> c, we can extract the full call stack from the memento.
        In this test, we make a -> b a batch call so that we override any speculative local
        execution that might happen in a sophisticated server.
        """
        result = fn_recursive_a()
        assert 30 == result
        mem_a = fn_recursive_a.memento()  # type: Memento
        assert mem_a is not None
        assert 2 == len(mem_a.invocation_metadata.invocations)
        for i in mem_a.invocation_metadata.invocations:
            assert (
                fn_recursive_b.fn_reference().qualified_name
                == i.fn_reference.qualified_name
            )
            mem_b = i.fn_reference.memento_fn.memento(x=i.kwargs["x"])  # type: Memento
            i_b = mem_b.invocation_metadata
            assert 1 == len(i_b.invocations)
            assert (
                fn_recursive_c.fn_reference().qualified_name
                == i_b.invocations[0].fn_reference.qualified_name
            )

    @staticmethod
    def test_memento_graph():
        """
        Test that, given a -> b -> c, we can extract the full call stack from the memento.
        In this test, we make a -> b a batch call so that we override any speculative local
        execution that might happen in a sophisticated server.
        """
        result = fn_recursive_a()
        assert 30 == result
        mem_a = fn_recursive_a.memento()  # type: Memento
        digraph = mem_a.graph()
        assert any("fn_recursive_a" in x for x in digraph.body)
        assert any("fn_recursive_b" in x for x in digraph.body)
        assert any("fn_recursive_c" in x for x in digraph.body)
