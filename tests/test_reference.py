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
from pickle import dumps, loads
from typing import cast
import pandas as pd
import shutil
import tempfile
import pytest
import pytz

from twosigma.memento import memento_function, Environment, MementoFunction
from twosigma.memento.exception import DependencyNotFoundError
from twosigma.memento.external import UnboundExternalMementoFunction
from twosigma.memento.reference import (
    FunctionReference,
    FunctionReferenceWithArguments,
    ArgumentHasher,
)

_called = False


@memento_function
def fn1(a):
    return a


@memento_function
def _test_method():
    return 1


# version number should be auto-converted into a str
@memento_function(version=2)
def _test_method_2(x: int, y: int) -> int:
    return x + y


@memento_function(auto_dependencies=False)
def _test_dep_depends_on_nonexistent() -> int:
    return 42


@memento_function()
def _test_dep_depends_on_nonexistent_auto() -> int:
    """
    This function depends on something that looks like a possible function to Memento's
    dependency auto-detection logic, but does not exist as a function.

    """

    def internal_fn():
        return 42

    return internal_fn()


@memento_function()
def _test_dep_a() -> int:
    return _test_dep_c()


@memento_function(dependencies=[_test_dep_a])
def _test_dep_b() -> int:
    return 42


@memento_function()
def _test_dep_c() -> int:
    return 42


@memento_function()
def _test_dep_d() -> int:
    # noinspection PyUnresolvedReferences
    return _test_dep_e()


@memento_function()
def _test_dep_circular_1() -> int:
    # Reference circular_2
    type(_test_dep_circular_2)
    return 42


@memento_function()
def _test_dep_circular_2() -> int:
    # Reference circular_1
    type(_test_dep_circular_1)
    return 42


@memento_function()
def _test_dep_changes() -> int:
    return 42


@memento_function()
def _test_dep_nonexistent() -> int:
    """This function gets removed later"""
    return 42


class TestReference:
    def setup_method(self):
        global _called
        self.env_before = Environment.get()
        self.env_dir = tempfile.mkdtemp(prefix="memoizeTest")
        env_file = "{}/env.json".format(self.env_dir)
        with open(env_file, "w") as f:
            print("""{"name": "test"}""", file=f)
        Environment.set(env_file)
        _called = False

    def teardown_method(self):
        shutil.rmtree(self.env_dir)
        Environment.set(self.env_before)

    def test_parse_qualified_name(self):
        parts = FunctionReference.parse_qualified_name(
            "cluster::module.name:fn_name#hash"
        )
        assert "cluster" == parts["cluster"]
        assert "module.name" == parts["module"]
        assert "fn_name" == parts["function"]
        assert "hash" == parts["version"]

    def test_reference(self):
        ref = FunctionReference.from_qualified_name("tests.test_reference:_test_method")
        assert (
            "tests.test_reference:_test_method"
            == ref.qualified_name[0 : ref.qualified_name.find("#")]
        )
        assert (
            "tests.test_reference:_test_method"
            == ref.qualified_name_without_cluster[
                0 : ref.qualified_name_without_cluster.find("#")
            ]
        )
        assert "tests.test_reference" == ref.module
        assert "_test_method" == ref.function_name
        assert ref.cluster_name is None
        assert ref.memento_fn.version() is not None

        ref = FunctionReference(_test_method)
        assert (
            "tests.test_reference:_test_method"
            == ref.qualified_name[0 : ref.qualified_name.find("#")]
        )
        assert (
            "tests.test_reference:_test_method"
            == ref.qualified_name_without_cluster[
                0 : ref.qualified_name_without_cluster.find("#")
            ]
        )
        assert "tests.test_reference" == ref.module
        assert "_test_method" == ref.function_name
        assert ref.cluster_name is None
        assert ref.memento_fn.version() is not None

        ref = FunctionReference(_test_method, cluster_name="cluster1")
        assert (
            "cluster1::tests.test_reference:_test_method"
            == ref.qualified_name[0 : ref.qualified_name.find("#")]
        )
        assert (
            "tests.test_reference:_test_method"
            == ref.qualified_name_without_cluster[
                0 : ref.qualified_name_without_cluster.find("#")
            ]
        )
        assert "tests.test_reference" == ref.module
        assert "_test_method" == ref.function_name
        assert "cluster1" == ref.cluster_name
        assert ref.memento_fn.version() is not None

    def test_version(self):
        ref = _test_method_2.fn_reference()
        assert "tests.test_reference:_test_method_2#2" == ref.qualified_name
        assert (
            "tests.test_reference:_test_method_2#2"
            == ref.qualified_name_without_cluster
        )
        assert "tests.test_reference" == ref.module
        assert "_test_method_2" == ref.function_name
        assert "2" == ref.memento_fn.version()
        assert ref.memento_fn.code_hash is None

        ref = FunctionReference(_test_method_2, cluster_name="cluster1", version="2")
        assert "cluster1::tests.test_reference:_test_method_2#2" == ref.qualified_name
        assert (
            "tests.test_reference:_test_method_2#2"
            == ref.qualified_name_without_cluster
        )
        assert "tests.test_reference" == ref.module
        assert "_test_method_2" == ref.function_name
        assert "2" == ref.memento_fn.version()
        assert ref.memento_fn.code_hash is None

        ref = FunctionReference.from_qualified_name(
            "tests.test_reference:_test_method_2#2"
        )
        assert "tests.test_reference:_test_method_2#2" == ref.qualified_name
        assert (
            "tests.test_reference:_test_method_2#2"
            == ref.qualified_name_without_cluster
        )
        assert "tests.test_reference" == ref.module
        assert "_test_method_2" == ref.function_name
        assert ref.cluster_name is None
        assert "2" == ref.memento_fn.version()
        assert ref.memento_fn.code_hash is None
        assert ref.memento_fn is not None
        assert _test_method_2 is ref.memento_fn

        # If the version does not match, check that reference is treated as external
        assert FunctionReference.from_qualified_name(
            "unknown_cluster::test_reference:_test_method_2#3",
            parameter_names=["x", "y"],
        ).external

    def test_find(self):
        ref = FunctionReference(_test_method, cluster_name="cluster1")
        ref2 = FunctionReference.from_qualified_name(ref.qualified_name)
        fn = ref2.memento_fn
        assert fn is not None
        assert fn.__qualname__ == _test_method.__qualname__
        # noinspection PyCallingNonCallable
        assert 1 == fn()

    def test_reference_with_args(self):
        ref1 = FunctionReference(_test_method_2, cluster_name="cluster1")
        ref1a = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(1,), kwargs={"y": 2}, context_args={"z": 3}
        )

        assert ref1.qualified_name == ref1a.fn_reference.qualified_name
        assert (1,) == ref1a.args
        assert {"y": 2} == ref1a.kwargs
        assert {"z": 3} == ref1a.context_args

        ref1b = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(), kwargs={"x": 1, "y": 2}, context_args={"z": 3}
        )

        assert ref1.qualified_name == ref1b.fn_reference.qualified_name
        assert () == ref1b.args
        assert {"x": 1, "y": 2} == ref1b.kwargs
        assert {"z": 3} == ref1b.context_args

        assert ref1a.arg_hash == ref1b.arg_hash

    def test_context_args_affect_hash(self):
        ref1 = FunctionReference(_test_method_2, cluster_name="cluster1")
        ref1a = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(1,), kwargs={"y": 2}
        )
        ref1b = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(1,), kwargs={"y": 2}, context_args={"z": 3}
        )
        assert ref1a.arg_hash != ref1b.arg_hash

    def test_compute_args(self):
        ref1 = FunctionReference(_test_method_2, cluster_name="cluster1")
        hash1 = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(1, 2), kwargs={}
        ).arg_hash
        hash2 = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(2, 3), kwargs={}
        ).arg_hash
        assert hash1 != hash2
        hash3 = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(2, 3), kwargs={}
        ).arg_hash
        assert hash2 == hash3
        hash4 = FunctionReferenceWithArguments(
            fn_reference=_test_method_2.partial(2).fn_reference(), args=(3,), kwargs={}
        ).arg_hash
        assert hash3 == hash4
        # Test that args are mapped to kwargs
        hash5 = FunctionReferenceWithArguments(
            fn_reference=ref1, args=(2,), kwargs={"y": 3}
        ).arg_hash
        assert hash2 == hash5
        hash6 = FunctionReferenceWithArguments(
            fn_reference=_test_method_2.partial(x=2).fn_reference(),
            args=(),
            kwargs={"y": 3},
        ).arg_hash
        assert hash2 == hash6

    def test_timestamp_arg(self):
        # Check that timestamp arg_hash is computed correctly
        t1 = pd.to_datetime("2019-01-01")
        t2 = t1.to_pydatetime()

        assert fn1(t1) == fn1(t2)
        assert (
            fn1.fn_reference().with_args(t1).arg_hash
            == fn1.fn_reference().with_args(t2).arg_hash
        )

    def test_arg_hasher_normalize(self):
        assert ArgumentHasher.normalize(None) is None
        assert ArgumentHasher.normalize(True)
        assert "abc 123" == ArgumentHasher.normalize("abc 123")
        assert 42 == ArgumentHasher.normalize(42)
        f = cast(float, ArgumentHasher.normalize(123.45))
        assert pytest.approx(123.45) == f
        assert datetime.date(2019, 4, 3) == ArgumentHasher.normalize(
            datetime.date(2019, 4, 3)
        )
        assert datetime.datetime(2019, 4, 3, 12, 34, 56) == ArgumentHasher.normalize(
            datetime.datetime(2019, 4, 3, 12, 34, 56)
        )
        assert datetime.datetime(
            2019, 4, 3, 12, 34, 56, 500000
        ) == ArgumentHasher.normalize(datetime.datetime(2019, 4, 3, 12, 34, 56, 500000))
        assert datetime.datetime(
            2019, 4, 3, 12, 34, 56, tzinfo=pytz.UTC
        ) == ArgumentHasher.normalize(
            datetime.datetime(2019, 4, 3, 12, 34, 56, tzinfo=pytz.UTC)
        )
        normalized_list = cast(list, ArgumentHasher.normalize([1, 2, 3]))
        assert [1, 2, 3] == normalized_list
        in_dict = {"a": 1, "b": 2, "c": [1, 2, 3], "d": {"e": "f"}}
        d = cast(dict, ArgumentHasher.normalize(in_dict))
        assert in_dict == d
        assert fn1 == ArgumentHasher.normalize(fn1)
        fn1_p = fn1.partial(a=7)
        # noinspection PyUnresolvedReferences
        normalized = ArgumentHasher.normalize(fn1_p).fn_reference()
        assert fn1_p.fn_reference().qualified_name == normalized.qualified_name
        assert fn1_p.fn_reference().partial_kwargs == normalized.partial_kwargs

    def test_arg_hasher_encode(self):
        assert ArgumentHasher._encode(None) is None
        assert ArgumentHasher._encode(True)
        assert "abc 123" == ArgumentHasher._encode("abc 123")
        assert 42 == ArgumentHasher._encode(42)
        f = cast(float, ArgumentHasher._encode(123.45))
        assert pytest.approx(123.45) == f
        assert {
            "_mementoType": "date",
            "iso8601": "2019-04-03",
        } == ArgumentHasher._encode(datetime.date(2019, 4, 3))
        assert {
            "_mementoType": "datetime",
            "iso8601": "2019-04-03T12:34:56",
        } == ArgumentHasher._encode(datetime.datetime(2019, 4, 3, 12, 34, 56))
        assert {
            "_mementoType": "datetime",
            "iso8601": "2019-04-03T12:34:56.500000",
        } == ArgumentHasher._encode(datetime.datetime(2019, 4, 3, 12, 34, 56, 500000))
        assert {
            "_mementoType": "datetime",
            "iso8601": "2019-04-03T12:34:56+00:00",
        } == ArgumentHasher._encode(
            datetime.datetime(2019, 4, 3, 12, 34, 56, tzinfo=pytz.UTC)
        )
        encoded_list = cast(list, ArgumentHasher._encode([1, 2, 3]))
        assert [1, 2, 3] == encoded_list
        in_dict = {"a": 1, "b": 2, "c": [1, 2, 3], "d": {"e": "f"}}
        d = cast(dict, ArgumentHasher._encode(in_dict))
        assert in_dict == d
        assert {
            "_mementoType": "FunctionReference",
            "parameterNames": ["a"],
            "qualifiedName": fn1.fn_reference().qualified_name,
            "partialArgs": None,
            "partialKwargs": {},
        } == ArgumentHasher._encode(fn1)
        assert {
            "_mementoType": "FunctionReference",
            "parameterNames": ["a"],
            "qualifiedName": fn1.partial(a=7).fn_reference().qualified_name,
            "partialArgs": None,
            "partialKwargs": {"a": 7},
        } == ArgumentHasher._encode(fn1.partial(a=7))

    def test_arg_hasher_normalized_json(self):
        assert "null" == ArgumentHasher._normalized_json(None)
        assert '"abc123"' == ArgumentHasher._normalized_json("abc123")
        assert "42" == ArgumentHasher._normalized_json(42)
        assert "123.45" == ArgumentHasher._normalized_json(123.45)
        assert "true" == ArgumentHasher._normalized_json(True)
        assert "[1,2,3]" == ArgumentHasher._normalized_json([1, 2, 3])
        in_dict = {"c": [1, 2, 3], "a": 1, "d": {"e": "f"}, "b": 2}
        assert (
            '{"a":1,"b":2,"c":[1,2,3],"d":{"e":"f"}}'
            == ArgumentHasher._normalized_json(in_dict)
        )

    # noinspection SpellCheckingInspection
    def test_arg_hasher_stability(self):
        """
        Ensure arg hasher is stable from release to release.

        """
        assert (
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
            == ArgumentHasher.compute_hash({})
        )
        assert (
            "4cc66ba3de661a1a9319c150542555d384af8d81724a3b64dab2001d85df06df"
            == ArgumentHasher.compute_hash({"a": 42})
        )
        assert (
            "d091f9c83c091f79652fe8786375b3fe4ce0861a56f5bfbafedbe431877ff0e8"
            == ArgumentHasher.compute_hash({"a": None})
        )
        assert (
            "f7f851f4ba8ef23c0a3f2c20548bcc4bac24c46bc1c2c9332f7be4a695f22275"
            == ArgumentHasher.compute_hash({"a": 123.45})
        )
        assert (
            "70621113b1eb7b8fbec0b1cb896e5f6adb32a9dbe08b5032c5edef18fca6002c"
            == ArgumentHasher.compute_hash({"a": "abc123"})
        )
        assert (
            "730bc329ebcd24c6c9663ca4bb0e199a090dbf9d9d1058651d8560236abb1095"
            == ArgumentHasher.compute_hash({"a": [1, 2, 3]})
        )
        in_dict = {"c": [1, 2, 3], "a": 1, "d": {"e": "f"}, "b": 2}
        assert (
            "52ea3bee36356ba2a31ff7931c95d69aee13f8f3d24727aa4fd9d456885ea00f"
            == ArgumentHasher.compute_hash({"a": in_dict})
        )

    def test_pickling(self):
        assert fn1 == loads(dumps(fn1.fn_reference())).memento_fn

    def test_code_hashed(self):
        """Test that, when a function is defined with no version, the code is hashed"""

        assert fn1.code_hash is not None

    def test_required_dependencies_fail_if_not_present(self):
        """Test that required dependencies cause evaluation to fail if not found."""

        orig_test_dep_nonexistent = _test_dep_nonexistent
        try:
            _test_dep_depends_on_nonexistent.required_dependencies.add(
                "_test_dep_nonexistent"
            )
            del globals()["_test_dep_nonexistent"]
            MementoFunction.increment_global_fn_generation()
            with pytest.raises(DependencyNotFoundError):
                _test_dep_depends_on_nonexistent()
        finally:
            globals()["_test_dep_nonexistent"] = orig_test_dep_nonexistent
            _test_dep_depends_on_nonexistent.required_dependencies.remove(
                "_test_dep_nonexistent"
            )

    def test_detected_dependencies_do_not_fail_if_not_present(self):
        """Test that detected dependencies do not cause evaluation to fail if not found."""

        assert 42 == _test_dep_depends_on_nonexistent_auto()

    def test_auto_dependencies(self):
        """
        Test, if auto_dependencies is True, the AST of the function is parsed and the list of
        names (and attributed-names) extracted. These names are added to the detected
        dependencies list.

        """

        assert "_test_dep_c" in _test_dep_a.detected_dependencies

    def test_manual_dependencies(self):
        """
        Test that, if the dependencies parameter is specified, the list of
        manually-specified dependencies is added to the required dependencies list.

        """

        assert "tests.test_reference:_test_dep_a" in _test_dep_b.required_dependencies

    def test_code_version_hash_recomputed(self):
        """
        Test that the version hash is not computed up-front. Instead, it is computed
        any time fn_reference is accessed.

        """

        hash_before = _test_dep_d.version()
        globals()["_test_dep_e"] = _test_dep_c
        MementoFunction.increment_global_fn_generation()
        hash_after = _test_dep_d.version()
        assert hash_before != hash_after

    def test_circular_references_work(self):
        """
        Test if there are any circular references between functions, the fn_reference
        computes.

        """

        assert 42 == _test_dep_circular_1()
        assert 42 == _test_dep_circular_2()
        assert {
            _test_dep_circular_2
        } == _test_dep_circular_1.dependencies().transitive_memento_fn_dependencies()
        assert {
            _test_dep_circular_2
        } == _test_dep_circular_1.dependencies().direct_memento_fn_dependencies()
        assert {
            _test_dep_circular_1
        } == _test_dep_circular_2.dependencies().transitive_memento_fn_dependencies()
        assert {
            _test_dep_circular_1
        } == _test_dep_circular_2.dependencies().transitive_memento_fn_dependencies()

        # Test the version numbers are stable if there are circular dependencies
        v1a = _test_dep_circular_1.version()
        v2a = _test_dep_circular_2.version()
        v1b = _test_dep_circular_1.version()
        v2b = _test_dep_circular_2.version()
        assert v1a == v1b
        assert v2a == v2b

    def test_dependency_hash(self):
        """
        Test that dependency hash incorporates both required and detected dependencies

        """

        hash_1 = _test_dep_changes.version()
        _test_dep_changes.required_dependencies.add("_test_dep_c")
        MementoFunction.increment_global_fn_generation()
        hash_2 = _test_dep_changes.version()
        assert hash_1 != hash_2
        _test_dep_changes.required_dependencies.remove("_test_dep_c")
        MementoFunction.increment_global_fn_generation()
        hash_3 = _test_dep_changes.version()
        assert hash_1 == hash_3

        _test_dep_changes.detected_dependencies.add("_test_dep_b")
        MementoFunction.increment_global_fn_generation()
        hash_4 = _test_dep_changes.version()
        assert hash_4 != hash_1
        assert hash_4 != hash_2
        _test_dep_changes.detected_dependencies.remove("_test_dep_b")
        MementoFunction.increment_global_fn_generation()
        hash_5 = _test_dep_changes.version()
        assert hash_1 == hash_5

    @staticmethod
    @memento_function()
    def _static_method_test():
        return _test_dep_a()

    def test_static_method_dependencies(self):
        assert 42 == self._static_method_test()
        assert {
            _test_dep_a,
            _test_dep_c,
        } == self._static_method_test.dependencies().transitive_memento_fn_dependencies()

    def test_qualified_name_without_version(self):
        ref1 = FunctionReference(_test_method_2, cluster_name="cluster1")
        assert (
            "cluster1::tests.test_reference:_test_method_2"
            == ref1.qualified_name_without_version
        )

    def test_parameter_names(self):
        fn_ref = _test_method_2.fn_reference()
        assert ["x", "y"] == fn_ref.parameter_names

    def test_explicit_parameter_names(self):
        memento_fn = UnboundExternalMementoFunction(
            cluster_name="unknown_cluster",
            version="3",
            module_name="test_reference",
            function_name="_test_method_2",
            parameter_names=["x", "y", "z"],
        )
        fn_ref = memento_fn.fn_reference()
        assert ["x", "y", "z"] == fn_ref.parameter_names

    def test_auto_external_reference(self):
        """Test that, if a function is not found, it is treated as external"""
        v = fn1.version()

        # Unknown module
        ref1 = FunctionReference.from_qualified_name(
            "unknown_cluster::unknown.module:fn1#" + v, parameter_names=["x", "y"]
        )
        assert ref1.external

        # Unknown function
        ref2 = FunctionReference.from_qualified_name(
            "unknown_cluster::tests.test_reference:fn1a#" + v, parameter_names=["a"]
        )
        assert ref2.external

        # Known function
        print(fn1.__module__)
        print(fn1.fn_reference())
        ref3 = FunctionReference.from_qualified_name(
            "unknown_cluster::tests.test_reference:fn1#" + v, parameter_names=["a"]
        )
        assert not ref3.external, str(fn1.fn_reference()) + " " + str(fn1.__module__)
