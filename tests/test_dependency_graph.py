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

import math
import shutil
import tempfile
import pandas as pd

from twosigma.memento.types import DependencyGraphType
from twosigma.memento import memento_function, Environment
from twosigma.memento.dependency_graph import DependencyGraph
from twosigma.memento.runner_test import fn_a, fn_c, fn_d, fn_e, fn_f


@memento_function()
def dep_a():
    # make sure the verbose dependency graph handles undefined symbols
    if math.sqrt(1) == 2:
        # noinspection PyUnresolvedReferences
        _ = no_such_symbol
    return dep_b()


@memento_function()
def dep_b():
    return dep_c()


@memento_function()
def dep_c():
    return global_var


# Global variable depended-on by a memento function. Should affect hash.
global_var = 42


# These three functions test that cycles in the memento function stack do not prevent
# the correct calculation of dependencies
@memento_function
def _fn_test_cycle_1(x):
    return _fn_test_cycle_2(x)


def _fn_test_cycle_2(x):
    return _fn_test_cycle_3(x)


@memento_function
def _fn_test_cycle_3(x):
    # Introduce a cycle in the call graph:
    if x < 1:
        return _fn_test_cycle_2(x + 1)
    return 0


class TestDependencyGraph:

    def setup_method(self):
        self.env_before = Environment.get()
        self.env_dir = tempfile.mkdtemp(prefix="dependencyGraphTest")
        env_file = "{}/env.json".format(self.env_dir)
        with open(env_file, "w") as f:
            print("""{"name": "test"}""", file=f)
        Environment.set(env_file)

    def teardown_method(self):
        shutil.rmtree(self.env_dir)
        Environment.set(self.env_before)

    def test_transitive_memento_fn_dependencies(self):
        deps = dep_a.dependencies()  # type: DependencyGraphType
        assert {dep_b, dep_c} == deps.transitive_memento_fn_dependencies()

    def test_direct_memento_fn_dependencies(self):
        deps = dep_a.dependencies()  # type: DependencyGraphType
        assert {dep_b} == deps.direct_memento_fn_dependencies()

    def test_graph(self):
        deps = dep_a.dependencies()  # type: DependencyGraphType
        assert "dep_c" in repr(deps.graph().body)

    def test_verbose_graph(self):
        deps = dep_a.dependencies(verbose=True)  # type: DependencyGraphType
        assert "dep_c" in repr(deps.graph().body)

    def test_df(self):
        deps = dep_a.dependencies(verbose=True)  # type: DependencyGraphType
        expected_df = pd.DataFrame(
            data=[
                {
                    "src": dep_a.qualified_name_without_version,
                    "target": dep_b.qualified_name_without_version,
                    "type": "MementoFunction",
                },
                {
                    "src": dep_a.qualified_name_without_version,
                    "target": "no_such_symbol",
                    "type": "UndefinedSymbol",
                },
                {
                    "src": dep_b.qualified_name_without_version,
                    "target": dep_c.qualified_name_without_version,
                    "type": "MementoFunction",
                },
                {
                    "src": dep_c.qualified_name_without_version,
                    "target": "global_var",
                    "type": "GlobalVariable",
                },
            ]
        ).sort_values(by="src")
        actual_df = deps.df().sort_values(by="src")
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_cycles_do_not_prevent_transitive_dep(self):
        """
        Check that cycles in the call graph do not prevent Memento from correctly
        calculating the dependency tree

        """
        dep = _fn_test_cycle_1.dependencies(verbose=True)
        assert {_fn_test_cycle_3} == dep.transitive_memento_fn_dependencies()
        dep_df = dep.df()
        assert any("_test_cycle_2" in name for name in dep_df.src.values)
        assert any("_test_cycle_2" in name for name in dep_df.target.values)
        assert (
            len(
                dep_df[
                    dep_df["src"].str.contains("_test_cycle_2")
                    & dep_df["target"].str.contains("_test_cycle_3")
                ]
            )
            == 1
        )

    def test_label_filter(self):
        graph = _fn_test_cycle_1.dependencies(label_filter=lambda x: "FILTERED").graph()
        body = "\n".join(graph.body)
        assert "FILTERED" in body

    def test_rules_until_first_memento_fn(self):
        def validate(expected, deps):
            for pair in expected:
                assert 1 == len(
                    [
                        r
                        for r in deps
                        if f"fn_{pair[0]}" in r.parent_symbol
                        and f"fn_{pair[1]}" in r.symbol
                    ]
                ), f"{pair} not found"
            assert len(expected) == len(deps)

        # noinspection PyTypeChecker
        validate(
            (("e", "c"), ("e", "d"), ("e", "G")),
            DependencyGraph._rules_until_first_memento_fn(fn_e),
        )

        # noinspection PyTypeChecker
        validate(
            (("c", "b"), ("b", "e"), ("b", "f"), ("b", "a")),
            DependencyGraph._rules_until_first_memento_fn(fn_c),
        )

    def test_complex_graph_verbose(self):
        # noinspection PyTypeChecker
        self.do_test_complex_graph_verbose(fn_a, fn_c, fn_d, fn_e, fn_f)

    @staticmethod
    def do_test_complex_graph_verbose(a, c, d, e, f):
        deps = c.dependencies(verbose=True)  # type: DependencyGraph
        expected_df = (
            pd.DataFrame(
                data=[
                    {
                        "src": a.qualified_name_without_version,
                        "target": e.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": "twosigma.memento.runner_test:fn_b",
                        "target": a.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": "twosigma.memento.runner_test:fn_b",
                        "target": e.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": "twosigma.memento.runner_test:fn_b",
                        "target": f.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": c.qualified_name_without_version,
                        "target": "twosigma.memento.runner_test:fn_b",
                        "type": "Function",
                    },
                    {
                        "src": d.qualified_name_without_version,
                        "target": a.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": e.qualified_name_without_version,
                        "target": "fn_G",
                        "type": "GlobalVariable",
                    },
                    {
                        "src": e.qualified_name_without_version,
                        "target": c.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": e.qualified_name_without_version,
                        "target": d.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                ]
            )
            .sort_values(by=["src", "target"])
            .reset_index(drop=True)
        )
        actual_df = deps.df().sort_values(by=["src", "target"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_complex_graph(self):
        # noinspection PyTypeChecker
        self.do_test_complex_graph(fn_a, fn_c, fn_d, fn_e, fn_f)

    @staticmethod
    def do_test_complex_graph(a, c, d, e, f):
        deps = c.dependencies()  # type: DependencyGraph
        expected_df = (
            pd.DataFrame(
                data=[
                    {
                        "src": a.qualified_name_without_version,
                        "target": e.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": c.qualified_name_without_version,
                        "target": a.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": c.qualified_name_without_version,
                        "target": e.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": c.qualified_name_without_version,
                        "target": f.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": d.qualified_name_without_version,
                        "target": a.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": e.qualified_name_without_version,
                        "target": c.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                    {
                        "src": e.qualified_name_without_version,
                        "target": d.qualified_name_without_version,
                        "type": "MementoFunction",
                    },
                ]
            )
            .sort_values(by=["src", "target"])
            .reset_index(drop=True)
        )
        actual_df = deps.df().sort_values(by=["src", "target"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_df, actual_df)
