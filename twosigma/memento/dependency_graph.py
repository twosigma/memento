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
import hashlib
import html
import pandas as pd

from typing import Set, Optional, Tuple, Dict, Callable, List
from graphviz import Digraph
from .code_hash import HashRule
from .types import MementoFunctionType, DependencyGraphType

"""
This module builds a transitive dependency graph of a `MementoFunctionType` and provides tools for
analyzing that graph, such as producing a graphviz diagram.

"""


def _parse_name(name: str) -> Tuple[str, str, str]:
    cluster = None
    module = None
    if "::" in name:
        (cluster, name) = name.split("::")
    if ":" in name:
        (module, name) = name.split(":")
    fn_name = name
    return cluster, module, fn_name


class Node:
    id = None  # type: Optional[str]
    label = None  # type: Optional[str]
    node_type = None  # type: Optional[str]
    edges = None  # type: Set[str]

    def __init__(self):
        self.id = None
        self.label = None
        self.node_type = None
        self.edges = set()


class DependencyGraph(DependencyGraphType):
    """
    Graph of dependencies of this function on other memento functions, plain functions and
    global variables. When invoked from Jupyter, and if graphviz is installed, this will draw
    a graphviz graph.

    The results of `transitive_memento_fn_dependencies` and `direct_memento_fn_dependencies`
    can be directly derived from the list of HashRules and so they compute quickly.

    The `df` and `graph` functions can be called to produce a complete graph
    which includes all edges. These graphs take slightly longer to produce and are derived by
    traversing the direct dependencies at each level. The verbose flag determines whether
    details like local functions and global variables are represented in the graph.
    """

    memento_fn = None  # type: MementoFunctionType
    _graph = None  # type: Optional[Dict[str, Node]]
    _verbose = None  # type: bool
    _all_rules = None  # type: List[HashRule]
    _label_filter = None  # type: Callable[[str], str]

    def __init__(
        self,
        memento_fn: MementoFunctionType = False,
        verbose: bool = False,
        label_filter: Callable[[str], str] = None,
    ):
        """
        Creates a new dependency graph for the given Memento Function

        :param memento_fn:  The function that serves as the root of the graph.
        :param verbose:  If `True`, captures all dependencies in the graph, else just memento
                         functions.
        :param label_filter:  If provided, this function will be called to transform the
            name of the label while rendering the graph.

        """
        self.memento_fn = memento_fn
        self._verbose = verbose
        self._all_rules = self.memento_fn.hash_rules()
        self._label_filter = label_filter
        self._graph = None

    def with_verbose(self, verbose: bool) -> "DependencyGraphType":
        return DependencyGraph(self.memento_fn, verbose, self._label_filter)

    def with_label_filter(
        self, label_filter: Callable[[str], str]
    ) -> "DependencyGraphType":
        return DependencyGraph(self.memento_fn, self._verbose, label_filter)

    def transitive_memento_fn_dependencies(self) -> Set[MementoFunctionType]:
        """The set of all transitive dependencies on other Memento functions"""
        # noinspection PyUnresolvedReferences
        return set(
            rule.memento_fn
            for rule in self._all_rules
            if hasattr(rule, "memento_fn") and rule.memento_fn != self.memento_fn
        )

    def direct_memento_fn_dependencies(self) -> Set[MementoFunctionType]:
        """The set of all direct dependencies on other Memento functions"""

        def is_direct_dependency(rule):
            is_memento_fn = hasattr(rule, "memento_fn") and rule.memento_fn is not None
            return (
                is_memento_fn
                and rule.memento_fn != self.memento_fn
                and rule.first_level
            )

        # noinspection PyTypeChecker,PyUnresolvedReferences
        return set(
            rule.memento_fn for rule in self._all_rules if is_direct_dependency(rule)
        )

    @classmethod
    def _rules_until_first_memento_fn(
        cls, memento_fn: MementoFunctionType
    ) -> List[HashRule]:
        """
        Set of all hash rules up through the first MementoFunctions in the stack but not beyond.

        This is useful for constructing the full dependency graphs.
        """
        result = list()  # type: List[HashRule]
        already_processed = set()  # type: Set[str]
        all_rules = memento_fn.hash_rules()  # type: List[HashRule]
        more_rules = list(r for r in all_rules if r.first_level)

        while more_rules:
            rules = more_rules
            more_rules = []
            for rule in rules:
                (node_type, _, node_name) = cls.parse_key(rule.key)
                # noinspection PyUnresolvedReferences
                if hasattr(rule, "memento_fn") and rule.memento_fn is not None:
                    if (
                        rule.memento_fn.qualified_name_without_version
                        == memento_fn.qualified_name_without_version
                    ):
                        # Exclude current function from result
                        continue
                else:
                    # add any direct parents to the list of rules to scan
                    for r in all_rules:
                        # Note: Converges to O(n^2) in degenerate very flat graphs
                        # This should be very small n in almost all realistic cases.
                        if node_name == r.parent_symbol:
                            if (
                                r.key not in already_processed
                            ):  # prevent infinite loop on cycle
                                already_processed.add(r.key)
                                more_rules.append(r)
                result.append(rule)

        return result

    @staticmethod
    def generate_graphviz(
        graph: Dict[str, Node], qualified_name_without_version: str
    ) -> Digraph:
        digraph = Digraph(
            format="svg",
            graph_attr={"rankdir": "BT"},
            node_attr={"shape": "box", "fontname": "Helvetica", "fontsize": "10"},
        )

        type_to_attrs = {
            "MementoFunction": {"shape": "rectangle"},
            "Function": {"shape": "rectangle", "style": "rounded"},
            "GlobalVariable": {"shape": "ellipse"},
            "UndefinedSymbol": {"shape": "octagon"},
        }

        def hash_name(name: str) -> str:
            return hashlib.md5(name.encode("utf-8")).hexdigest()[0:8]

        (_, _, root_fn_name) = DependencyGraph.parse_key(qualified_name_without_version)
        (root_cluster, root_module, _) = _parse_name(root_fn_name)

        for node in graph.values():
            # Parse name and only show cluster and module if different from root node
            (cluster, module, fn_name) = _parse_name(node.id)
            label = "<"
            label += (
                html.escape(cluster) + "<br/>"
                if cluster is not None and cluster != root_cluster
                else ""
            )
            label += (
                html.escape(module) + "<br/>"
                if module is not None and module != root_module
                else ""
            )
            label += html.escape(node.label)
            label += ">"
            attrs = type_to_attrs[
                node.node_type if node.node_type is not None else "MementoFunction"
            ]
            digraph.node(hash_name(node.id), label=label, **attrs)

            for edge in node.edges:
                digraph.edge(hash_name(node.id), hash_name(edge))

        return digraph

    def graph(self) -> Digraph:
        graph = self._get_graph()
        return DependencyGraph.generate_graphviz(
            graph, self.memento_fn.qualified_name_without_version
        )

    @staticmethod
    def generate_df(graph: Dict[str, Node]) -> pd.DataFrame:
        rows = []
        for node in graph.values():
            for edge in node.edges:
                rows.append(
                    {"src": node.id, "target": edge, "type": graph[edge].node_type}
                )

        return (
            pd.DataFrame(data=rows, columns=["src", "target", "type"])
            .sort_values(by=["src", "type"])
            .reset_index(drop=True)
        )

    def df(self):
        """
        Returns the list of transitive dependencies as a DataFrame, or None if nothing to report.

        The DataFrame will have four columns:
        * `src` (str): The name of the symbol
        * `target` (str): Name of the symbol this symbol depends on
        * `type` (str): The type of the target (`"MementoFunction"`, `"NonMementoFunction"`,
          `"GlobalVariable"` or `"UndefinedSymbol"`)
        """
        return DependencyGraph.generate_df(self._get_graph())

    def _repr_html_(self):
        return self.graph().pipe().decode("utf-8")

    @staticmethod
    def parse_key(key: str) -> Tuple[str, str, str]:
        """
        Key is of the form "type;parent;name". This function returns a
        tuple of (type, parent, name)

        """
        semi_index = key.find(";")
        semi_index_2 = key.find(";", semi_index + 1)
        node_type = key[0:semi_index]
        node_parent = key[semi_index + 1 : semi_index_2]
        node_name = key[semi_index_2 + 1 :]
        return node_type, node_parent, node_name

    def _get_graph(self) -> Dict[str, Node]:
        if self._graph is None:
            self._graph = dict()
            self.generate_graph(
                set(), self._graph, self.memento_fn, self._label_filter, self._verbose
            )

        return self._graph

    @classmethod
    def generate_graph(
        cls,
        processed: Set[str],
        graph: Dict[str, Node],
        root_fn: MementoFunctionType,
        label_filter: Callable[[str], str],
        verbose: bool,
    ):
        """
        Generates a tree of Nodes and returns the root node (which is always a
        MementoFunction node).

        If the root function was not found, returns `None`.

        :param processed:  Set of node names already processed (to prevent infinite recursion)
        :param graph:      The graph being built
        :param root_fn:    MementoFunction of the sub-graph being generated
        :param label_filter:  If specified, this Callable is called to customize the appearance
                           of labels in the graph.
        :param verbose:    If true, a verbose graph is generated, including local functions
                           and global variables. Otherwise, the graph collapses that detail and
                           focuses exclusively on MementoFunctions.
        """
        root_node_name = root_fn.qualified_name_without_version
        if root_node_name in processed:
            # Already processed - do not recurse infinitely
            return

        processed.add(root_node_name)

        # The list of HashRules is only for reachability and so it does not have all of the
        # detail needed to draw a full dependency graph. But it does have enough detail to
        # get to the first level of MementoFunction dependencies. Here, we analyze
        # the hash_rules and extract only those rules between the root and the first
        # level of MementoFunctions. Then, we recursively analyze the rest of the dependency
        # graph from there.
        first_rules = cls._rules_until_first_memento_fn(root_fn)

        def get_node(name: str, lbl_filter: Callable[[str], str]):
            n = graph.get(name)
            if n is None:
                n = Node()
                n.id = name
                (_, _, fn_name) = _parse_name(name)
                n.label = lbl_filter(name) if lbl_filter is not None else fn_name
                graph[name] = n

            return n

        if verbose:
            # Add all detail
            for rule in first_rules:
                (node_type, _, node_name) = cls.parse_key(rule.key)
                node = get_node(node_name, label_filter)
                node.node_type = node_type
                parent_node = get_node(rule.parent_symbol, label_filter)
                parent_node.edges.add(node_name)
                # noinspection PyUnresolvedReferences
                if hasattr(rule, "memento_fn") and rule.memento_fn is not None:
                    cls.generate_graph(
                        processed, graph, rule.memento_fn, label_filter, verbose
                    )
        else:
            # Only add edges from root to MementoFunctions
            root_node = get_node(root_node_name, label_filter)
            root_node.node_type = "MementoFunction"
            for rule in first_rules:
                # noinspection PyUnresolvedReferences
                if hasattr(rule, "memento_fn") and rule.memento_fn is not None:
                    (node_type, _, node_name) = cls.parse_key(rule.key)
                    node = get_node(node_name, label_filter)
                    node.node_type = node_type
                    root_node.edges.add(node_name)
                    cls.generate_graph(
                        processed, graph, rule.memento_fn, label_filter, verbose
                    )
