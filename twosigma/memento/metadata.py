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
from functools import lru_cache

import datetime
import sys

import graphviz
import pandas as pd
import numpy as np
import inspect
from typing import List, Dict, Optional, MutableSet, ClassVar, Set
from enum import Enum

from .logging import log
from .resource import ResourceHandle
from .exception import MementoException
from .partition import Partition
from .reference import (
    FunctionReferenceWithArguments,
    FunctionReference,
    FunctionReferenceWithArgHash,
)
from .types import VersionedDataSourceKey, MementoFunctionType


class ResultType(Enum):
    exception = (0,)
    null = (1,)
    boolean = (2,)
    string = (3,)
    binary = (4,)
    number = (5,)
    date = (6,)
    timestamp = (7,)
    list_result = (8,)  # list is a reserved word
    dictionary = (9,)
    array_boolean = (10,)
    array_int8 = (11,)
    array_int16 = (12,)
    array_int32 = (13,)
    array_int64 = (14,)
    array_float32 = (15,)
    array_float64 = (16,)
    index = (17,)
    series = (18,)
    data_frame = (19,)
    partition = (20,)
    memento_function = 21  # not a valid return type, but valid argument type

    @staticmethod
    def from_object(obj) -> "ResultType":
        if isinstance(obj, MementoException):
            return ResultType.exception
        if obj is None:
            return ResultType.null
        if isinstance(obj, bool):
            return ResultType.boolean
        if isinstance(obj, str):
            return ResultType.string
        if isinstance(obj, bytes):
            return ResultType.binary
        if isinstance(obj, int) or isinstance(obj, float):
            return ResultType.number
        if isinstance(obj, complex):
            raise ValueError("Memento cannot [de]serialize a complex")
        if isinstance(
            obj, datetime.datetime
        ):  # pd.Timestamp also extends datetime.datetime
            return ResultType.timestamp
        if isinstance(obj, datetime.date):
            return ResultType.date
        if isinstance(obj, list):
            return ResultType.list_result
        if isinstance(obj, dict):
            return ResultType.dictionary
        if isinstance(obj, pd.Index):
            return ResultType.index
        if isinstance(obj, pd.Series):
            return ResultType.series
        if isinstance(obj, pd.DataFrame):
            return ResultType.data_frame
        if isinstance(obj, np.ndarray):
            if obj.dtype == "bool":
                return ResultType.array_boolean
            if obj.dtype == "int8":
                return ResultType.array_int8
            if obj.dtype == "int16":
                return ResultType.array_int16
            if obj.dtype == "int32":
                return ResultType.array_int32
            if obj.dtype == "int64":
                return ResultType.array_int64
            if obj.dtype == "float32":
                return ResultType.array_float32
            if obj.dtype == "float64":
                return ResultType.array_float64
            raise ValueError(
                "Memento cannot [de]serialize a ndarray of type {}".format(obj.dtype)
            )
        if isinstance(obj, Partition):
            return ResultType.partition
        raise ValueError("Memento cannot [de]serialize a {}".format(type(obj)))

    @staticmethod
    def from_annotation(annotation: ClassVar) -> "ResultType":
        """
        Converts a given python class to a generic memento type representation.
        :param annotation:  Annotation of a function/variable
        :return:            Memento type representation (e.g. int becomes number)
        """

        def check_type(t):
            if hasattr(annotation, "__origin__") and annotation.__origin__ is not None:
                return issubclass(annotation.__origin__, t)
            if not inspect.isclass(annotation):
                raise ValueError("{} is not a class".format(annotation))
            return issubclass(annotation, t)

        # noinspection PyProtectedMember
        if annotation is None or annotation == inspect._empty:
            return ResultType.null
        if check_type(MementoException):
            return ResultType.exception
        if check_type(bool):
            return ResultType.boolean
        if check_type(str):
            return ResultType.string
        if check_type(bytes):
            return ResultType.binary
        if check_type(int) or check_type(float):
            return ResultType.number
        if check_type(complex):
            raise ValueError("Memento cannot [de]serialize a complex")
        if check_type(datetime.datetime):  # pd.Timestamp extends datetime.datetime
            return ResultType.timestamp
        if check_type(datetime.date):
            return ResultType.date
        if check_type(list) or check_type(List):
            return ResultType.list_result
        if check_type(dict) or check_type(Dict):
            return ResultType.dictionary
        if check_type(pd.Index):
            return ResultType.index
        if check_type(pd.Series):
            return ResultType.series
        if check_type(pd.DataFrame):
            return ResultType.data_frame
        # Not sure how to represent np.ndarray type hints
        if check_type(Partition):
            return ResultType.partition
        if check_type(MementoFunctionType):
            return ResultType.memento_function
        raise ValueError("Memento cannot [de]serialize a {}".format(type(annotation)))


class InvocationMetadata:
    """
    Defines metadata that is tracked for invocations made during a
    memento function call. This is a subset of the metadata
    tracked for a top-level call and is needed to ensure there is
    no stale data upstream.

    The metadata includes the full call tree so that transitive dependencies
    and a full graph visualization can be retrieve in O(1) time.

    The metadata also includes a list of resources that are depended-on, as
    detected by calls to resource functions.

    """

    fn_reference_with_args = None  # type: FunctionReferenceWithArguments
    invocations = None  # type: List[FunctionReferenceWithArguments]
    resources = None  # type: List[ResourceHandle]
    runtime = None  # type: Optional[datetime.timedelta]
    result_type = None  # type: Optional[ResultType]

    def __sizeof__(self):
        return (
            sys.getsizeof(self.fn_reference_with_args)
            + sum([sys.getsizeof(x) for x in self.invocations])
            + sum([sys.getsizeof(x) for x in self.resources])
            + sys.getsizeof(self.runtime)
            + sys.getsizeof(self.result_type)
        )

    def __init__(
        self,
        fn_reference_with_args: FunctionReferenceWithArguments,
        invocations: List[FunctionReferenceWithArguments],
        resources: List[ResourceHandle],
        runtime: Optional[datetime.timedelta],
        result_type: Optional[ResultType],
    ):
        self.fn_reference_with_args = fn_reference_with_args
        self.invocations = invocations
        self.resources = resources
        self.runtime = runtime
        self.result_type = result_type

    def __repr__(self):
        return (
            "InvocationMetadata(fn_reference_with_args={}, invocations={}, runtime={}, "
            "result_type={})".format(
                repr(self.fn_reference_with_args),
                repr(self.invocations),
                repr(self.runtime),
                repr(self.result_type),
            )
        )

    def __str__(self):
        return self.__repr__()


def _label_fn_ref_args(fn_with_args: FunctionReferenceWithArguments) -> str:
    kwarg_str = ""
    for key, value in fn_with_args.effective_kwargs.items():
        if kwarg_str != "":
            kwarg_str += ", "
        kwarg_str += key + "=" + repr(value)

    return kwarg_str


class Memento:
    """
    Defines the metadata that is tracked for each function invocation.

    """

    time = None  # type: datetime.datetime
    invocation_metadata = None  # type: InvocationMetadata
    function_dependencies = None  # type: MutableSet[FunctionReference]
    runner = None  # type: Dict[str, object]
    correlation_id = None  # type: str
    content_key = None  # type: Optional[VersionedDataSourceKey]
    """If specified, this key overrides the default key for a Memento result"""

    def __init__(
        self,
        time: datetime.datetime,
        invocation_metadata: InvocationMetadata,
        function_dependencies: MutableSet[FunctionReference],
        runner: Dict[str, object],
        correlation_id: str,
        content_key: Optional[VersionedDataSourceKey],
    ):
        """
        Creates a new Memento, tracking the metadata for the function invocation.

        :param time:                        The time the function was invoked
        :param invocation_metadata:         Metadata about the invocation
        :param runner:                      String representation of the metadata of the runner
                                            that was used to run this function
        :param correlation_id:              Identifier used to trace the root request that caused
                                            this result to be computed. The same correlation id
                                            will appear for each memento that was part of the same
                                            root request.
        :param content_key:                 Location of the full contents of the result, allowing
                                            content-addressable storage.

        """
        self.time = time
        self.invocation_metadata = invocation_metadata
        self.function_dependencies = function_dependencies
        self.runner = runner
        self.correlation_id = correlation_id
        self.content_key = content_key

    def forget(self):
        """
        Forget the memoized results.
        """

        from .configuration import Environment

        env = Environment.get()

        fn_reference_with_args = self.invocation_metadata.fn_reference_with_args
        fn_reference = fn_reference_with_args.fn_reference
        cluster_config = env.get_cluster(fn_reference.cluster_name)
        storage_backend = cluster_config.storage

        # Forget only the results for the provided parameters
        arg_hash = fn_reference_with_args.fn_reference_with_arg_hash()
        log.info(
            "Forgetting {} for arg hash {}".format(
                fn_reference.qualified_name, arg_hash
            )
        )
        storage_backend.forget_call(arg_hash)

    def forget_exceptions_recursively(self, dry_run=False):
        """
        Forget an exceptional result in this function and its recursive invocations.

        For cases where a memento function raises an exception it is often useful to
        forget that exception as well as any exceptions that were raised in downstream
        dependents.

        If this memento is not an exceptional result, nothing is done. Otherwise,
        the invocation chain is scanned recursively to find all exceptional results
        and each of these is forgotten, clearing the way for a re-computation.

        Use this method with caution as it may affect analysis and reproducibility of
        other call chains that depend on one or more of the same dependents.

        :param dry_run:  If `True`, does not delete, only logs warnings about what would be
            forgotten.
        """

        from .configuration import Environment

        env = Environment.get()

        warned = set()  # type: Set[str]
        to_forget = set()  # type: Set[FunctionReferenceWithArguments]

        @lru_cache(maxsize=1024)
        def get_cluster(fn_reference: FunctionReference):
            cluster_name = fn_reference.cluster_name
            result = env.get_cluster(cluster_name)
            if result is None:
                if cluster_name not in warned:
                    warned.add(cluster_name)
                    log.warning(
                        f"Cannot find cluster {cluster_name} in default environment"
                    )
            return result

        @lru_cache(maxsize=10240)
        def get_memento(
            cluster_storage, fn_with_arg_hash: FunctionReferenceWithArgHash
        ) -> Memento:
            return cluster_storage.get_memento(fn_with_arg_hash)

        def add_result(invocation: InvocationMetadata):
            if invocation.result_type != ResultType.exception:
                return

            to_forget.add(invocation.fn_reference_with_args)

            for inv in invocation.invocations:
                fn_cluster = get_cluster(inv.fn_reference)
                if fn_cluster is not None:
                    memento = get_memento(
                        fn_cluster.storage, inv.fn_reference_with_arg_hash()
                    )
                    if memento is not None:
                        add_result(memento.invocation_metadata)

        add_result(self.invocation_metadata)

        for fn_with_args in to_forget:
            if not dry_run:
                cluster = get_cluster(fn_with_args.fn_reference)
                if cluster is not None:
                    cluster.storage.forget_call(
                        fn_with_args.fn_reference_with_arg_hash()
                    )
            log.warning(f"{'Would forget' if dry_run else 'Forgot'} {fn_with_args}")

    def trace(self, max_depth=None, only_exceptions=False) -> str:
        """
        Return an ASCII string containing a visualization of the call stack
        and runtime statistics for a given memento.

        :param max_depth:  Maximum call depth for trace, inclusive of root node
        :param only_exceptions:  If `True`, only trace mementos with exception results
        """

        from .configuration import Environment

        env = Environment.get()

        warned = set()  # type: Set[str]

        def label(m: InvocationMetadata) -> str:
            """
            Convert the given invocation metadata to a label for the stack trace

            """
            return "{}({}) [{}] -> {}".format(
                m.fn_reference_with_args.fn_reference.qualified_name,
                _label_fn_ref_args(m.fn_reference_with_args),
                m.runtime,
                str(m.result_type),
            )

        @lru_cache(maxsize=10240)
        def get_memento(
            cluster_storage, fn_with_arg_hash: FunctionReferenceWithArgHash
        ) -> Memento:
            return cluster_storage.get_memento(fn_with_arg_hash)

        def add_row(rows: List[str], indent: int, invocation: InvocationMetadata):
            rows.append((" " * (indent * 4)) + label(invocation))

            for invocation in invocation.invocations:
                cluster_name = invocation.fn_reference.cluster_name
                cluster = env.get_cluster(cluster_name)
                if cluster is None:
                    if cluster_name not in warned:
                        warned.add(cluster_name)
                        log.warning(
                            f"Cannot find cluster {cluster_name} in default environment"
                        )
                elif max_depth is None or indent < (max_depth - 1):
                    memento = get_memento(
                        cluster.storage, invocation.fn_reference_with_arg_hash()
                    )
                    if memento is not None and (
                        not only_exceptions
                        or memento.invocation_metadata.result_type
                        == ResultType.exception
                    ):
                        add_row(rows, indent + 1, memento.invocation_metadata)
                elif not only_exceptions and (
                    max_depth is not None and indent >= (max_depth - 1)
                ):
                    rows.append(
                        (" " * ((indent + 1) * 4))
                        + "{}({})".format(
                            invocation.fn_reference.qualified_name,
                            _label_fn_ref_args(invocation),
                        )
                    )

        result_rows = []
        add_row(result_rows, 0, self.invocation_metadata)

        return "\n".join(result_rows)

    def graph(self, max_depth=None, only_exceptions=False) -> graphviz.Digraph:
        """
        Return a graphviz digraph containing a visualization of the call stack for a given memento.

        :param max_depth:  Maximum call depth for trace, inclusive of root node
        :param only_exceptions:  If `True`, only trace mementos with exception results
        """

        from .configuration import Environment

        env = Environment.get()

        graph = graphviz.Digraph(
            graph_attr={"rankdir": "LR", "splines": "ortho"},
            node_attr={"shape": "box", "fontname": "Helvetica", "fontsize": "10"},
        )

        node_id_holder = [0]
        warned = set()  # type: Set[str]

        @lru_cache(maxsize=10240)
        def get_memento(
            cluster_storage, fn_with_arg_hash: FunctionReferenceWithArgHash
        ) -> Memento:
            return cluster_storage.get_memento(fn_with_arg_hash)

        def graph_node(
            fn_reference_with_args: FunctionReferenceWithArguments,
            m: Optional[InvocationMetadata],
        ):
            node_id_holder[0] += 1
            node_id = "n" + str(node_id_holder[0])
            kwarg_str = _label_fn_ref_args(fn_reference_with_args)
            kwarg_short = kwarg_str
            max_len = 40
            if len(kwarg_str) > max_len:
                kwarg_short = kwarg_str[0:max_len] + "..."
            if m is not None:
                label = "{}({})\n-> {}\n{}".format(
                    fn_reference_with_args.fn_reference.function_name,
                    kwarg_short,
                    m.result_type.name,
                    m.runtime,
                )
            else:
                label = "{}({})".format(
                    fn_reference_with_args.fn_reference.function_name, kwarg_short
                )
            tooltip = "{}({})".format(
                fn_reference_with_args.fn_reference.function_name, kwarg_str
            )
            graph.node(node_id, label=label, tooltip=tooltip)
            return node_id

        def add_node(m: InvocationMetadata, depth: int) -> str:
            node_id = graph_node(m.fn_reference_with_args, m)

            for invocation in m.invocations:
                cluster_name = invocation.fn_reference.cluster_name
                cluster = env.get_cluster(cluster_name)
                if cluster is None:
                    if cluster_name not in warned:
                        warned.add(cluster_name)
                        log.warning(
                            f"Cannot find cluster {cluster_name} in default environment"
                        )
                elif max_depth is None or depth < (max_depth - 1):
                    cluster_storage = cluster.storage
                    memento = get_memento(
                        cluster_storage, invocation.fn_reference_with_arg_hash()
                    )

                    if memento is not None and (
                        not only_exceptions
                        or memento.invocation_metadata.result_type
                        == ResultType.exception
                    ):
                        other_node_id = add_node(memento.invocation_metadata, depth + 1)
                        graph.edge(node_id, other_node_id)
                elif not only_exceptions and (
                    max_depth is not None and depth >= (max_depth - 1)
                ):
                    other_node_id = graph_node(invocation, None)
                    graph.edge(node_id, other_node_id)

            return node_id

        add_node(self.invocation_metadata, 0)

        return graph

    def __sizeof__(self):
        return (
            sys.getsizeof(self.time)
            + sys.getsizeof(self.invocation_metadata)
            + sys.getsizeof(self.runner)
            + sys.getsizeof(self.correlation_id)
            + sys.getsizeof(self.content_key)
        )

    def __repr__(self):
        return (
            "Memento(time={}, invocation_metadata={}, function_dependencies={}, runner={}, "
            "correlation_id={}, content_key={})".format(
                repr(self.time),
                repr(self.invocation_metadata),
                repr(self.function_dependencies),
                repr(self.runner),
                repr(self.correlation_id),
                repr(self.content_key),
            )
        )

    def __str__(self):
        return self.__repr__()
