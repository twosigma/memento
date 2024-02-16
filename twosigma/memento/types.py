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
from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd
from graphviz import Digraph
from typing import Callable, Tuple, Any, Dict, Union, List, Optional, Set

from twosigma.memento.context import InvocationContext


ContentAddressableHash = namedtuple("ContentAddressableHash", ["key"])
DataSourceKey = namedtuple("DataSourceKey", ["key"])
VersionedDataSourceKey = namedtuple("VersionedDataSourceKey", ["key", "version"])


class DependencyGraphType(ABC):
    """
    Common base class for the return type from `MementoFunctionType.dependencies`.
    """

    @abstractmethod
    def transitive_memento_fn_dependencies(self) -> Set["MementoFunctionType"]:
        pass

    @abstractmethod
    def direct_memento_fn_dependencies(self) -> Set["MementoFunctionType"]:
        pass

    @abstractmethod
    def graph(self) -> Digraph:
        pass

    @abstractmethod
    def df(self):
        pass

    @abstractmethod
    def with_verbose(self, verbose: bool) -> "DependencyGraphType":
        pass

    @abstractmethod
    def with_label_filter(
        self, label_filter: Callable[[str], str]
    ) -> "DependencyGraphType":
        pass


class MementoFunctionType(ABC):
    """
    This is an interface for all Memento Functions. It also serves as a marker class to allow
    testing if an object is an instance of MementoFunction without resolving circular references.

    """

    fn = None  # type: Callable
    src_fn = None  # type: Callable
    function_type = None  # type: str
    code_hash = None  # type: Optional[str]
    qualified_name_without_version = None  # type: str
    context = None  # type: InvocationContext
    partial_args = None  # type: Tuple[Any]
    partial_kwargs = None  # type: Dict[str, Any]
    cluster_name = None  # type: str
    required_dependencies = None  # type: Set[str]
    detected_dependencies = None  # type: Set[str]
    auto_dependencies = None  # type: bool
    explicit_version = None  # type: Optional[str]

    @abstractmethod
    def hash_rules(self) -> List:
        pass

    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def fn_reference(self):
        pass

    @abstractmethod
    def clone_with(
        self,
        fn: Callable = None,
        src_fn: Callable = None,
        cluster_name: str = None,
        version: str = None,
        calculated_version: str = None,
        context: InvocationContext = None,
        partial_args: Tuple[Any] = None,
        partial_kwargs: Dict[str, Any] = None,
        auto_dependencies: bool = True,
        dependencies: List[Union[str, "MementoFunctionType"]] = None,
        version_code_hash: str = None,
        version_salt: str = None,
    ) -> "MementoFunctionType":
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def call_batch(
        self, kwargs_list: List[Dict[str, Any]], raise_first_exception=True
    ) -> List[Any]:
        pass

    @abstractmethod
    def map_over_range(self, **kwargs) -> Dict:
        pass

    @abstractmethod
    def forget(self, *args, **kwargs):
        pass

    @abstractmethod
    def forget_all(self):
        pass

    @abstractmethod
    def get_metadata(self, key: str, args=None, kwargs=None) -> Optional[bytes]:
        pass

    @abstractmethod
    def with_prevent_further_calls(self, prevent_calls: bool):
        pass

    @abstractmethod
    def with_context_args(self, context_args: Dict[str, Any]):
        pass

    @abstractmethod
    def ignore_result(self, ignore: bool = True):
        pass

    @abstractmethod
    def list_mementos(self, limit: int = None):
        pass

    @abstractmethod
    def logs(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def memento(self, *args, **kwargs):
        pass

    @abstractmethod
    def monitor_progress(self, monitor: bool = True):
        pass

    @abstractmethod
    def partial(self, *partial_args, **partial_kwargs) -> "MementoFunctionType":
        pass

    @abstractmethod
    def put_metadata(self, key: str, value: bytes, *args, **kwargs):
        pass

    @abstractmethod
    def force_local(self, local: bool = True):
        pass

    @abstractmethod
    def dependencies(
        self, verbose=False, label_filter: Callable[[str], str] = None
    ) -> DependencyGraphType:
        pass

    @abstractmethod
    def list(self, *args, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def _filter_call(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def _call_with_contexts_from_context_args(self, fn: Callable) -> Any:
        pass


class FunctionNotFoundError(ValueError):
    """
    Thrown if there was an error mapping a stored reference to a function
    to an actual function.

    """

    pass
