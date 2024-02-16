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
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

from twosigma.memento import FunctionReference
from twosigma.memento.base import MementoFunctionBase
from twosigma.memento.code_hash import HashRule
from twosigma.memento.context import InvocationContext
from twosigma.memento.dependency_graph import DependencyGraph
from twosigma.memento.types import MementoFunctionType, DependencyGraphType

_registered_function_type_classes = {}  # type: Dict[str, type(MementoFunctionBase)]


class ExternalMementoFunctionBase(MementoFunctionBase, ABC):
    """
    Stub to a MementoFunction that is external to this process.
    """

    _fn_reference = None  # type: FunctionReference
    _version = None  # type: str

    code_hash = None  # type: Optional[str]
    "Hash of the code of the function. Only computed if version is not specified."

    qualified_name_without_version = None  # type: str
    """
    Qualified name of the function, except for the version. Use fn_reference().qualified_name
    to include the version number.
    """

    @property
    def partial_args(self):
        """If this is a partial function, the arguments for the partial function"""
        return self._fn_reference.partial_args

    @property
    def partial_kwargs(self):
        """If this is a partial function, the keyword arguments for the partial function"""
        return self._fn_reference.partial_kwargs

    @property
    def cluster_name(self):
        """Name of the cluster to which this function belongs"""
        return self._fn_reference.cluster_name

    context = None  # type: InvocationContext
    "Invocation context for this function"

    _hash_rules = None  # type: List[HashRule]

    def __init__(
        self,
        fn_reference: FunctionReference,
        context: InvocationContext,
        function_type: str,
        hash_rules: List[HashRule],
    ):
        """
        Creates a new ExternalMementoFunction for the given function reference.

        :param fn_reference: Reference to the external function, must have `external=True`
        :param context: Invocation context
        :param function_type:  One of the registered external function types
        :param hash_rules:  Pre-computed list of hash rules, used for dependency graphs
        """
        super(ExternalMementoFunctionBase, self).__init__()
        assert fn_reference.external, "FunctionReference must be external"
        self._fn_reference = fn_reference
        parts = FunctionReference.parse_qualified_name(fn_reference.qualified_name)
        self._version = parts["version"]
        self.context = context
        self.qualified_name_without_version = (
            self._fn_reference.qualified_name_without_version
        )
        self.code_hash = None
        self.function_type = function_type
        self._hash_rules = hash_rules

    def _clone_fn_ref(
        self,
        fn: Callable = None,
        src_fn: Callable = None,
        cluster_name: str = None,
        version: str = None,
        calculated_version: str = None,
        partial_args: Tuple[Any] = None,
        partial_kwargs: Dict[str, Any] = None,
        auto_dependencies: bool = True,
        dependencies: List[Union[str, "MementoFunctionType"]] = None,
        version_code_hash: str = None,
        version_salt: str = None,
    ) -> FunctionReference:
        assert (
            fn is None
        ), "External function may not refer to a function in the local process"
        assert src_fn is None, (
            "External function may not refer to a source function in the "
            "local process"
        )
        assert (
            calculated_version is None
        ), "External functions always have fixed versions"
        assert (
            auto_dependencies
        ), "Cannot disable auto_dependencies for external functions"
        assert dependencies is None, "Cannot set dependencies for external functions"
        assert (
            version_code_hash is None
        ), "Cannot set version code hash for external functions"
        assert version_salt is None, "Cannot set version_salt for external functions"
        return FunctionReference(
            memento_fn=self,
            cluster_name=cluster_name or self._fn_reference.cluster_name,
            version=version or self._version,
            partial_args=partial_args or self._fn_reference.partial_args,
            partial_kwargs=partial_kwargs or self._fn_reference.partial_kwargs,
            module_name=self._fn_reference.module,
            function_name=self._fn_reference.function_name,
            parameter_names=self._fn_reference.parameter_names,
            external=True,
        )

    def hash_rules(self) -> List[HashRule]:
        return self._hash_rules

    def version(self) -> str:
        return self._version

    def fn_reference(self):
        return self._fn_reference

    def dependencies(
        self, verbose=False, label_filter: Callable[[str], str] = None
    ) -> DependencyGraphType:
        return DependencyGraph(self, verbose=verbose, label_filter=label_filter)

    def _filter_call(self, *args, **kwargs) -> Any:
        raise RuntimeError("Cannot filter calls for external functions")

    @classmethod
    def register(cls, function_type, clazz):
        """
        Static method to register a new type of external function. This is
        used by the client to instantiate the correct handler for the function given its type.

        Note that if two providers register a backend with the same name,
        the last one to register wins.

        :param function_type: The name of the function type (lowercase)
        :param clazz: The class to handle functions of this type. Must have a constructor
            with signature `(fn_reference: FunctionReference, context: InvocationContext)`
        """
        global _registered_function_type_classes
        _registered_function_type_classes[function_type] = clazz

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "ExternalMementoFunction({})".format(repr(self.fn_reference()))

    @classmethod
    def get_registered_function_type_classes(
        cls,
    ) -> Dict[str, type(MementoFunctionBase)]:
        return _registered_function_type_classes


class UnboundExternalMementoFunction(ExternalMementoFunctionBase):
    """
    ExternalMementoFunction which is not bound to a particular server endpoint.
    """

    def __init__(
        self,
        context: Optional[InvocationContext] = None,
        cluster_name: Optional[str] = None,
        module_name: Optional[str] = None,
        function_name: Optional[str] = None,
        version: Optional[str] = None,
        partial_args: Optional[Tuple[Any]] = None,
        partial_kwargs: Optional[Dict[str, Any]] = None,
        parameter_names: Optional[List[str]] = None,
        fn_reference: Optional[FunctionReference] = None,
    ):
        assert fn_reference or cluster_name is not None, "Cluster name is required"

        if fn_reference is None:
            fn_reference = FunctionReference(
                memento_fn=self,
                cluster_name=cluster_name,
                module_name=module_name,
                function_name=function_name,
                version=version,
                partial_args=partial_args,
                partial_kwargs=partial_kwargs,
                parameter_names=parameter_names,
                external=True,
            )

        if context is None:
            context = InvocationContext()

        super().__init__(fn_reference, context, "unbound", hash_rules=list())

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
        dependencies: List[Union[str, MementoFunctionType]] = None,
        version_code_hash: str = None,
        version_salt: str = None,
    ) -> MementoFunctionType:
        fn_ref = self._clone_fn_ref(
            fn=fn,
            src_fn=src_fn,
            cluster_name=cluster_name,
            version=version,
            calculated_version=calculated_version,
            partial_args=partial_args,
            partial_kwargs=partial_kwargs,
            auto_dependencies=auto_dependencies,
            dependencies=dependencies,
            version_code_hash=version_code_hash,
            version_salt=version_salt,
        )
        return UnboundExternalMementoFunction(
            context=context or self.context, fn_reference=fn_ref
        )


ExternalMementoFunctionBase.register("unbound", UnboundExternalMementoFunction)
