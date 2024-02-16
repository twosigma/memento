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

"""Core memento implementation."""
import functools
import hashlib
import inspect
from collections import namedtuple
from typing import Callable, Dict, Any, Tuple, List, Union, Optional, Set, cast

from .configuration import Environment, ENVIRONMENT_HASH_BYTES
from .base import MementoFunctionBase
from .dependency_graph import DependencyGraph
from .exception import UndeclaredDependencyError
from .call_stack import CallStack
from .context import InvocationContext
from .external import ExternalMementoFunctionBase
from .logging import log
from .types import MementoFunctionType, DependencyGraphType
from .reference import FunctionReference
from .code_hash import (
    fn_code_hash,
    resolve_to_symbolic_names,
    HashRule,
    MementoFunctionHashRule,
    list_dotted_names,
)
from .metadata import ResultType


_MementoFunctionVersionCacheEntry = namedtuple(
    "_MementoFunctionVersionCacheEntry", ["as_of_generation", "version"]
)


class MementoFunction(MementoFunctionBase):
    """
    Pure function whose execution is scheduled to run on a configurable
    :class:`memento.runner.RunnerBackend` with an
    :class:`memento.context.InvocationContext`, and whose results are
    memoized and written to a particular :class:`memento.storage.StorageBackend`.

    This class is not typically instantiated directly, but rather via
    the :func:`memento.memento_function` decorator.

    The function starts out with a default invocation context. This context
    can be modified by calling the function modifier functions (e.g.
    :meth:`force_local` which forces the function to run in the local process
    instead of the cluster's configured runner). Each modifier returns a new
    MementoFunctionType object with a modified context, which makes it easy to
    chain multiple modifiers together (e.g.
    `my_function.force_local().ignore_result().call()`). The :meth:`call`
    method is equivalent to calling the function, when done and
    looks better, syntactically, than `modifier()()`.

    If desired, the caller can force the wrapped function `fn` to be run
    using a local (in-process) runner by calling `fn.force_local().call(...)`.

    When a MementoFunctionType is defined, its code is hashed, and two sets of dependencies
    are assembled. The first is a set of detected dependencies that are auto-detected. The second
    is a set of required dependencies that are manually-specified. Detected dependencies that
    cannot be found do not prevent a function from evaluating whereas required dependencies do.
    - If auto_dependencies parameter is not specified, it defaults to True.
    - If auto_dependencies is True, the AST of the function is parsed and the list of
      names (and attributed-names) extracted. These names are added to the detected dependencies
      list.
    - If the dependencies parameter is specified, the list of manually-specified dependencies
      is added to the required dependencies list.

    """

    _global_fn_generation = 0  # type: int
    """
    Global generation number, used as an optimization to help prevent unnecessary version
    hash computation. This number is incremented every time a new function is defined, which
    forces all version hashes to be recomputed.
    """

    _global_fn_version_cache = (
        dict()
    )  # type: Dict[str, _MementoFunctionVersionCacheEntry]
    """
    Cache that maps from function name to a version cache entry that contains the
    generation number as of when this was current and the function version number.
    """

    fn = None  # type: Callable
    "The function being wrapped"

    src_fn = None  # type: Callable
    "The function containing the source code to be scanned for dependencies"

    code_hash = None  # type: Optional[str]
    "Hash of the code of the function. Only computed if version is not specified."

    qualified_name_without_version = None  # type: str
    """
    Qualified name of the function, except for the version. Use fn_reference().qualified_name
    to include the version number.
    """

    context = None  # type: InvocationContext
    "Invocation context for this function"

    partial_args = None  # type: Tuple[Any]
    "If this is a partial function, the arguments for the partial function"

    partial_kwargs = None  # type: Dict[str, Any]
    "If this is a partial function, the keyword arguments for the partial function"

    cluster_name = None  # type: str
    "Name of the cluster to which this function belongs"

    required_dependencies = None  # type: Set[str]
    "Set of manually-specified dependencies that are required for this function"

    detected_dependencies = None  # type: Set[str]
    "Set of automatically-detected dependencies for this function"

    auto_dependencies = None  # type: bool
    "If True, dependencies will be searched for automatically"

    _constructor_provided_dependencies = (
        None
    )  # type: List[Union[str, MementoFunctionType]]
    "The explicit list of dependencies provided by the user"

    _constructor_provided_version_code_hash = None  # type: str
    "The explicit version code hash provided by the user"

    _constructor_provided_version_salt = None  # type: str
    "The explicit version salt provided by the user"

    _hash_rules = None  # type: List[HashRule]

    def hash_rules(self) -> List[HashRule]:
        """Ordered list of hash rules from which the hash was computed"""
        self._update_dependencies()
        return self._hash_rules

    explicit_version = None  # type: Optional[str]
    _calculated_version = None  # type: Optional[str]

    def version(self) -> str:
        """Version of this function, usually computed using the code hash and dependencies"""
        if self.explicit_version is not None:
            return self.explicit_version

        self._update_dependencies()
        return self._calculated_version

    _fn_reference = None  # type: Optional[FunctionReference]

    def fn_reference(self) -> FunctionReference:
        """Reference to the function"""

        self._update_dependencies()
        return self._fn_reference

    def supports_kwargs(self) -> bool:
        """True if function supports kwargs, False otherwise"""
        fn = self.fn
        # noinspection PyUnresolvedReferences
        return hasattr(fn, "__code__") and (
            fn.__code__.co_flags & inspect.CO_VARKEYWORDS != 0
        )

    def get_args(self) -> List[Dict[str, str]]:
        """
        Extracts argument names and types from a function signature and return in dictionary form.

        :return:    List of dictionaries where each dictionary represents an argument which has
                    a name and an argument type.
        """
        params = inspect.signature(self.fn).parameters
        args = []
        for param_name in params:
            param = params[param_name]
            arg = {
                "name": param.name,
                "argumentType": ResultType.from_annotation(param.annotation).name,
            }
            args.append(arg)
        return args

    def __init__(
        self,
        fn: Callable,
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
        register_fn: bool = True,
    ):
        """
        Creates a new MementoFunction that wraps the provided `fn`.

        :param fn:              The function to call when this MementoFunction is invoked.
        :param src_fn:          The function that contains the relevant source code to be
                                scanned for dependencies. This is often the same as `fn` but
                                can differ if the function is wrapped. Defaults to `fn`.
        :param cluster_name:    The name of the cluster to which this function belongs.
                                This determines the storage backend mechanism for this function's
                                results. If not specified, this function will be placed in the
                                `default` cluster.
        :param version:         Version number of this function. When a version number changes, all
                                previously computed data is ignored and new data is computed.
                                Unless explicitly forgotten, the old data is still available for
                                previous callers. If no version is specified, a version is
                                automatically computed from the code hash and dependencies.
        :param calculated_version:  If a version was previously calculated for this function,
                                it can be provided.
        :param context:         The invocation context for this function. This starts out with
                                a default context, which can be modified by calling function
                                modifiers.
        :param partial_args:    If this is a partially-evaluated function, these are the partial
                                arguments to prepend.
        :param partial_kwargs:  If this is a partially-evaluated function, these are the partial
                                kwargs to prepend.
        :param auto_dependencies:  If `True`, the code of the function is analyzed in order to
                                automatically compute a list of detected dependencies on other
                                Memento functions. This is used to compute a robust version hash.
                                Only MementoFunctions are added. Defaults to `True`.
        :param dependencies:    Allows manual specification of required dependencies. Provide a
                                list of str, or MementoFunctions. Note that any
                                specified dependency that cannot be found will cause the
                                function reference computation to fail.
        :param version_code_hash:  If specified, use the provided value instead of automatically
                                computing the code hash. This allows the user to assert that
                                some code change does not affect behavior.
        :param version_salt:    If specified, add the provided value to the hash of the function.
                                This allows the user to assert that the function or one of its
                                dependencies has changed, even if the auto-detection mechanism
                                didn't detect it.
        :param register_fn:     If `True`, register this function with the environment. This is
                                set to `False` when constructing based on a modifier where
                                the original function is already registered.

        """
        assert inspect.isfunction(fn), "fn {} is not a function".format(fn)
        assert not isinstance(
            fn, MementoFunctionType
        ), "Cannot create a MementoFunction that wraps another MementoFunction"

        self._hash_rules = []  # type: List[HashRule]
        self.fn = fn
        self.src_fn = src_fn if src_fn is not None else fn
        self.function_type = "memento_function"
        self._constructor_provided_version_code_hash = version_code_hash
        self._constructor_provided_version_salt = version_salt
        self.explicit_version = version
        self._calculated_version = calculated_version
        if version is not None:
            code_hash = None
        elif version_code_hash is not None:
            code_hash = version_code_hash
        else:
            code_hash = fn_code_hash(
                fn, salt=version_salt, environment=ENVIRONMENT_HASH_BYTES
            )
        self.code_hash = code_hash
        self.context = context or InvocationContext()

        self.cluster_name = cluster_name
        self.qualified_name_without_version = ""
        if cluster_name is not None:
            self.qualified_name_without_version = cluster_name + "::"
        # noinspection PyUnresolvedReferences
        self.qualified_name_without_version += fn.__module__ + ":" + fn.__qualname__

        self.auto_dependencies = auto_dependencies

        # Resolve required dependencies to symbolic names so evaluation can be deferred.
        # This allows re-binding of functions later.
        self._constructor_provided_dependencies = dependencies
        self.required_dependencies = (
            resolve_to_symbolic_names(dependencies) if dependencies else set()
        )
        assert self.required_dependencies is None or all(
            isinstance(dep, str) for dep in self.required_dependencies
        ), "Could not resolve all functions in dependencies to symbolic names"
        self.detected_dependencies = (
            list_dotted_names(self.src_fn) if auto_dependencies else set()
        )

        self.partial_args = partial_args
        self.partial_kwargs = partial_kwargs

        self._fn_reference = None

        if "<locals>" in self.qualified_name_without_version:
            raise ValueError(
                "Memento functions must be top-level functions, "
                "not local to another function."
            )

        functools.update_wrapper(self, fn)

        if register_fn:
            # Increase generation number so other functions can update their version number
            # if necessary
            MementoFunction.increment_global_fn_generation(
                reason="registered new function {}".format(
                    self.qualified_name_without_version
                )
            )
            Environment.register_function(cluster_name, self)

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
        """Re-constructs a clone of this function, modifying one or more attributes"""
        return MementoFunction(
            fn=fn or self.fn,
            src_fn=src_fn or self.src_fn,
            cluster_name=cluster_name or self.cluster_name,
            version=version or self.version(),
            calculated_version=calculated_version or self._calculated_version,
            context=context or self.context,
            partial_args=partial_args or self.partial_args,
            partial_kwargs=partial_kwargs or self.partial_kwargs,
            auto_dependencies=auto_dependencies or self.auto_dependencies,
            dependencies=dependencies or self._constructor_provided_dependencies,
            version_code_hash=version_code_hash
            or self._constructor_provided_version_code_hash,
            version_salt=version_salt or self._constructor_provided_version_salt,
            register_fn=False,
        )

    def call(self, *args, **kwargs):
        self._validate_dependency()
        return super(MementoFunction, self).call(*args, **kwargs)

    def call_batch(
        self, kwargs_list: List[Dict[str, Any]], raise_first_exception=True
    ) -> List[Any]:
        self._validate_dependency()
        return super(MementoFunction, self).call_batch(
            kwargs_list=kwargs_list, raise_first_exception=raise_first_exception
        )

    def dependencies(
        self, verbose=False, label_filter: Callable[[str], str] = None
    ) -> DependencyGraphType:
        """
        Return an object that allows the caller to explore the dependencies of this function
        on other memento functions, plain functions and global variables. When invoked from
        Jupyter, and if graphviz is installed, this will draw a graphviz graph.

        :param verbose:  If `True`, shows more details about the dependencies.
        :param label_filter:  If specified, filters in the graph will be run through this
            function. This allows some customization of the graph output.

        """
        self._update_dependencies()
        return DependencyGraph(self, verbose=verbose, label_filter=label_filter)

    def _filter_call(self, *args, **kwargs) -> Any:
        """
        Intercept the actual call to the Memento function, updating inputs, contexts or
        outputs.

        This gets called when the MementoFunction decides there are no mementos and the
        result needs to be computed. The computation happens in the current process and
        thread. The implementation must call `super()._filter_call()` to actually call
        the function. It may modify any of the arguments before calling the super function.
        """
        return self.fn(*args, **kwargs)

    def _update_fn_reference(self):
        """Update the _fn_reference attribute based on the latest computed version"""
        self._fn_reference = FunctionReference(
            self,
            cluster_name=self.cluster_name,
            version=self.version(),
            partial_args=self.partial_args,
            partial_kwargs=self.partial_kwargs,
        )

    def _update_dependencies(self):
        """Assemble dependencies and update the version and fn_reference"""

        # If version is explicitly specified, function reference is static.
        if self.explicit_version is not None:
            if self._fn_reference is None:
                self._update_fn_reference()
            return

        # Do not recompute version if the cluster is locked
        if self._calculated_version is not None:
            cluster = Environment.get().get_cluster(cluster_name=self.cluster_name)
            if cluster is not None and cluster.locked:
                return

        # Check the version cache to see if we need to recompute the version
        entry = None  # type: Optional[_MementoFunctionVersionCacheEntry]
        if (
            self.qualified_name_without_version
            in MementoFunction._global_fn_version_cache
        ):
            entry = MementoFunction._global_fn_version_cache[
                self.qualified_name_without_version
            ]
            if entry.as_of_generation == MementoFunction._global_fn_generation:
                changed_rules = [rule for rule in self._hash_rules if rule.did_change()]
                if len(changed_rules) > 0:
                    # Global variables or local functions may have changed since the last time
                    # this function was run - check that they haven't before assuming we
                    # can use the cached version.
                    MementoFunction.increment_global_fn_generation(
                        reason="function {} hash rules changed: {}".format(
                            self.qualified_name_without_version,
                            [rule.describe() for rule in changed_rules],
                        )
                    )
                else:
                    if self._calculated_version is None:
                        self._calculated_version = entry.version()
                        self._update_fn_reference()
                    return

        # Otherwise, it needs to be calculated based on code hash and dependencies
        version = self._recompute_version()

        if self._calculated_version != version:
            self._calculated_version = version
            self._update_fn_reference()

            if entry is None or (entry is not None and entry.version != version):
                # Notify the user about the new version
                log.debug(
                    "At generation {}, calculated version for fn {} as {}.".format(
                        MementoFunction._global_fn_generation,
                        self.qualified_name_without_version,
                        version,
                    )
                )

        # Update the cache entry
        MementoFunction._global_fn_version_cache[
            self.qualified_name_without_version
        ] = _MementoFunctionVersionCacheEntry(
            as_of_generation=MementoFunction._global_fn_generation, version=version
        )

    def _recompute_version(self):
        """Collect dependencies and [re]compute the version of this function"""

        # Code hash is already computed during construction, so it does not need to be recomputed

        hash_rules = set()  # type: Set[HashRule]
        # Collect dependencies
        self_rule = MementoFunctionHashRule(
            parent_symbol=None,
            symbol=self.qualified_name_without_version,
            resolver=lambda: self,
            obj=self,
            first_level=True,
        )
        self_rule.collect_transitive_dependencies(
            result=hash_rules,
            root_fn=self,
            package_scope={inspect.getmodule(self.src_fn).__package__},
            blacklist=[memento_function],
        )

        # Order hash rules
        ordered_hash_rules = sorted(hash_rules)
        self._hash_rules = ordered_hash_rules

        # Compute hash by evaluating each hash rule
        sha256 = hashlib.sha256()
        for rule in ordered_hash_rules:
            hash_code = rule.compute_hash()
            rule.rule_hash = hash_code
            if hash_code is not None:
                sha256.update(rule.rule_hash.encode("utf-8"))
        version = sha256.hexdigest()[0:16]

        return version

    @staticmethod
    def _extract_fn_ref_args(
        caller_args: Tuple, caller_kwargs: Dict, caller_context_args: Dict
    ) -> Set[str]:
        """
        Assemble a list of arguments that are FunctionReferences, as these are valid to call.
        Note these can be nested in data structures like Lists or Dicts.

        :return: Each element of the set is the fully qualified name of a memento function

        """
        result = set()  # type: Set[str]

        def extract_refs(arg) -> Set[str]:
            refs = set()

            if isinstance(arg, MementoFunctionType):
                refs.add(arg.fn_reference().qualified_name)
            elif hasattr(arg, "__wrapped__"):
                refs |= extract_refs(arg.__wrapped__)
            elif isinstance(arg, list) or isinstance(arg, tuple):
                for element in arg:
                    refs |= extract_refs(element)
            elif isinstance(arg, dict):
                for element in arg.values():
                    refs |= extract_refs(element)

            return refs

        if caller_args is not None:
            result |= extract_refs(caller_args)

        if caller_kwargs is not None:
            result |= extract_refs(caller_kwargs)

        if caller_context_args is not None:
            result |= extract_refs(caller_context_args)

        return result

    def _validate_dependency(self):
        """
        Validate that this function has declared or detected a dependency on the provided
        function and thus it is okay to call. It is important that a function not call another
        function that is not a detected dependency so that we can ensure the version was
        computed correctly. If the version was overridden, this validation is not performed.

        If this is the top of the invocation chain, `parent` will be `None` and the invocation
        will always be valid.

        """
        frame = CallStack.get().get_calling_frame()
        if frame is None:
            # Top of stack, so no caller. Any call is allowed.
            return

        caller_ref = (
            frame.memento.invocation_metadata.fn_reference_with_args.fn_reference
        )
        caller = cast(MementoFunctionType, caller_ref.memento_fn)
        if caller.explicit_version is not None:
            # Caller has declared version explicitly, so there is no need to worry that
            # dependencies were not detected properly. Carry on.
            return

        caller_args = frame.memento.invocation_metadata.fn_reference_with_args.args
        caller_kwargs = frame.memento.invocation_metadata.fn_reference_with_args.kwargs
        caller_context_args = (
            frame.memento.invocation_metadata.fn_reference_with_args.context_args
        )
        fn_ref_args = self._extract_fn_ref_args(
            caller_args, caller_kwargs, caller_context_args
        )

        # Any dependency is a valid function
        valid_fns = {
            fn.fn_reference().qualified_name
            for fn in caller.dependencies().transitive_memento_fn_dependencies()
        }

        # Any argument that is a function reference is also valid
        valid_fns |= fn_ref_args

        if (
            caller.qualified_name_without_version != self.qualified_name_without_version
            and self.fn_reference().qualified_name not in valid_fns
        ):
            raise UndeclaredDependencyError(
                "{target} is not declared or detected to be a dependency of {src}. "
                "Solution: Add @memento_function(dependencies=[{target}]) to {src}.".format(
                    src=caller.qualified_name_without_version,
                    target=self.qualified_name_without_version,
                )
            )

    @classmethod
    def increment_global_fn_generation(cls, reason=None):
        """
        Called when a material change has been made to one or more functions definitions
        that could have an impact on the dependency tree. This forces functions to
        recompute their version based on their dependencies.

        """
        cls._global_fn_generation += 1
        log.debug(
            "New global generation{}: {}".format(
                " ({})".format(reason) if reason else "", cls._global_fn_generation
            )
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "{}.memento_fn".format(repr(self.fn_reference()))


def memento_function(
    *plain_fn,
    cluster: str = None,
    version: Any = None,
    auto_dependencies: bool = True,
    dependencies: List[Union[Callable, MementoFunctionType]] = None,
    version_code_hash: str = None,
    version_salt: str = None,
) -> Union[MementoFunctionType, Callable[..., MementoFunctionType]]:
    """
    Decorator that causes a function to be treated as a memento function.
    If it is called with the same parameters in the future, the result will be memoized and not
    need to be recomputed.

    This decorator can be invoked in one of the following ways:

    .. code-block:: python

        import twosigma.memento as m

        @m.memento_function
        def a():
            pass

        @m.memento_function()
        def b():
            pass

        @m.memento_function():
        def c():
            pass

    Once decorated, the memento function can be run as any other function.

    :param cluster:         The name of the cluster to which this function belongs.
                            This determines the storage backend mechanism for this function's
                            results. If not specified, this function will be placed in the
                            `default` cluster.
    :param version:         If specified, overrides the automatic version computation logic
                            and uses a static version string for the function. This is useful
                            for cases where the function author wants to explicitly control
                            when function changes cause recomputation. Note that the version
                            does not have to be monotonic. If version is not a string, it is
                            converted to a string. If the version is specified, a code hash is
                            not computed. Instead, the version number doubles as the code hash.
    :param auto_dependencies: See `MementoFunction` constructor
    :param dependencies:    See `MementoFunction` constructor
    :param version_code_hash:  See `MementoFunction` constructor
    :param version_salt:    See `MementoFunction` constructor

    """

    if version and not isinstance(version, str):
        version = str(version)

    # This logic is to support both @memento_function and @memento_function() consistently:
    def decorator(fn) -> MementoFunction:
        return MementoFunction(
            fn=fn,
            cluster_name=cluster,
            version=version,
            auto_dependencies=auto_dependencies,
            dependencies=dependencies,
            version_code_hash=version_code_hash,
            version_salt=version_salt,
        )

    if cluster is None and len(plain_fn) == 1:
        # Decorator Invoked without arguments
        return decorator(plain_fn[0])
    else:
        return decorator


class ExternalMementoFunction(ExternalMementoFunctionBase):
    def __init__(
        self,
        fn_reference: FunctionReference,
        context: InvocationContext,
        hash_rules: List[HashRule],
    ):
        super().__init__(fn_reference, context, "memento_function", hash_rules)

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
        return ExternalMementoFunction(
            fn_reference=fn_ref,
            context=context or self.context,
            hash_rules=self._hash_rules,
        )


ExternalMementoFunctionBase.register("memento_function", ExternalMementoFunction)


def forget_cluster(cluster_name: str = None):
    """
    Forget everything about every function in the given cluster

    :param cluster_name:    The cluster to forget. If None, forgets everything in
                            the default cluster.
    """

    cluster_config = Environment.get().get_cluster(cluster_name)
    if cluster_config is None:
        raise ValueError("Cluster with name '{}' not found".format(cluster_name))
    storage_backend = cluster_config.storage
    log.info(
        "Forgetting all functions for all arg hashes in cluster_name {}".format(
            cluster_name
        )
    )
    storage_backend.forget_everything()


def list_memoized_functions(cluster_name: str = None) -> List[FunctionReference]:
    """
    List the functions memoized in the given cluster

    :param cluster_name:    The name of the cluster to list, or the default cluster
                            if omitted
    :return: A list of FunctionReference objects

    """
    cluster = Environment.get().get_cluster(cluster_name=cluster_name)
    if cluster is None:
        raise ValueError("Cluster with name '{}' not found".format(cluster_name))
    storage = cluster.storage
    return storage.list_functions()
