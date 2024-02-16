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

import ast
import base64
import hashlib
import inspect
import json
import textwrap
from abc import ABC, abstractmethod
from types import CodeType
from typing import Callable, Set, Optional, List, Union, Type
from weakref import WeakKeyDictionary

from twosigma.memento.reference import FunctionReference
from twosigma.memento.exception import DependencyNotFoundError
from twosigma.memento.logging import log
from twosigma.memento.serialization import MementoCodec
from twosigma.memento.types import MementoFunctionType


def fn_code_hash(fn: Callable, salt: str = None, environment: bytes = None) -> str:
    """
    Compute a hex digest of the code for a function.

    This can be used, for example, as the parameter to the version parameter so that
    the function is invalidated whenever its code changes.

    :param fn:      The function to hash
    :param salt:    If specified, the salt is prepended to the code before it is hashed.
                    This allows changing the hash to accommodate other factors.
    :param environment:  An encoded description of the environment, allowing the hash to
                    be modified when the environment changes in some way.

    """

    def hash_if_code_object(o):
        """
        If the parameter is a code object, return a hash, else return the object.

        """
        if isinstance(o, CodeType):
            sha256 = hashlib.sha256()
            if environment:
                sha256.update(environment)
            attr_values = [
                o.co_argcount,
                base64.b64encode(o.co_code).decode("utf-8"),
                o.co_cellvars,
                tuple([hash_if_code_object(x) for x in o.co_consts]),
                # may contain embedded code objects
                # o.co_filename", # results in every cell re-execution being a different hash
                # o.co_firstlineno", # results in hash changing when function moves within cell
                o.co_flags,
                # o.co_lnotab", # bytecode->offset lookup does not affect behavior
                o.co_freevars,
                o.co_kwonlyargcount,
                o.co_name,
                o.co_names,
                o.co_nlocals,
                o.co_stacksize,
                o.co_varnames,
            ]
            if salt:
                sha256.update(salt.encode("utf-8"))
            sha256.update(json.dumps(attr_values, sort_keys=True).encode("utf-8"))
            return sha256.hexdigest()[0:16]
        else:
            return repr(o)

    if isinstance(fn, MementoFunctionType):
        memento_fn = fn  # type: MementoFunctionType
        fn = memento_fn.fn
    assert callable(fn), "Must provide a function to hash"
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    if hasattr(fn, "__code__"):
        code = getattr(fn, "__code__")  # type: code
        result = hash_if_code_object(code)
        return result
    else:
        # If we can't get the code for the function, then return the name of the function
        return repr(fn)


def resolve_to_symbolic_names(
    dependencies: List[Union[str, MementoFunctionType]]
) -> Set[str]:
    """
    Takes a set of str and MementoFunctionType and resolves each to a symbolic name,
    represented as a str.

    """
    if dependencies is None:
        return set()

    def resolve_to_symbol(dep) -> str:
        if isinstance(dep, str):
            return dep
        elif isinstance(dep, MementoFunctionType):
            q_name = dep.fn_reference().qualified_name
            q_name = q_name[0 : q_name.find("#")] if "#" in q_name else q_name
            return q_name
        else:
            raise ValueError(
                "Each dependency must be either a str or "
                "MementoFunctionType. Got {}".format(dep)
            )

    return set(resolve_to_symbol(dep) for dep in dependencies)


_dotted_names_cache = WeakKeyDictionary()  # type: WeakKeyDictionary
"""Cache for list_dotted_names to improve performance"""


def list_dotted_names(fn: Callable) -> Set[str]:
    """
    Parse the AST of a function and return a set of all dotted names that are not local variables.

    A dotted name, in this context, is a name that is resolved as part of a chain of
    attribute operations (e.g. `twosigma.memento.MementoFunction` is a dotted name).

    Because parsing the AST twice will produce the same result, this function utilizes a cache.

    """
    if fn in _dotted_names_cache:
        return _dotted_names_cache[fn]

    class ReferenceExtractor(ast.NodeVisitor):
        references = None  # type: Set[str]

        def __init__(self):
            self.references = set()

        # noinspection PyPep8Naming
        def visit_Attribute(self, node):
            def eval_attr(n) -> Optional[str]:
                if isinstance(n, ast.Attribute):
                    attr_result = eval_attr(n.value)
                    if attr_result is None:
                        return None
                    return attr_result + "." + n.attr
                elif isinstance(n, ast.Call):
                    # Take everything before the call
                    return eval_attr(n.func)
                elif isinstance(n, ast.Name):
                    return n.id
                return None

            eval_attr_result = eval_attr(node)
            if eval_attr_result is not None:
                self.references.add(eval_attr_result)
            return self.generic_visit(node)

        # noinspection PyPep8Naming
        def visit_Name(self, node):
            self.references.add(node.id)
            return None  # stop traversing this subtree

    try:
        source = inspect.getsource(fn)
        # If this is a static method or some other inline function, dedent it first.
        source = textwrap.dedent(source)
        parsed = ast.parse(source)

        extractor = ReferenceExtractor()
        extractor.visit(parsed)
        result = extractor.references
        # Remove any local variables and cell variables
        if hasattr(fn, "__code__"):
            code_obj = fn.__code__
            local_vars = set()  # type: Set[str]
            local_vars.update(code_obj.co_varnames)
            local_vars.update(code_obj.co_cellvars)
            result.difference_update(local_vars)
            # Also remove anything that dereferences a local variable
            to_remove = {
                symbol
                for symbol in result
                if "." in symbol and symbol[0 : symbol.find(".")] in local_vars
            }
            result.difference_update(to_remove)

        _dotted_names_cache[fn] = result
        return result
    except (OSError, TypeError) as e:
        # Will be thrown if not a module, class, method, function, traceback, frame, or code object
        # or if the source could not be loaded
        log.debug("Skipping {} in code hash because {}".format(fn.__qualname__, e))
        _dotted_names_cache[fn] = set()
        return set()


class HashRule(ABC):
    """
    Base class for hash rules. Each hash rule computes the hash of a different type of
    dependency. A hash rule starts from a symbol and ends at one or more `MementoFunctionType`s
    that are detected to be called from that symbol.

    """

    all_rules = []  # type: List[Type[HashRule]]
    """All known HashRule instances, in order of evaluation"""

    key = None  # type: str
    """A string used to uniquely identify, and canonically order, this hash rule."""

    parent_symbol = None  # type: Optional[str]
    """The name of the parent of the variable, used as a namespace for global symbols"""

    symbol = None  # type: str
    """The name of the variable that this rule is computing for"""

    first_level = None  # type: bool
    """If `True`, this is a first-level (direct) dependency"""

    rule_hash = None  # type: Optional[str]
    """The hash computed for this hash rule, in the format of a hex string, or `None`"""

    def __init__(
        self, key: str, parent_symbol: Optional[str], symbol: str, first_level: bool
    ):
        self.key = key
        self.parent_symbol = parent_symbol
        self.symbol = symbol
        self.first_level = first_level
        self.rule_hash = None  # this is typically computed and stored later

    def describe(self) -> str:
        """Return a textual description of the component of the hash"""
        return "{}{}{}".format(
            self.key,
            "#" + self.rule_hash[0:16] if self.rule_hash is not None else "",
            " (direct)" if self.first_level else "",
        )

    @abstractmethod
    def collect_transitive_dependencies(
        self,
        result: Set["HashRule"],
        root_fn: MementoFunctionType,
        package_scope: Set[str],
        blacklist: List[object],
    ):
        """
        Analyze the entity behind this symbol and recursively descend to collect a full
        tree of transitive dependencies into the `results` variable. Cycles are broken by
        not recursively descending if the `HashRule` is already in the set.

        The caller should not put this rule in result before calling, as this method
        will add itself to the result as the first step.

        :param result:  The collected dependencies
        :param root_fn:  The Memento Function at the root of the dependency analysis.
        :param package_scope:  Set of package names for which to descend to look
            for transitive dependencies.
        :param blacklist:  List of objects to exclude (recursively) from dependency list

        """
        pass

    @abstractmethod
    def compute_hash(self) -> Optional[str]:
        """
        Compute the hash code for this rule and return a hex string representation.
        or `None` to represent that the hash should not change as a result of this rule.

        """
        pass

    @abstractmethod
    def did_change(self) -> bool:
        """
        Return True if the object pointed to by this rule changed since the hash was last
        computed

        """
        pass

    @staticmethod
    def _visit_dependency(
        result: Set["HashRule"],
        src_fn: Callable,
        parent_symbol: str,
        symbol: str,
        required: bool,
        root_fn: MementoFunctionType,
        first_level: bool,
        package_scope: Set[str],
        blacklist: List[object],
    ):
        """
        Evaluates the given symbol (of the form a.b.c) using the globals scope of the provided
        function and adds a `HashRule` to the result set, if found. The `HashRule` also gets
        its `collect_transitive_dependencies` method called.

        :param result:  The set of `HashRule`s used to hash the entity represented by the symbol.
            The returned set includes all hash rules for the function itself, plus transitive
            dependencies.
        :param src_fn:  The function that contains the source code.
        :param parent_symbol:  The symbol of the parent of this symbol. This namespaces the scope
            (e.g. two global variables can have the same name but might be different for
            different fns because of a different import)
        :param symbol:  The symbol. If it has a `':'`, this represents a fully qualified name of
            a `MementoFunctionType`. Otherwise, this is a "dotted name" like `a.b.c.fn`.
        :param required:  If `True`, a `DependencyNotFoundError` will be raised if the given symbol
            could not be resolved.
        :param root_fn:  The `MementoFunctionType` at the root of the dependency analysis
        :param first_level: If `True`, this should be considered a direct dependency.
        :param package_scope:  Set of names of packages for which to descend to collect
            transitive dependencies. This prevents Memento from descending into irrelevant
            packages.
        :param blacklist:  List of objects to exclude from dependency sets (recursively).

        """
        if hasattr(src_fn, "__globals__"):
            global_table = src_fn.__globals__
        else:
            # Cannot visit this dependency if there is no global scope
            return

        # If ":" is in the symbol name, this is a qualified name from memento.
        # Construct via a FunctionReference
        if ":" in symbol:

            def memento_fn_resolver():
                return FunctionReference.from_qualified_name(symbol).memento_fn

            memento_fn = memento_fn_resolver()
            rule = MementoFunctionHashRule(
                parent_symbol=parent_symbol,
                symbol=symbol,
                resolver=lambda: memento_fn_resolver,
                obj=memento_fn,
                first_level=first_level,
            )
            # collect_transitive_dependencies will add this rule to result
            rule.collect_transitive_dependencies(
                result=result,
                root_fn=root_fn,
                package_scope=package_scope,
                blacklist=blacklist,
            )
            return

        # Otherwise, treat this as a "dotted name" (e.g. a.b.c.fn())
        parts = symbol.split(".")
        if parts[0] not in global_table:
            if required:
                raise DependencyNotFoundError(
                    "Could not find required dependency {} for function {}. "
                    "Failed to resolve {} in global table.".format(
                        symbol, src_fn.__name__, parts[0]
                    )
                )
            result.add(
                UndefinedSymbolHashRule(
                    global_table,
                    parent_symbol=parent_symbol,
                    symbol=parts[0],
                    first_level=first_level,
                    ref_is_global_table=True,
                )
            )
            return

        first_part = parts[0]

        def resolver():
            return global_table[first_part] if first_part in global_table else None

        ref = resolver()

        def resolve_symbol(
            parent_sym: str, sym: str, resolver_fn, reference: object
        ) -> Optional[HashRule]:
            if any(x is reference for x in blacklist):
                # Do not include some symbols (e.g. memento library itself)
                return None

            for strategy in HashRule.all_rules:
                # noinspection PyUnresolvedReferences
                found = strategy.try_resolve(
                    parent_sym, sym, resolver_fn, reference, first_level=first_level
                )
                if found is not None:
                    return found
            return None

        rule = resolve_symbol(parent_symbol, symbol, resolver, ref)
        if rule is not None:
            # collect_transitive_dependencies will add this rule to result
            rule.collect_transitive_dependencies(
                result=result,
                root_fn=root_fn,
                package_scope=package_scope,
                blacklist=blacklist,
            )
            return

        symbol_part = parts[0]
        for i in range(1, len(parts)):
            if not hasattr(ref, parts[i]):
                if required:
                    raise DependencyNotFoundError(
                        "Could not find required dependency {} for function {}. "
                        "Failed to resolve {}.{}.".format(
                            symbol, src_fn.__name__, symbol_part, parts[i]
                        )
                    )
                result.add(
                    UndefinedSymbolHashRule(
                        ref,
                        parent_symbol=parent_symbol,
                        symbol=parts[i],
                        first_level=first_level,
                        ref_is_global_table=False,
                    )
                )
                return
            symbol_part += "."
            part_i = parts[i]
            symbol_part += part_i
            ref_to_resolve = ref

            def resolver():
                return getattr(ref_to_resolve, part_i)

            ref = resolver()
            rule = resolve_symbol(parent_symbol, symbol_part, resolver, ref)
            if rule is not None:
                # collect_transitive_dependencies will add this rule to result
                rule.collect_transitive_dependencies(
                    result=result,
                    root_fn=root_fn,
                    package_scope=package_scope,
                    blacklist=blacklist,
                )
                return

        if required:
            raise DependencyNotFoundError(
                "Could not find required dependency {} for function {}-{}. "
                "No hash rules matched.".format(symbol, src_fn.__name__, symbol_part)
            )

    @abstractmethod
    def clone(self) -> "HashRule":
        pass

    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        return self.describe()


class UndefinedSymbolHashRule(HashRule):
    """
    Hash rule for cases where a symbol is not defined.

    """

    # No try_resolve for UndefinedSymbolHashRule since these are constructed when a symbol
    # doesn't exist

    ref = None  # type: object
    """Base reference from which the symbol resolution failed"""

    ref_is_global_table = None  # type: bool
    """If true, ref is the global table, so we should use an `in` check instead of `hasattr`"""

    def __init__(
        self,
        ref: object,
        parent_symbol: str,
        symbol: str,
        first_level: bool,
        ref_is_global_table: bool,
    ):
        # noinspection PyUnresolvedReferences
        super().__init__(
            key="UndefinedSymbol;{};{}".format(parent_symbol, symbol),
            parent_symbol=parent_symbol,
            symbol=symbol,
            first_level=first_level,
        )
        self.ref = ref
        self.ref_is_global_table = ref_is_global_table

    def clone(self) -> HashRule:
        return UndefinedSymbolHashRule(
            self.ref,
            self.parent_symbol,
            self.symbol,
            self.first_level,
            self.ref_is_global_table,
        )

    def collect_transitive_dependencies(
        self,
        result: Set[HashRule],
        root_fn: MementoFunctionType,
        package_scope: Set[str],
        blacklist: List[object],
    ):
        # An undefined symbol cannot have transitive dependencies
        pass

    def compute_hash(self) -> Optional[str]:
        # Undefined symbols do not impact the hash until they are defined
        return None

    def did_change(self) -> bool:
        # Considered changed if the symbol now points to something.
        if self.ref_is_global_table:
            return self.symbol in self.ref

        return hasattr(self.ref, self.symbol)

    def __repr__(self):
        return "UndefinedSymbolHashRule(parent_symbol={parent_symbol}, symbol={symbol})".format(
            parent_symbol=repr(self.parent_symbol), symbol=repr(self.symbol)
        )


class MementoFunctionHashRule(HashRule):
    """
    Hash rule for the case where a variable points to a Memento Function

    """

    memento_fn = None  # type: MementoFunctionType
    resolver = None  # type: Callable
    parent_symbol = None  # type: str

    # noinspection PyUnusedLocal
    @staticmethod
    def try_resolve(
        parent_symbol: str,
        symbol: str,
        resolver: Callable,
        ref: object,
        first_level: bool,
    ) -> Optional["MementoFunctionHashRule"]:
        # Memento functions may be wrapped by decorators, so check each level of wrapping
        # to decide if this is a MementoFunctionType.
        while True:
            if isinstance(ref, MementoFunctionType):
                return MementoFunctionHashRule(
                    parent_symbol, symbol, resolver, ref, first_level
                )
            elif hasattr(ref, "__wrapped__"):
                ref = ref.__wrapped__
            else:
                return None

    def __init__(
        self,
        parent_symbol: Optional[str],
        symbol: str,
        resolver: Callable,
        obj: MementoFunctionType,
        first_level: bool,
    ):
        # noinspection PyUnresolvedReferences
        super().__init__(
            key="MementoFunction;{};{}".format(
                parent_symbol, obj.qualified_name_without_version
            ),
            parent_symbol=parent_symbol,
            symbol=symbol,
            first_level=first_level,
        )
        self.memento_fn = obj
        self.resolver = resolver

    def clone(self) -> "HashRule":
        return MementoFunctionHashRule(
            self.parent_symbol,
            self.symbol,
            self.resolver,
            self.memento_fn,
            self.first_level,
        )

    def collect_transitive_dependencies(
        self,
        result: Set[HashRule],
        root_fn: MementoFunctionType,
        package_scope: Set[str],
        blacklist: List[object],
    ):
        # Make sure self is not already accounted for:
        if self in result:
            return

        # Always add self, even if this function is not in package scope. Memento Functions
        # are tracked across modules and packages.
        result.add(self)

        # Add transitive dependencies:
        memento_fn = self.memento_fn

        for dep in memento_fn.required_dependencies:
            HashRule._visit_dependency(
                result=result,
                src_fn=memento_fn.src_fn,
                parent_symbol=memento_fn.qualified_name_without_version,
                symbol=dep,
                required=True,
                root_fn=root_fn,
                first_level=memento_fn is root_fn,
                package_scope=package_scope,
                blacklist=blacklist,
            )

        for dep in memento_fn.detected_dependencies:
            HashRule._visit_dependency(
                result=result,
                src_fn=memento_fn.src_fn,
                parent_symbol=memento_fn.qualified_name_without_version,
                symbol=dep,
                required=False,
                root_fn=root_fn,
                first_level=memento_fn is root_fn,
                package_scope=package_scope,
                blacklist=blacklist,
            )

    def compute_hash(self) -> Optional[str]:
        return (
            self.memento_fn.explicit_version
            if self.memento_fn.explicit_version is not None
            else self.memento_fn.code_hash
        )

    def did_change(self) -> bool:
        # Changes to the definition of a MementoFunctionType are more robust and detected using a
        # different mechanism (the global counter), but it is possible that a symbol
        # pointing to a memento function is now pointing to something else, or even undefined
        # so detect if that happened, else return `False`.
        new_fn = self.resolver()
        return not isinstance(new_fn, MementoFunctionType)

    def __repr__(self):
        return f"MementoFunctionHashRule(key={repr(self.key)})"


class GlobalVariableHashRule(HashRule):
    """
    Hash rule for the case where a variable points to a global variable.

    """

    var = None  # type: object
    resolver = None  # type: Callable
    last_value = None  # type: bytes

    @staticmethod
    def try_resolve(
        parent_symbol: str,
        symbol: str,
        resolver: Callable,
        ref: object,
        first_level: bool,
    ) -> Optional["GlobalVariableHashRule"]:
        val = GlobalVariableHashRule._serialize_value(ref)
        return (
            GlobalVariableHashRule(
                parent_symbol, symbol, resolver, ref, val, first_level
            )
            if val is not None
            else None
        )

    def __init__(
        self,
        parent_symbol: str,
        symbol: str,
        resolver: Callable,
        ref: object,
        last_value: bytes,
        first_level: bool,
    ):
        super().__init__(
            key="GlobalVariable;{};{}".format(parent_symbol, symbol),
            parent_symbol=parent_symbol,
            symbol=symbol,
            first_level=first_level,
        )
        self.var = ref
        self.resolver = resolver
        self.last_value = last_value

    def clone(self):
        return GlobalVariableHashRule(
            self.parent_symbol,
            self.symbol,
            self.resolver,
            self.var,
            self.last_value,
            self.first_level,
        )

    def collect_transitive_dependencies(
        self,
        result: Set["HashRule"],
        root_fn: MementoFunctionType,
        package_scope: Set[str],
        blacklist: List[object],
    ):
        # Variables cannot have transitive dependencies, so just add self and return
        result.add(self)

    def compute_hash(self) -> Optional[str]:
        # If this is a type that Memento understands, use its stable hashing semantics
        return hashlib.sha256(self.last_value).hexdigest()[0:16]

    def did_change(self) -> bool:
        """Returns `True` if the variable has changed since this rule was last computed"""
        if self.last_value is None:
            return False
        # Re-resolve the symbol so we get changes to by-value semantics
        new_var = self.resolver()
        new_value = self._serialize_value(new_var)
        return self.last_value != new_value

    @staticmethod
    def _serialize_value(var: object) -> Optional[bytes]:
        try:
            return json.dumps(MementoCodec.encode_arg(var), sort_keys=True).encode(
                "utf-8"
            )
        except (TypeError, ValueError):
            # not a type that Memento understands or can hash. Do not hash.
            return None

    def describe(self) -> str:
        return "{} {}".format(super().describe(), self.last_value.decode("utf-8"))

    def __repr__(self):
        return f"GlobalVariableHashRule(key={repr(self.key)})"


class NonMementoFunctionHashRule(HashRule):
    """
    Hash rule for the case where a variable points to a non-memento function.

    """

    src_fn = None  # type: Callable
    resolver = None  # type: Callable

    @staticmethod
    def try_resolve(
        parent_symbol: str,
        symbol: str,
        resolver: Callable,
        ref: object,
        first_level: bool,
    ) -> Optional["NonMementoFunctionHashRule"]:
        return (
            NonMementoFunctionHashRule(
                parent_symbol, symbol, resolver, ref, first_level
            )
            if callable(ref) and hasattr(ref, "__globals__")
            else None
        )

    def __init__(
        self,
        parent_symbol: str,
        symbol: str,
        resolver: Callable,
        obj: Callable,
        first_level: bool,
    ):
        # noinspection PyUnresolvedReferences
        super().__init__(
            key="Function;{};{}".format(
                parent_symbol, obj.__module__ + ":" + obj.__qualname__
            ),
            parent_symbol=parent_symbol,
            symbol=symbol,
            first_level=first_level,
        )
        self.src_fn = obj
        self.resolver = resolver

    def clone(self) -> HashRule:
        return NonMementoFunctionHashRule(
            self.parent_symbol,
            self.symbol,
            self.resolver,
            self.src_fn,
            self.first_level,
        )

    def collect_transitive_dependencies(
        self,
        result: Set[HashRule],
        root_fn: MementoFunctionType,
        package_scope: Set[str],
        blacklist: List[object],
    ):
        # Make sure self is not already accounted for:
        if self in result:
            return

        # Only add this function and descend if it is within the package scope.
        if inspect.getmodule(self.src_fn).__package__ not in package_scope:
            return

        # Add self
        result.add(self)

        # Add transitive dependencies:
        src_fn = self.src_fn

        for dep in list_dotted_names(src_fn):
            # noinspection PyUnresolvedReferences
            symbol_parent = src_fn.__module__ + ":" + src_fn.__qualname__
            HashRule._visit_dependency(
                result=result,
                src_fn=src_fn,
                parent_symbol=symbol_parent,
                symbol=dep,
                required=False,
                root_fn=root_fn,
                first_level=False,
                package_scope=package_scope,
                blacklist=blacklist,
            )

    def compute_hash(self) -> Optional[str]:
        return fn_code_hash(self.src_fn)

    def did_change(self) -> bool:
        """
        Returns `True` if the function has changed since this rule was last computed.
        We use the function reference to detect changes.

        """
        new_fn = self.resolver()
        return self.src_fn != new_fn

    def __repr__(self):
        return f"NonMementoFunctionHashRule(key={self.key})"


HashRule.all_rules = [
    MementoFunctionHashRule,
    NonMementoFunctionHashRule,
    GlobalVariableHashRule,  # must go last
]
