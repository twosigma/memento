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

import collections
import datetime

import dateutil.parser as date_parser
import hashlib
import importlib
import inspect
import json
import re
from typing import Tuple, Dict, Any, cast, Optional, List  # noqa: F401

from .logging import log
from .types import MementoFunctionType, FunctionNotFoundError


class ArgumentHasher:
    """
    This class includes an implementation for the Memento-standard, cross-language logic
    for computing argument hashes, which is as follows:

    * Use SHA-256 as the hash computation algorithm
    * Normal arguments are converted to kwargs (called "effective kwargs")
    * Partial args are added to the effective kwargs.
    * kwargs are assembled into a dict of argument name to argument value
    * The dict is encoded into a JSON-friendly object as follows:
      - None, boolean, string, int, float: No special encoding
      - date / datetime.date: Encode as an object with the following structure:
        {
            "_mementoType": "date",
            "iso8601": <iso8601-date>
        }
      - datetime / datetime.datetime: Encode as an object with the following structure:
        {
            "_mementoType": "datetime",
            "iso8601": <iso8601-datetime>
        }
        Where iso8601-datetime is a string containing the ISO-8601 standard
        representation of a datetime (e.g. '2019-04-03T21:15:00'). Additional rules for datetime:
        - Fractions of a second are only encoded if non-zero, and always have 6 digits of
          precision.
        - Timezone-naive datetimes should not contain a timezone component.
        - Timezone-aware datetimes should include a standard timezone designator.
      - Lists are encoded as arrays, with each value encoded as per these rules.
      - Dictionaries are encoded as a dict with string keys and values encoded as per these rules.
      - References to a memento function are encoded as an object with the following structure.
        Partial args and kwargs are merged and encoded into the effectiveKwargs object.
        {
            "_mementoType": "FunctionReference",
            "qualifiedName", <qualified-name>,
            "partialArgs": <encoded-object-with-args>,
            "partialKwargs": <encoded-object-with-kwargs>
        }
    * The resulting JSON object is normalized by using JSON representation with sorted keys in
      objects, and removing all unnecessary whitespace.
    * The normalized encoding of the JSON object is hashed.
    * The hash is converted into a hex digest, in lowercase.

    """

    @staticmethod
    def normalize(obj: object) -> object:
        """
        Encode and then decode the given object to make sure the resulting types
        are what would be seen when deserializing.

        """
        result = ArgumentHasher._decode(ArgumentHasher._encode(obj))
        return result

    @staticmethod
    def _encode(arg: object) -> object:
        """
        Encodes the provided argument as an object using the memento standard encoding rules.

        """
        if (
            arg is None
            or isinstance(arg, bool)
            or isinstance(arg, str)
            or isinstance(arg, int)
            or isinstance(arg, float)
        ):
            return arg

        if isinstance(arg, datetime.datetime):
            return {"_mementoType": "datetime", "iso8601": arg.isoformat()}

        if isinstance(arg, datetime.date):
            return {"_mementoType": "date", "iso8601": arg.isoformat()}

        if isinstance(arg, list):
            return [ArgumentHasher._encode(x) for x in arg]

        if isinstance(arg, dict):
            return {k: ArgumentHasher._encode(v) for (k, v) in arg.items()}

        if isinstance(arg, MementoFunctionType):
            fn_reference = arg.fn_reference()  # type: FunctionReference
            partial_args = fn_reference.partial_args
            return {
                "_mementoType": "FunctionReference",
                "qualifiedName": fn_reference.qualified_name,
                "partialArgs": ArgumentHasher._encode(
                    list(partial_args) if partial_args else None
                ),
                "partialKwargs": ArgumentHasher._encode(fn_reference.partial_kwargs),
                "parameterNames": fn_reference.parameter_names,
            }

        raise ValueError(
            "Illegal argument type for memento argument: {}".format(type(arg))
        )

    @staticmethod
    def _decode(arg: object) -> object:
        """
        Decodes the provided argument as an object using the memento standard encoding rules.

        """
        if (
            arg is None
            or isinstance(arg, bool)
            or isinstance(arg, str)
            or isinstance(arg, int)
            or isinstance(arg, float)
        ):
            return arg

        if isinstance(arg, list):
            return [ArgumentHasher._decode(x) for x in arg]

        if isinstance(arg, dict):
            if "_mementoType" in arg:
                memento_type = arg["_mementoType"]
                if memento_type == "FunctionReference":
                    qualified_name = arg["qualifiedName"]
                    partial_args_list = cast(
                        list, ArgumentHasher._decode(arg["partialArgs"])
                    )
                    partial_kwargs = cast(
                        dict, ArgumentHasher._decode(arg["partialKwargs"])
                    )
                    parameter_names = cast(list, arg["parameterNames"])
                    return FunctionReference.from_qualified_name(
                        qualified_name=qualified_name,
                        partial_args=(
                            tuple(partial_args_list) if partial_args_list else None
                        ),
                        partial_kwargs=partial_kwargs,
                        parameter_names=parameter_names,
                    ).memento_fn
                elif memento_type == "datetime":
                    return date_parser.isoparse(arg["iso8601"])
                elif memento_type == "date":
                    return date_parser.isoparse(arg["iso8601"]).date()
                else:
                    raise ValueError("Unknown memento type {}".format(memento_type))
            else:
                return {k: ArgumentHasher._decode(v) for (k, v) in arg.items()}

        raise ValueError(
            "Illegal argument type for memento argument: {}".format(type(arg))
        )

    @staticmethod
    def _normalized_json(obj: object) -> str:
        """
        Compute the normalized json version of the given object

        """
        if (
            obj is None
            or isinstance(obj, bool)
            or isinstance(obj, str)
            or isinstance(obj, int)
            or isinstance(obj, float)
        ):
            return json.dumps(obj)

        if isinstance(obj, list):
            return (
                "[" + ",".join([ArgumentHasher._normalized_json(x) for x in obj]) + "]"
            )

        if isinstance(obj, dict):
            return (
                "{"
                + ",".join(
                    [
                        json.dumps(k) + ":" + ArgumentHasher._normalized_json(v)
                        for (k, v) in list(sorted(obj.items(), key=lambda t: t[0]))
                    ]
                )
                + "}"
            )

        raise ValueError(
            "Illegal object type for normalized json: {}".format(type(obj))
        )

    @staticmethod
    def compute_hash(effective_kwargs: dict) -> str:
        """
        Compute the standard Memento hash of a normalized json string.

        """
        arg_hash = hashlib.sha256()
        encoded_effective_kwargs = ArgumentHasher._encode(effective_kwargs)
        effective_kwargs_normalized_json = ArgumentHasher._normalized_json(
            encoded_effective_kwargs
        )
        arg_hash.update(effective_kwargs_normalized_json.encode("utf-8"))
        result = arg_hash.hexdigest()
        log.debug(
            "Computed hash of normalized kwargs {} = {}".format(
                effective_kwargs_normalized_json, result
            )
        )
        return result


class FunctionReference:
    """
    Reference to a memento function.

    Captures the cluster name (optional), module name, and function name.

    After construction, the `qualified_name` field will contain a string
    of the form `[cluster::]module:function.name#version`. The cluster can be
    retrieved through the `cluster` field.

    The function reference also contains any partially-evaluated
    args or kwargs.

    """

    external = False  # type: bool
    """
    Whether this refers to a function that is external to the local process, in which
    case function versions may not match those from the local process and it will not be
    possible to return the :py:class:`MementoFunction` instance referenced.
    """

    _qualified_name = None  # type: str

    @property
    def qualified_name(self):
        """
        The fully-qualified name of the function, including the cluster,
        module and function name as well as the version.

        """
        return self._qualified_name

    _qualified_name_without_cluster = None  # type: str

    @property
    def qualified_name_without_cluster(self):
        return self._qualified_name_without_cluster

    qualified_name_without_version = None  # type: str

    _memento_fn = None  # type: Optional[MementoFunctionType]

    @property
    def memento_fn(self) -> Optional[MementoFunctionType]:
        """The function to which this reference points"""
        return self._memento_fn

    _cluster_name = None  # type: Optional[str]

    @property
    def cluster_name(self):
        return self._cluster_name

    _module = None  # type: str

    @property
    def module(self):
        return self._module

    _function_name = None  # type: str

    @property
    def function_name(self):
        return self._function_name

    _partial_args = None  # type: Tuple[Any]

    @property
    def partial_args(self):
        """
        Stores partial arguments to this function, provided by
        :meth:`MementoFunctionType.partial`. These arguments are prepended to
        the function argument list during invocation.

        """
        return self._partial_args

    _partial_kwargs = None  # type: Dict[str, Any]

    @property
    def partial_kwargs(self):
        """
        Stores partial keyword arguments to this function, provided
        by :meth:`MementoFunctionType.partial`. These arguments are
        provided to the function kwargs during invocation.

        """
        return self._partial_kwargs

    parameter_names = None  # type: List[str]
    """Formal parameter names of this function, in order"""

    def __init__(
        self,
        memento_fn: Optional[MementoFunctionType],
        cluster_name: str = None,
        version: str = None,
        partial_args: Tuple[Any] = None,
        partial_kwargs: Dict[str, Any] = None,
        module_name: str = None,
        function_name: str = None,
        parameter_names: List[str] = None,
        external=False,
    ):
        """
        Construct a function reference, either by passing in the function
        or the qualified name of the function.

        A FunctionReference must refer to the same version of the function that is locally
        registered or it must be declared external, meaning the reference is for a function
        external to this process, in which case the MementoFunction is a stub.
        """
        # Get the information from the MementoFunction
        assert memento_fn is not None, "memento_fn must not be None"
        assert isinstance(
            memento_fn, MementoFunctionType
        ), "memento_fn must be a MementoFunctionType"
        self.external = external

        # Get cluster_name
        if cluster_name is not None:
            self._cluster_name = cluster_name
        else:
            self._cluster_name = (
                memento_fn.cluster_name if memento_fn is not None else None
            )

        # Get module
        # noinspection PyUnresolvedReferences
        if module_name is not None:
            self._module = module_name
        elif memento_fn is not None and hasattr(memento_fn.fn, "__module__"):
            self._module = memento_fn.fn.__module__
        else:
            assert (
                module_name is not None
            ), "Either memento_fn or module_name must be specified"

        # Get function name
        if function_name is not None:
            self._function_name = function_name
        elif memento_fn is not None:
            self._function_name = memento_fn.fn.__name__
        else:
            assert (
                function_name is not None
            ), "Either memento_fn or function_name must be specified"

        # Get version number
        if version is None:
            version = memento_fn.version()

        # Construct the qualified name
        qualified_name = (
            memento_fn.qualified_name_without_version
            if memento_fn is not None and memento_fn.fn is not None
            else self._module + ":" + self._function_name
        )
        if version is not None:
            qualified_name += "#" + version
        if cluster_name is not None and "::" not in qualified_name:
            qualified_name = cluster_name + "::" + qualified_name
        self._qualified_name = qualified_name

        self._qualified_name_without_cluster = (
            self.qualified_name
            if "::" not in self.qualified_name
            else self.qualified_name[self.qualified_name.find("::") + 2 :]
        )

        self.qualified_name_without_version = self.module + ":" + self.function_name
        if cluster_name is not None:
            self.qualified_name_without_version = (
                self.cluster_name + "::" + self.qualified_name_without_version
            )

        normalized_partial_args = (
            ArgumentHasher.normalize(list(partial_args)) if partial_args else []
        )  # type: list
        self._partial_args = tuple(normalized_partial_args)
        normalized_partial_kwargs = (
            ArgumentHasher.normalize(partial_kwargs) if partial_kwargs else {}
        )  # type: dict
        self._partial_kwargs = normalized_partial_kwargs

        # Record function
        self._memento_fn = memento_fn

        # Record formal parameter names, from function signature or from provided value.
        if parameter_names is not None:
            self.parameter_names = parameter_names
        elif memento_fn is not None:
            self.parameter_names = list(
                inspect.signature(memento_fn.fn).parameters.keys()
            )
        else:
            assert (
                parameter_names is not None
            ), "Must specify parameter_names if memento_fn not provided"

    @staticmethod
    def parse_qualified_name(qualified_name: str) -> Dict:
        """
        Parses a qualified name and returns a dict containing cluster, module, function, version.
        """
        assert isinstance(qualified_name, str), "Qualified name must be a str"

        # Parse information from the string
        match = re.match(
            r"((?P<cluster>.*)::)?(?P<module>.*):(?P<function>[^#]*)(#(?P<version>.*))?",
            qualified_name,
        )
        if not match:
            raise ValueError(
                "fn_or_name '{}' is not a valid qualified name".format(qualified_name)
            )
        return match.groupdict()

    @staticmethod
    def from_qualified_name(
        qualified_name: str,
        partial_args: Tuple[Any] = None,
        partial_kwargs: Dict[str, Any] = None,
        parameter_names: List[str] = None,
        external=False,
    ) -> "FunctionReference":
        """
        Attempts to find a function with the given qualified name.

        :raises FunctionNotFoundError: if external is False and the function could not be found,
            for example if the wrong version of the function is present.
        """
        parts = FunctionReference.parse_qualified_name(qualified_name)
        cluster_name = parts["cluster"]
        module = parts["module"]
        function_name = parts["function"]
        version = parts["version"]

        if not external:
            try:
                memento_fn = FunctionReference._find_function(
                    module=module,
                    function_name=function_name,
                    version=version,
                    partial_args=partial_args,
                    partial_kwargs=partial_kwargs,
                )
                return FunctionReference(
                    memento_fn,
                    cluster_name=cluster_name,
                    version=version,
                    partial_args=partial_args,
                    partial_kwargs=partial_kwargs,
                )
            except (ModuleNotFoundError, ValueError, AttributeError):
                # Cannot find module or function. Treat as an external function reference.
                external = True

        if external:
            from .external import UnboundExternalMementoFunction

            # We don't know which server this function is bound to, yet.
            # We also may not know the parameter names, so pass [] if unknown
            unbound_fn = UnboundExternalMementoFunction(
                cluster_name=cluster_name,
                module_name=module,
                function_name=function_name,
                version=version,
                partial_args=partial_args,
                partial_kwargs=partial_kwargs,
                parameter_names=parameter_names if parameter_names is not None else [],
            )
            return unbound_fn.fn_reference()

    @staticmethod
    def _find_function(
        module: str,
        function_name: str,
        version: str,
        partial_args: Tuple[Any] = None,
        partial_kwargs: Dict[str, Any] = None,
    ) -> MementoFunctionType:
        """
        Find the function to which this reference points.

        If there are any partially-evaluated args, return a partial function.

        This should always return a MementoFunctionType, not the wrapped function, else a
        ValueError is raised.

        Note: If self.version() is None, the function will find the currently linked version of
        the function.

        :raises ValueError: If the function, or the version of the function, could not be found.

        """
        module = importlib.import_module(module)
        ref = module
        if function_name.find("<locals>") != -1:
            raise ValueError(
                "Memento functions must be top-level. Cannot find a "
                "function that is local to another function."
            )
        for part in function_name.split("."):
            ref = getattr(ref, part)
        if not callable(ref):
            raise ValueError("{} does not refer to a function".format(function_name))
        if not isinstance(ref, MementoFunctionType):
            raise ValueError(
                "{} did not resolve to a MementoFunctionType".format(function_name)
            )
        memento_fn = ref

        # Check version
        if version is not None and memento_fn.version() != version:
            raise ValueError(
                "Function version does not match for {}: Expected {} but "
                "registered function is {} with dependencies {}".format(
                    function_name,
                    version,
                    memento_fn.version(),
                    memento_fn.dependencies().transitive_memento_fn_dependencies(),
                )
            )

        if partial_args or partial_kwargs:
            normalized_partial_args = (
                ArgumentHasher.normalize(list(partial_args)) if partial_args else []
            )  # type: list
            normalized_partial_kwargs = (
                ArgumentHasher.normalize(partial_kwargs) if partial_kwargs else {}
            )  # type: dict
            return memento_fn.partial(
                *tuple(normalized_partial_args), **normalized_partial_kwargs
            )

        return memento_fn

    def with_args(
        self, *args, _memento_context_args: Dict[str, Any] = None, **kwargs
    ) -> "FunctionReferenceWithArguments":
        return FunctionReferenceWithArguments(
            fn_reference=self,
            args=args,
            kwargs=kwargs,
            context_args=_memento_context_args,
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        # fn attribute does not serialize consistently
        return "FunctionReference({}, partial_args={}, partial_kwargs={}, external={})".format(
            repr(self.qualified_name),
            repr(self.partial_args),
            repr(
                collections.OrderedDict(sorted(self.partial_kwargs.items()))
                if self.partial_kwargs
                else None
            ),
            repr(self.external),
        )

    def __eq__(self, other):
        if not isinstance(other, FunctionReference):
            return False

        fn_ref = cast(FunctionReference, other)
        return (
            fn_ref.qualified_name == self.qualified_name
            and fn_ref.partial_args == self.partial_args
            and fn_ref.partial_kwargs == self.partial_kwargs
            and fn_ref.external == self.external
        )

    def __hash__(self):
        return hash(self.qualified_name)

    def __getstate__(self):
        return (
            self.qualified_name,
            self.partial_args,
            self.partial_kwargs,
            self.external,
        )

    def __setstate__(self, state):
        qualified_name, partial_args, partial_kwargs, external = state
        fn_ref = FunctionReference.from_qualified_name(
            qualified_name,
            partial_args=partial_args,
            partial_kwargs=partial_kwargs,
            external=external,
        )
        FunctionReference.__init__(
            self,
            fn_ref.memento_fn,
            cluster_name=fn_ref.cluster_name,
            version=fn_ref.memento_fn.version(),
            partial_args=fn_ref.partial_args,
            partial_kwargs=fn_ref.partial_kwargs,
            external=fn_ref.external,
        )


class FunctionReferenceWithArgHash:
    """
    Function Reference with argument hash.

    This is useful for cases where the reference and arg hash are known
    but the actual arguments are not.

    """

    fn_reference = None  # type: FunctionReference
    arg_hash = None  # type: str

    def __init__(self, fn_reference: FunctionReference, arg_hash: str):
        self.fn_reference = fn_reference
        self.arg_hash = arg_hash

    def __eq__(self, o: "FunctionReferenceWithArgHash") -> bool:
        return (
            isinstance(o, FunctionReferenceWithArgHash)
            and self.fn_reference.qualified_name == o.fn_reference.qualified_name
            and self.arg_hash == o.arg_hash
        )

    def __hash__(self):
        return hash((self.fn_reference.qualified_name, self.arg_hash))

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "FunctionReferenceWithArgHash(fn_reference={}, arg_hash={})".format(
            repr(self.fn_reference), repr(self.arg_hash)
        )


class FunctionReferenceWithArguments:
    """
    Function reference, bound with arguments.

    """

    fn_reference = None  # type: FunctionReference
    args = None  # type: Tuple[Any]
    kwargs = None  # type: Dict[str, Any]
    context_args = None  # type: Dict[str, Any]
    effective_kwargs = None  # type: Dict[str, Any]
    arg_hash = None  # type: str

    def __init__(
        self,
        fn_reference: FunctionReference,
        args: Tuple,
        kwargs: Dict[str, Any],
        context_args: Optional[Dict[str, Any]] = None,
    ):
        if not fn_reference.memento_fn:
            raise FunctionNotFoundError(
                "Cannot create a FunctionReferenceWithArguments if the underlying function "
                "reference cannot be mapped to a real function. This could be because the "
                "memento function is not in the path or a function version mismatch. "
                "Reference: {}".format(repr(fn_reference))
            )
        self.fn_reference = fn_reference
        normalized_args = (
            ArgumentHasher.normalize(list(args)) if args else []
        )  # type: list
        self.args = tuple(normalized_args)
        normalized_kwargs = (
            ArgumentHasher.normalize(kwargs) if kwargs else {}
        )  # type: dict
        self.kwargs = normalized_kwargs
        normalized_context_args = (
            ArgumentHasher.normalize(context_args) if context_args else {}
        )  # type: dict
        self.context_args = normalized_context_args
        self.effective_kwargs = self._compute_effective_kwargs()
        self.effective_kwargs_with_context_args = (
            self._compute_effective_kwargs_with_context_args()
        )
        self.arg_hash = ArgumentHasher.compute_hash(
            self.effective_kwargs_with_context_args
        )
        validate_args(
            *self.args, **self.kwargs, _memento_context_args=self.context_args
        )

    def fn_reference_with_arg_hash(self) -> FunctionReferenceWithArgHash:
        """
        Returns this FunctionReferenceWithArguments as a FunctionReferenceWithArgHash

        """
        return FunctionReferenceWithArgHash(self.fn_reference, self.arg_hash)

    def _compute_effective_kwargs(self) -> Dict[str, Any]:
        # Start with any partial kwargs
        result = dict(self.fn_reference.partial_kwargs)

        # Fill in partial args
        parameter_names = self.fn_reference.parameter_names
        partial_args = self.fn_reference.partial_args
        if len(parameter_names) < len(partial_args):
            raise ValueError(
                f"More partial arguments provided ({len(partial_args)} "
                f"than the arguments for the function ({parameter_names})"
            )
        for i in range(0, len(partial_args)):
            result[parameter_names[i]] = partial_args[i]

        # Which parameter names are left after partial?
        remaining_parameter_names = [
            name for name in parameter_names if name not in result
        ]

        # Now fill in args
        if len(remaining_parameter_names) < len(self.args):
            raise ValueError(
                f"More arguments provided ({len(self.args)} "
                f"than the remaining arguments for the "
                f"function ({remaining_parameter_names})"
            )

        for i in range(0, len(self.args)):
            result[remaining_parameter_names[i]] = self.args[i]

        # And remaining kwargs
        result.update(self.kwargs)

        return result

    def _compute_effective_kwargs_with_context_args(self) -> Dict[str, Any]:
        hash_kwargs = self.effective_kwargs.copy()

        # Apply context args as a single kwarg
        if self.context_args is not None and len(self.context_args) > 0:
            hash_kwargs["_memento_context_args"] = self.context_args

        return hash_kwargs

    def __eq__(self, o: "FunctionReferenceWithArguments") -> bool:
        return (
            isinstance(o, FunctionReferenceWithArguments)
            and self.fn_reference.qualified_name == o.fn_reference.qualified_name
            and self.arg_hash == o.arg_hash
        )

    def __hash__(self):
        return hash((self.fn_reference.qualified_name, self.arg_hash))

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return (
            "FunctionReferenceWithArguments(fn_reference={}, args={}, kwargs={}, "
            "context_args={})".format(
                repr(self.fn_reference),
                repr(self.args),
                repr(self.kwargs),
                repr(self.context_args),
            )
        )


def validate_args(*args, _memento_context_args: Dict[str, Any] = None, **kwargs):
    """
    Validate that the provided arguments are of types valid to Memento

    """

    def validate_arg(a):
        return (
            a is None
            or isinstance(a, bool)
            or isinstance(a, str)
            or isinstance(a, int)
            or isinstance(a, float)
            or isinstance(a, datetime.date)
            or isinstance(a, datetime.datetime)
            or (
                isinstance(a, dict)
                and False not in [validate_arg(v) for (k, v) in a.items()]
            )
            or (isinstance(a, list) and False not in [validate_arg(v) for v in a])
            or isinstance(a, MementoFunctionType)
        )

    if _memento_context_args is not None and not validate_arg(_memento_context_args):
        raise AssertionError(
            "Memento cannot handle context arg. Value: {}".format(
                type(_memento_context_args)
            )
        )

    for idx, arg in enumerate(args):
        if not validate_arg(arg):
            raise AssertionError(
                "Memento cannot handle function argument type {} at index {}. Value: {}".format(
                    type(arg), idx, args
                )
            )

    for key, arg in kwargs.items():
        if not validate_arg(arg):
            raise AssertionError(
                "Memento cannot handle function argument type {} for kwarg {}. Value: {}".format(
                    type(arg), key, kwargs
                )
            )
