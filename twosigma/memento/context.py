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

"""
Invocation context for running memento functions. There are two scopes
of context:

* local - applies to the current call only
* recursive - applies to this call and all recursive calls

The recursive context is carried forward to all future calls in this
stack. Any local context settings override the recursive settings
for the current call.

The :class:`InvocationContext` class encapsulates all scopes.

The :class:`ScopedContext` abstract base class encapsulates all settings
common to all scopes.

The :class:`LocalContext` and :class:`RecursiveContext` classes extend
:class:`ScopedContext` and provide any settings specific to each scope.

The context objects are intended to be read-only. To update a field, use
one of the update methods.

"""
from abc import ABC
from typing import Any, Dict


class ScopedContext(ABC):
    """
    Abstract base class encapsulating portions of context that apply to
    all scopes.
    """

    def __init__(self) -> None:
        pass


class RecursiveContext(ScopedContext):
    """
    Scope that applies to the current call and all recursive calls.
    """

    correlation_id = None  # type: str

    retry_on_remote_call = None  # type: bool
    """
    If true, the Memento framework will raise a RemoteCallException if an
    attempt is made to make a remote call. This is to prevent worker starvation
    scenarios where remote workers are waiting on other remote workers. The
    typical behavior is to fail and retry later once the dependent work is
    complete. Note that not all remote runners implement or need to
    implement this feature.
    """

    prevent_further_calls = None  # type: bool
    """
    If true, Memento calls on this thread after this one are blocked with a
    `RuntimeError`. Defaults to `False`.
    """

    context_args = None  # type: Dict[str, Any]
    """
    Implicit arguments that are included in the argument hash but not passed
    explicitly as parameters. Note that this dict should never be updated
    in-place. If a new context arg is needed, replace the dict with a copy
    including the new args.
    """

    # Note: If additional attributes ar added, be sure to update
    # serialization.encode_recursive_context.

    def __init__(
        self,
        correlation_id: str = None,
        retry_on_remote_call: bool = False,
        prevent_further_calls: bool = False,
        context_args: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self.__dict__["correlation_id"] = correlation_id
        self.__dict__["retry_on_remote_call"] = retry_on_remote_call
        self.__dict__["prevent_further_calls"] = prevent_further_calls
        self.__dict__["context_args"] = context_args

    def __setattr__(self, key, value):
        raise ValueError("To update a context field, use the update methods")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "RecursiveContext({})".format(self.__dict__)

    def copy(self) -> "RecursiveContext":
        result = RecursiveContext()
        result.__dict__.update(self.__dict__)
        return result

    def update(self, key: str, value: Any) -> "RecursiveContext":
        if key not in self.__dict__:
            raise ValueError("No such property {}".format(key))
        result = self.copy()
        result.__dict__[key] = value
        return result


class LocalContext(ScopedContext):
    """
    Scope that applies to only the current call.

    """

    ignore_result = None  # type: bool
    """
    If true, the function result is ignored, and `None` is returned. This
    optimizes performance for cases where we do not care about the return
    value. Defaults to `False`.
    """

    force_local = None  # type: bool
    """
    If true, overrides the runner for this call to use the local process
    runner, even if a remote runner is configured. Defaults to `False`.
    """

    monitor_progress = None  # type: bool
    """
    If true, a progress indicator should be rendered during the computation
    of this result.
    """

    def __init__(
        self,
        ignore_result: bool = False,
        force_local: bool = False,
        monitor_progress: bool = False,
    ) -> None:
        super().__init__()
        self.__dict__["ignore_result"] = ignore_result
        self.__dict__["force_local"] = force_local
        self.__dict__["monitor_progress"] = monitor_progress

    def __setattr__(self, key, value):
        raise ValueError("To update a context field, use the update methods")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "LocalContext({})".format(self.__dict__)

    def copy(self) -> "LocalContext":
        result = LocalContext()
        result.__dict__.update(self.__dict__)
        return result

    def update(self, key: str, value: Any) -> "LocalContext":
        if key not in self.__dict__:
            raise ValueError("No such property {}".format(key))
        result = self.copy()
        result.__dict__[key] = value
        return result


class InvocationContext:
    """
    Contextual information for an invocation, consisting of sub-contexts
    for different scopes (see :module:`context`).
    """

    recursive = None  # type: RecursiveContext
    local = None  # type: LocalContext

    def __init__(
        self, recursive: RecursiveContext = None, local: LocalContext = None
    ) -> None:
        self.__dict__["recursive"] = recursive or RecursiveContext()
        self.__dict__["local"] = local or LocalContext()

    def __setattr__(self, key, value):
        raise ValueError("To update a context field, use the update methods")

    def __str__(self):
        return "InvocationContext({}, {})".format(self.recursive, self.local)

    def update_local(self, key: str, value: Any) -> "InvocationContext":
        """
        Create a copy of this context with the given property updated in the
        local part of the context.
        """
        return InvocationContext(self.recursive, self.local.update(key, value))

    def update_recursive(self, key: str, value: Any) -> "InvocationContext":
        """
        Create a copy of this context with the given property updated in the
        recursive part of the context.
        """
        return InvocationContext(self.recursive.update(key, value), self.local)
