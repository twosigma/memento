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
Defines the MementoException class, which represents that a
function invocation resulted in an exception.

"""
import importlib
import inspect
import re
import traceback
from typing import List

from twosigma.memento.reference import FunctionReferenceWithArgHash

_MEMENTO_EXCEPTION_REGEX = r"([^:]*)::([^:]*):?([^:]*)"


class MementoException(RuntimeError):
    """
    Represents that a function invocation resulted in an exception.
    The caller can retrieve the name of the exception via
    `exception_name` and the message via `message`.

    The exception name is in the format `language::module:name`.
    If the language is `python`, an attempt will be made to
    reconstitute the original exception class when calling
    `to_exception`.

    """

    exception_name = None  # type: str
    message = None  # type: str

    stack_trace = None  # type: str
    "A string representation of the stack trace"

    def __init__(self, exception_name: str, message: str, stack_trace: str):
        super().__init__(
            "{}: {}. Original stack trace follows:\n{}".format(
                exception_name, message, stack_trace
            )
        )
        self.exception_name = exception_name
        self.message = message
        self.stack_trace = stack_trace
        global _MEMENTO_EXCEPTION_REGEX
        match = re.match(_MEMENTO_EXCEPTION_REGEX, self.exception_name)
        if not match:
            raise ValueError("exception name must be in the form language::module:name")

    def to_exception(self) -> Exception:
        """
        Attempt to reconstitute the exception. If that attempt fails,
        returns `self`.

        """
        global _MEMENTO_EXCEPTION_REGEX
        match = re.match(_MEMENTO_EXCEPTION_REGEX, self.exception_name)
        if match:
            language = match.group(1)
            module_name = match.group(2)
            name = match.group(3)
            if language == "python":
                module = importlib.import_module(module_name)
                ref = module
                for part in name.split("."):
                    ref = getattr(ref, part)
                if not inspect.isclass(ref):
                    return self
                try:
                    # noinspection PyCallingNonCallable
                    return ref(
                        "{}. Original stack trace follows:\n{}".format(
                            self.message, self.stack_trace
                        )
                    )
                except TypeError:
                    # If we couldn't construct the exception (e.g. it has required parameters),
                    # just return this as a MementoException
                    return self
            else:
                # If this is an exception from another language, return this as a MementoException
                return self
        # If we couldn't match the name to the expected pattern, just return this as a
        # MementoException
        return self

    @staticmethod
    def from_exception(e: Exception) -> "MementoException":
        """
        Convert the provided exception into a MementoException,
        attempting to preserve enough data to re-constitute it.

        """
        # Fully qualify the name of the exception so we can
        # attempt to reconstitute it
        exc_class = e.__class__
        language = "python"
        module = exc_class.__module__
        qual_name = exc_class.__qualname__
        full_qual_name = "{}::{}:{}".format(language, module, qual_name)
        return MementoException(
            full_qual_name,
            str(e),
            "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        )


class NonMemoizedException(RuntimeError):
    """
    Exception thrown by the framework that is not to be memoized.
    """

    def __init__(self, message: str):
        super().__init__(message)


class MementoNotFoundError(NonMemoizedException):
    """
    Thrown in some cases when a memento could not be found (e.g. when writing a log message
    but couldn't associate it with a Memento)

    """

    def __init__(self, message: str):
        super().__init__(message)


class RemoteCallException(RuntimeError):
    """
    Thrown when `context.recursive.retry_on_remote_call` is `True` and a remote
    call is made. The set of dependent functions are included in the
    exception to allow the framework to record what needs to be computed
    first before retrying this function.

    """

    dependencies = None  # type: List[FunctionReferenceWithArgHash]

    def __init__(self, dependencies: List[FunctionReferenceWithArgHash]):
        super().__init__()
        self.dependencies = dependencies


class UndeclaredDependencyError(RuntimeError):
    """
    Thrown when a Memento function calls, either directly or indirectly, a function
    that is not declared or detected to be in its dependency chain.

    """

    def __init__(self, message: str):
        super().__init__(message)


class DependencyNotFoundError(RuntimeError):
    """
    Thrown when a Memento function declares a required dependency on a symbol and the symbol
    cannot be found.

    """

    def __init__(self, message: str):
        super().__init__(message)
