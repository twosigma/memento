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
Information about the stack of active memento functions in each thread of execution.
Memento uses thread local storage to keep the history of invocations that led up
to the execution of the current `MementoFunctionType`. This allows tracking of provenance
and staleness computation.

"""
import datetime
import threading
from typing import List  # noqa: F401

from .context import RecursiveContext
from .runner import RunnerBackend
from .metadata import InvocationMetadata, Memento
from .reference import FunctionReferenceWithArguments

# Use a ThreadLocal to track the call stack
# TODO: This is not safe if the function implementation spawns a thread
# TODO: This is not safe with coroutines - perhaps use aiocontextvars
_call_stack_thread_local = threading.local()


class StackFrame:
    """
    Tracks recursive invocations of memento functions to facilitate
    invalidation of downstream results.

    """

    memento = None  # type: Memento
    recursive_context = None  # type: RecursiveContext

    def __init__(
        self,
        fn_reference_with_args: FunctionReferenceWithArguments,
        runner: RunnerBackend,
        recursive_context: RecursiveContext,
    ):
        self.memento = Memento(
            time=datetime.datetime.now(datetime.timezone.utc),
            invocation_metadata=InvocationMetadata(
                runtime=None,
                fn_reference_with_args=fn_reference_with_args,
                result_type=None,
                invocations=[],
                resources=[],
            ),
            function_dependencies={fn_reference_with_args.fn_reference},
            runner=runner.to_dict(),
            correlation_id=recursive_context.correlation_id,
            content_key=None,
        )
        self.recursive_context = recursive_context

    def __repr__(self):
        return f"StackFrame(memento={self.memento}, recursive_context={self.recursive_context})"

    def __str__(self):
        return repr(self)


class CallStack:
    """
    Information about the active memento function calls that led to the current invocation.

    The call stack starts out empty when the first function is called and is once again empty
    when it returns. The `CallStack` is only available in-process - when a remote call is made,
    a new `CallStack` is created.

    """

    _frames = None  # type: List[StackFrame]

    def __init__(self):
        self._frames = []

    def depth(self) -> int:
        """
        Returns the number of frames in the stack
        """
        return len(self._frames) if self._frames else 0

    def get_calling_frame(self) -> StackFrame:
        """
        Returns the frame of the calling function, or `None` if this is the root function in the
        call chain.

        """
        return self._frames[-1] if self._frames else None

    def push_frame(self, frame: StackFrame):
        """
        Pushes the provided stack frame onto the call stack.

        """
        self._frames.append(frame)

    def pop_frame(self) -> StackFrame:
        """
        Removes the last stack frame from the call stack.

        """
        return self._frames.pop()

    @staticmethod
    def get() -> "CallStack":
        """
        Returns the instance of the call stack for this thread.

        """
        if not hasattr(_call_stack_thread_local, "call_stack"):
            _call_stack_thread_local.call_stack = CallStack()
        return _call_stack_thread_local.call_stack

    @staticmethod
    def swap(new_call_stack: "CallStack") -> "CallStack":
        """
        Returns the instance of the call stack for this thread and swaps
        it for another one.

        """
        if not hasattr(_call_stack_thread_local, "call_stack"):
            result = None
        else:
            result = _call_stack_thread_local.call_stack

        _call_stack_thread_local.call_stack = new_call_stack

        return result
