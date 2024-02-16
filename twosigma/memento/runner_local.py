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
Implements a runner backend that executes functions in the local process.
Functions are not run in parallel.

"""
import datetime
from collections import defaultdict
from threading import RLock
from typing import Any, List, Dict, Tuple, Optional  # noqa: F401

from tqdm.auto import tqdm

from .call_stack import CallStack, StackFrame
from .exception import MementoException, RemoteCallException, NonMemoizedException
from .context import InvocationContext
from .logging import log
from .metadata import Memento, ResultType
from .reference import FunctionReferenceWithArguments
from .result import KeyOverrideResult
from .runner import RunnerBackend, process_existing_memento, ExistingMementoResult
from .storage import StorageBackend

_memento_fn_mutex_lock = (
    RLock()
)  # Lock to protect the defaultdict since it is not ThreadSafe
# with the lambda
_memento_fn_mutex = defaultdict(lambda: RLock())  # type: Dict[Tuple[str, str], RLock]


def _mutex_for_invocation(
    fn_reference_with_args: FunctionReferenceWithArguments,
) -> RLock:
    """
    Check if any other callers in this process are calling this function at the same
    time and, if so, wait for them to complete.

    Note that the current implementation leaves Lock objects around for each
    historical invocation, which is a slow memory leak.

    """
    with _memento_fn_mutex_lock:
        return _memento_fn_mutex[
            (
                fn_reference_with_args.fn_reference.qualified_name,
                fn_reference_with_args.arg_hash,
            )
        ]


class LocalRunnerBackend(RunnerBackend):
    """
    Implements a runner backend that executes functions in the local process.
    Functions are not run in parallel.

    """

    def __init__(self, config: dict = None):
        super().__init__("local", config=config)

    def batch_run(
        self,
        context: InvocationContext,
        storage_backend: StorageBackend,
        fn_reference_with_args: List[FunctionReferenceWithArguments],
        log_runner_backend: RunnerBackend,
        caller_memento: Optional[Memento],
    ) -> List[Any]:
        context = self.ensure_correlation_id(context, fn_reference_with_args)

        arg_list = fn_reference_with_args
        results = []  # type: List[Any]

        # Bulk query for existing mementos
        existing_mementos = storage_backend.get_mementos(
            [f.fn_reference_with_arg_hash() for f in fn_reference_with_args]
        )

        if (
            context.local.monitor_progress
            and CallStack.get().get_calling_frame() is None
        ):
            # Only show progress bar if monitor_progress is set and this is the root call
            arg_list = tqdm(arg_list)

        # Since memento_run_local handles updating the invocation list of the caller already, the
        # local runner can ignore caller_memento.
        for idx, f in enumerate(arg_list):
            existing_memento_result = ExistingMementoResult(
                result=None, valid_result=False
            )
            existing_memento = existing_mementos[idx]
            if existing_memento:
                existing_memento_result = process_existing_memento(
                    storage_backend, existing_memento, context.local.ignore_result
                )
            if existing_memento_result.valid_result:
                results.append(existing_memento_result.result)

                # Have to make sure that this memoized function is included in the
                # invocation list of the calling function.
                call_stack = CallStack.get()
                calling_frame = call_stack.get_calling_frame()
                if calling_frame:
                    propagate_dependencies(
                        caller_memento=calling_frame.memento,
                        result_memento=existing_memento,
                    )

            else:
                try:
                    results.append(
                        memento_run_local(
                            context=context,
                            fn_reference_with_args=f,
                            storage_backend=storage_backend,
                            log_runner_backend=log_runner_backend,
                        )
                    )
                except Exception as e:
                    results.append(e)
        return results

    def to_dict(self):
        config = {"type": "local"}
        return config


RunnerBackend.register("local", LocalRunnerBackend)


def memento_run_batch(
    context: InvocationContext,
    fn_reference_with_args: List[FunctionReferenceWithArguments],
    storage_backend: StorageBackend,
    runner_backend: RunnerBackend,
    log_runner_backend: RunnerBackend,
) -> List[Any]:
    """
    Run a batch of Memento functions, using the given runner, with arguments.
    All calls to MementoFunctionTypes pass through this function, on the client side.

    :param context:                 Invocation context for this function run
    :param fn_reference_with_args:  `Iterable` of function reference with arguments baked in
    :param storage_backend:         Store the memoized results using this storage backend
    :param runner_backend:          Run the function using this runner
    :param log_runner_backend:      Use this runner when writing the log. This is used
                                    when the function is run in a cluster - initial call
                                    will be a cluster runner and then the function is run
                                    again on the local machine. We want to log the original
                                    runner.
    :return:                        A list, the same list as fn_reference_with_args, with
                                    whatever each function returns. If the function raised
                                    an exception, the exception is placed in that slot in
                                    the list.

    """
    # Submit the batch to the runner
    runner = LocalRunnerBackend() if context.local.force_local else runner_backend

    call_stack = CallStack.get()
    calling_frame = call_stack.get_calling_frame()
    caller_memento = calling_frame.memento if calling_frame else None

    if calling_frame and calling_frame.recursive_context.prevent_further_calls:
        raise RuntimeError("Further Memento calls are prevented in this context.")

    if caller_memento:
        # If the caller memento exists, pass down its correlation id and other fields
        context = context.update_recursive(
            "correlation_id", caller_memento.correlation_id
        )
        context = context.update_recursive(
            "retry_on_remote_call", calling_frame.recursive_context.retry_on_remote_call
        )
        if context.recursive.context_args is None:
            # Only update the context args from the call stack if not overridden in this call
            context = context.update_recursive(
                "context_args", calling_frame.recursive_context.context_args
            )

            # Update the fn_reference_with_args since the context args could change the
            # argument hash
            fn_reference_with_args = [
                FunctionReferenceWithArguments(
                    ref.fn_reference,
                    ref.args,
                    ref.kwargs,
                    context.recursive.context_args,
                )
                for ref in fn_reference_with_args
            ]

    return runner.batch_run(
        context=context,
        storage_backend=storage_backend,
        fn_reference_with_args=fn_reference_with_args,
        log_runner_backend=log_runner_backend,
        caller_memento=caller_memento,
    )


def propagate_dependencies(caller_memento: Memento, result_memento: Memento):
    """
    Add this invocation to the list of invocations made by the previous function, and
    propagate function_dependencies up the stack.

    :param caller_memento:  The caller's memento, to be updated
    :param result_memento:  The memento for the function that was just invoked

    """
    fn_reference_with_args = result_memento.invocation_metadata.fn_reference_with_args

    caller_memento.invocation_metadata.invocations.append(fn_reference_with_args)

    parent_dependencies = caller_memento.function_dependencies
    parent_dependencies.add(fn_reference_with_args.fn_reference)
    parent_dependencies |= result_memento.function_dependencies


def memento_run_local(
    context: InvocationContext,
    fn_reference_with_args: FunctionReferenceWithArguments,
    storage_backend: StorageBackend,
    log_runner_backend: RunnerBackend,
) -> Any:
    """
    Run a single Memento function in the local process, with arguments.
    This is typically not called directly, but by a runner implementation.
    This is where the logic of reading and generating memoized results lives.

    :param context:                 Invocation context for this function run
    :param fn_reference_with_args:  `Iterable` of function reference with arguments baked in
    :param storage_backend:         Store the memoized results using this storage backend
    :param log_runner_backend:      Use this runner when writing the log. This is used
                                    when the function is run in a cluster - initial call
                                    will be a cluster runner and then the function is run
                                    again on the local machine. We want to log the original
                                    runner.
    :return:                        A list, the same list as fn_reference_with_args, with
                                    whatever each function returns. If the function raised
                                    an exception, the exception is placed in that slot in
                                    the list.

    """

    # Ensure we have a correlation id - this is generally set by the runner in batch_run if not
    # already set
    assert context.recursive.correlation_id
    correlation_id = context.recursive.correlation_id

    # Log what we're calling
    log.debug(
        "{}: Calling {} with context {}".format(
            correlation_id, fn_reference_with_args, context
        )
    )

    # Create a stack frame for the invocation
    log_runner = (
        LocalRunnerBackend() if context.local.force_local else log_runner_backend
    )
    call_stack = CallStack.get()
    stack_frame = StackFrame(fn_reference_with_args, log_runner, context.recursive)

    # Acquire a lock for the invocation
    with _mutex_for_invocation(fn_reference_with_args):
        try:
            call_stack.push_frame(stack_frame)

            existing_memento = storage_backend.get_memento(
                fn_reference_with_args.fn_reference_with_arg_hash()
            )
            if existing_memento:
                existing_memento_result = process_existing_memento(
                    storage_backend, existing_memento, context.local.ignore_result
                )
                if existing_memento_result.valid_result:
                    stack_frame.memento = existing_memento
                    return existing_memento_result.result

            # Otherwise, compute, store and return
            time_start = datetime.datetime.now(datetime.timezone.utc)
            exception_result = None
            # noinspection PyBroadException
            try:
                log.debug("{}: Function call begins".format(correlation_id))
                # noinspection PyProtectedMember
                result = fn_reference_with_args.fn_reference.memento_fn._filter_call(
                    **fn_reference_with_args.effective_kwargs
                )
            except RemoteCallException:
                # Special exception thrown if a remote call is made while processing
                # the function. This should immediately stop processing and not memoize
                # the result. The function will be retried by the framework later.
                log.debug(
                    "Remote call detected during function execution. Retrying later."
                )
                raise
            except NonMemoizedException:
                # If the exception is marked not to be memoized, just raise it
                raise
            except Exception as e:
                result = MementoException.from_exception(e)
                exception_result = e
            time_end = datetime.datetime.now(datetime.timezone.utc)
            # TODO: There is an issue with this runtime - it includes the runtime of its
            #       children, but depending on whether the children are cached, this will come
            #       out different. We should change this to subtract out the runtime of children
            stack_frame.memento.invocation_metadata.runtime = time_end - time_start

            # Unwrap KeyOverrideResult
            key_override = None
            if isinstance(result, KeyOverrideResult):
                key_override = result.key_override
                result = result.result

            stack_frame.memento.invocation_metadata.result_type = (
                ResultType.from_object(result)
            )

            # Memoize the result if it is not already present.
            # There are two primary reasons a memoized result could have appeared while
            # the function was being invoked:
            #     1. This function was invoked in another thread or process and we lost the race
            #     2. A runner is being used that memoized the result in another process (possibly
            #        on another machine in a compute cluster) and the result is already memoized.
            if not storage_backend.is_memoized(
                fn_reference_with_args.fn_reference, fn_reference_with_args.arg_hash
            ):
                try:
                    storage_backend.memoize(key_override, stack_frame.memento, result)
                    memoization_status = "successfully memoized"
                except IOError:
                    log.warning(
                        "IO Error while writing memoized result.", exc_info=True
                    )
                    memoization_status = "memoization failed to write result"
            else:
                memoization_status = (
                    "memoized elsewhere while we were computing the result"
                )

            if (
                context.local.ignore_result
                and stack_frame.memento.invocation_metadata.result_type
                != ResultType.exception
            ):
                log.debug(
                    "{}: Result was computed, {} and is ignored".format(
                        correlation_id, memoization_status
                    )
                )
                return

            # If an exception occurred, raise it instead of returning the result
            if exception_result is not None:
                log.debug(
                    "{}: Result was computed, {} and is an exception: {}: {}".format(
                        correlation_id,
                        memoization_status,
                        type(exception_result).__name__,
                        exception_result,
                    )
                )
                return exception_result

            log.debug(
                "{}: Result was computed, {} and is of type {}".format(
                    correlation_id,
                    memoization_status,
                    stack_frame.memento.invocation_metadata.result_type.name,
                )
            )
            return result
        finally:
            log.debug("{}: Function call ends".format(correlation_id))
            call_stack.pop_frame()
            calling_frame = call_stack.get_calling_frame()
            if calling_frame:
                propagate_dependencies(
                    caller_memento=calling_frame.memento,
                    result_memento=stack_frame.memento,
                )
