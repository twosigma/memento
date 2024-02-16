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
from typing import List, Dict, Any, Optional, Callable

from pandas import DataFrame

from twosigma.memento import Environment, Memento
from twosigma.memento.exception import MementoNotFoundError
from twosigma.memento.logging import log
from twosigma.memento.reference import FunctionReferenceWithArguments, validate_args
from twosigma.memento.runner_local import memento_run_batch
from twosigma.memento.types import MementoFunctionType


class MementoFunctionBase(MementoFunctionType, ABC):
    """
    Common base class for MementoFunctions
    """

    def call(self, *args, **kwargs):
        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        if cluster_config is None:
            raise ValueError(
                "No cluster found with name {}".format(fn_reference.cluster_name)
            )
        storage_backend = cluster_config.storage
        runner_backend = cluster_config.runner

        # Call function as a batch function of size one.
        results = memento_run_batch(
            context=self.context,
            fn_reference_with_args=[
                FunctionReferenceWithArguments(
                    fn_reference,
                    args,
                    kwargs,
                    context_args=self.context.recursive.context_args,
                )
            ],
            storage_backend=storage_backend,
            runner_backend=runner_backend,
            log_runner_backend=runner_backend,
        )

        result = results[0]
        if isinstance(result, Exception):
            raise result

        return result

    def call_batch(
        self, kwargs_list: List[Dict[str, Any]], raise_first_exception=True
    ) -> List[Any]:
        """
        Evaluates this function several times, in batch with the provided arguments.

        The runner backend for the function may decide to parallelize the computation,
        if supported.

        :param kwargs_list:     The list of arguments to evaluate. Each element of the list
                                is a dict of kwarg keys to values.
        :param raise_first_exception:  If at least one result is an exception, raise the first
                                exception.
        :return:  A list of results, in the same order as `kwargs_list`. If any
                  of the functions raised an exception, the exception instance
                  will be returned in that slot.

        """

        # Ensure all args have strings as keys
        for kwargs in kwargs_list:
            if any([type(key) for key in kwargs.keys() if type(key) != str]):
                raise TypeError(
                    "Keys must be strings for all kwargs in kwargs list. Got {}".format(
                        kwargs_list
                    )
                )

        # Get cluster configuration
        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        if cluster_config is None:
            raise ValueError(
                "No cluster found with name {}".format(fn_reference.cluster_name)
            )
        storage_backend = cluster_config.storage
        runner_backend = cluster_config.runner

        # Construct a list of FunctionReferenceWithArguments
        fns = [
            FunctionReferenceWithArguments(
                fn_reference,
                args=(),
                kwargs=kwargs,
                context_args=self.context.recursive.context_args,
            )
            for kwargs in kwargs_list
        ]

        # Invoke the functions in a batch
        result = memento_run_batch(
            context=self.context,
            fn_reference_with_args=fns,
            storage_backend=storage_backend,
            runner_backend=runner_backend,
            log_runner_backend=runner_backend,
        )

        if raise_first_exception:
            for r in result:
                if isinstance(r, Exception):
                    raise r

        return result

    def map_over_range(self, **kwargs) -> Dict:
        # noinspection PyUnresolvedReferences
        """
        Evaluates this function over the provided range of arguments.
        Exactly one `kwarg` must be specified, whose name must match the name of a
        function parameter, and whose value is expected to be an `Iterable`,
        enumerating each of the values for which to call the function. All other
        parameters to the function must be partially evaluated.

        The runner backend for the function may decide to parallelize the computation,
        if supported.

        Example:

        >>> @memento_function
        >>> def add(x, y):
        ...     # noinspection PyUnresolvedReferences
        ...     return x + y
        >>>
        >>> add2 = add.partial(x=2)
        >>>
        >>> result = add2.map_over_range(y=range(1, 4))
        >>>
        >>> assert result[1] == 3
        >>> assert result[2] == 4
        >>> assert result[3] == 5

        :param kwargs:      A single `kwarg` specifying the name of the parameter and
                            `Iterable` over the range of values for which to call the function.
        :return:  A map from parameter value to the value returned for that invocation.

        """

        assert kwargs is not None, "kwargs must not be None"

        keys = kwargs.keys()
        assert (
            len(keys) == 1
        ), "kwargs must contain exactly one key, corresponding to a fn parameter name"

        name = next(x for x in keys)
        values = kwargs[name]

        # Build list of things to evaluate
        value_list = list(values)
        to_evaluate = [{name: val} for val in value_list]

        # Evaluate in batch:
        result_list = self.call_batch(to_evaluate)

        # Return results in a dict
        return {value_list[idx]: result_list[idx] for idx in range(0, len(value_list))}

    def forget(self, *args, **kwargs):
        """
        Instead of running the function, forget the memoized results for running
        the function with the given arguments.

        To forget the entire function, call :meth:`forget_all`
        """

        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        storage_backend = cluster_config.storage

        # Forget only the results for the provided parameters
        fn_reference_with_args = fn_reference.with_args(
            *args, **kwargs, _memento_context_args=self.context.recursive.context_args
        )
        arg_hash = fn_reference_with_args.arg_hash

        log.info(
            "Forgetting {} for arg hash {}".format(
                fn_reference.qualified_name, arg_hash
            )
        )
        storage_backend.forget_call(fn_reference_with_args.fn_reference_with_arg_hash())

    def forget_all(self):
        """
        Instead of running the function, forget the memoized results for running
        the function. This version forgets memoization for all arguments.

        """

        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        storage_backend = cluster_config.storage
        log.info("Forgetting {} for all arg hashes".format(fn_reference.qualified_name))
        storage_backend.forget_function(fn_reference)

    def get_metadata(self, key: str, args=None, kwargs=None) -> Optional[bytes]:
        """
        Read custom metadata for the given arguments from the given
        key. This is useful, for example, for reading logs
        associated with a given memoized result.

        :param key:         The metadata key
        :param args:        The arguments for the invocation
        :param kwargs:      The kwargs for the invocation
        :return:            The data, or None if no such metadata exists.
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        if cluster_config is None:
            raise ValueError(
                "No cluster found with name {}".format(fn_reference.cluster_name)
            )
        storage_backend = cluster_config.storage
        fa = FunctionReferenceWithArguments(
            fn_reference,
            args=args,
            kwargs=kwargs,
            context_args=self.context.recursive.context_args,
        )
        memento = storage_backend.get_memento(fa.fn_reference_with_arg_hash())
        if memento:
            return storage_backend.read_metadata(
                memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
                key,
            )
        else:
            return None

    def with_prevent_further_calls(self, prevent_calls: bool):
        """
        Sets whether calls after this one are prevented (`True`) or allowed (`False`).

        """
        new_context = self.context.update_recursive(
            "prevent_further_calls", prevent_calls
        )
        return self.clone_with(context=new_context)

    def with_context_args(self, context_args: Dict[str, Any]):
        """
        Add context arguments that are passed recursively to any downstream invocations.
        These arguments are typically used by frameworks. The arguments do not appear
        in the arg or kwargs list but they do affect the argument hash.

        The arguments provided completely replace the existing context args. To call this
        properly, retrieve the existing context args and clone the dict before adding it.
        Never modify the dict in place.

        """
        assert (
            context_args is not self.context.recursive.context_args
        ), "Context arg dict must be cloned, not modified in-place"
        new_context = self.context.update_recursive("context_args", context_args)
        return self.clone_with(context=new_context)

    def ignore_result(self, ignore: bool = True):
        """
        Function modifier to ignore the return value of the invocation. This optimizes
        performance for cases where the runner may be remote and we do not care to
        deserialize the return value because it will not be used.

        Exceptions will still be propagated even if ignore_result is set.

        """
        new_context = self.context.update_local("ignore_result", ignore)
        return self.clone_with(context=new_context)

    def list_mementos(self, limit: int = None) -> List[Memento]:
        """
        List the arguments memoized for this function.

        :param limit:   If specified, limits the number of returned results

        :return:        A list of :class:`Memento`s.

        """
        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        storage = cluster_config.storage
        return storage.list_mementos(fn_reference, limit=limit)

    def logs(self, *args, **kwargs) -> str:
        """
        Returns the logs for the invocation to this function with the given arguments,
        if any logs exist, or None if no logs exist or if the function was never
        run with the given arguments.

        """
        log_metadata = self.get_metadata("log", args=args, kwargs=kwargs)
        if log_metadata is None:
            raise ValueError("No logs found for provided arguments")
        return log_metadata.decode("utf-8")

    def memento(self, *args, **kwargs) -> Memento:
        """
        Returns the memento for the invocation with the given args and kwargs, or
        `None` if does not exist or if the memento could not be read, or there was an issue
        finding a referenced function or version of a function.

        """
        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        storage = cluster_config.storage
        fa = FunctionReferenceWithArguments(
            fn_reference,
            args=args,
            kwargs=kwargs,
            context_args=self.context.recursive.context_args,
        )
        return storage.get_memento(fa.fn_reference_with_arg_hash())

    def monitor_progress(self, monitor: bool = True):
        """
        Function modifier to indicate that a progress indicator should be
        rendered during the invocation of this function.

        """
        new_context = self.context.update_local("monitor_progress", monitor)
        return self.clone_with(context=new_context)

    def partial(self, *partial_args, **partial_kwargs) -> MementoFunctionType:
        # noinspection PyUnresolvedReferences
        """
        Partially evaluates a :class:`MementoFunctionType`, producing another
        :class:`MementoFunctionType` that has some of its parameters partially
        evaluated. For example, if we have a function:

        >>> import twosigma.memento as m
        >>> @m.memento_function
        >>> def add(x, y):
        ...     return x + y

        Then we can define a new function :code:`add3 = add.partial(3)`.
        :code:`add3` is a one-parameter function, where :code:`add3(y) = 3 + y`

        >>> add3 = add.partial(x=3)
        >>> assert 4 == add3(1)

        Use this as an alternative to passing code like
        :code:`lambda y: add(3, y)` because a lambda function cannot be
        hashed properly as an argument to a memento function.

        Successive calls to partial append args and update kwargs in the
        context.

        """

        validate_args(*partial_args, **partial_kwargs)
        fn_reference = self.fn_reference()
        new_partial_args = fn_reference.partial_args or ()
        new_partial_args += partial_args
        new_partial_kwargs = (
            dict(fn_reference.partial_kwargs)
            if fn_reference.partial_kwargs is not None
            else {}
        )
        new_partial_kwargs.update(partial_kwargs)
        return self.clone_with(
            partial_args=new_partial_args, partial_kwargs=new_partial_kwargs
        )

    def put_metadata(
        self, key: str, value: bytes, *args, store_with_data: bool = False, **kwargs
    ):
        """
        Write custom metadata for the given arguments to the given
        key. This is useful, for example, for writing logs to be
        associated with a given output.

        Any previously written data will be overwritten.

        Note that not all runners support metadata.

        :param key:         The metadata key
        :param value:       The value to write
        :param store_with_data:  If True, the metadata is stored with the data instead of in
                            the metadata store.
        :param args:        The arguments for the invocation
        :param kwargs:      The kwargs for the invocation
        """
        fn_reference = self.fn_reference()
        cluster_config = Environment.get().get_cluster(fn_reference.cluster_name)
        if cluster_config is None:
            raise ValueError(
                "No cluster found with name {}".format(fn_reference.cluster_name)
            )
        storage_backend = cluster_config.storage
        fa = FunctionReferenceWithArguments(
            fn_reference,
            args=args,
            kwargs=kwargs,
            context_args=self.context.recursive.context_args,
        )
        memento = storage_backend.get_memento(fa.fn_reference_with_arg_hash())
        if memento is None:
            raise MementoNotFoundError("No memento found with provided arguments")

        store_with_content_key = None
        if store_with_data:
            store_with_content_key = memento.content_key

        storage_backend.write_metadata(
            memento.invocation_metadata.fn_reference_with_args.fn_reference_with_arg_hash(),
            key,
            value,
            store_with_content_key=store_with_content_key,
        )

    def force_local(self, local: bool = True):
        """
        Function modifier to force this function be run with a
        :class:`LocalRunnerBackend` even if a different runner is configured for this
        cluster. This is useful for debugging and cases where the overhead of making
        the invocation remote is not worthwhile.

        """
        new_context = self.context.update_local("force_local", local)
        return self.clone_with(context=new_context)

    def list(self, *args, **kwargs) -> DataFrame:
        """
        Return a DataFrame containing a list of parameter combinations for previously-memoized
        results.

        Only memoized results for the current context are returned.

        If any kwargs are specified, results will be filtered to those rows matching the
        provided kwargs.

        The resulting dataframe contains one column for each argument. Args are expanded to
        effective kwargs in the results. An additional column is provided called result_type
        containing the type of result.

        If an invocation of a function that takes no args is made, only a single row with the
        result type is returned.

        If no results match, `None` is returned.
        """
        results = []
        effective_kwargs = (
            self.fn_reference().with_args(*args, **kwargs).effective_kwargs
        )
        context_args = self.context.recursive.context_args
        if context_args is None:
            context_args = {}
        for memento in self.list_mementos():
            fn_reference_with_args = memento.invocation_metadata.fn_reference_with_args
            if context_args == fn_reference_with_args.context_args:
                # Filter out any non-matching kwargs
                memento_kwargs = fn_reference_with_args.effective_kwargs
                match = True
                row = {}
                for key, value in effective_kwargs.items():
                    if key in memento_kwargs and memento_kwargs[key] != value:
                        match = False
                        break
                if match:
                    for key, value in memento_kwargs.items():
                        row[key] = value
                    row["result_type"] = memento.invocation_metadata.result_type.name
                    results.append(row)

        return DataFrame(data=results) if len(results) > 0 else None

    def _call_with_contexts_from_context_args(self, fn: Callable) -> Any:
        """
        Used by the framework to call the provided function with the appropriate
        contexts in place given the context for this function.

        The default behavior is to just call the function, but other frameworks can
        override this behavior. This is particularly useful for server-side invocations
        to restore the context embedded in the request.
        """
        return fn()

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
