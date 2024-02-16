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
Base serialization capabilities

"""
import base64
import datetime
from typing import Dict, List, Union, Any, Callable, Optional, Tuple  # noqa: F401

import dateutil.parser
import numpy as np
import re

from twosigma.memento.context import RecursiveContext
from twosigma.memento.metadata import ResultType, InvocationMetadata, Memento
from twosigma.memento.reference import (
    FunctionReferenceWithArguments,
    FunctionReference,
    FunctionReferenceWithArgHash,
)
from twosigma.memento.resource import ResourceHandle
from twosigma.memento.types import (
    FunctionNotFoundError,
    MementoFunctionType,
    VersionedDataSourceKey,
)


class MementoCodec:
    """
    Handles serializing and deserializing a Memento object in a JSON format that is cross-language.

    """

    @classmethod
    def encode_datetime(cls, obj: Union[datetime.datetime, datetime.date]) -> str:
        return obj.isoformat().replace("+00:00", "Z")

    @classmethod
    def decode_datetime(cls, state: str) -> Union[datetime.date, datetime.datetime]:
        result = dateutil.parser.parse(state)
        if re.match(r"^\d\d\d\d-\d\d-\d\d$", state):
            return result.date()
        return result

    @classmethod
    def encode_memento(cls, memento: Memento) -> Dict:
        return {
            "time": cls.encode_datetime(memento.time),
            "invocationMetadata": cls.encode_invocation_metadata(
                memento.invocation_metadata
            ),
            "functionDependencies": (
                [cls.encode_fn_reference(x) for x in memento.function_dependencies]
                if memento is not None
                else []
            ),
            "runner": memento.runner,
            "correlationId": memento.correlation_id,
            "contentKey": cls.encode_versioned_data_source_key(memento.content_key),
        }

    @classmethod
    def decode_memento(cls, state: Dict) -> Memento:
        return Memento(
            time=cls.decode_datetime(state["time"]),
            invocation_metadata=cls.decode_invocation_metadata(
                state["invocationMetadata"]
            ),
            function_dependencies=(
                set(cls.decode_fn_reference(x) for x in state["functionDependencies"])
                if state["functionDependencies"] is not None
                else {}
            ),
            runner=state["runner"],
            correlation_id=state["correlationId"],
            content_key=cls.decode_versioned_data_source_key(state["contentKey"]),
        )

    @classmethod
    def encode_invocation_metadata(cls, obj: InvocationMetadata) -> Dict:
        return {
            "fnReferenceWithArgs": cls.encode_fn_reference_with_args(
                obj.fn_reference_with_args
            ),
            "invocations": (
                [cls.encode_fn_reference_with_args(x) for x in obj.invocations]
                if obj.invocations is not None
                else None
            ),
            "resources": (
                [cls.encode_resource_handle(x) for x in obj.resources]
                if obj.resources is not None
                else None
            ),
            "runtimeSeconds": obj.runtime.total_seconds(),
            "resultType": obj.result_type.name,
        }

    @classmethod
    def decode_invocation_metadata(cls, state: Dict) -> InvocationMetadata:
        return InvocationMetadata(
            fn_reference_with_args=cls.decode_fn_reference_with_args(
                state["fnReferenceWithArgs"]
            ),
            invocations=(
                [cls.decode_fn_reference_with_args(x) for x in state["invocations"]]
                if state["invocations"] is not None
                else None
            ),
            resources=(
                [cls.decode_resource_handle(x) for x in state["resources"]]
                if state["resources"] is not None
                else None
            ),
            runtime=datetime.timedelta(seconds=state["runtimeSeconds"]),
            result_type=ResultType[state["resultType"]],
        )

    @classmethod
    def encode_fn_reference_with_args(cls, obj: FunctionReferenceWithArguments) -> Dict:
        return {
            "fnReference": cls.encode_fn_reference(obj.fn_reference),
            "args": (
                [cls.encode_arg(x) for x in obj.args] if obj.args is not None else None
            ),
            "kwargs": (
                {k: cls.encode_arg(v) for (k, v) in obj.kwargs.items()}
                if obj.kwargs is not None
                else None
            ),
            "contextArgs": (
                {k: cls.encode_arg(v) for (k, v) in obj.context_args.items()}
                if obj.context_args is not None
                else None
            ),
        }

    @classmethod
    def decode_fn_reference_with_args(
        cls, state: Dict
    ) -> FunctionReferenceWithArguments:
        return FunctionReferenceWithArguments(
            fn_reference=cls.decode_fn_reference(state["fnReference"]),
            args=(
                tuple([cls.decode_arg(x) for x in state["args"]])
                if state["args"] is not None
                else None
            ),
            kwargs=(
                {k: cls.decode_arg(v) for (k, v) in state["kwargs"].items()}
                if state["kwargs"] is not None
                else None
            ),
            context_args=(
                {k: cls.decode_arg(v) for (k, v) in state["contextArgs"].items()}
                if state["contextArgs"] is not None
                else None
            ),
        )

    @classmethod
    def encode_fn_reference_with_arg_hash(
        cls, obj: FunctionReferenceWithArgHash
    ) -> Dict:
        return {
            "fnReference": cls.encode_fn_reference(obj.fn_reference),
            "argHash": obj.arg_hash,
        }

    @classmethod
    def decode_fn_reference_with_arg_hash(
        cls, state: Dict
    ) -> FunctionReferenceWithArgHash:
        return FunctionReferenceWithArgHash(
            fn_reference=cls.decode_fn_reference(state["fnReference"]),
            arg_hash=state["argHash"],
        )

    @classmethod
    def encode_resource_handle(cls, obj: ResourceHandle) -> Dict:
        return {
            "resourceType": obj.resource_type,
            "url": obj.url,
            "version": obj.version,
        }

    @classmethod
    def decode_resource_handle(cls, state: Dict) -> ResourceHandle:
        return ResourceHandle(
            resource_type=state["resourceType"],
            url=state["url"],
            version=state["version"],
        )

    @classmethod
    def encode_fn_reference(cls, obj: FunctionReference) -> Dict:
        return {
            "qualifiedName": obj.qualified_name,
            "partialArgs": (
                [cls.encode_arg(x) for x in obj.partial_args]
                if obj.partial_args is not None
                else None
            ),
            "partialKwargs": (
                {k: cls.encode_arg(v) for (k, v) in obj.partial_kwargs.items()}
                if obj.partial_kwargs is not None
                else None
            ),
            "parameterNames": obj.parameter_names,
        }

    @classmethod
    def decode_fn_reference(cls, state: Dict) -> FunctionReference:
        return FunctionReference.from_qualified_name(
            qualified_name=state["qualifiedName"],
            partial_args=(
                tuple([cls.decode_arg(x) for x in state["partialArgs"]])
                if state["partialArgs"] is not None
                else None
            ),
            partial_kwargs=(
                {k: cls.decode_arg(v) for (k, v) in state["partialKwargs"].items()}
                if state["partialKwargs"] is not None
                else None
            ),
            parameter_names=state["parameterNames"],
        )

    @classmethod
    def encode_recursive_context(cls, obj: RecursiveContext) -> Dict:
        return {
            "correlationId": obj.correlation_id,
            "retryOnRemoteCall": obj.retry_on_remote_call,
            "preventFurtherCalls": obj.prevent_further_calls,
            "contextArgs": cls.encode_arg(obj.context_args),
        }

    @classmethod
    def decode_recursive_context(cls, state: Dict) -> RecursiveContext:
        return RecursiveContext(
            correlation_id=state["correlationId"],
            retry_on_remote_call=state["retryOnRemoteCall"],
            prevent_further_calls=state["preventFurtherCalls"],
            context_args=cls.decode_arg(state["contextArgs"]),
        )

    @classmethod
    def encode_versioned_data_source_key(
        cls, content_key: VersionedDataSourceKey
    ) -> Optional[str]:
        if content_key is None:
            return None
        return "{}#{}".format(content_key.key, content_key.version)

    @classmethod
    def decode_versioned_data_source_key(
        cls, state: Optional[str]
    ) -> Optional[VersionedDataSourceKey]:
        if state is None:
            return None
        hash_index = state.rfind("#")
        return VersionedDataSourceKey(
            key=state[0:hash_index], version=state[hash_index + 1 :]
        )

    @classmethod
    def encode_arg(cls, obj: Any) -> Dict:
        """
        If the arg is a function, replace it with a FunctionReference, else if a tuple, list or
        map, look recursively for functions and convert them to FunctionReferences.

        """
        if obj is None:
            return {"type": ResultType.null.name}
        elif isinstance(obj, bool):
            return {"type": ResultType.boolean.name, "value": obj}
        elif isinstance(obj, str):
            return {"type": ResultType.string.name, "value": obj}
        elif isinstance(obj, bytes):
            return {"type": ResultType.binary.name, "value": base64.b64encode(obj)}
        elif isinstance(obj, int) or isinstance(obj, float):
            return {"type": ResultType.number.name, "value": obj}
        elif isinstance(obj, np.ndarray):
            if obj.ndim > 1:
                raise ValueError(
                    "Memento does not support serializing array arguments of "
                    "more than 1 dimension"
                )

            if obj.dtype == np.bool:
                type_str = ResultType.array_boolean.name
            elif obj.dtype == np.int8:
                type_str = ResultType.array_int8.name
            elif obj.dtype == np.int16:
                type_str = ResultType.array_int16.name
            elif obj.dtype == np.int32:
                type_str = ResultType.array_int32.name
            elif obj.dtype == np.int64:
                type_str = ResultType.array_int64.name
            elif obj.dtype == np.float32:
                type_str = ResultType.array_float32.name
            elif obj.dtype == np.float64:
                type_str = ResultType.array_float64.name
            else:
                raise ValueError("Unknown numpy array type: {}".format(obj.dtype))

            return {"type": type_str, "value": [x for x in obj]}
        elif isinstance(obj, Callable):
            if not isinstance(obj, MementoFunctionType):
                raise ValueError(
                    "{} is callable but not a MementoFunctionType".format(obj)
                )
            return {
                "type": "twosigma.memento.FunctionReference",
                "value": cls.encode_fn_reference(obj.fn_reference()),
            }
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return {
                "type": ResultType.list_result.name,
                "value": [cls.encode_arg(x) for x in obj],
            }
        elif isinstance(obj, dict):
            return {
                "type": ResultType.dictionary.name,
                "value": {k: cls.encode_arg(v) for (k, v) in obj.items()},
            }
        elif isinstance(obj, datetime.datetime):
            return {
                "type": ResultType.timestamp.name,
                "value": cls.encode_datetime(obj),
            }
        elif isinstance(obj, datetime.date):
            return {"type": ResultType.date.name, "value": cls.encode_datetime(obj)}
        else:
            raise ValueError("Cannot encode argument of type {}".format(type(obj)))

    @classmethod
    def decode_arg(cls, state: Dict) -> Any:
        """
        If the arg is a FunctionReference, replace it with the function it refers to,
        else if a tuple, list or map, look recursively for FunctionReferences and convert
        them to their realized functions.

        """
        if type(state) is not dict:
            raise ValueError(
                "During deserialization of argument, state was not a dict."
            )

        obj_type = state["type"]
        if not obj_type:
            raise ValueError(
                "During deserialization of argument, state did not have 'type' attribute."
            )

        if obj_type == ResultType.null.name:
            return None

        value = state["value"]

        if (
            obj_type == ResultType.boolean.name
            or obj_type == ResultType.string.name
            or obj_type == ResultType.number.name
        ):
            return value
        elif obj_type == ResultType.binary.name:
            return base64.b64decode(value)
        elif obj_type == ResultType.array_boolean.name:
            return np.array(value, dtype=np.bool)
        elif obj_type == ResultType.array_int8.name:
            return np.array(value, dtype=np.int8)
        elif obj_type == ResultType.array_int16.name:
            return np.array(value, dtype=np.int16)
        elif obj_type == ResultType.array_int32.name:
            return np.array(value, dtype=np.int32)
        elif obj_type == ResultType.array_int64.name:
            return np.array(value, dtype=np.int64)
        elif obj_type == ResultType.array_float32.name:
            return np.array(value, dtype=np.float32)
        elif obj_type == ResultType.array_float64.name:
            return np.array(value, dtype=np.float64)
        elif obj_type == "twosigma.memento.FunctionReference":
            fn_reference = cls.decode_fn_reference(value)
            if fn_reference.memento_fn is None:
                raise FunctionNotFoundError(
                    "Could not deserialize: Could not replace fn_reference with fn for {}. "
                    "This could be because the function is not in the path or because of a "
                    "version mismatch.".format(fn_reference.qualified_name)
                )
            return fn_reference.memento_fn
        elif obj_type == ResultType.list_result.name:
            return [cls.decode_arg(x) for x in value]
        elif obj_type == ResultType.dictionary.name:
            return {k: cls.decode_arg(v) for (k, v) in value.items()}
        elif obj_type == ResultType.date.name:
            return cls.decode_datetime(value)
        elif obj_type == ResultType.timestamp.name:
            return cls.decode_datetime(value)
        else:
            raise ValueError("Cannot decode argument of type {}".format(obj_type))
