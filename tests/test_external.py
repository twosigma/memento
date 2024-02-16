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

from typing import Callable, Tuple, Any, Dict, List, Union

from twosigma.memento import FunctionReference
from twosigma.memento.context import InvocationContext

from twosigma.memento.external import ExternalMementoFunctionBase
from twosigma.memento.types import MementoFunctionType


class SampleExternalMementoFunction(ExternalMementoFunctionBase):
    def __init__(self, fn_reference: FunctionReference, context: InvocationContext):
        super().__init__(fn_reference, context, "test", hash_rules=list())

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
        pass


class TestExternal:
    """Class to test external function handlers."""

    def test_register_external(self):
        types = ExternalMementoFunctionBase.get_registered_function_type_classes()
        assert "memento_function" in types.keys()
        assert "test" not in types.keys()
        ExternalMementoFunctionBase.register("test", SampleExternalMementoFunction)
        assert "memento_function" in types.keys()
        assert "test" in types.keys()
        assert SampleExternalMementoFunction == types.get("test")
