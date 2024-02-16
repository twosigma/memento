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
Implements a runner backend that refuses to execute functions
and instead forces either reading memoized data or fails.

"""
from typing import Any, List, Optional

from .metadata import Memento
from .context import InvocationContext
from .reference import FunctionReferenceWithArguments
from .storage import StorageBackend
from .runner import RunnerBackend


class NullRunnerBackend(RunnerBackend):
    def __init__(self, config: dict = None):
        super().__init__("null", config=config)

    def batch_run(
        self,
        context: InvocationContext,
        storage_backend: StorageBackend,
        fn_reference_with_args: List[FunctionReferenceWithArguments],
        log_runner_backend: RunnerBackend,
        caller_memento: Optional[Memento],
    ) -> List[Any]:
        raise RuntimeError("Null runner refusing to run functions")

    def to_dict(self):
        config = {"type": "null"}
        return config


RunnerBackend.register("null", NullRunnerBackend)
