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

"""Public interface for memento module."""
import importlib

from importlib.metadata import entry_points

from .logging import set_log_level

from . import serialization  # noqa: F401

from .storage import StorageBackend
from . import storage_null  # noqa: F401
from . import storage_filesystem  # noqa: F401
from . import storage_memory  # noqa: F401

from .runner import RunnerBackend
from . import runner_local  # noqa: F401
from . import runner_null  # noqa: F401

from .reference import FunctionReference

from .configuration import ConfigurationRepository, FunctionCluster, Environment

from .metadata import Memento, InvocationMetadata

from .memento import (
    forget_cluster,
    list_memoized_functions,
    memento_function,
    MementoFunction,
)

from .resource_function import file_resource

__all__ = [
    "set_log_level",
    "StorageBackend",
    "RunnerBackend",
    "FunctionReference",
    "forget_cluster",
    "file_resource",
    "list_memoized_functions",
    "memento_function",
    "MementoFunction",
    "Memento",
    "InvocationMetadata",
    "ConfigurationRepository",
    "FunctionCluster",
    "Environment",
]


def _load_plugins():
    """
    Dynamically load plugins by importing their modules.
    """
    for entry_point in entry_points():
        if "group" in entry_point and entry_point.group == "twosigma.memento.plugin":
            importlib.import_module(entry_point.name)


_load_plugins()
