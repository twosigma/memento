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


# The set of known resource types, updated whenever the @resource_function decorator is used.
from typing import Callable, Dict  # noqa: F401

_registered_resource_types = {}  # type: Dict[str, Callable]


class ResourceHandle:
    """
    Handle that embeds the URL and version of a resource that is depended upon.

    """

    resource_type = None  # type: str
    url = None  # type: str
    version = None  # type: str

    def __init__(self, resource_type: str, url: str, version: str):
        self.resource_type = resource_type
        self.url = url
        self.version = version

    def __eq__(self, o: "ResourceHandle"):
        return isinstance(o, ResourceHandle) and self.__dict__ == o.__dict__

    def __hash__(self):
        return hash((self.resource_type, self.url, self.version))

    def __repr__(self):
        return "ResourceHandle(resource_type={}, url={}, version={})".format(
            repr(self.resource_type), repr(self.url), repr(self.version)
        )

    def __str__(self):
        return self.__repr__()


def register_resource_type(resource_type: str, resource_fn: Callable):
    """
    Registers that the given resource type should use the given resource function to compute
    versions. The resource_fn is an instance of `ResourceFunction` which cannot be imported due
    to circular references.

    """
    _registered_resource_types[resource_type] = resource_fn
