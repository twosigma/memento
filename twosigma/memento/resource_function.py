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
Functions that represent a dependency on an external resource
such as a file. The framework will treat invocations of
these functions as returning versions that change when the resource changes.

"""
import os
import pathlib
import urllib.parse
from typing import Callable
from urllib.request import url2pathname

from .resource import ResourceHandle, register_resource_type
from .call_stack import CallStack


class ResourceFunction:
    """
    Wrapper that registers that a caller calling a `resource_function`
    depends on a specific version of that resource.

    This class is not typically instantiated directly, but rather via
    the :func:`memento.resource_function` decorator.

    """

    fn = None  # type: Callable[[str], ResourceHandle]
    "The resource function being wrapped"

    def __init__(self, fn: Callable[[str], ResourceHandle]):
        self.fn = fn

    def __call__(self, *args, **kwargs) -> ResourceHandle:
        handle = self.fn(*args, **kwargs)
        call_stack = CallStack.get()
        caller_frame = call_stack.get_calling_frame()
        if caller_frame:
            caller_frame.memento.invocation_metadata.resources.append(handle)
        return handle


def resource_function(
    resource_type: str,
) -> Callable[[Callable[[str], ResourceHandle]], ResourceFunction]:
    """
    Decorator that causes a function to be treated as a Memento resource function.
    A resource function is a special type of function in Memento that represents a
    dependency on an external resource. The function accepts a url and returns a string that
    summarizes the state of the resource (e.g. a timestamp of a file).

    Resource functions do not belong to a cluster and they are not memoized.

    In general, one of the pre-defined resource functions can be used, but to
    create a new resource function, this decorator can be invoked in one of the following ways:

    .. code-block:: python

        import twosigma.memento as m

        @m.resource_function(resource_type="type")
        def b(url):
            pass

    :param resource_type:   The name of the resource

    """

    def decorator(f: Callable[[str], ResourceHandle]) -> ResourceFunction:
        result = ResourceFunction(f)
        register_resource_type(resource_type, result)
        return result

    return decorator


@resource_function(resource_type="file")
def file_resource(url: str) -> ResourceHandle:
    """
    Call this function to indicate a dependency on the file at the given path.
    The function will return a handle representing the version of the resource
    that is depended upon. The `ResourceHandle` embeds the URL and version of the
    file (a string representing the last modified time in milliseconds since epoch).

    :param url:     The path to the file, either as a `file:/` URI or an absolute pathname.

    """
    if not url.startswith("file:///"):
        url = pathlib.Path(url).absolute().as_uri()

    # convert URL to a plain path
    # See https://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
    parsed = urllib.parse.urlparse(url)
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    path = pathlib.Path(
        os.path.abspath(
            os.path.join(host, url2pathname(urllib.parse.unquote(parsed.path)))
        )
    )

    if not path.exists():
        version = "deleted"
    else:
        version = str(int(round(os.path.getmtime(path) * 1000)))

    return ResourceHandle("file", url, version)
