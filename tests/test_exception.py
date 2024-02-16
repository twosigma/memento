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

import pytest

from twosigma.memento.exception import MementoException


class TestMementoException:
    """Class to test MementoException."""

    def test_attr(self):
        e = MementoException(
            "python::builtins:ValueError", "test message", "test stack"
        )
        assert "python::builtins:ValueError" == e.exception_name
        assert "test message" == e.message
        assert "test stack" == e.stack_trace

    def test_validate_format_exception_name(self):
        with pytest.raises(ValueError):
            MementoException("bad_format", "message", "stack")

    def test_to_exception(self):
        e = MementoException(
            "python::builtins:ValueError", "test message", "stack trace"
        )
        assert isinstance(e.to_exception(), ValueError)
        assert -1 != str(e.to_exception()).find("test message")

    def test_from_exception(self):
        e = ValueError("test message")
        me = MementoException.from_exception(e)
        assert isinstance(me, MementoException)
        assert -1 != me.message.find("test message")

    def test_exception_from_another_language(self):
        e = MementoException(
            "java::java.lang.IllegalArgumentException", "test message", "stack trace"
        )
        assert isinstance(e.to_exception(), MementoException)
        assert -1 != str(e.to_exception()).find("test message")
