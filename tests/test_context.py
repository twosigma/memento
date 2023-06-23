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

from twosigma.memento.context import InvocationContext


class TestContext:
    """Class to test context objects."""

    def test_update_local(self):
        context = InvocationContext()
        assert not context.local.ignore_result
        assert not context.local.force_local
        context2 = context.update_local("force_local", True)
        assert not context.local.ignore_result
        assert not context.local.force_local
        assert not context2.local.ignore_result
        assert context2.local.force_local

    def test_readonly_fields(self):
        context = InvocationContext()
        assert not context.local.ignore_result

        def update_field(ctx: InvocationContext):
            ctx.local.ignore_result = True

        with pytest.raises(ValueError):
            update_field(context)
        assert not context.local.force_local

    def test_valid_property_name_enforced(self):
        context = InvocationContext()
        with pytest.raises(ValueError):
            context.local.update("non_existent_field", True)
