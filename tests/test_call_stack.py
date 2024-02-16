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

import threading

from twosigma.memento import Environment, memento_function
from twosigma.memento.call_stack import CallStack, StackFrame
from twosigma.memento.context import RecursiveContext


@memento_function
def fn_sample_1():
    return 1


class TestCallStack:
    """Class to test Memento call stack."""

    def setup_method(self):
        self.call_stack_1 = None
        self.call_stack_2 = None

    def test_get_call_stack(self):
        call_stack = CallStack.get()
        assert call_stack is not None

    def test_threads_have_different_call_stacks(self):
        def get_stack_1():
            self.call_stack_1 = CallStack.get()

        def get_stack_2():
            self.call_stack_2 = CallStack.get()

        t1 = threading.Thread(target=get_stack_1)
        t2 = threading.Thread(target=get_stack_2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert self.call_stack_1 is not None
        assert self.call_stack_2 is not None
        assert self.call_stack_1 is not self.call_stack_2

    def test_no_caller(self):
        call_stack = CallStack.get()
        assert call_stack.get_calling_frame() is None
        assert call_stack.depth() == 0

    def test_push_pop_caller(self):
        corr_id = "12345"
        recursive_context = RecursiveContext()
        recursive_context.update("correlation_id", corr_id)
        call_stack = CallStack.get()
        frame1 = StackFrame(
            fn_sample_1.fn_reference().with_args(),
            Environment.get().default_cluster.runner,
            recursive_context,
        )
        frame2 = StackFrame(
            fn_sample_1.fn_reference().with_args(),
            Environment.get().default_cluster.runner,
            recursive_context,
        )
        call_stack.push_frame(frame1)
        assert call_stack.depth() == 1
        assert frame1 is call_stack.get_calling_frame()
        call_stack.push_frame(frame2)
        assert call_stack.depth() == 2
        assert frame2 is call_stack.get_calling_frame()
        assert frame2 is call_stack.pop_frame()
        assert call_stack.depth() == 1
        assert frame1 is call_stack.pop_frame()
        assert call_stack.depth() == 0
        assert call_stack.get_calling_frame() is None
