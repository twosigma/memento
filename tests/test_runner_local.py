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

import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep

import twosigma.memento as m

from twosigma.memento import (
    RunnerBackend,
    Environment,
    ConfigurationRepository,
    FunctionCluster,
)  # noqa: F401
from twosigma.memento.context import InvocationContext  # noqa: F401
from twosigma.memento.runner_local import LocalRunnerBackend
from twosigma.memento.runner_test import (
    set_runner_fn_test_1_called,
    get_runner_fn_test_1_called,
    runner_fn_test_1,
    fn_A,
    fn_B,
)
from twosigma.memento.storage_filesystem import FilesystemStorageBackend
from tests.test_runner_backend import RunnerBackendTester


changed = None
times_run = None
barrier = None


# Turn off auto-dependencies, else global variable changed will introduce version change
@m.memento_function(auto_dependencies=False, cluster="cluster1")
def fn1():
    global changed
    changed = True
    return 1


# Turn off auto-dependencies, else global variable changed will introduce version change
@m.memento_function(auto_dependencies=False, cluster="cluster1")
def long_running_fn(a):
    sleep(0.1)
    # noinspection PyUnresolvedReferences
    times_run[a] = times_run[a] + 1
    return a + 1


class TestRunnerLocal(RunnerBackendTester):
    """
    Test the local runner backend.

    """

    backend = None  # type: RunnerBackend

    def setup_method(self):
        super().setup_method()
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_runner_local_test")
        self.data_path = "{}/data".format(self.base_path)
        m.Environment.set(
            Environment(
                name="test1",
                base_dir=self.base_path,
                repos=[
                    ConfigurationRepository(
                        name="repo1",
                        clusters={
                            "cluster1": FunctionCluster(
                                name="cluster1",
                                storage=FilesystemStorageBackend(path=self.data_path),
                                runner=LocalRunnerBackend(),
                            ),
                            "memento.unit_test": FunctionCluster(
                                name="memento.unit_test",
                                storage=FilesystemStorageBackend(path=self.data_path),
                                runner=LocalRunnerBackend(),
                            ),
                        },
                    )
                ],
            )
        )

        self.cluster = m.Environment.get().get_cluster("cluster1")
        self.backend = self.cluster.runner

    def teardown_method(self):
        shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)
        super().teardown_method()

    def test_memoize_in_process(self):
        global changed
        # This variable is mutated, to prove the change happened in-process
        changed = False

        assert not changed
        assert 1 == fn1()
        assert changed

    def test_ignore_result(self):
        # Test that ignoring return memoizes the result but does not return it right away
        set_runner_fn_test_1_called(False)

        # Ensure context is properly modified
        context = runner_fn_test_1.context  # type: InvocationContext
        assert not context.local.ignore_result
        context = runner_fn_test_1.ignore_result().context  # type: InvocationContext
        assert context.local.ignore_result

        # Ensure context of original MementoFunction is not modified
        context = runner_fn_test_1.context  # type: InvocationContext
        assert not context.local.ignore_result

        assert runner_fn_test_1.ignore_result().call(1, 2) is None

        # should have been called locally
        assert get_runner_fn_test_1_called()

        set_runner_fn_test_1_called(False)
        result = runner_fn_test_1(1, 2)

        # should have retrieved memoized results
        assert not get_runner_fn_test_1_called()

        assert 2 == result["b"]

    def test_memoized_invocations(self):
        """
        Test memoized invocations.
        When a function A calls another function E that has already been
        memoized, then function E should be added to A's invocation list
        """
        fn_B()
        fn_A()
        mem_a = fn_A.memento()
        assert len(mem_a.invocation_metadata.invocations) > 0
        assert len(mem_a.function_dependencies) > 0

    def test_mutex(self):
        """
        Tests that the runner enforces a mutex on already-running jobs, so that
        effort is not duplicated.

        """
        global times_run, barrier
        times_run = defaultdict(lambda: 0)
        barrier = Barrier(3)

        def run_long_running_fn(a):
            # Wait for all of the threads to be ready and then call the function
            # this reduces the likelihood there was some delay with the ThreadPoolExecutor
            barrier.wait(timeout=10)
            return long_running_fn(a)

        with ThreadPoolExecutor(max_workers=3) as tpe:
            future1a = tpe.submit(run_long_running_fn, a=1)
            future2 = tpe.submit(run_long_running_fn, a=2)
            future1b = tpe.submit(run_long_running_fn, a=1)

        assert 2 == future1a.result(timeout=10)
        assert 3 == future2.result(timeout=10)
        assert 2 == future1b.result(timeout=10)
        assert 1 == times_run[1]
        assert 1 == times_run[2]
