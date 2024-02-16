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
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    canonical_version = (3, 11)
    current_version = (sys.version_info.major, sys.version_info.minor)
    canonical_version_running = current_version == canonical_version

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_non_canonical_version = pytest.mark.skip(
        reason=f"need python canonical version {canonical_version} to run (running {current_version})"
    )

    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)

        if "needs_canonical_version" in item.keywords and not canonical_version_running:
            item.add_marker(skip_non_canonical_version)
