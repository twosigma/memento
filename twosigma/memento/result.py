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


class KeyOverrideResult:
    """
    Wraps a result and indicates it should be written at the given key instead of the
    automatically-computed key.
    """

    result = None  # type: object
    key_override = None  # type: str

    def __init__(self, result: object, key_override: str):
        self.result = result
        self.key_override = key_override
