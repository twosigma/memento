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

from twosigma.memento.partition import InMemoryPartition
from twosigma.memento.storage_filesystem import OnDiskPartition


class TestPartitionTest:

    def test_inmemory_partition(self):
        partition = InMemoryPartition({"a": 1, "b": [1, 2, 3]})
        assert ["a", "b"] == sorted(partition.list_keys())
        assert 1 == partition.get("a")
        assert [1, 2, 3] == partition.get("b")

    def test_ondisk_partition(self):
        partition = OnDiskPartition()
        partition["a"] = 1
        partition["b"] = [1, 2, 3]
        assert ["a", "b"] == sorted(partition.list_keys())
        assert 1 == partition.get("a")
        assert [1, 2, 3] == partition.get("b")

    def test_inmemory_partition_not_in_list(self):
        partition = InMemoryPartition({"a": 1, "b": [1, 2, 3]})
        with pytest.raises(ValueError):
            partition.get("c")

    def test_ondisk_partition_not_in_list(self):
        partition = OnDiskPartition()
        partition["a"] = 1
        partition["b"] = [1, 2, 3]
        with pytest.raises(ValueError):
            partition.get("c")

    def test_list_keys_without_merge_parent_inmemory(self):
        # Note: There is a pickle version of this test in test_storage_backend
        partition_a = InMemoryPartition({"a": 1, "b": 2})
        partition_b = InMemoryPartition({"b": 3, "c": 4})
        partition_b._merge_parent = partition_a
        assert ["a", "b", "c"] == sorted(partition_b.list_keys())
        assert ["b", "c"] == sorted(partition_b.list_keys(_include_merge_parent=False))

    def test_list_keys_without_merge_parent_ondisk(self):
        partition_a = OnDiskPartition()
        partition_a["a"] = 1
        partition_a["b"] = 2
        partition_b = OnDiskPartition()
        partition_b["b"] = 3
        partition_b["c"] = 4
        partition_b._merge_parent = partition_a
        assert ["a", "b", "c"] == sorted(partition_b.list_keys())
        assert ["b", "c"] == sorted(partition_b.list_keys(_include_merge_parent=False))
