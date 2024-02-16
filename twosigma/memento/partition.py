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
When re-shuffling data it is often useful to be able to
have a function return multiple outputs, each of which
is efficiently independently loadable.

A partition has a string key and arbitrary values.

When written, all the data are provided and
the values are written to different storage blocks for
each key.

When read, a handle to the partition is returned. No
data is transferred until a key is given, at which point
the data for that key is retrieved.

Example
=======

>>> from twosigma.memento import memento_function
>>> from collections import defaultdict
>>>
>>> @memento_function
>>> def words_by_letter():
>>>     result = defaultdict(lambda: [])
>>>     # provided by `apt-get install wamerican`
>>>     with open("/usr/share/dict/words") as f:
>>>         for word in f.readlines():
>>>             first_letter = word[0].upper()
>>>             result[first_letter].append(word.rstrip())
>>>     return InMemoryPartition(result)
>>>
>>> r = words_by_letter()
>>>
>>> # List keys in the result
>>> r.list_keys()
dict_keys(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Å', 'É'])
>>> # Get only the words that begin with A
>>> r.get("A")
['A', "A's", 'AMD', "AMD's", 'AOL']

"""
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional


class Partition(ABC):
    """
    Abstract base class for Partition implementations. See the documentation for
    :py:mod:`twosigma.memento.partition` for more details.

    """

    _index_bytes = None  # type: Optional[bytes]
    """Encoded version of the index, for serialization"""

    _merge_parent = None  # type: Optional[Partition]
    """
    If set, self's index will be merged with this other partition's index upon serialization,
    resulting in a merged partition.
    """

    @abstractmethod
    def __init__(self):
        self._index_bytes = None
        self._merge_parent = None

    @abstractmethod
    def get(self, key: str) -> object:
        """
        Return the object associated with the given key

        """
        pass

    @abstractmethod
    def list_keys(self, _include_merge_parent: bool = True) -> Iterable[str]:
        """
        List all of the keys present in this partition

        """
        pass


class InMemoryPartition(Partition):
    """
    Partition that stores all results in memory. This is typically
    the partition used by functions when they wish to return data.

    """

    _results = None  # type: Dict[str, object]

    _output_keys = None  # type: Optional[Dict]
    """
    If set, tracks where the output keys were written when this partition was last serialized.
    This allows the partition to be stored in the in-memory cache and still serve as a parent
    for a merged partition.
    """

    _parent_data_source = None
    """
    If set, tracks the data source where output keys were written when this partition was last
    serialized.
    """

    def __init__(self, results: Dict[str, object]):
        """
        Creates an in-memory partition with all the results stored in
        the provided dictionary.

        :param results:     The results. The key must be a string.

        """
        super().__init__()
        self._results = results
        self._output_keys = None
        self._parent_data_source = None

    def get(self, key: str) -> object:
        if key not in self._results:
            if self._merge_parent:
                return self._merge_parent.get(key)
            raise ValueError("Key '{}' not in key list for partition".format(key))
        return self._results[key]

    def list_keys(self, _include_merge_parent: bool = True) -> Iterable[str]:
        if _include_merge_parent and self._merge_parent:
            result = set(self._merge_parent.list_keys())
            result.update(self._results.keys())
            return sorted(list(result))

        return sorted(list(self._results.keys()))
