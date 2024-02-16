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

import datetime
from typing import List, Dict
import pandas as pd
import numpy as np
import pytest

from twosigma.memento import MementoFunction
from twosigma.memento.partition import InMemoryPartition
from twosigma.memento.exception import MementoException
from twosigma.memento.metadata import ResultType


class TestMetadata:

    def test_result_type_from_object(self):
        assert ResultType.exception == ResultType.from_object(
            MementoException("python::builtins:ValueError", "message", "stack_trace")
        )
        assert ResultType.null == ResultType.from_object(None)
        assert ResultType.boolean == ResultType.from_object(True)
        assert ResultType.string == ResultType.from_object("foo")
        assert ResultType.binary == ResultType.from_object(b"foo")
        assert ResultType.number == ResultType.from_object(1)
        assert ResultType.number == ResultType.from_object(1.2)
        with pytest.raises(ValueError):
            ResultType.from_object((3 + 4j))
        assert ResultType.timestamp == ResultType.from_object(
            datetime.datetime.now(datetime.timezone.utc)
        )
        assert ResultType.date == ResultType.from_object(datetime.date.today())
        assert ResultType.list_result == ResultType.from_object([1, 2])
        assert ResultType.dictionary == ResultType.from_object({"a": "b"})
        assert ResultType.index == ResultType.from_object(pd.Index([1, 2]))
        assert ResultType.series == ResultType.from_object(pd.Series([1, 2]))
        assert ResultType.data_frame == ResultType.from_object(
            pd.DataFrame({"a": [1, 2]})
        )
        assert ResultType.array_boolean == ResultType.from_object(
            np.array([True, False], dtype=np.dtype("?"))
        )
        assert ResultType.array_int8 == ResultType.from_object(
            np.array([1, 2], dtype=np.dtype("i1"))
        )
        assert ResultType.array_int16 == ResultType.from_object(
            np.array([1, 2], dtype=np.dtype("i2"))
        )
        assert ResultType.array_int32 == ResultType.from_object(
            np.array([1, 2], dtype=np.dtype("i4"))
        )
        assert ResultType.array_int64 == ResultType.from_object(
            np.array([1, 2], dtype=np.dtype("i8"))
        )
        assert ResultType.array_float32 == ResultType.from_object(
            np.array([1, 2], dtype=np.dtype("f4"))
        )
        assert ResultType.array_float64 == ResultType.from_object(
            np.array([1, 2], dtype=np.dtype("f8"))
        )
        assert ResultType.partition == ResultType.from_object(
            InMemoryPartition({"a": 1})
        )

    def test_result_type_from_annotation(self):
        assert ResultType.exception == ResultType.from_annotation(MementoException)
        assert ResultType.null == ResultType.from_annotation(None)
        assert ResultType.boolean == ResultType.from_annotation(bool)
        assert ResultType.string == ResultType.from_annotation(str)
        assert ResultType.binary == ResultType.from_annotation(bytes)
        assert ResultType.number == ResultType.from_annotation(int)
        assert ResultType.number == ResultType.from_annotation(float)
        with pytest.raises(ValueError):
            ResultType.from_annotation(complex)
        assert ResultType.timestamp == ResultType.from_annotation(datetime.datetime)
        assert ResultType.date == ResultType.from_annotation(datetime.date)
        assert ResultType.list_result == ResultType.from_annotation(list)
        assert ResultType.list_result == ResultType.from_annotation(List)
        assert ResultType.list_result == ResultType.from_annotation(List[str])
        assert ResultType.dictionary == ResultType.from_annotation(dict)
        assert ResultType.dictionary == ResultType.from_annotation(Dict)
        assert ResultType.index == ResultType.from_annotation(pd.Index)
        assert ResultType.series == ResultType.from_annotation(pd.Series)
        assert ResultType.data_frame == ResultType.from_annotation(pd.DataFrame)
        # Not sure how to represent np.ndarray type hints
        assert ResultType.partition == ResultType.from_annotation(InMemoryPartition)
        assert ResultType.memento_function == ResultType.from_annotation(
            MementoFunction
        )
