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
Configures logging for Memento.

All memento modules share a common logger. By default, INFO and up are
logged for memento modules, but this can be overridden by the
MEMENTO_LOG_LEVEL environment variable.

The default handler for memento log messages is to stream to stderr.
This can be overridden by an application using memento using the standard
Python logging APIs.

"""

import logging
import logging.config
import os


def _init_logging():
    result = logging.getLogger("memento")
    result.setLevel(os.getenv("MEMENTO_LOG_LEVEL", "WARN").upper())
    return result


log = _init_logging()


def set_log_level(level: int):
    """
    Convenience API for setting Memento's log level.
    Use constants from logging (e.g. `logging.INFO` is the default)

    Another way to set the log level is to use the `MEMENTO_LOG_LEVEL` environment
    variable.

    """
    global log
    log.setLevel(level)
