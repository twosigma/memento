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
Configuration management for Memento.

Memento defines the concept of environments, configuration repositories and
data function clusters.

To set the environment, use :py:method:`Environment.set`

"""
import hashlib
import json
import io
import os
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict, Optional, Set  # noqa: F401

import pandas
import yaml
from jinja2 import Template

from twosigma.memento.types import MementoFunctionType
from .logging import log
from .runner import RunnerBackend
from .storage import StorageBackend

_DEFAULT_STORAGE_TYPE = "filesystem"
_DEFAULT_STORAGE_CONFIG = {}

_DEFAULT_RUNNER_TYPE = "local"
_DEFAULT_RUNNER_CONFIG = {}

# Bytes used to salt all function hashes.
# WARNING: Changing anything in this environment hash will force re-evaluation of everything!
ENVIRONMENT_HASH_BYTES = hashlib.sha256(
    json.dumps(
        {
            "memento_serialization_version": 7,  # Updated when a backwards-incompatible
            # change is made to the file format
            "packages": {
                "pandas": pandas.__version__[
                    0 : pandas.__version__.find(".")
                ]  # major version
            },
        },
        sort_keys=True,
    ).encode("utf-8")
).digest()

# Global collection of registered functions, persists beyond environment creation and destruction
_registered_function_names = set()  # type: Set[str]
_registered_functions = defaultdict(
    lambda: list()
)  # type: Dict[str, List[MementoFunctionType]]

# environment is lazily-loaded the first time Environment.get() is called.
environment = None


def _load_config(base_dir: str, config: Union[str, Dict], **kwargs) -> Dict:
    """
    Return the configuration, if already an object, or if a string, load the
    configuration file at the given path, in either json or yaml format.
    If this is parsed from a file, the property `base_dir` is added to the
    config so that paths can be resolved relative to this config file, if present.

    :param base_dir:    The base directory to evaluate paths relative to
    :param config:      The object or the path to the configuration file.
    :param kwargs:      If specified, expand these kwargs, treating the file
                        as a jinja2 template.
    :return: A Python object with the deserialized configuration
    :raises IOError: If the file format could not be detected or if a directory was specified

    """
    base_dir_path = Path(base_dir) if base_dir else None

    if isinstance(config, str):
        config_path = Path(config)
        if not config_path.is_absolute():
            if base_dir_path is None:
                raise FileNotFoundError(
                    "Could not evaluate relative path '{}' "
                    "as there is no base directory defined".format(config_path)
                )
            # this is a relative path. Prepend base_dir
            config_path = base_dir_path.joinpath(config_path)

        if config_path.is_dir():
            raise FileNotFoundError(
                "Expected file but got directory: {}".format(config_path)
            )

        if not config_path.is_file():
            raise FileNotFoundError(
                "Could not find configuration file: {}".format(config_path)
            )

        # Open as a jinja2 template and perform parameter substitution
        with config_path.open("r") as f:
            substituted_file = io.StringIO(Template(f.read()).render(**kwargs))

        if config_path.name.endswith(".json"):
            result = json.load(substituted_file)
        elif (
            config_path.name.endswith(".yaml")
            or config_path.name.endswith(".yml")
            or config_path.name.endswith(".jinja")
        ):
            result = yaml.safe_load(substituted_file)
        else:
            raise IOError(
                "Unknown config file extension for file {}".format(config_path)
            )

        result["base_dir"] = str(config_path.parent)

        return result
    else:
        return config


class FunctionCluster:
    """
    Each data function cluster has configuration that binds the data function
    to a specific backend and specifies the location and connection parameters
    for that backend. It also stores optional metadata for the cluster.

    A data function cluster is configured using a JSON file using the following
    format:

    .. code-block:: json

        {
            "name": "com.example.vendor.foo.product1",
            "description": "Functions about product 1",
            "maintainer": "Team Name",
            "documentation": "https://...",
            "storage": {
                "type": "filesystem",
                "url": "file:///dev/null",
                "readonly": true
            }
        }

    """

    config = None  # type: Dict
    name = None  # type: str
    description = None  # type: str
    maintainer = None  # type: str
    documentation = None  # type: str
    storage = None  # type: StorageBackend
    runner = None  # type: RunnerBackend

    locked = None  # type: bool
    """
    A locked cluster allows no additional functions to be defined and freezes the versions of the
    existing functions. This improves performance when the user is positive that no further
    function updates will occur to the cluster.

    """

    def __init__(
        self,
        config: Dict = None,
        name: str = None,
        description: str = None,
        maintainer: str = None,
        documentation: str = None,
        storage: StorageBackend = None,
        runner: RunnerBackend = None,
    ):
        """
        Create a new DataFunctionCluster from the provided configuration.

        :param config:          The configuration object for this data function cluster
        :param name:            Name of the function cluster. Must be provided either
                                in the `config` parameter or in this parameter. If provided,
                                this parameter overrides the `name` from the `config` parameter.
        :param description:     Description for this function cluster. If provided,
                                overrides the `description` from the `config` parameter.
        :param maintainer:      Maintainer of this function cluster. If provided,
                                overrides the `maintainer` from the `config` parameter.
        :param documentation:   Documentation for this function cluster. If provided,
                                overrides the `documentation` from the `config` parameter.
        :param storage:         Storage to use for this function cluster. If provided,
                                overrides the `storage` from the `config` parameter.
                                If provided in neither place, this defaults to filesystem
                                storage.
        :param runner:          Runner to use for this function cluster. If provided,
                                overrides the `runner` from the `config` parameter.
                                If provided in neither place, this defaults to a local,
                                in-process runner.

        """
        self.locked = False
        self.config = config if config is not None else {}

        # Process configuration
        self.name = self.config.get("name", None)
        self.description = self.config.get("description", None)
        self.maintainer = self.config.get("maintainer", None)
        self.documentation = self.config.get("documentation", None)

        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if maintainer is not None:
            self.maintainer = maintainer
        if documentation is not None:
            self.documentation = documentation

        if self.name is None:
            raise ValueError("Must provide a name for FunctionCluster")

        if storage is not None:
            self.storage = storage
        elif "storage" not in self.config:
            self.storage = StorageBackend.create(
                _DEFAULT_STORAGE_TYPE, _DEFAULT_STORAGE_CONFIG
            )
        else:
            storage_config = self.config["storage"]
            if "type" not in storage_config:
                raise ValueError(
                    "Missing required parameter 'type' in storage "
                    "configuration {}".format(self.name)
                )
            storage_type = storage_config["type"]
            self.storage = StorageBackend.create(storage_type, storage_config)

        if runner is not None:
            self.runner = runner
        elif "runner" not in self.config:
            self.runner = RunnerBackend.create(
                _DEFAULT_RUNNER_TYPE, _DEFAULT_RUNNER_CONFIG
            )
        else:
            runner_config = self.config["runner"]
            if "type" not in runner_config:
                raise ValueError(
                    "Missing required parameter 'type' in runner "
                    "configuration {}".format(self.name)
                )
            runner_type = runner_config["type"]
            self.runner = RunnerBackend.create(runner_type, runner_config)

    def to_dict(self):
        """
        Return a dict representation of this function cluster.

        """

        config = {
            "name": self.name,
            "storage": self.storage.to_dict(),
            "runner": self.runner.to_dict(),
        }
        if self.description is not None:
            config["description"] = self.description
        if self.maintainer is not None:
            config["maintainer"] = self.maintainer
        if self.documentation is not None:
            config["documentation"] = self.documentation
        return config


class _DefaultFunctionCluster(FunctionCluster):
    """
    The function cluster functions belong to when no cluster is specified.

    """

    def __init__(self, env: "Environment"):

        env_path = env.get_base_dir() or str(
            Path("~").expanduser().joinpath(".memento", "env", env.name)
        )
        base_path = Path(env_path).joinpath("cluster", "default")
        super().__init__(
            {
                "name": "default",
                "description": "Default function cluster",
                "storage": {
                    "type": "filesystem",
                    "path": str(base_path),
                    "readonly": False,
                },
            }
        )


class ConfigurationRepository:
    """
    Repository containing all the information needed to locate the code,
    data and metadata for data function clusters.

    Configuration Repositories are placed in priority order into an
    :py:class:`Environment`.

    A configuration repo is configured using a JSON file, normally called
    `memento.json`. The format of the file is as follows:

    .. code-block:: json
        {
            "name": "raw-vendor-data",
            "description": "Raw data from the vendor",
            "maintainer": "Team Name",
            "documentation": "https://...",
            "modules": [
                "a.b.c"
            ],
            "clusters": {
                "com.example.vendor.foo.product1": "/abspath/to/memento/product1.json",
                "com.example.vendor.foo.product1": "relpath/to/memento/product2.json",
                "com.example.vendor.foo.product2": {...config for product 3...}
            }
        }

    """

    config = None  # type: Dict
    name = None  # type: str
    base_dir = None  # type: str
    description = None  # type: str
    maintainer = None  # type: str
    documentation = None  # type: str
    clusters = None  # type: Dict[str, FunctionCluster]
    modules = None  # type: List[str]

    def __init__(
        self,
        config: Dict = None,
        name: str = None,
        base_dir: str = None,
        description: str = None,
        maintainer: str = None,
        documentation: str = None,
        clusters: Dict[str, FunctionCluster] = None,
        modules: List[str] = None,
    ):
        """
        Create a new ConfigurationRepository from the config file located at
        the provided path.

        :param config:          Configuration object. Must have a "name" at a minimum.
        :param name:            Name of the configuration repository. Must be provided either
                                in the `config` parameter or in this parameter. If provided,
                                this parameter overrides the `name` from the `config` parameter.
        :param base_dir:        Base directory from which to evaluate relative paths. If
                                provided, overrides the `base_dir` from the `config` parameter.
        :param description:     Description for this configuration repository. If provided,
                                overrides the `description` from the `config` parameter.
        :param maintainer:      Maintainer of this configuration repository. If provided,
                                overrides the `maintainer` from the `config` parameter.
        :param documentation:   Documentation for this configuration repository. If provided,
                                overrides the `documentation` from the `config` parameter.
        :param clusters:        Map of cluster name to `FunctionCluster` for this configuration
                                repository. If provided, overrides the `clusters` from the
                                `config` parameter.
        :param modules:         List of Python modules to import in order to instantiate the
                                Memento functions in this environment.

        """
        # Process configuration
        self.config = config if config is not None else {}
        self.name = self.config.get("name", None)
        self.base_dir = self.config.get("base_dir", None)
        self.description = self.config.get("description", None)
        self.maintainer = self.config.get("maintainer", None)
        self.documentation = self.config.get("documentation", None)
        self.clusters = dict()
        for cluster_name, config_path in self.config.get("clusters", {}).items():
            self.clusters[cluster_name] = FunctionCluster(
                _load_config(self.base_dir, config_path)
            )
        self.modules = self.config.get("modules", None)

        if name is not None:
            self.name = name
        if base_dir is not None:
            self.base_dir = base_dir
        if description is not None:
            self.description = description
        if maintainer is not None:
            self.maintainer = maintainer
        if documentation is not None:
            self.documentation = documentation
        if clusters is not None:
            self.clusters = clusters
        if modules is not None:
            self.modules = modules

        if self.modules is None:
            self.modules = []

        if self.name is None:
            raise ValueError("Must provide a name for ConfigurationRepository")

    def to_dict(self):
        """
        Return a dict representation of this configuration repository

        """
        config = {
            "name": self.name,
            "modules": self.modules,
            "clusters": {k: v.to_dict() for (k, v) in self.clusters.items()},
        }
        if self.base_dir is not None:
            config["base_dir"] = self.base_dir
        if self.description is not None:
            config["description"] = self.description
        if self.maintainer is not None:
            config["maintainer"] = self.maintainer
        if self.documentation is not None:
            config["documentation"] = self.documentation
        return config

    @staticmethod
    def from_file(path: str, **kwargs) -> "ConfigurationRepository":
        """
        Read configuration from a file, using a jinja2 template to substitute
        any kwargs specified.

        """
        return ConfigurationRepository(
            _load_config(os.path.dirname(path), os.path.basename(path), **kwargs)
        )


class Environment:
    """
    Each Memento Environment maintains a prioritized collection of
    configuration repositories.

    To switch environments, use the `MEMENTO_ENV` environment variable
    or call :py:meth:`Environment.set`.

    If `MEMENTO_ENV` is not specified, Memento will scan for a default
    configuration file in ~/.memento/env/default/env.json or env.yaml.
    If not found, a minimal default configuration will be used.

    """

    config = None  # type: Dict
    name = None  # type: str
    base_dir = None  # type: str
    repos = None  # type: List[ConfigurationRepository]
    default_cluster = None  # type: FunctionCluster
    _last_env_log = None  # type: str

    """
    String the last time the environment was logged - this is so we can log
    a message when the environment changes
    """

    def __init__(
        self,
        config: Dict = None,
        name: str = None,
        base_dir: str = None,
        repos: List[ConfigurationRepository] = None,
    ):
        """
        Create a new environment from the provided configuration object.
        The expected form of the config object is:

        .. code-block:: json
            {
                "name": "EnvName",
                "repos": [
                    "/abspath/to/repo1",
                    "relpath/to/repo2",
                    {...repo config object...},
                    ...
                ]
            }

        Repos contains a list of strings and objects. Strings are interpreted to be
        file paths that are either absolute (if they begin with `/`) or relative.
        Objects are interpreted to be inline configuration.

        :param config:      Configuration object
        :param name:        The name of the environment. If specified, overrides the
                            `name` from the `config` parameter.
        :param base_dir:    The base directory to evaluate relative paths. If specified,
                            overrides the `base_dir` from the `config` parameter.
        :param repos:       A list of configuration repositories as ConfigurationRepository
                            objects. If specified, overrides the `repos` from the
                            config parameter.


        """
        config = config if config is not None else {}
        self.config = config
        self.name = config.get("name", "default")
        self.base_dir = config.get("base_dir", None)
        self.repos = [
            ConfigurationRepository(_load_config(self.base_dir, repo_config))
            for repo_config in config.get("repos", [])
        ]

        if name is not None:
            self.name = name
        if base_dir is not None:
            self.base_dir = base_dir
        if repos is not None:
            self.repos = repos
        self.default_cluster = _DefaultFunctionCluster(self)

    def get_cluster(self, cluster_name: Optional[str]) -> Optional[FunctionCluster]:
        """
        Searches the environment for a function cluster with the given name and
        returns its configuration.

        :param cluster_name:    The name of the cluster to find, or None if the default
                                cluster should be returned
        :return:                The function cluster, or None if no cluster was found

        """
        if cluster_name is None:
            return self.default_cluster

        for repo in self.repos:
            if cluster_name in repo.clusters:
                return repo.clusters[cluster_name]
        return None

    @staticmethod
    def get_registered_clusters() -> List[str]:
        """
        Return a list of registered clusters. The "" cluster is the default cluster.

        """
        return list(sorted(_registered_functions.keys()))

    @staticmethod
    def is_function_registered(qualified_name: str) -> bool:
        return qualified_name in _registered_function_names

    @staticmethod
    def register_function(cluster_name: Optional[str], fn: MementoFunctionType):
        """
        Registers that the given function claims to be part of the cluster with
        the given name. Pass `None` as cluster_name for the default cluster.

        Note the cluster may or may not exist at the time of the registration, but
        the registration is tracked regardless.

        """
        qn = fn.fn_reference().qualified_name
        if Environment.is_function_registered(qn):
            # Already registered
            return

        # Check if cluster is locked
        cluster = Environment.get().get_cluster(cluster_name)
        if cluster is not None and cluster.locked:
            raise ValueError(
                "Cluster {} is locked to new functions".format(cluster_name)
            )

        # Use "" as a key if this is the default cluster
        if cluster_name is None:
            cluster_name = ""

        if qn not in _registered_function_names:
            _registered_function_names.add(qn)
            _registered_functions[cluster_name].append(fn)
            log.debug("Registered memento function for {}".format(qn))

    @staticmethod
    def get_registered_functions(cluster_name: str) -> List[MementoFunctionType]:
        """
        Returns a list of registered functions associated with this function cluster.
        Pass "" as cluster_name for the default cluster.

        """
        if cluster_name is None:
            cluster_name = ""
        return _registered_functions[cluster_name]

    def get_base_dir(self):
        """
        Returns the base directory of this environment, or `None` if the environment
        was constructed from an object.

        """
        return self.base_dir

    def get_repo(self, repo_name) -> Optional[ConfigurationRepository]:
        """
        return the repository with the given name if it exists in the environment,
        or `None` if it does not exist.

        """
        for r in self.repos:
            if r.name == repo_name:
                return r
        return None

    def append_repo(self, repo: ConfigurationRepository):
        """
        Adds the given configuration repository with lower priority than the other
        repositories registered.

        """
        self.repos.append(repo)

    def prepend_repo(self, repo: ConfigurationRepository):
        """
        Adds the given configuration repository with higher priority than the other
        repositories registered.

        """
        self.repos.insert(0, repo)

    def to_dict(self):
        """
        Return a dict representation of this environment

        """
        config = {"name": self.name, "repos": [repo.to_dict() for repo in self.repos]}
        if self.base_dir is not None:
            config["base_dir"] = self.base_dir
        return config

    @classmethod
    def set(cls, config: ["Environment", str, object]):
        """
        Switch Memento's default environment.

        This is a programmatic alternative to using the `MEMENTO_ENV`
        environment variable.

        :param config The path to the environment configuration file,
                      or a configuration object, or an Environment object.

        """
        global environment
        environment = (
            config
            if isinstance(config, Environment)
            else Environment(_load_config(os.getcwd(), config))
        )

    @classmethod
    def get(cls) -> "Environment":
        """
        Return Memento's current default environment.

        """
        global environment
        if environment is None:
            environment = _load_environment()

        env_str = str(environment.to_dict())
        if env_str != cls._last_env_log:
            cls._last_env_log = env_str
            log.info("Using environment: {}".format(env_str))
        return environment

    @classmethod
    def from_file(cls, memento_env_path: str) -> "Environment":
        base_dir = os.path.basename(memento_env_path)
        return Environment(_load_config(base_dir, memento_env_path))


def _load_environment() -> Environment:
    memento_env = os.getenv("MEMENTO_ENV")
    if not memento_env:
        # Search for default environment file
        default_config_file = (
            Path("~").expanduser().joinpath(".memento", "env", "default", "env")
        )
        for ext in [".json", ".yaml"]:
            filename = default_config_file.with_name(default_config_file.name + ext)
            if filename.is_file():
                memento_env = str(filename)
                break
    if memento_env:
        return Environment.from_file(memento_env)
    else:
        # No config file found. Use a suitable default
        return Environment(name="default")
