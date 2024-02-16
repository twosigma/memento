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

import json
import os
import shutil
import tempfile
from typing import cast

import pytest

import twosigma.memento as m
from twosigma.memento import ConfigurationRepository, Environment, FunctionCluster
from twosigma.memento.storage_filesystem import FilesystemStorageBackend  # noqa: F401


@m.memento_function(cluster="test_cluster")
def registered_fn_1():
    pass


def not_yet_registered_fn():
    pass


class TestConfiguration:
    """Class to test Memento configuration."""

    def setup_method(self):
        self.original_env = m.Environment.get()
        self.test_dir = tempfile.mkdtemp(prefix="configurationTest")

    def teardown_method(self):
        m.Environment.set(self.original_env)
        shutil.rmtree(self.test_dir)

    def test_environment_default_init(self):
        env = m.Environment.get()
        assert env is not None
        assert "default" == env.name
        assert [] == env.repos
        assert env.base_dir is None

    def test_environment_set_static(self):
        m.Environment.set({"name": "test1", "repos": []})
        env = m.Environment.get()
        assert "test1" == env.name
        assert [] == env.repos
        assert env.base_dir is None

    def test_environment_set_file_json(self):
        d = None
        try:
            d = tempfile.mkdtemp(prefix="memento_test_configuration")
            config_file = "{}/env.json".format(d)
            with open(config_file, "w") as f:
                print("""{"name": "test2", "repos": []}""", file=f)
            m.Environment.set(config_file)
            env = m.Environment.get()
            assert "test2" == env.name
            assert [] == env.repos
            assert os.path.dirname(config_file) == env.base_dir
        finally:
            if d:
                shutil.rmtree(d)

    def test_environment_set_file_yaml(self):
        d = None
        try:
            d = tempfile.mkdtemp(prefix="memento_test_configuration")
            config_file = "{}/env.yaml".format(d)
            with open(config_file, "w") as f:
                print("""name: test3\nrepos: []\n""", file=f)
            m.Environment.set(config_file)
            env = m.Environment.get()
            assert "test3" == env.name
            assert [] == env.repos
            assert os.path.dirname(config_file) == env.base_dir
        finally:
            if d:
                shutil.rmtree(d)

    @staticmethod
    def get_sample_config():
        return {
            "name": "config1",
            "description": "description1",
            "maintainer": "maintainer1",
            "documentation": "doc1",
            "clusters": {},
            "modules": ["a.b.c"],
        }

    @staticmethod
    def assert_sample_config(config):
        assert "config1" == config.name
        assert "description1" == config.description
        assert "maintainer1" == config.maintainer
        assert "doc1" == config.documentation
        assert {} == config.clusters
        assert ["a.b.c"] == config.modules

    def test_config_repo_set_static(self):
        self.assert_sample_config(m.ConfigurationRepository(self.get_sample_config()))

    def test_config_repo_env_static(self):
        m.Environment.set({"name": "test", "repos": [self.get_sample_config()]})
        self.assert_sample_config(m.Environment.get().repos[0])

    def test_config_repo_env_file(self):
        d = None
        try:
            d = tempfile.mkdtemp(prefix="memento_test_configuration")
            config_file = "{}/config.json".format(d)
            with open(config_file, "w") as f:
                json.dump(self.get_sample_config(), f)
            m.Environment.set({"name": "test", "repos": [config_file]})
            self.assert_sample_config(m.Environment.get().repos[0])
        finally:
            if d:
                shutil.rmtree(d)

    def test_config_repo_env_file_relative_path(self):
        d = None
        try:
            d = tempfile.mkdtemp(prefix="memento_test_configuration")
            env_file = "{}/env.json".format(d)
            with open(env_file, "w") as f:
                json.dump({"name": "test", "repos": ["subdir/config.json"]}, f)
            os.mkdir("{}/subdir".format(d))
            config_file = "{}/subdir/config.json".format(d)
            with open(config_file, "w") as f:
                json.dump(self.get_sample_config(), f)
            m.Environment.set(env_file)
            self.assert_sample_config(m.Environment.get().repos[0])
        finally:
            if d:
                shutil.rmtree(d)

    @staticmethod
    def get_sample_cluster():
        return {
            "name": "cluster1",
            "description": "description1",
            "maintainer": "maintainer1",
            "documentation": "doc1",
            "storage": {"type": "null", "readonly": True},
        }

    @staticmethod
    def assert_sample_cluster(cluster):
        assert "cluster1" == cluster.name
        assert "description1" == cluster.description
        assert "maintainer1" == cluster.maintainer
        assert "doc1" == cluster.documentation
        assert cluster.storage.read_only

    def test_function_cluster_static(self):
        self.assert_sample_cluster(m.FunctionCluster(self.get_sample_cluster()))

    def test_function_cluster_env_file(self):
        d = None
        try:
            d = tempfile.mkdtemp(prefix="memento_test_configuration")
            cluster_file = "{}/cluster.json".format(d)
            with open(cluster_file, "w") as f:
                json.dump(self.get_sample_cluster(), f)
            m.Environment.set(
                {
                    "name": "test",
                    "repos": [
                        {"name": "config1", "clusters": {"cluster1": cluster_file}}
                    ],
                }
            )
            self.assert_sample_cluster(
                m.Environment.get().repos[0].clusters["cluster1"]
            )
        finally:
            if d:
                shutil.rmtree(d)

    def test_function_cluster_env_file_relative_path(self):
        d = None
        try:
            d = tempfile.mkdtemp(prefix="memento_test_configuration")
            config_file = "{}/config.json".format(d)
            with open(config_file, "w") as f:
                json.dump(
                    {
                        "name": "config1",
                        "clusters": {"cluster1": "cluster_subdir/cluster.json"},
                    },
                    f,
                )
            cluster_file = "{}/cluster_subdir/cluster.json".format(d)
            os.mkdir("{}/cluster_subdir".format(d))
            with open(cluster_file, "w") as f:
                json.dump(self.get_sample_cluster(), f)
            m.Environment.set({"name": "test", "repos": [config_file]})
            self.assert_sample_cluster(
                m.Environment.get().repos[0].clusters["cluster1"]
            )
        finally:
            if d:
                shutil.rmtree(d)

    def test_get_cluster(self):
        m.Environment.set(
            {
                "name": "test",
                "repos": [
                    {
                        "name": "repo1",
                        "clusters": {
                            "A": {
                                "name": "A",
                                "description": "1",
                                "storage": {"type": "null"},
                            }
                        },
                    },
                    {
                        "name": "repo2",
                        "clusters": {
                            "A": {
                                "name": "A",
                                "description": "2",
                                "storage": {"type": "null"},
                            },
                            "B": {
                                "name": "B",
                                "description": "3",
                                "storage": {"type": "null"},
                            },
                        },
                    },
                ],
            }
        )
        env = m.Environment.get()
        assert "1" == env.get_cluster("A").description
        assert "3" == env.get_cluster("B").description
        assert env.get_cluster("C") is None
        assert "default" == env.get_cluster(None).name

    def test_default_cluster_storage_is_filesystem(self):
        m.Environment.set({})
        env = m.Environment.get()
        cluster = env.default_cluster
        assert "filesystem" == cluster.storage.storage_type

    def test_default_storage_is_filesystem(self):
        m.Environment.set(
            {
                "name": "test",
                "repos": [
                    {
                        "name": "repo1",
                        "clusters": {"A": {"name": "A", "description": "1"}},
                    }
                ],
            }
        )
        env = m.Environment.get()
        cluster = env.get_cluster("A")
        assert "filesystem" == cluster.storage.storage_type

    def test_cluster_get_registered_functions(self):
        fns = m.Environment.get_registered_functions("test_cluster")
        assert 1 == len(fns)
        assert "registered_fn_1" == fns[0].fn_reference().function_name

    def test_get_registered_clusters(self):
        assert "test_cluster" in m.Environment.get_registered_clusters()

    def test_get_repo(self):
        m.Environment.set(
            {
                "name": "test",
                "repos": [
                    {
                        "name": "repo1",
                        "clusters": {"A": {"name": "A", "description": "1"}},
                    }
                ],
            }
        )
        env = m.Environment.get()
        assert "repo1" == env.get_repo("repo1").name
        assert env.get_repo("repo2") is None

    def test_append_repo(self):
        m.Environment.set(
            {
                "name": "test",
                "repos": [
                    {
                        "name": "repo1",
                        "clusters": {"A": {"name": "A", "description": "1"}},
                    }
                ],
            }
        )
        env = m.Environment.get()
        env.append_repo(ConfigurationRepository(name="repo2"))
        assert "repo2" == env.repos[-1].name

    def test_prepend_repo(self):
        m.Environment.set(
            {
                "name": "test",
                "repos": [
                    {
                        "name": "repo1",
                        "clusters": {"A": {"name": "A", "description": "1"}},
                    }
                ],
            }
        )
        env = m.Environment.get()
        env.prepend_repo(ConfigurationRepository(name="repo2"))
        assert "repo2" == env.repos[0].name

    def test_env_param_override(self):
        env = Environment({"name": "name1"}, name="name2")
        assert "name2" == env.name

    def test_config_repo_param_override(self):
        cr = ConfigurationRepository({"name": "name1"}, name="name2")
        assert "name2" == cr.name

    def test_function_cluster_param_override(self):
        fc = FunctionCluster({"name": "name1"}, name="name2")
        assert "name2" == fc.name

    def test_is_function_registered(self):
        qn = registered_fn_1.fn_reference().qualified_name
        assert Environment.is_function_registered(qn)
        assert not Environment.is_function_registered(qn + "x")

    def test_env_to_dict(self):
        m.Environment.set(
            {
                "name": "e_name",
                "base_dir": "e_basedir",
                "repos": [
                    {
                        "name": "r_name",
                        "base_dir": "r_basedir",
                        "description": "r_description",
                        "maintainer": "r_maintainer",
                        "clusters": {
                            "c_name": {
                                "name": "c_name",
                                "description": "c_description",
                                "maintainer": "c_maintainer",
                                "documentation": "c_doc",
                                "storage": {"type": "filesystem", "path": "/tmp"},
                                "runner": {"type": "local"},
                            }
                        },
                    }
                ],
            }
        )
        env = m.Environment.get()
        env_dict = env.to_dict()
        env2 = m.Environment(env_dict)
        assert "e_name" == env2.name
        assert "e_basedir" == env2.base_dir
        assert "r_name" == env2.repos[0].name
        assert "r_basedir" == env2.repos[0].base_dir
        assert "r_description" == env2.repos[0].description
        assert "r_maintainer" == env2.repos[0].maintainer
        c = env2.repos[0].clusters["c_name"]
        assert "c_name" == c.name
        assert "c_description" == c.description
        assert "c_maintainer" == c.maintainer
        assert "c_doc" == c.documentation
        assert "filesystem" == c.storage.storage_type
        fs = cast(FilesystemStorageBackend, c.storage)
        assert "/tmp" == fs.config_path
        assert "local" == c.runner.runner_type

    def test_jinja2_template(self):
        config_file = "{}/test.json".format(self.test_dir)
        repo = {
            "name": "{{param1}}",
        }
        with open(config_file, "w") as f:
            json.dump(repo, f)
        cr = ConfigurationRepository.from_file(config_file, param1="subtest")
        assert "subtest" == cr.name

    def test_locked_cluster(self):
        cluster = Environment.get().get_cluster(None)
        cluster.locked = True
        try:
            with pytest.raises(ValueError):
                Environment.register_function(
                    None, m.memento_function(not_yet_registered_fn)
                )
        finally:
            cluster.locked = False
