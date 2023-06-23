***************
Memento plugins
***************

The Memento framework abstracts how a function is run and where the cached data is stored,
allowing the function author to focus on the logic of the function. Runner plugins allow
changing how a function is run (for example in the local process vs. in an external process)
and storage plugins allow changing where the memoized data is stored (for example, in
memory vs. on disk).

The core :py:mod:`twosigma.memento` package includes several implementations of these plugins:

Built-in storage plugins
========================
The following storage plugins are built-in:

*  filesystem (default): :py:class:`twosigma.memento.storage_filesystem`: Stores memoized results
   on the local filesystem
*  memory: :py:class:`twosigma.memento.storage_memory`: Stores memoized results in the heap
*  null: :py:class:`twosigma.memento.storage_null`: Disables memoization of results

Built-in runner plugins
=======================
The following runner plugins are built-in:

*  local (default): :py:class:`twosigma.memento.runner_local`: Runs functions in the local process
*  null: :py:class:`twosigma.memento.runner_null`: Runs functions


Writing Memento Plugins
=======================

Memento can be extended with additional plugins by installing Python modules. Upon initialization,
Memento scans for plugins installed in the Python environment. To ensure it is properly
registered, modules should be configured with `setuptools <https://pypi.org/project/setuptools/>`_
and include the following directive in :code:`setup()`:

.. code-block:: python

   setup(
       ...
       entry_points={'twosigma.memento.plugin': 'shortname = module.name.goes.here'},
   )

The :code:`__init__.py` for the module should import each of the modules needed to register the plugins
that appear in that package. These will be imported whenever :py:mod:`twosigma.memento` is
imported.


Implementing a storage plugin
-----------------------------
To implement a new storage plugin:

1. Define a short name for the module which will be used to identify the module in config files
2. Define a Python module to store the plugin
3. Define a class which extends :py:class:`twosigma.memento.storage.StorageBackend`
   Consider extending :py:class:`twosigma.memento.storage_base.StorageBackendBase` (see below)
4. Design which parameters can be passed to the storage plugin in the Memento configuration
   file. These should be parsed in the constructor of the backend.
5. Call :py:meth:`twosigma.memento.storage.StorageBackend.register` to register the
   plugin so that Memento recognizes it in configuration files.

The :py:class:`twosigma.memento.storage_base.StorageBackendBase` class makes it easier to implement
a storage backend by providing two additional abstractions: a data store and a metadata store.
Also included is a default memory cache implementation. Either the data store or the metadata
store can be borrowed from other implementations (for example, if the author wishes to only
innovate on the metadata implementation).

Implementing a runner plugin
----------------------------
Implementing a new runner plugin is similar to implementing a storage plugin:

1. Define a short name for the module which will be used to identify the module in config files
2. Define a Python module to store the plugin
3. Define a class which extends :py:class:`twosigma.memento.runner.RunnerBackend`
4. Design which parameters can be passed to the runner plugin in the Memento configuration
   file. These should be parsed in the constructor of the backend.
5. Call :py:meth:`twosigma.memento.runner.RunnerBackend.register` to register the
   plugin so that Memento recognizes it in configuration files.

All runners eventually end up calling :py:meth:`twosigma.memento.runner_local.memento_run_local`
when the function should actually be called. This ensures proper handling of memoization.
