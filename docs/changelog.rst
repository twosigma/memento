.. _changelog:

Changelog
=========

All notable changes to Memento will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this
project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Note that Memento is still in the `0` major version and so backwards-compatible changes are
expected even in minor releases.

pending
-------
Changed
```````
* Serialization format changed to use pickle 5. This will cause a regeneration of all memoized data.
* Added native support for Windows
* Instead of creating symlinks, use `.link` files, so that a Windows filesystem can be used

0.27.0 - 2021-06-23
-------------------
Added
`````
* Serialization of byte streams and strings with size >= 2GiB now supported,
  with the new `MementoLongBytes` and `MementoLongString` Proto

Changed
```````
* Serialization format changed. This will cause a regeneration of all memoized data.
* Only the major version of `pandas` and `pyarrow` are included in the environment hash

0.26.2 - 2021-05-28
-------------------
Fixed
`````
* Dependency graphs now render correctly when `verbose=True`, even when there is an
  undefined symbol in the function.

0.26.1 - 2021-05-14
-------------------
Added
`````
* fn.list() returns a DataFrame with a list of memoized args

Changed
```````
* Output warning when raising a memoized exception
* MementoNotFoundError is now a `NonMemoizedException`


0.26.0 - 2021-05-11
-------------------
Changed
```````
* Dependency graph is now complete and now longer limits each dependency to one incoming arrow.
  This also impacts the code hash of functions, which justifies the minor version bump.
* API changes for `DependencyGraph` and removal of dependency graph `Node` class


0.25.0 - 2021-05-05
-------------------
Added
`````
* Enable ingestors to be run remotely on a Memento Server


Unreleased
----------
Unreleased changes are summarized in this section before the next release.
