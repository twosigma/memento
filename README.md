# Two Sigma Memento
Memento is a framework and lightweight set of standards that encourage discipline in the way data
is incrementally transformed through code. The goal of Memento is to ensure that data is
reproducible and that accurate provenance is maintained in metadata. The Memento Framework
does not pin itself to a specific programming language, back-end storage technology or compute
framework. Rather, it focuses on a technique for generating and accessing data that preserves
accurate metadata throughout.

Memento can be extended with plugins which customize where memoized data is stored and how
distributed compute is executed.

This codebase hosts several independent products, which are all synchronized in their version.
This product is the core memento framework, which is usable by itself or in combination with
various plugins.

The core framework includes three storage backends, filesystem (the default), memory, and
null (never store). The core framework also includes two runner backends, local (in-process),
and null (never run). To use memento for distributed computation or shared storage, other
plugins can be used.


# Quick Start
The following instructions should get you from a git clone to a working build of Memento.
These instructions are tested on Linux and Windows but should work in other environments
as well.

## Prerequisites
You need a Python environment with hatch installed:

```bash
$ python --version
Python 3.11.1
$ pip install hatch
```

## Build Memento
To build, simply run the following command. You will get a dist directory with a pip package.

```bash
$ hatch build
```


## Test Memento
Memento has an extensive suite of unit tests. To run the tests to ensure the build is working,
run the following.

```bash
$ hatch run cov
```

There are some tests that will not run by default because they require a lot of RAM and are slow
to run. To include those tests, include the `--runslow` parameter.

## Run Memento
If you have an existing environment, you can install memento with:

```bash
$ pip install -e .
```

If you prefer to start an isolated environment with a barebones Python with just Memento installed:

```bash
$ hatch shell
$ python
```

# Quick example
Try a simple cached function definition in a python repl:

```bash
$ hatch run python
Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32                                                                                                                                                                                                               
Type "help", "copyright", "credits" or "license" for more information.
>>> from twosigma.memento import memento_function
>>> @memento_function
... def f(x):
...     print("eval f")
...     return x + 1
...
>>> f(1)
eval f
2
>>> # The second time we run it should not print "eval f" because it reads the cached result
>>> f(1)
2
>>>
```
