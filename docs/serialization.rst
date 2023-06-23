*********************
Memento Serialization
*********************

Memento memoizes the return values of methods marked by the **@memento_function**
decorator (or derived decorators in other modules).  Depending on your
configuration, these meoized values by be held
directly in memory or serialized out to disk (local storage, GCS bucket, etc).
This document describes the mechanism Memento uses to serialize these values.

Supported Types
===============

Due to the need to serialize/deserialize values in a language agnostic way, only
a finite number of types are supported by Memento.  Currently, supported return
types from memoized functions are:

* boolean
* string
* bytes
* number
* datetime.date
* datetime.datetime
* datetime.timedelta
* `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
* `pandas.Series <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_
* `pandas.Index <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html>`_
* `numpy.ndarray <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_ (1-dimensional)
* list of any of these types
* dictionary of any of these types
* `Partition` of keys to any of these types

Returning any other values from your **@memento_function** will result in an
`Error` being raised.  These types were selected for their common use in data
science research and the ability to efficiently encode and decode values with
limited memory usage in a cross platform format.


File format
===========

.. warning:: This section describes the internal data serialization strategy used by Memento.  This is subject to change and should not be relied upon by external users.

Serialized values on disk are stored in pickle 5 format.
