"""Transactional message middleware laboratory implementation.

The package intentionally depends only on the Python standard library.  It is
designed for a local Windows (or POSIX) loopback environment and keeps all
authoritative state in SQLite.
"""

__version__ = "1.0.0"
FORMAT_VERSION = 1

