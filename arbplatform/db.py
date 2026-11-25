"""SQLite connection adapter.

The platform requires SQLite >= 3.40. The Ubuntu 22.04 system library is
3.37, so the project virtualenv installs ``pysqlite3-binary`` (bundled SQLite
3.51). This shim prefers that package and falls back to the stdlib module only
when its runtime version already satisfies the requirement.
"""
from __future__ import annotations

import sqlite3

REQUIRED_SQLITE = (3, 40, 0)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in text.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def connect(database: str) -> sqlite3.Connection:
    try:
        import pysqlite3  # type: ignore
        module = pysqlite3
    except ImportError:
        module = sqlite3
    if _version_tuple(module.sqlite_version) < REQUIRED_SQLITE:
        raise RuntimeError(
            f"SQLite >= {'.'.join(map(str, REQUIRED_SQLITE))} required, "
            f"found {module.sqlite_version}; install pysqlite3-binary"
        )
    conn = module.connect(database)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = module.Row
    return conn


def sqlite_version() -> str:
    try:
        import pysqlite3  # type: ignore
        return pysqlite3.sqlite_version
    except ImportError:
        return sqlite3.sqlite_version
