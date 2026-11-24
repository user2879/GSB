from __future__ import annotations

import os
import signal
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    database = Path(sys.argv[1])
    marker = Path(sys.argv[2])
    view_id = sys.argv[3]
    key = sys.argv[4]
    version = int(sys.argv[5])

    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        "INSERT INTO results(view_id, key, view_version, status, summary_units, "
        "summary_digest, adopted_json, rejected_json, bound_json, attribution_json, "
        "slice_versions_json, result_json, result_digest, is_stale, committed_monotonic_ms) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            view_id,
            key,
            version,
            "OK",
            123,
            "uncommitted",
            "[]",
            "[]",
            "{}",
            "{}",
            "{}",
            "{}",
            "uncommitted",
            0,
            1,
        ),
    )
    marker.write_text("ready", encoding="utf-8")
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.TerminateProcess(
            ctypes.windll.kernel32.GetCurrentProcess(), 1
        )
    else:
        os.kill(os.getpid(), signal.SIGKILL)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
