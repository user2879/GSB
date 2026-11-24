"""Fault injection chain.

Injection only ever alters input files or parameters, never the engine logic:

* ``truncate``  - cuts a stream file short          -> E_INTEGRITY
* ``order``     - makes a sequence regress          -> E_ORDER
* ``snapshot``  - corrupts the stored content digest -> E_SNAPSHOT

Each injected run writes a structured JSONL log entry carrying the business
code and the offending line/field.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from .errors import ObsFlowError


def _copy_stream(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def inject_truncation(stream_path: str, work_path: str,
                      keep_lines: Optional[int] = None) -> str:
    """Truncate a stream so the last record is incomplete."""
    _copy_stream(stream_path, work_path)
    with open(stream_path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if keep_lines is None:
        keep_lines = max(2, len(lines) - 1)
    with open(work_path, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines[:keep_lines]:
            handle.write(line + "\n")
        # Half-written final record (invalid JSON -> E_INTEGRITY).
        handle.write('{"seq": 999, "val')
    return work_path


def inject_order_regression(stream_path: str, work_path: str) -> str:
    """Rewrite the last record so its sequence regresses."""
    _copy_stream(stream_path, work_path)
    with open(stream_path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    last = json.loads(lines[-1])
    last["seq"] = 0
    lines[-1] = json.dumps(last, sort_keys=True)
    with open(work_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return work_path


def inject_snapshot_digest(ledger_path: str) -> None:
    """Tamper with the stored snapshot content digest (parameter only)."""
    import sqlite3

    conn = sqlite3.connect(ledger_path)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='content_digest'").fetchone()
        if row is None:
            raise FileNotFoundError("ledger has no content_digest to tamper")
        stored = json.loads(row[0])
        tampered = ("0" if not stored.startswith("0") else "1") + stored[1:]
        conn.execute("UPDATE meta SET value=? WHERE key='content_digest'",
                     (json.dumps(tampered),))
        conn.commit()
    finally:
        conn.close()


def describe_error(exc: ObsFlowError) -> dict:
    return exc.as_dict()
