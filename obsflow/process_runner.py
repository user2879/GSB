"""Subprocess helper for force-kill / recovery demonstrations.

The worker is a real, independently launched OS process (``sys.executable``)
that opens the same SQLite ledger, begins a batch and is then hard killed by
the parent. The surviving ledger therefore contains an ``in_progress``
batch, exercising the crash-recovery path based solely on committed data.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass


WORKER_SOURCE = textwrap.dedent("""
    import json
    import sqlite3
    import sys

    ledger_path = sys.argv[1]
    conn = sqlite3.connect(ledger_path)
    next_id = conn.execute(
        "SELECT COALESCE(MAX(batch_id), 0) + 1 FROM batches").fetchone()[0]
    conn.execute(
        "INSERT INTO batches(batch_id, status, committed_ms, source_ids, "
        "record_count, seq_cursor, content_digest) "
        "VALUES(?, 'in_progress', 0, '[]', 0, 0, ?)",
        (next_id, json.dumps("uncommitted-crash-batch")))
    conn.commit()
    # Keep the process alive until the parent hard-kills it.
    print("READY", flush=True)
    sys.stdin.read()
""")


@dataclass(frozen=True, slots=True)
class CrashResult:
    returncode: int
    stdout: str
    stderr: str


def launch_crashed_worker(ledger_path: str,
                          kill_after_ms: int = 400) -> CrashResult:
    """Start a worker that leaves an in_progress batch, then SIGKILL it."""
    import time

    cmd = [sys.executable, "-c", WORKER_SOURCE, os.path.abspath(ledger_path)]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 5
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.strip() == "READY":
            break
    time.sleep(kill_after_ms / 1000.0)
    proc.kill()
    stdout, stderr = proc.communicate(timeout=10)
    return CrashResult(returncode=proc.returncode, stdout=stdout,
                       stderr=stderr)
