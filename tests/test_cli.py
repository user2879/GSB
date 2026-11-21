from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import ROOT, SAMPLE_SPANS, make_nodes, write_json


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return env


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "trace_arbiter", *args],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=20,
    )


def _last_log(path: Path) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_cli_injections_are_structured_and_use_required_codes(tmp_path: Path) -> None:
    nodes = write_json(tmp_path / "nodes.json", make_nodes())
    corrupt_log = tmp_path / "corrupt.log.jsonl"
    corrupt = _run(
        "adjudicate",
        "--nodes",
        str(nodes),
        "--spans",
        str(SAMPLE_SPANS),
        "--db",
        str(tmp_path / "corrupt.sqlite3"),
        "--log",
        str(corrupt_log),
        "--inject",
        "corrupt-message",
    )
    assert corrupt.returncode == 2
    corrupt_record = _last_log(corrupt_log)
    assert corrupt_record["code"] == "E_SPEC"
    assert corrupt_record["line"] == 1

    parent_log = tmp_path / "parent.log.jsonl"
    parent = _run(
        "adjudicate",
        "--nodes",
        str(nodes),
        "--spans",
        str(SAMPLE_SPANS),
        "--db",
        str(tmp_path / "parent.sqlite3"),
        "--log",
        str(parent_log),
        "--inject",
        "parent-missing",
    )
    assert parent.returncode == 9
    assert _last_log(parent_log)["code"] == "E_CHAIN"

    db_path = tmp_path / "ledger.sqlite3"
    normal = _run(
        "adjudicate",
        "--nodes",
        str(nodes),
        "--spans",
        str(SAMPLE_SPANS),
        "--db",
        str(db_path),
        "--log",
        str(tmp_path / "normal.log.jsonl"),
        "--global-concurrency",
        "3",
        "--per-node-concurrency",
        "1",
        "--batch-size",
        "3",
    )
    assert normal.returncode == 0, normal.stderr
    snapshot_log = tmp_path / "snapshot.log.jsonl"
    snapshot = _run(
        "replay",
        "--db",
        str(db_path),
        "--log",
        str(snapshot_log),
        "--inject",
        "snapshot-mismatch",
    )
    assert snapshot.returncode == 8
    assert _last_log(snapshot_log)["code"] == "E_SNAPSHOT"
