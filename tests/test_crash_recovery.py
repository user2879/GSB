from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from driftwatch.config import load_inputs
from driftwatch.engine import DriftEngine, recover_ledger
from driftwatch.ledger import Ledger
from driftwatch.processes import port_is_free

from .conftest import free_port, write_json


def _configs(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    ports = [free_port(), free_port()]
    snapshots = {
        "snapshots": [
            {
                "seq": 1,
                "parent_seq": None,
                "hosts": {
                    "h1": {
                        "expected": {"cpu_quota": 100000, "memory_pages": 64000, "processes": 500},
                        "tolerance_bps": 500,
                    }
                },
            },
            {
                "seq": 2,
                "parent_seq": 1,
                "hosts": {
                    "h1": {
                        "expected": {"cpu_quota": 101000, "memory_pages": 65000, "processes": 505},
                        "tolerance_bps": 500,
                    }
                },
            },
        ]
    }
    profiles = {
        "h1": {
            "cpu_quota": {"base": 98000, "spread": 800},
            "memory_pages": {"base": 63700, "spread": 300},
            "processes": {"base": 497, "spread": 4},
        }
    }
    agents = {
        "agents": [
            {
                "agent_id": "a1",
                "host": "127.0.0.1",
                "port": ports[0],
                "seed": 10,
                "aggregation": {"kind": "median", "window": 5, "version": "v1"},
                "hosts": ["h1"],
                "profiles": profiles,
            },
            {
                "agent_id": "a2",
                "host": "127.0.0.1",
                "port": ports[1],
                "seed": 20,
                "aggregation": {"kind": "mean_floor", "window": 5, "version": "v1"},
                "hosts": ["h1"],
                "profiles": profiles,
            },
        ]
    }
    criteria = {
        "global_concurrency": 2,
        "per_agent_concurrency": 1,
        "min_sources": 2,
        "scale_exp": 3,
        "socket_timeout_ms": 1500,
        "handshake_timeout_ms": 5000,
        "max_retries": 1,
        "retry_backoff_ms": 5,
        "boundary_rule": "tolerance_floor",
    }
    return (
        write_json(tmp_path / "snapshots.json", snapshots),
        write_json(tmp_path / "agents.json", agents),
        write_json(tmp_path / "criteria.json", criteria),
        str(tmp_path / "crash.sqlite3"),
    )


def _wait_until(predicate, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition did not become true before timeout")


def test_force_kill_recovery_uses_only_committed_batches_and_replays(tmp_path):
    snapshots, agents, criteria, db = _configs(tmp_path)
    chain, agent_specs, parsed_criteria = load_inputs(str(snapshots), str(agents), str(criteria))
    ports = [agent.port for agent in agent_specs]
    marker = tmp_path / "paused.marker"
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not env.get("PYTHONPATH") else root + os.pathsep + env["PYTHONPATH"]
    worker = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "_crash_worker.py"),
            "--snapshots",
            str(snapshots),
            "--agents",
            str(agents),
            "--criteria",
            str(criteria),
            "--db",
            db,
            "--marker",
            str(marker),
            "--snapshot",
            "2",
        ],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_until(marker.exists, 12.0)
        assert worker.poll() is None
        worker.kill()
        worker.wait(timeout=8.0)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5.0)
        if worker.stdout is not None:
            worker.stdout.close()
        if worker.stderr is not None:
            worker.stderr.close()

    _wait_until(lambda: all(port_is_free("127.0.0.1", port) for port in ports), 8.0)

    with sqlite3.connect(db) as raw:
        rows = raw.execute(
            "SELECT snapshot_seq, status FROM batches ORDER BY started_ms"
        ).fetchall()
        observations = raw.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        uncommitted_observations = raw.execute(
            """
            SELECT COUNT(*) FROM observations o JOIN batches b ON b.batch_id=o.batch_id
            WHERE b.snapshot_seq=2
            """
        ).fetchone()[0]
    assert (1, "COMMITTED") in rows
    assert (2, "RUNNING") in rows
    assert observations > 0
    assert uncommitted_observations == 0

    with Ledger(db) as ledger:
        recovered = ledger.recover_incomplete()
        assert recovered
        assert {item["snapshot_seq"] for item in recovered} == {2}
        assert len(ledger.required_recollection(ledger.active_generation() or 0)) == 3
        statuses = {
            row["snapshot_seq"]: row["status"]
            for row in ledger.connection.execute("SELECT snapshot_seq, status FROM batches ORDER BY started_ms")
        }
        assert statuses[1] == "COMMITTED"
        assert statuses[2] == "ABORTED"

    recovered_report = recover_ledger(db)
    report = recovered_report["report"]
    assert recovered_report["source"] == "ledger_declarations"
    assert report["replay"] is not None and report["replay"]["ok"] is True
    assert report["replay"]["batch_count"] == 1
    assert report["recovered_unknown"]
    assert report["view_summary"]["merged_rows"] == 6
    assert report["view_summary"]["drift_rows"] == 6
    with Ledger(db) as engine_ledger:
        view = engine_ledger.session_snapshot(report["generation"])
        assert {row["snapshot_seq"] for row in view["drift_view"]} == {1, 2}
        assert engine_ledger.required_recollection(report["generation"]) == []
        assert all(row["decision"] in {"ACCEPT", "REJECT"} for row in view["drift_view"])
