from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from modeldash.nodes import NodeSupervisor, SocketDispatcher
from modeldash.protocol import is_port_free
from modeldash.replay import ReplayVerifier
from modeldash.scheduler import Scheduler
from modeldash.spec import load_spec, validate_spec
from modeldash.storage import Ledger
from modeldash.util import StructuredLog

from conftest import free_port
from test_scheduler import minimal_raw


ROOT = Path(__file__).resolve().parent.parent


def write_spec(path: Path, raw) -> Path:
    spec = validate_spec(raw)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def test_real_multi_node_preemption_and_live_replay(tmp_path):
    raw = minimal_raw(
        nodes=2,
        quota=1,
        global_concurrency=2,
        shards=3,
        backoff=[5],
        max_retries=1,
        tasks=[
            {"id": "low-a", "config_version": 1, "node": "n1", "priority": 1},
            {"id": "low-b", "config_version": 1, "node": "n2", "priority": 1},
            {"id": "high-a", "config_version": 1, "node": "n1", "priority": 9, "release_after_ms": 35},
        ],
    )
    for node in raw["nodes"]:
        node["shard_delay_ms"] = 20
    spec = validate_spec(raw)
    db_path = tmp_path / "ledger.sqlite3"
    logger = StructuredLog(tmp_path / "logs" / "live.jsonl")
    ledger = Ledger(db_path)
    activation = ledger.activate_spec(spec)
    supervisor = NodeSupervisor(spec, tmp_path, logger)
    try:
        supervisor.start_all()
        summary = Scheduler(
            spec,
            ledger,
            activation.session_id,
            SocketDispatcher(spec, logger),
            snapshot_hook=lambda state, label: ledger.create_snapshot(activation.session_id, state, label),
        ).run()
    finally:
        supervisor.stop_all()
    assert all(task["state"] == "DONE" for task in summary.tasks)
    assert any(item["status"] == "SUSPENDED" for item in summary.preemptions)
    ledger.create_snapshot(activation.session_id, ledger.latest_state(activation.session_id), "final")
    snapshot_id = ledger.list_snapshots(activation.session_id)[0]["snapshot_id"]

    supervisor = NodeSupervisor(spec, tmp_path, logger)
    try:
        supervisor.start_all()
        replay = ReplayVerifier(
            spec,
            ledger,
            activation.session_id,
            snapshot_id=snapshot_id,
            dispatcher=SocketDispatcher(spec, logger),
            redispatch=True,
        ).verify()
    finally:
        supervisor.stop_all()
        ledger.close()
    assert replay.redispatch_count == 9
    assert replay.remaining_task_ids == ["high-a", "low-a", "low-b"]


def test_forced_termination_marks_inflight_unknown_and_reruns(tmp_path):
    raw = minimal_raw(nodes=1, quota=1, global_concurrency=1, shards=2)
    raw["nodes"][0]["shard_delay_ms"] = 500
    raw["budget"]["timeout_ms"] = 3000
    spec_path = write_spec(tmp_path / "spec.json", raw)
    data_dir = tmp_path / "runtime"
    data_dir.mkdir()
    (data_dir / "logs").mkdir()
    db_path = data_dir / "ledger.sqlite3"
    runner = ROOT / "tests" / "support" / "run_session.py"
    process = subprocess.Popen(
        [sys.executable, str(runner), str(spec_path), str(db_path), str(data_dir)],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    saw_inflight = False
    while time.monotonic() < deadline and process.poll() is None:
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM rounds WHERE kind='DISPATCH_INFLIGHT'"
                    ).fetchone()
                    saw_inflight = row is not None and row[0] > 0
                finally:
                    conn.close()
            except sqlite3.Error:
                pass
        if saw_inflight:
            break
        time.sleep(0.02)
    assert saw_inflight
    process.kill()
    process.wait(timeout=5)
    free_deadline = time.monotonic() + 3.0
    while time.monotonic() < free_deadline and not is_port_free("127.0.0.1", raw["nodes"][0]["port"]):
        time.sleep(0.02)
    assert is_port_free("127.0.0.1", raw["nodes"][0]["port"])

    spec = load_spec(spec_path)
    ledger = Ledger(db_path)
    try:
        activation = ledger.activate_spec(spec)
        assert activation.action == "resumed"
        recovered = activation.state
        assert recovered is not None
        active = [
            task
            for task in recovered["tasks"].values()
            if task["state"] in {"RUNNING", "PREEMPT_REQUESTED"}
        ]
        assert active
        supervisor = NodeSupervisor(spec, data_dir)
        try:
            supervisor.start_all()
            summary = Scheduler(
                spec,
                ledger,
                activation.session_id,
                SocketDispatcher(spec),
                recovered_state=recovered,
            ).run()
        finally:
            supervisor.stop_all()
        task = summary.tasks[0]
        assert task["state"] == "DONE"
        assert task["unknown_shards"]
        assert task["recovery_reruns"] >= 1
        assert summary.recovery_notes
    finally:
        ledger.close()
