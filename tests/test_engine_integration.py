from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cachelease.engine import Engine
from cachelease.ledger import Ledger
from cachelease.spec import load_config, load_events

from .helpers import write_sample


def test_full_pipeline_arbitration_rollback_quota_and_queue(tmp_path: Path) -> None:
    config_path, events_path, _, _ = write_sample(tmp_path, seed=811)
    config = load_config(config_path)
    events = load_events(events_path, config)
    ledger = Ledger(tmp_path / "ledger.sqlite", config)
    try:
        report = Engine(config, events, ledger, tmp_path / "nodes").run()
        assert report["ok"] is True
        assert report["progress"]["last_ordinal"] == len(events)
        assert report["replay"]["events"] == len(events)
        assert report["replay"]["watermarks"] == len(events) * len(config.nodes)

        connection = ledger.connection
        connection.row_factory = sqlite3.Row
        outcomes = {
            int(row["seq"]): json.loads(str(row["outcome_json"]))
            for row in connection.execute("SELECT seq, outcome_json FROM events ORDER BY ordinal")
        }
        assert outcomes[1]["node_id"] == outcomes[5]["node_id"]
        assert outcomes[5]["node_id"] == "node-a"
        assert any(item["digest_match"] for item in outcomes[5]["candidates"])
        assert any(item["reason"] == "lower_priority" for item in outcomes[5]["candidates"])

        assert outcomes[14]["status"] == "error"
        assert outcomes[14]["error_code"] == "E_EXPIRED"
        expired_reservation = connection.execute(
            "SELECT status, disposition FROM reservations WHERE apply_seq=14"
        ).fetchone()
        assert expired_reservation["status"] == "rolled_back"
        assert "grace_expired" in expired_reservation["disposition"]
        assert connection.execute("SELECT COUNT(*) FROM leases WHERE apply_seq=14").fetchone()[0] == 0
        assert outcomes[15]["status"] == "ok"

        stale_watermarks = connection.execute(
            "SELECT COUNT(*) FROM watermarks WHERE node_id='node-a' AND quota_version=1 AND stale=1"
        ).fetchone()[0]
        current_watermarks = connection.execute(
            "SELECT COUNT(*) FROM watermarks WHERE node_id='node-a' AND quota_version=2 AND stale=0"
        ).fetchone()[0]
        assert stale_watermarks > 0
        assert current_watermarks > 0
        stale_leases = connection.execute(
            "SELECT COUNT(*) FROM leases WHERE node_id='node-a' AND quota_version=1 AND stale=1"
        ).fetchone()[0]
        assert stale_leases > 0

        queue_rows = [dict(row) for row in connection.execute("SELECT * FROM queue_observations")]
        assert queue_rows
        for row in queue_rows:
            assert row["wait_ms"] == row["wait_end_ms"] - row["wait_start_ms"]
        for metric in report["scheduler_metrics"]:
            assert metric["max_global_in_flight"] <= config.runtime.global_concurrency
            for node in config.nodes:
                assert metric["max_node_in_flight"][node.node_id] <= node.request_concurrency
            for stage in config.stages:
                assert metric["max_stage_in_flight"][stage.stage] <= stage.concurrency

        assert report["content_digest"] and len(report["content_digest"]) == 64
    finally:
        ledger.close()


def test_nonstrictly_increasing_seq_never_collides_lease_ids(tmp_path: Path) -> None:
    from .helpers import one_node_config, write_events

    _, config_path = one_node_config(tmp_path, capacity_slots=4, lease_limit=4)
    config = load_config(config_path)
    events_path = write_events(
        tmp_path / "duplicate-seq.ndjson",
        [
            {"seq": 1, "batch": 1, "type": "write", "artifact": "core", "lease_ms": 1000},
            {"seq": 1, "batch": 2, "type": "write", "artifact": "core", "lease_ms": 1000},
        ],
    )
    events = load_events(events_path, config)
    ledger = Ledger(tmp_path / "duplicate-seq.sqlite", config)
    try:
        Engine(config, events, ledger, tmp_path / "duplicate-seq-nodes").run()
        leases = ledger.connection.execute(
            "SELECT lease_id, ordinal, apply_seq FROM leases ORDER BY ordinal"
        ).fetchall()
        assert len(leases) == 2
        assert leases[0]["lease_id"] != leases[1]["lease_id"]
        assert leases[0]["apply_seq"] == leases[1]["apply_seq"] == 1
        assert ledger.progress()["last_ordinal"] == 2
    finally:
        ledger.close()


def test_pipeline_queue_is_real_and_matches_ledger(tmp_path: Path) -> None:
    from .helpers import one_node_config, write_events

    _, config_path = one_node_config(
        tmp_path,
        capacity_slots=4,
        lease_limit=4,
        request_concurrency=1,
    )
    config = load_config(config_path)
    events_path = write_events(
        tmp_path / "queue.ndjson",
        [
            {"seq": 1, "batch": 1, "type": "write", "artifact": "core", "lease_ms": 5000, "hold_ms": 80},
            {"seq": 2, "batch": 1, "type": "write", "artifact": "api", "lease_ms": 5000},
        ],
    )
    events = load_events(events_path, config)
    ledger = Ledger(tmp_path / "queue.sqlite", config)
    try:
        report = Engine(config, events, ledger, tmp_path / "queue-nodes").run()
        rows = [dict(row) for row in ledger.connection.execute("SELECT * FROM queue_observations ORDER BY ordinal")]
        queued = next(row for row in rows if row["ordinal"] == 2)
        assert queued["wait_ms"] == queued["wait_end_ms"] - queued["wait_start_ms"]
        assert queued["wait_ms"] >= 40
        assert queued["node_in_flight"] <= 1
        assert report["scheduler_metrics"][0]["max_node_in_flight"]["node-a"] == 1
    finally:
        ledger.close()
