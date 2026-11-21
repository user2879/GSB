from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from trace_arbiter.canonical import canonical_text, digest_text
from trace_arbiter.engine import (
    EngineConfig,
    recover_incomplete,
    replay_snapshot,
    review_current,
    run_adjudication,
    view_current,
)
from trace_arbiter.errors import ArbiterError
from trace_arbiter.jsonlog import JsonlLogger
from trace_arbiter.ledger import (
    LedgerDB,
    STAT_QUANTIZATION_ERROR_NS,
    STAT_QUANTIZATION_ERROR_US,
    STAT_ROUNDING,
    STAT_SCALE_FACTOR,
)
from trace_arbiter.spec import load_dataset

from conftest import ROOT, SAMPLE_SPANS, make_nodes, write_json


def _versioned_datasets(tmp_path: Path):
    base = make_nodes()
    ports = {node["node_id"]: node["port"] for node in base["nodes"]}
    v1_nodes = write_json(tmp_path / "nodes_v1.json", base)
    evolved = make_nodes(
        ruleset_version=2,
        nodes=[
            {
                "node_id": "api",
                "host": "127.0.0.1",
                "port": ports["api"],
                "propagation_rule_version": 3,
                "sample_ratio_bps": 0,
                "random_seed": 303,
            },
            {
                "node_id": "worker",
                "host": "127.0.0.1",
                "port": ports["worker"],
                "propagation_rule_version": 4,
                "sample_ratio_bps": 10000,
                "random_seed": 404,
            },
            {
                "node_id": "static",
                "host": "127.0.0.1",
                "port": ports["static"],
                "propagation_rule_version": 5,
                "sample_ratio_bps": 5000,
                "random_seed": 505,
            },
        ],
    )
    v2_nodes = write_json(tmp_path / "nodes_v2.json", evolved)
    return (
        load_dataset(str(v1_nodes), str(SAMPLE_SPANS)),
        load_dataset(str(v2_nodes), str(SAMPLE_SPANS)),
        ports,
    )


def _config() -> EngineConfig:
    return EngineConfig(
        global_concurrency=3,
        per_node_concurrency=1,
        batch_size=2,
        request_timeout_ms=3000,
        retries=1,
        handshake_timeout_ms=3000,
    )


def test_versioned_ledger_flip_and_unchanged_entry_bytes(tmp_path: Path) -> None:
    dataset_v1, dataset_v2, _ports = _versioned_datasets(tmp_path)
    db = LedgerDB(str(tmp_path / "ledger.sqlite3"))
    try:
        config = _config()
        with JsonlLogger(str(tmp_path / "run1.log.jsonl")) as logger:
            first = run_adjudication(db, dataset_v1, config, logger)
        first_view = view_current(db)
        before_static = next(
            row for row in first_view["traces"] if row["trace_id"] == "trace-static"
        )
        before_static_json = canonical_text(before_static["entry"])
        first_snapshot = first["snapshot_id"]
        assert first_view["view_mode"] == "full_replacement"
        assert first_view["authority"] == "cached_ledger_view_only"
        assert first_view["snapshot_digest_valid"] is True
        assert first_view["snapshot_rule_versions_valid"] is True

        with JsonlLogger(str(tmp_path / "run2.log.jsonl")) as logger:
            second = run_adjudication(db, dataset_v2, config, logger)
        second_view = view_current(db)
        after_static = next(
            row for row in second_view["traces"] if row["trace_id"] == "trace-static"
        )
        after_static_json = canonical_text(after_static["entry"])

        assert second["snapshot_id"] != first_snapshot
        assert second_view["current_ruleset_version"] == 2
        assert second_view["previous_ruleset_version"] == 1
        assert before_static_json == after_static_json
        assert before_static["entry_digest"] == after_static["entry_digest"]
        with sqlite3.connect(db.path) as conn:
            raw_static = conn.execute(
                """
                SELECT s.snapshot_id, le.entry_json
                FROM snapshots s
                JOIN snapshot_traces st ON st.snapshot_id=s.snapshot_id
                JOIN ledger_entries le ON le.entry_id=st.entry_id
                WHERE st.trace_id='trace-static'
                ORDER BY s.snapshot_id
                """
            ).fetchall()
        assert len(raw_static) == 2
        assert raw_static[0][1] == raw_static[1][1]
        flip_by_trace = {flip["trace_id"]: flip for flip in second_view["flips"]}
        assert "trace-static" not in flip_by_trace
        assert flip_by_trace["trace-api"]["change_kind"] == "changed"
        assert "sampled_node_ids" in flip_by_trace["trace-api"]["diff_fields"]
        assert flip_by_trace["trace-api"]["before_rule_versions"] == [1]
        assert flip_by_trace["trace-api"]["after_rule_versions"] == [3]
        now = {row["trace_id"]: row for row in second_view["traces"]}
        assert now["trace-cross"]["entry"]["sampled_node_ids"] == ["worker"]
        assert now["trace-three-node"]["entry"]["root_span_id"] == "tt1"
        assert len(second_view["changed_traces"]) >= 4

        with sqlite3.connect(db.path) as conn:
            old = conn.execute(
                "SELECT is_current, is_stale FROM snapshots WHERE snapshot_id=?",
                (first_snapshot,),
            ).fetchone()
            stale_current_stats = conn.execute(
                "SELECT COUNT(*) FROM latency_stats WHERE snapshot_id=? AND is_stale=0",
                (first_snapshot,),
            ).fetchone()[0]
            stale_current_samples = conn.execute(
                "SELECT COUNT(*) FROM latency_samples WHERE snapshot_id=? AND is_stale=0",
                (first_snapshot,),
            ).fetchone()[0]
        assert old == (0, 1)
        assert stale_current_stats == 0
        assert stale_current_samples == 0
    finally:
        db.close()


def test_ruleset_change_requires_node_version_increment(tmp_path: Path) -> None:
    dataset_v1, _dataset_v2, _ports = _versioned_datasets(tmp_path)
    nodes = dataset_v1.nodes_payload()
    nodes["ruleset_version"] = 3
    for index, node in enumerate(nodes["nodes"]):
        if node["node_id"] == "api":
            nodes["nodes"][index]["sample_ratio_bps"] = 0
    # Keep api's rule version at 1 while changing its ratio.
    changed_path = write_json(tmp_path / "bad_rule_version.json", nodes)
    changed = load_dataset(str(changed_path), str(SAMPLE_SPANS))
    db = LedgerDB(str(tmp_path / "ledger.sqlite3"))
    try:
        with JsonlLogger(None) as logger:
            run_adjudication(db, dataset_v1, _config(), logger)
        with pytest.raises(ArbiterError) as exc:
            with JsonlLogger(None) as logger:
                run_adjudication(db, changed, _config(), logger)
        assert exc.value.code == "E_SPEC"
        assert exc.value.path == "/nodes/0/propagation_rule_version"
    finally:
        db.close()


def test_authoritative_review_diverges_on_tampered_ledger(tmp_path: Path) -> None:
    dataset, _dataset_v2, _ports = _versioned_datasets(tmp_path)
    db = LedgerDB(str(tmp_path / "ledger.sqlite3"))
    try:
        with JsonlLogger(None) as logger:
            run_adjudication(db, dataset, _config(), logger)
            reviewed = review_current(db, _config(), logger)
        assert reviewed["ok"] is True
        assert reviewed["truth_source"] == "service_decision_summary"

        with sqlite3.connect(db.path) as conn:
            row = conn.execute(
                """
                SELECT le.entry_id, le.entry_json
                FROM ledger_entries le JOIN snapshot_traces st ON st.entry_id=le.entry_id
                JOIN snapshots s ON s.snapshot_id=st.snapshot_id AND s.is_current=1
                WHERE st.trace_id='trace-api'
                """
            ).fetchone()
            entry_id, entry_json = row
            payload = json.loads(entry_json)
            payload["root_span_id"] = "tampered-root"
            tampered_json = canonical_text(payload)
            tampered_digest = digest_text(tampered_json)
            conn.execute(
                "UPDATE ledger_entries SET entry_json=?, entry_digest=? WHERE entry_id=?",
                (tampered_json, tampered_digest, entry_id),
            )
            conn.commit()
        with pytest.raises(ArbiterError) as exc:
            with JsonlLogger(None) as logger:
                review_current(db, _config(), logger)
        assert exc.value.code == "E_DIVERGE"
        assert exc.value.trace_id == "trace-api"
        assert exc.value.field == "root_span_id"
    finally:
        db.close()


def test_replay_success_and_snapshot_digest_mismatch(tmp_path: Path) -> None:
    dataset, _dataset_v2, _ports = _versioned_datasets(tmp_path)
    db = LedgerDB(str(tmp_path / "ledger.sqlite3"))
    try:
        with JsonlLogger(None) as logger:
            run_adjudication(db, dataset, _config(), logger)
            replay = replay_snapshot(db, _config(), logger)
        assert replay["ok"] is True
        assert replay["trace_count"] == 5
        assert replay["truth_source"] == "service_decision_summary"
        with sqlite3.connect(db.path) as conn:
            row = conn.execute(
                """
                SELECT le.entry_id, le.entry_json
                FROM ledger_entries le JOIN snapshot_traces st ON st.entry_id=le.entry_id
                JOIN snapshots s ON s.snapshot_id=st.snapshot_id AND s.is_current=1
                WHERE st.trace_id='trace-cross'
                """
            ).fetchone()
            entry_id, entry_json = row
            payload = json.loads(entry_json)
            summary = json.loads(payload["decision_summary"])
            summary["root_span_id"] = "replay-tampered-root"
            payload["root_span_id"] = "replay-tampered-root"
            payload["decision_summary"] = canonical_text(summary)
            payload["decision_digest"] = digest_text(payload["decision_summary"])
            tampered_json = canonical_text(payload)
            conn.execute(
                "UPDATE ledger_entries SET entry_json=?, entry_digest=? WHERE entry_id=?",
                (tampered_json, digest_text(tampered_json), entry_id),
            )
            conn.commit()
        with pytest.raises(ArbiterError) as exc:
            with JsonlLogger(None) as logger:
                replay_snapshot(db, _config(), logger)
        assert exc.value.code == "E_REPLAY"
        assert exc.value.trace_id == "trace-cross"
        assert exc.value.field == "root_span_id"
        with pytest.raises(ArbiterError) as exc:
            with JsonlLogger(None) as logger:
                replay_snapshot(db, _config(), logger, inject_snapshot_mismatch=True)
        assert exc.value.code == "E_SNAPSHOT"
    finally:
        db.close()


def test_fixed_point_latency_statistics(tmp_path: Path) -> None:
    dataset, _dataset_v2, _ports = _versioned_datasets(tmp_path)
    db = LedgerDB(str(tmp_path / "ledger.sqlite3"))
    try:
        with JsonlLogger(None) as logger:
            run = run_adjudication(db, dataset, _config(), logger)
        snapshot_id = run["snapshot_id"]
        values = [("api", 10_000), ("api", 20_000), ("api", 30_000), ("worker", 40_000)]
        for node_id, elapsed_us in values:
            db.record_latency(
                snapshot_id=snapshot_id,
                operation="manual",
                node_id=node_id,
                started_ns=1_000_000,
                ended_ns=1_000_000 + elapsed_us * 1000,
                rebuild=False,
            )
        stats = db.rebuild_latency_stats(snapshot_id, operation="manual")
        all_stats = next(
            row for row in stats if row["operation"] == "manual" and row["node_id"] == "ALL"
        )
        assert all_stats["sample_count"] == 4
        assert all_stats["mean_us"] == 25_000
        assert all_stats["p95_us"] == 40_000
        assert all_stats["max_us"] == 40_000
        assert all_stats["scale_factor"] == STAT_SCALE_FACTOR
        assert all_stats["rounding"] == STAT_ROUNDING
        assert all_stats["quantization_error_us"] == STAT_QUANTIZATION_ERROR_US
        assert all_stats["quantization_error_ns"] == STAT_QUANTIZATION_ERROR_NS
        api_stats = next(
            row for row in stats if row["operation"] == "manual" and row["node_id"] == "api"
        )
        assert api_stats["mean_us"] == 20_000
        assert api_stats["p95_us"] == 30_000
    finally:
        db.close()


def _wait_ports_released(ports: list[int], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    remaining = list(ports)
    while remaining and time.monotonic() < deadline:
        still_busy: list[int] = []
        for port in remaining:
            with __import__("socket").socket(__import__("socket").AF_INET, __import__("socket").SOCK_STREAM) as sock:
                sock.setsockopt(__import__("socket").SOL_SOCKET, __import__("socket").SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    still_busy.append(port)
        remaining = still_busy
        if remaining:
            time.sleep(0.1)
    if remaining:
        raise AssertionError(f"ports still busy: {remaining}")


def test_forced_exit_recovery_and_replay(tmp_path: Path) -> None:
    dataset, _dataset_v2, ports = _versioned_datasets(tmp_path)
    db_path = tmp_path / "recovery.sqlite3"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    command = [
        sys.executable,
        "-m",
        "trace_arbiter",
        "adjudicate",
        "--nodes",
        dataset.nodes_path,
        "--spans",
        dataset.spans_path,
        "--db",
        str(db_path),
        "--log",
        str(tmp_path / "crash.log.jsonl"),
        "--global-concurrency",
        "2",
        "--per-node-concurrency",
        "1",
        "--batch-size",
        "1",
        "--fault-after-commits",
        "1",
    ]
    crashed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=20)
    assert crashed.returncode == 137
    _wait_ports_released(list(ports.values()))
    db = LedgerDB(str(db_path))
    try:
        incomplete = db.incomplete_run()
        assert incomplete is not None
        assert incomplete["status"] == "running"
        with JsonlLogger(str(tmp_path / "recover.log.jsonl")) as logger:
            recovery = recover_incomplete(db, _config(), logger)
        assert recovery["status"] == "recovered"
        assert recovery["batch_indices_rerun"]
        assert recovery["replay"]["ok"] is True
        with sqlite3.connect(db.path) as conn:
            unknown_count = conn.execute(
                "SELECT COUNT(*) FROM run_batches WHERE was_unknown=1"
            ).fetchone()[0]
            run = conn.execute("SELECT status, progress_cursor, batch_count FROM runs").fetchone()
            current = conn.execute(
                "SELECT COUNT(*) FROM snapshots WHERE is_current=1"
            ).fetchone()[0]
        assert unknown_count > 0
        assert run[0] == "completed"
        assert run[1] == run[2]
        assert current == 1
    finally:
        db.close()
