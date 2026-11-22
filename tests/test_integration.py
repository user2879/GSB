from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from driftwatch.config import load_inputs
from driftwatch.engine import DriftEngine, replay_ledger
from driftwatch.processes import port_is_free

from .conftest import free_port, write_json


def _configuration(tmp_path: Path, ports: list[int]) -> tuple[Path, Path, Path]:
    snapshots = {
        "snapshots": [
            {
                "seq": 1,
                "parent_seq": None,
                "hosts": {
                    "host-a": {
                        "expected": {"cpu_quota": 100000, "memory_pages": 64000, "processes": 500},
                        "tolerance_bps": 500,
                    },
                    "host-b": {
                        "expected": {"cpu_quota": 82000, "memory_pages": 52000, "processes": 430},
                        "tolerance_bps": 700,
                    },
                },
            },
            {
                "seq": 2,
                "parent_seq": 1,
                "hosts": {
                    "host-a": {
                        "expected": {"cpu_quota": 102000, "memory_pages": 65000, "processes": 505},
                        "tolerance_bps": 200,
                    },
                    "host-b": {
                        "expected": {"cpu_quota": 84000, "memory_pages": 53000, "processes": 435},
                        "tolerance_bps": 300,
                    },
                },
            },
        ]
    }
    profiles = {
        host: {
            metric: {"base": base, "spread": spread}
            for metric, (base, spread) in zip(
                ("cpu_quota", "memory_pages", "processes"),
                [(100000, 1200), (63600, 300), (498, 4)],
            )
        }
        for host in ("host-a", "host-b")
    }
    agents = {
        "agents": [
            {
                "agent_id": "a1",
                "host": "127.0.0.1",
                "port": ports[0],
                "seed": 11,
                "aggregation": {"kind": "median", "window": 5, "version": "median-v1"},
                "hosts": ["host-a", "host-b"],
                "profiles": profiles,
            },
            {
                "agent_id": "a2",
                "host": "127.0.0.1",
                "port": ports[1],
                "seed": 22,
                "aggregation": {"kind": "mean_floor", "window": 5, "version": "mean-v1"},
                "hosts": ["host-a"],
                "profiles": {"host-a": profiles["host-a"]},
            },
            {
                "agent_id": "a3",
                "host": "127.0.0.1",
                "port": ports[2],
                "seed": 33,
                "aggregation": {"kind": "trimmed_mean", "window": 7, "version": "trim-v1"},
                "hosts": ["host-b"],
                "profiles": {"host-b": profiles["host-b"]},
            },
        ]
    }
    criteria = {
        "global_concurrency": 3,
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
    )


def test_end_to_end_merge_drift_and_replay(tmp_path):
    ports = [free_port(), free_port(), free_port()]
    snapshots, agents, criteria = _configuration(tmp_path, ports)
    db = str(tmp_path / "ledger.sqlite3")
    engine = DriftEngine.from_files(str(snapshots), str(agents), str(criteria), db)
    try:
        report = engine.run()
        assert report.view_summary["merged_rows"] == 12
        assert report.view_summary["drift_rows"] == 12
        assert report.replay is None
        view = engine.ledger.session_snapshot(report.generation)
        source_counts = {(row["snapshot_seq"], row["host_id"], row["metric_id"]): row["source_count"] for row in view["merged_view"]}
        assert source_counts[(1, "host-a", "cpu_quota")] == 2
        assert source_counts[(1, "host-b", "cpu_quota")] == 2
        assert all(row["authority_agent"] is not None for row in view["merged_view"])
        replay = engine.run(verify_replay=True)
        assert replay.replay["ok"] is True
        assert replay.replay["matched_rows"] == 12
        assert replay.replay["batch_count"] == 2
    finally:
        engine.close()


def test_aggregation_evolution_replaces_whole_generation_and_tracks_affected(tmp_path):
    ports = [free_port(), free_port(), free_port(), free_port()]
    snapshots, agents_path, criteria_path = _configuration(tmp_path, ports)
    chain, agents, criteria = load_inputs(str(snapshots), str(agents_path), str(criteria_path))
    db = str(tmp_path / "ledger.sqlite3")
    first_engine = DriftEngine(chain, agents, criteria, db)
    try:
        first = first_engine.run()
        old_generation = first.generation
    finally:
        first_engine.close()

    agents_payload = json.loads(agents_path.read_text())
    agents_payload["agents"][1]["aggregation"] = {"kind": "mean_floor", "window": 9, "version": "mean-v2"}
    agents_payload["agents"].append(
        {
            "agent_id": "a4",
            "host": "127.0.0.1",
            "port": ports[3],
            "seed": 44,
            "aggregation": {"kind": "median", "window": 5, "version": "median-v1"},
            "hosts": ["host-b"],
        }
    )
    evolved_path = write_json(tmp_path / "agents-v2.json", agents_payload)
    evolved_chain, evolved_agents, evolved_criteria = load_inputs(str(snapshots), str(evolved_path), str(criteria_path))
    second_engine = DriftEngine(evolved_chain, evolved_agents, evolved_criteria, db)
    try:
        second = second_engine.run()
        assert second.replaced is True
        assert second.generation != old_generation
        affected = {(item["snapshot_seq"], item["host_id"], item["metric_id"]): item["reasons"] for item in second.affected}
        reasons = affected[(1, "host-a", "cpu_quota")]
        assert "aggregation_version_changed:a2:mean-v1->mean-v2" in reasons
        assert "agent_added:a4" not in reasons
        assert "agent_added:a4" in affected[(1, "host-b", "cpu_quota")]
        assert any(item["kind"] == "agent_added" for item in second.differences)
        view = second_engine.ledger.session_snapshot(second.generation)
        assert len(view["merged_view"]) == 12
        assert not any(row["stale"] for row in view["merged_view"])
    finally:
        second_engine.close()

    replay = replay_ledger(db)
    assert replay["replay"]["ok"] is True
    assert replay["generation"] == second.generation


def test_all_ports_are_closed_after_run(tmp_path):
    ports = [free_port(), free_port(), free_port()]
    snapshots, agents, criteria = _configuration(tmp_path, ports)
    engine = DriftEngine.from_files(str(snapshots), str(agents), str(criteria), str(tmp_path / "ledger.sqlite3"))
    try:
        engine.run()
    finally:
        engine.close()
    assert all(port_is_free("127.0.0.1", port) for port in ports)


def test_replay_mismatch_reports_first_structured_location(tmp_path):
    from driftwatch.errors import DriftError
    import sqlite3
    import pytest

    ports = [free_port(), free_port(), free_port()]
    snapshots, agents, criteria = _configuration(tmp_path, ports)
    db = str(tmp_path / "ledger.sqlite3")
    engine = DriftEngine.from_files(str(snapshots), str(agents), str(criteria), db)
    try:
        report = engine.run()
        generation = report.generation
    finally:
        engine.close()

    with sqlite3.connect(db) as raw:
        row = raw.execute(
            """
            SELECT batch_id, host_id, metric_id FROM drift_rows
            WHERE generation=?
            ORDER BY snapshot_seq, host_id, metric_id LIMIT 1
            """,
            (generation,),
        ).fetchone()
        raw.execute(
            "UPDATE drift_rows SET decision='REJECT', status='DRIFT' WHERE batch_id=? AND host_id=? AND metric_id=?",
            row,
        )
        raw.commit()

    with pytest.raises(DriftError) as caught:
        replay_ledger(db)
    error = caught.value
    assert error.code == "E_REPLAY"
    assert error.details["snapshot_seq"] == 1
    assert error.details["host_id"] == "host-a"
    assert error.details["metric_id"] in {"cpu_quota", "memory_pages", "processes"}
    assert error.details["field"].startswith("drift.")


def test_force_kill_does_not_commit_partial_observations(tmp_path):
    ports = [free_port(), free_port(), free_port()]
    snapshots, agents, criteria = _configuration(tmp_path, ports)
    db = str(tmp_path / "ledger.sqlite3")
    engine = DriftEngine.from_files(str(snapshots), str(agents), str(criteria), db)

    def hook(event: str, payload: dict) -> None:
        if event == "batch_begun" and payload["snapshot_seq"] == 2:
            raise RuntimeError("simulated interruption before collection")

    try:
        try:
            engine.run(lifecycle_hook=hook)
        except RuntimeError:
            pass
        with sqlite3.connect(db) as raw:
            committed = raw.execute("SELECT COUNT(*) FROM batches WHERE status='COMMITTED'").fetchone()[0]
            running = raw.execute("SELECT COUNT(*) FROM batches WHERE status='RUNNING'").fetchone()[0]
            partial = raw.execute(
                """
                SELECT COUNT(*) FROM observations o JOIN batches b ON b.batch_id=o.batch_id
                WHERE b.status='RUNNING'
                """
            ).fetchone()[0]
        assert committed == 1
        assert running == 1
        assert partial == 0
        recovered = engine.ledger.recover_incomplete()
        assert {item["snapshot_seq"] for item in recovered} == {2}
    finally:
        engine.close()
