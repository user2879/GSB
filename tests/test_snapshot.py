import json
import os

import pytest

from edge_cache.config import load_config_dir
from edge_cache.errors import E_SNAPSHOT_VERSION, EdgeCacheError
from edge_cache.ledger import Ledger
from edge_cache.runner import RunOrchestrator
from edge_cache.snapshot import (
    STAGES,
    SnapshotStore,
    build_diagnostic_report,
    write_exit_record,
)


def test_partial_temp_directory_is_ignored_on_recovery(tmp_workdir):
    store = SnapshotStore(os.path.join(tmp_workdir, "snapshots"))
    handle = store.begin("torn")
    store.write_stage(handle, "config", {"hello": "world"})
    # No commit: simulate SIGKILL leaving the temp directory behind.
    assert store.current_id() is None
    cleaned = store.cleanup_temp()
    assert cleaned == ["torn"]
    recovery = store.load_current()
    assert recovery["status"] == "no-snapshot"
    assert recovery["missing_stages"] == list(STAGES)


def test_committed_snapshot_round_trips_and_reports_cursor(sample_config_dir, tmp_workdir):
    work_dir = os.path.join(tmp_workdir, "run")
    RunOrchestrator(sample_config_dir, work_dir, "c3").run()
    store = SnapshotStore(os.path.join(work_dir, "snapshots"))
    config = load_config_dir(sample_config_dir)
    recovery = store.load_current(expected_config_hash=config.source_hash)
    assert recovery["status"] == "recovered"
    assert recovery["missing_stages"] == []
    manifest = recovery["manifest"]
    assert manifest["global_version"] >= 1
    assert manifest["config_hash"] == config.source_hash
    assert set(manifest["stages"]) == set(STAGES)
    assert manifest["cursor"]["completed_shards"]
    assert isinstance(manifest["cursor"]["unfinished_shards"], list)


def test_config_hash_mismatch_is_rejected(sample_config_dir, tmp_workdir):
    work_dir = os.path.join(tmp_workdir, "run")
    RunOrchestrator(sample_config_dir, work_dir, "c3").run()
    store = SnapshotStore(os.path.join(work_dir, "snapshots"))
    with pytest.raises(EdgeCacheError) as exc:
        store.load_current(expected_config_hash="deadbeef")
    assert exc.value.failure.code == E_SNAPSHOT_VERSION


def test_diagnostic_report_counts_codes_advances_and_formats(sample_config_dir, tmp_workdir):
    work_dir = os.path.join(tmp_workdir, "run")
    report = RunOrchestrator(sample_config_dir, work_dir, "c3").run()
    assert report["recovery_status"] == "clean-run"
    assert report["format_counts"]["c3"] > 0
    assert isinstance(report["business_code_counts"], dict)
    assert sum(report["node_generation_advances"].values()) > 0
    assert report["business_entries"] == len(load_config_dir(sample_config_dir).shards)


def test_exit_record_is_processable_and_persisted(sample_config_dir, tmp_workdir):
    work_dir = os.path.join(tmp_workdir, "run")
    RunOrchestrator(sample_config_dir, work_dir, "c3").run()
    ledger = Ledger(os.path.join(work_dir, "ledger"))
    record = write_exit_record(
        ledger,
        code="E_LEASE_EXPIRED",
        message="forced termination handled on restart",
        detail={"stage": "leases"},
    )
    assert record["code"] == "E_LEASE_EXPIRED"
    rows = Ledger._read_jsonl(os.path.join(work_dir, "ledger", "exits.jsonl"))
    assert rows[-1]["code"] == "E_LEASE_EXPIRED"


def test_missing_stage_is_named_in_partial_snapshot(tmp_workdir):
    store = SnapshotStore(os.path.join(tmp_workdir, "snapshots"))
    handle = store.begin("partial")
    store.write_stage(handle, "config", {"v": 1})
    store.write_stage(handle, "leases", {"leases": []})
    store.commit(
        handle,
        global_version=1,
        config_hash="hash-1",
        cursor={"completed_shards": ["s1"], "unfinished_shards": ["s2"],
                "budget_consumed": 3},
    )
    recovery = store.load_current(expected_config_hash="hash-1")
    assert recovery["status"] == "partial"
    assert "conclusions" in recovery["missing_stages"]
    ledger = Ledger(os.path.join(tmp_workdir, "ledger"))
    report = build_diagnostic_report(
        recovery, ledger,
        format_counts={"c1": 1, "c2": 0, "c3": 0},
        node_advances={"node-a": 2},
        budget_consumed=3,
    )
    assert report["unfinished_shards"] == ["s2"]
    assert report["completed_shards"] == ["s1"]
    assert report["missing_stages"]
