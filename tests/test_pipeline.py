from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from log_sampler.pipeline import Pipeline, RuntimeConfig
from log_sampler.snapshot import SnapshotManager
from log_sampler.state import SessionStore


def test_single_bucket_edit_reuses_all_other_buckets(sample_bundle, tmp_path):
    state = tmp_path / "incremental"
    store = SessionStore(state)
    store.initialize(
        sample_bundle["config"], sample_bundle["batches"], sample_bundle["injections"]
    )
    config, batches, injections = store.effective_input()
    baseline = Pipeline(
        config,
        batches,
        injections,
        RuntimeConfig(state_root=state),
    ).run()
    assert baseline.logical_result["final_samples"]

    command = {
        "id": "capacity-only",
        "type": "set_bucket_capacity",
        "device_id": "edge-a",
        "severity": "warning",
        "capacity": 2,
    }
    store.apply_commands([command])
    config, batches, injections = store.effective_input()
    result = Pipeline(
        config,
        batches,
        injections,
        RuntimeConfig(state_root=state),
    ).run()
    ledger = result.ledger
    misses = [
        item["bucket_id"]
        for item in ledger["buckets"]
        if item["status"] == "cache_miss"
    ]
    hits = [
        item["bucket_id"]
        for item in ledger["buckets"]
        if item["status"] == "cache_hit"
    ]
    assert misses == ["edge-a|warning"]
    assert len(hits) == 8
    assert ledger["final_sample"]["status"] == "cache_miss"
    assert any(
        item["reason"] == "bucket_parameters_changed"
        for item in ledger["buckets"]
        if item["bucket_id"] == "edge-a|warning"
    )
    final_sources = {item["log_id"]: item for item in ledger["final_sources"]}
    for sample in result.logical_result["final_samples"]:
        source = final_sources[sample["log_id"]]
        assert source["slot_no"] == sample["slot_no"]
        assert source["bucket_id"] == sample["bucket_id"]
    assert all(
        record["source_slot"] is not None
        for record in ledger["records"]
        if record["in_final_sample"]
    )


def test_manifest_has_stage_versions_and_no_pending_work(sample_bundle, tmp_path):
    result = Pipeline(
        copy.deepcopy(sample_bundle["config"]),
        copy.deepcopy(sample_bundle["batches"]),
        copy.deepcopy(sample_bundle["injections"]),
        RuntimeConfig(state_root=tmp_path / "run"),
    ).run()
    manifest = result.manifest
    assert manifest["status"] == "committed"
    assert set(manifest["stage_versions"]) == {
        "validation",
        "buckets",
        "truncation",
        "serialization",
    }
    assert manifest["pending_buckets"] == []
    assert manifest["consumed_budget"] == len(result.logical_result["final_samples"])
    assert manifest["completed_batch_cursor"] == 6
    for version, artifact in {
        "p1": "evidence.p1.txt",
        "p2": "evidence.p2.json",
        "p3": "evidence.p3.bin",
    }.items():
        assert (result.version_dir / artifact).exists()
        assert (result.version_dir / f"{artifact}.metadata.json").exists()
    assert (result.version_dir / "logical_result.json").exists()
    assert (result.version_dir / "ledger.json").exists()
    assert (result.version_dir / "diagnostics.json").exists()


def test_executor_never_exceeds_configured_workers(sample_bundle, tmp_path):
    config = copy.deepcopy(sample_bundle["config"])
    config["max_workers"] = 2
    observed: list[int] = []
    real_executor = ThreadPoolExecutor

    def factory(*args, **kwargs):
        observed.append(kwargs.get("max_workers"))
        return real_executor(*args, **kwargs)

    with patch("log_sampler.pipeline.ThreadPoolExecutor", side_effect=factory):
        result = Pipeline(
            config,
            copy.deepcopy(sample_bundle["batches"]),
            copy.deepcopy(sample_bundle["injections"]),
            RuntimeConfig(state_root=tmp_path / "run"),
        ).run()
    assert observed
    assert max(observed) <= 2
    task_keys = [item["sort_key"] for item in result.ledger["execution_tasks"]]
    assert task_keys == sorted(task_keys)


def test_removed_whole_bucket_is_recorded_in_ledger(sample_bundle, tmp_path):
    state = tmp_path / "run"
    baseline = Pipeline(
        copy.deepcopy(sample_bundle["config"]),
        copy.deepcopy(sample_bundle["batches"]),
        copy.deepcopy(sample_bundle["injections"]),
        RuntimeConfig(state_root=state),
    ).run()
    assert "edge-b|debug" in baseline.logical_result["buckets"]

    batches = copy.deepcopy(sample_bundle["batches"])
    for batch in batches:
        batch["records"] = [
            record
            for record in batch["records"]
            if not (
                record["device_id"] == "edge-b"
                and record["severity"] == "debug"
            )
        ]
    second = Pipeline(
        copy.deepcopy(sample_bundle["config"]),
        batches,
        copy.deepcopy(sample_bundle["injections"]),
        RuntimeConfig(state_root=state),
    ).run()
    assert "edge-b|debug" not in second.logical_result["buckets"]
    removed_bucket = next(
        item
        for item in second.ledger["buckets"]
        if item["bucket_id"] == "edge-b|debug"
    )
    assert removed_bucket["reason"] == "bucket_removed_no_current_records"
    removed_records = [
        item
        for item in second.ledger["records"]
        if item["bucket_id"] == "edge-b|debug"
    ]
    assert removed_records
    assert all(item["status"] == "removed" for item in removed_records)


def test_staging_reuses_matching_manifest_and_rebuilds_on_input_change(tmp_path):
    manager = SnapshotManager(tmp_path / "snapshot")
    first = manager.begin(
        config_hash="config-a",
        input_hash="input-a",
        random_source_digest="random-a",
    )
    assert not first.resumed
    manager.save(first)
    resumed = manager.begin(
        config_hash="config-a",
        input_hash="input-a",
        random_source_digest="random-a",
    )
    assert resumed.resumed
    rebuilt = manager.begin(
        config_hash="config-b",
        input_hash="input-b",
        random_source_digest="random-b",
    )
    assert not rebuilt.resumed
    assert rebuilt.manifest["config_hash"] == "config-b"
    assert rebuilt.manifest["input_hash"] == "input-b"
