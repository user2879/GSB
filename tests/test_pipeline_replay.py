from __future__ import annotations

import json
from reco_shard_diag.coordinator import ForcedTermination, run_pipeline
from reco_shard_diag.fixtures import build_demo_config
from reco_shard_diag.models import ProjectConfig
from reco_shard_diag.persistence import SnapshotStore, termination_classification
from reco_shard_diag.state import materialize


def _config(raw, event_count=None):
    item = json.loads(json.dumps(raw))
    if event_count is not None:
        item["events"] = item["events"][:event_count]
    return ProjectConfig.from_raw(item)


def _view(result):
    return {
        "phase": result.get("phase"),
        "metadata": result.get("metadata"),
        "per_shard": {
            shard_id: {
                "ready": value.get("ready"),
                "aggregate": value.get("aggregate"),
                "receipt": value.get("receipt"),
                "watermark": value.get("watermark"),
                "cache_keys": value.get("cache_keys"),
            }
            for shard_id, value in sorted(result.get("per_shard", {}).items())
        },
        "ready_wait_history": result.get("ready_wait_history"),
        "ready_wait_queue": result.get("ready_wait_queue"),
        "errors": result.get("errors"),
        "error_counts": result.get("error_counts"),
        "budget": result.get("budget"),
        "batch_plan": result.get("batch_plan"),
        "completed_cursor": result.get("completed_cursor"),
        "reuse_ledger": result.get("reuse_ledger"),
        "invalidation_ledger": result.get("invalidation_ledger"),
        "reuse_plan": result.get("reuse_plan"),
        "invalidation_plan": result.get("invalidation_plan"),
    }


def test_incremental_full_and_forced_recovery_are_identical(tmp_path):
    raw = build_demo_config(include_events=True)
    config = _config(raw)
    state = materialize(config)
    full = run_pipeline(
        config,
        state,
        tmp_path / "full",
        mode="full",
        use_previous=False,
    )
    baseline_config = _config(raw, 3)
    baseline = run_pipeline(
        baseline_config,
        materialize(baseline_config),
        tmp_path / "incremental",
        mode="incremental",
        use_previous=False,
    )
    assert baseline["phase"] == "COMPLETE"
    incremental = run_pipeline(
        config,
        state,
        tmp_path / "incremental",
        mode="incremental",
        use_previous=True,
    )
    assert _view(incremental) == _view(full)

    crash_work = tmp_path / "crash"
    try:
        run_pipeline(
            config,
            state,
            crash_work,
            mode="incremental",
            abort_after=2,
        )
    except ForcedTermination as exc:
        assert exc.completed == 2
    store = SnapshotStore(
        crash_work,
        global_version=config.global_version,
        base_config_hash=config.base_config_hash,
    )
    snapshot = store.load()
    assert snapshot["phase"] == "RUNNING"
    assert (
        termination_classification(snapshot)["kind"]
        == "RECOVERED_AFTER_FORCED_TERMINATION"
    )
    resumed = run_pipeline(
        config,
        state,
        crash_work,
        mode="resume",
        use_previous=True,
    )
    assert _view(resumed) == _view(full)
    assert resumed["execution"]["snapshot_status"]["loaded"] is True


def test_selective_reuse_and_downstream_invalidation(tmp_path):
    raw = build_demo_config(include_events=True)
    baseline_config = _config(raw, 0)
    selected_config = _config(raw, 1)
    work = tmp_path / "select"
    run_pipeline(
        baseline_config,
        materialize(baseline_config),
        work,
        mode="full",
        use_previous=False,
    )
    result = run_pipeline(
        selected_config,
        materialize(selected_config),
        work,
        mode="incremental",
        use_previous=True,
    )
    assert [entry["shard_id"] for entry in result["reuse_plan"]] == ["shard-e"]
    assert [entry["shard_id"] for entry in result["invalidation_plan"]] == [
        "shard-a",
        "shard-b",
        "shard-c",
        "shard-d",
    ]
    stages = {
        entry["stage"]
        for entry in result["cache_hit_ledger"]
        if entry["shard_id"] == "shard-e"
    }
    assert stages == {"readiness", "watermark", "aggregate", "receipt"}
    assert all(
        entry["downstream_affected"]
        for entry in result["invalidation_plan"]
        if entry["shard_id"] != "shard-a"
    )
