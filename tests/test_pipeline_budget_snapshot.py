from __future__ import annotations

import json

from corpus_workbench.codes import (
    E_GLOBAL_BUDGET_EXHAUSTED,
    E_SNAPSHOT_STALE,
    E_SOURCE_QUOTA_EXCEEDED,
)
from corpus_workbench.demo_data import generate_demo_workspace
from corpus_workbench.pipeline import Workbench


def _workbench(tmp_path, output_name: str, run_id: str, scheduler_name: str, dedup_name: str = "dedup.json"):
    return Workbench(
        data_path=tmp_path / "data" / "corpus",
        balance_path=tmp_path / "configs" / "balance.json",
        dedup_path=tmp_path / "configs" / dedup_name,
        scheduler_path=tmp_path / "configs" / scheduler_name,
        injection_path=tmp_path / "configs" / "injection.json",
        output_root=tmp_path / output_name,
        run_id=run_id,
    )


def test_budget_exhaustion_snapshot_and_resume(tmp_path):
    generate_demo_workspace(tmp_path)
    baseline = _workbench(tmp_path, "out-baseline", "base", "scheduler.json").run(reset=True)
    assert baseline["status"] == "complete"

    tight = _workbench(tmp_path, "out-budget", "budget", "scheduler_tight.json")
    first = tight.run(reset=True)
    assert first["status"] == "budget_exhausted"
    assert first["complete"] is False
    assert first["pending_queue"]
    assert first["budget"]["consumed_points"] <= 160
    codes = {failure["code"] for failure in first["failures"]}
    assert E_SOURCE_QUOTA_EXCEEDED in codes
    assert E_GLOBAL_BUDGET_EXHAUSTED in codes

    first_version = tight.snapshot_store.current_version()
    manifest_path = (
        tight.snapshot_store.versions_root
        / f"v{first_version:06d}"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["samples"] == first["sampling"]["samples"]
    assert manifest["pending_queue"] == first["pending_queue"]
    assert manifest["consumed_points"] == first["budget"]["consumed_points"]

    expanded = _workbench(tmp_path, "out-budget", "budget", "scheduler_expanded.json")
    second = expanded.run(resume=True)
    assert second["status"] == "complete"
    assert second["complete"] is True
    assert second["pending_queue"] == []
    assert second["budget"]["consumed_points"] > first["budget"]["consumed_points"]
    assert expanded.snapshot_store.current_version() > first_version
    assert {
        (item["source"], item["doc_id"]) for item in second["dedup"]["retained_documents"]
    } == {
        (item["source"], item["doc_id"]) for item in baseline["dedup"]["retained_documents"]
    }
    assert second["deterministic_result_digest"] == baseline["deterministic_result_digest"]


def test_stale_version_or_config_refuses_old_artifacts(tmp_path):
    generate_demo_workspace(tmp_path)
    base = _workbench(tmp_path, "out-stale", "stale", "scheduler.json")
    base_report = base.run(reset=True)
    assert base_report["status"] == "complete"
    version_before = base.snapshot_store.current_version()

    changed_config = _workbench(
        tmp_path,
        "out-stale",
        "stale",
        "scheduler.json",
        dedup_name="dedup_threshold_090.json",
    )
    stale = changed_config.run(resume=True)
    assert stale["status"] == "stale"
    assert stale["complete"] is False
    assert stale["failures"][0]["code"] == E_SNAPSHOT_STALE
    assert stale["global_version"] == version_before
    assert changed_config.snapshot_store.current_version() == version_before
    assert "dedup configuration hash changed" in stale["snapshot"]["stale_details"]["reasons"][0]

    wrong_version = _workbench(tmp_path, "out-stale", "stale", "scheduler.json")
    stale_version = wrong_version.run(resume=True, expected_version=version_before + 1)
    assert stale_version["status"] == "stale"
    assert stale_version["failures"][0]["code"] == E_SNAPSHOT_STALE
    assert wrong_version.snapshot_store.current_version() == version_before


def test_checkpoint_interruption_can_resume(tmp_path):
    generate_demo_workspace(tmp_path)
    workbench = _workbench(tmp_path, "out-checkpoint", "checkpoint", "scheduler.json")
    first = workbench.run(reset=True, stop_after_batches=1)
    assert first["status"] == "interrupted"
    assert first["pending_queue"]
    assert first["budget"]["batch_count"] == 1
    assert first["budget"]["cursor"]["next_pending_stable_key"] is not None

    resumed = _workbench(tmp_path, "out-checkpoint", "checkpoint", "scheduler.json").run(resume=True)
    assert resumed["status"] == "complete"
    assert resumed["pending_queue"] == []
    assert resumed["budget"]["batch_count"] > 1
