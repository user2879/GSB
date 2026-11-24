from __future__ import annotations

import json
from pathlib import Path

from fcdiag.errors import E_SNAPSHOT_VERSION_MISMATCH
from fcdiag.experiment import run_injections
from fcdiag.runner import Runner
from fcdiag.snapshot import SnapshotStore


def test_snapshot_commit_and_resume_decision(tmp_path, context):
    result = Runner(context["state"], context["receipts"][3]).run()
    store = SnapshotStore(str(tmp_path / "snapshots"))
    manifest_base = {
        "config_hash": context["state"].config_hash(),
        "sample_fingerprint": context["state"].sample_fingerprint(),
        "receipt_fingerprint": context["state"].receipt_fingerprint(),
        "completed_stages": [],
    }
    stage = "interact"
    artifacts = tuple(item for item in result.artifacts.values() if item.stage == stage)
    target = store.commit_stage(1, stage, artifacts, manifest_base, result.budget_used)
    assert target.exists()
    assert (tmp_path / "snapshots" / "manifest.json").exists()
    decision = store.resume_check(
        1,
        context["state"].config_hash(),
        context["state"].sample_fingerprint(),
        context["state"].receipt_fingerprint(),
    )
    assert decision.resume is True and decision.code is None
    changed = store.resume_check(
        2,
        context["state"].config_hash(),
        context["state"].sample_fingerprint(),
        context["state"].receipt_fingerprint(),
    )
    assert changed.resume is False and changed.code == E_SNAPSHOT_VERSION_MISMATCH
    loaded = store.load_stage(1, stage)
    assert loaded == artifacts


def test_snapshot_uses_versioned_directory_not_file_overwrite(tmp_path, context):
    result = Runner(context["state"], context["receipts"][3]).run()
    store = SnapshotStore(str(tmp_path / "snapshots"))
    manifest_base = {
        "config_hash": context["state"].config_hash(),
        "sample_fingerprint": context["state"].sample_fingerprint(),
        "receipt_fingerprint": context["state"].receipt_fingerprint(),
        "completed_stages": [],
    }
    artifacts = tuple(item for item in result.artifacts.values() if item.stage == "bucketize")
    first = store.commit_stage(1, "bucketize", artifacts, manifest_base, {})
    second = store.commit_stage(2, "bucketize", artifacts, manifest_base, {})
    assert first.parent != second.parent
    assert first.exists() and second.exists()


def test_runner_reuses_snapshot_store_for_resume_decision(tmp_path, context):
    store = SnapshotStore(str(tmp_path / "snapshots"))
    first = Runner(context["state"], context["receipts"][3], snapshot_store=store).run()
    second = Runner(context["state"], context["receipts"][3], snapshot_store=store).run()
    assert first.resume_decision["resume"] is False
    assert second.resume_decision["resume"] is True
    assert second.resume_decision["code"] is None


def test_injection_report_maps_every_fault_to_business_code(context):
    sample_dir = Path("samples")
    receipt_texts = {
        version: (sample_dir / "receipts" / f"v{version}.txt").read_text(encoding="utf-8")
        for version in (1, 2, 3)
    }
    reports = run_injections(
        (sample_dir / "samples.jsonl").read_text(encoding="utf-8"),
        (sample_dir / "pipeline.yaml").read_text(encoding="utf-8"),
        json.loads((sample_dir / "config.json").read_text(encoding="utf-8")),
        receipt_texts,
        json.loads((sample_dir / "injections.json").read_text(encoding="utf-8")),
    )
    assert len(reports) == 5
    assert [report.injection_id for report in reports] == [
        "INF-001",
        "INF-002",
        "INF-003",
        "INF-004",
        "INF-005",
    ]
    assert all(report.hit for report in reports)
    assert all(report.observed_code == report.expected_code for report in reports)
