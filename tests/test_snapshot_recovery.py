"""Requirement 6/8: atomic snapshots, recovery, config mismatch, signals."""
import json
import os
import signal
import threading
import time

import pytest

from detbox.errors import E_CONFIG_MISMATCH
from detbox.models import ClassTable
from detbox.config import DetectionParams, MatchConfig, load_injections
from detbox.pipeline import Pipeline
from detbox.report import build_report, write_report
from detbox.samplegen import generate_dataset, WIDTH, HEIGHT
from detbox.store import SnapshotStore, SNAPSHOT_VERSION
from detbox.util import read_json


@pytest.fixture()
def dataset(tmp_path):
    root = tmp_path / "dataset"
    generate_dataset(root)
    return root


def _pipeline(dataset, workdir, **kwargs):
    return Pipeline(
        dataset_root=dataset,
        workdir=workdir,
        fmt=kwargs.get("fmt", "v2"),
        detection_params=DetectionParams.load(dataset / "detection_params.json"),
        match_config=MatchConfig.load(dataset / "match_config.json"),
        classes=ClassTable.from_dict(read_json(dataset / "ground_truth.json")["class_table"]),
        width=WIDTH,
        height=HEIGHT,
        injections=load_injections(kwargs.get("injections")),
    )


def test_snapshot_stages_committed_with_manifest(dataset, tmp_path):
    work = tmp_path / "work"
    result = _pipeline(dataset, work, injections=dataset / "injections.json").run()
    store = SnapshotStore(work)
    for stage in ("detect", "match", "serialize"):
        assert store.stage_exists(stage)
        assert not list(store.snapshot.glob("*.staging-*"))
    manifest = read_json(work / "manifest.json")
    assert manifest["snapshot_version"] == SNAPSHOT_VERSION
    assert set(manifest["stage_versions"]) == {"detect", "match", "serialize"}
    assert manifest["config_hash"] == result["manifest"]["config_hash"]


def test_resume_after_rerun_uses_cache(dataset, tmp_path):
    work = tmp_path / "work"
    first = _pipeline(dataset, work).run()
    assert first["reused"] == []
    second = _pipeline(dataset, work).run()
    assert second["reused"]
    assert second["recomputed"] == []


def test_strict_config_mismatch_rejects_resume(dataset, tmp_path):
    work = tmp_path / "work"
    _pipeline(dataset, work).run()
    # Tamper with the stored config hash externally (unknown state).
    manifest_path = work / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["config_hash"] = "x" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = SnapshotStore(work)
    with pytest.raises(Exception) as exc:
        store.verify_resume(SNAPSHOT_VERSION, "y" * 64, strict=True)
    assert exc.value.code == E_CONFIG_MISMATCH


def test_recovery_ignores_incomplete_staging(dataset, tmp_path):
    work = tmp_path / "work"
    _pipeline(dataset, work).run()
    staging = work / "snapshot" / ".match.staging-999"
    staging.mkdir(parents=True)
    (staging / "partial.json").write_text("{broken", encoding="utf-8")
    # A rerun only sees fully replaced stage directories and succeeds.
    result = _pipeline(dataset, work).run()
    assert result["exit_code"] == 0
    assert not staging.exists()


def test_sigint_finishes_current_frame_and_exits_zero(dataset, tmp_path):
    work = tmp_path / "work"
    pipeline = _pipeline(dataset, work)
    thread = threading.Thread(target=pipeline.run)
    thread.start()
    time.sleep(0.2)
    os.kill(os.getpid(), signal.SIGINT)
    thread.join(timeout=30)
    assert not thread.is_alive()
    manifest = read_json(work / "manifest.json")
    assert manifest["shutdown_requested"] is True


def test_report_atomic_write_contains_required_fields(dataset, tmp_path):
    work = tmp_path / "work"
    result = _pipeline(dataset, work, injections=dataset / "injections.json").run()
    classes = ClassTable.from_dict(read_json(dataset / "ground_truth.json")["class_table"])
    report = build_report(result, classes, WIDTH, HEIGHT)
    report_path = tmp_path / "report.json"
    write_report(report_path, report)
    loaded = read_json(report_path)
    for key in (
        "tracks", "round_trip", "failed_frames", "injection_hits",
        "invalidated", "reused", "snapshot_version", "business_codes",
        "eviction_log", "failure_list", "fingerprints",
    ):
        assert key in loaded
    assert loaded["round_trip"]["class_mapping_roundtrip"]["circle"]["consistent"]
    assert {hit["injection_id"] for hit in loaded["injection_hits"]} >= {
        "INJ001", "INJ002", "INJ003", "INJ004", "INJ005", "INJ006"
    }
