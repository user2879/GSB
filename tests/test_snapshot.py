from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quant_workbench.engine import WorkbenchEngine
from quant_workbench.errors import E_SNAPSHOT_VERSION_MISMATCH, WorkbenchError
from quant_workbench.loading import load_calibration, load_commands, load_config, load_weight_pack
from quant_workbench.snapshot import SnapshotStore
from quant_workbench.util import canonical_json


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def new_engine(sample_paths: dict[str, Path], snapshot: Path, *, expected_version=None):
    config = load_config(sample_paths["config"])
    config.latency_budget_ms = 1_000
    config.point_batch_budget = 1_000_000
    config.worker_delay_ms = 0
    weights, w_issues = load_weight_pack(sample_paths["weights"])
    calibration, c_issues = load_calibration(sample_paths["calibration"])
    return WorkbenchEngine(
        config,
        weights,
        calibration,
        snapshot_store=SnapshotStore(snapshot),
        expected_version=expected_version,
        base_issues=w_issues + c_issues,
        command_base_dir=sample_paths["commands"].parent,
    )


def test_snapshot_resume_is_whole_snapshot_and_reuses_artifacts(
    sample_paths: dict[str, Path], tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    engine = new_engine(sample_paths, snapshot)
    try:
        commands, _ = load_commands(sample_paths["commands"])
        for command in commands:
            engine.process_command(command)
        engine.drain_all()
        expected = engine.result_payload()
        manifest = engine.commit_snapshot()
        assert manifest is not None
        assert manifest["completed_layers"] == [f"layer.{index}" for index in range(5)]
        assert manifest["artifact_versions"].keys() >= {
            "ingest",
            "calibration",
            "quantization",
            "evaluation",
            "deployment",
        }
        assert (snapshot / "runtime_state.json").exists()
    finally:
        engine.close()

    resumed = new_engine(sample_paths, snapshot)
    try:
        actual = resumed.result_payload()
        assert canonical_json(expected) == canonical_json(actual)
        assert resumed._needs_work_layers() == []
        assert len(resumed.diagnostics()["cache_stats"]["misses"]) == 0
    finally:
        resumed.close()

    before = digest_tree(snapshot)
    config = load_config(sample_paths["config"])
    config.snapshot_version += 1
    weights, _ = load_weight_pack(sample_paths["weights"])
    calibration, _ = load_calibration(sample_paths["calibration"])
    with pytest.raises(WorkbenchError) as error:
        WorkbenchEngine(
            config,
            weights,
            calibration,
            snapshot_store=SnapshotStore(snapshot),
        )
    assert error.value.code == E_SNAPSHOT_VERSION_MISMATCH
    assert digest_tree(snapshot) == before


def test_expected_version_mismatch_does_not_touch_snapshot(
    sample_paths: dict[str, Path], tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    engine = new_engine(sample_paths, snapshot)
    try:
        engine.drain_all()
        engine.commit_snapshot()
    finally:
        engine.close()
    before = digest_tree(snapshot)
    with pytest.raises(WorkbenchError) as error:
        new_engine(sample_paths, snapshot, expected_version=99)
    assert error.value.code == E_SNAPSHOT_VERSION_MISMATCH
    assert digest_tree(snapshot) == before


def test_config_hash_mismatch_is_detected_without_overwrite(
    sample_paths: dict[str, Path], tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    engine = new_engine(sample_paths, snapshot)
    try:
        engine.drain_all()
        engine.commit_snapshot()
    finally:
        engine.close()
    before = digest_tree(snapshot)

    changed_config = tmp_path / "config-hash-changed.yaml"
    changed_config.write_text(
        sample_paths["config"].read_text(encoding="utf-8") + "\n# changed file hash\n",
        encoding="utf-8",
    )
    config = load_config(changed_config)
    weights, _ = load_weight_pack(sample_paths["weights"])
    calibration, _ = load_calibration(sample_paths["calibration"])
    with pytest.raises(WorkbenchError) as error:
        WorkbenchEngine(
            config,
            weights,
            calibration,
            snapshot_store=SnapshotStore(snapshot),
        )
    assert error.value.code == E_SNAPSHOT_VERSION_MISMATCH
    assert digest_tree(snapshot) == before
