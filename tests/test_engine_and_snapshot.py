from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from remote_seg.engine import RunOptions, SegmentationEngine, run_until_complete
from remote_seg.errors import E_SNAPSHOT_STALE, SegmentationError
from remote_seg.snapshot import SnapshotStore
from remote_seg.samplegen import generate_sample_workspace
from remote_seg.verify import compare_states, load_state

from .conftest import write_config


def _options(sample_workspace, state_dir: Path, config_path: Path | None = None) -> RunOptions:
    return RunOptions(
        index_path=sample_workspace["index"],
        config_path=config_path or sample_workspace["config"],
        fusion_versions_path=sample_workspace["fusion"],
        state_dir=state_dir,
        report_path=state_dir.parent / f"{state_dir.name}-report.json",
        max_batches=100,
    )


def test_budget_pending_then_resume_prioritises_queue(sample_workspace, tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "budget-config.json", max_workers=2, recompute_budget=2)
    state_dir = tmp_path / "state"
    first_options = _options(sample_workspace, state_dir, config_path)
    first = SegmentationEngine(first_options).run()
    assert first["status"] == "budget_pending"
    assert "E_RECOMPUTE_BUDGET_PENDING" in first["codes"]
    assert first["version_convergence"][-1]["pending"]
    pending_keys = [entry["key"] for entry in first["version_convergence"][-1]["pending"]]

    second_options = _options(sample_workspace, state_dir, config_path)
    second = run_until_complete(second_options)
    assert second["status"] == "complete"
    assert second["recovery_before_run"]["global_version"] >= 1
    first_batch_deductions = [entry["key"] for entry in second["budget_deductions"] if entry["batch"] == 1]
    assert first_batch_deductions == pending_keys[: len(first_batch_deductions)]


def test_concurrent_and_serial_outputs_match(sample_workspace, tmp_path: Path) -> None:
    serial_config = write_config(tmp_path / "serial.json", max_workers=1, recompute_budget=50)
    parallel_config = write_config(tmp_path / "parallel.json", max_workers=4, recompute_budget=50)
    serial = run_until_complete(_options(sample_workspace, tmp_path / "serial-state", serial_config))
    parallel = run_until_complete(
        _options(sample_workspace, tmp_path / "parallel-state", parallel_config)
    )
    assert serial["status"] == parallel["status"] == "complete"
    serial_order = [(entry["kind"], entry["key"]) for entry in serial["budget_deductions"]]
    parallel_order = [(entry["kind"], entry["key"]) for entry in parallel["budget_deductions"]]
    assert serial_order == parallel_order
    comparison = compare_states(load_state(tmp_path / "serial-state"), load_state(tmp_path / "parallel-state"))
    assert comparison["equal"], comparison


def test_snapshot_recovery_and_stale_hash(sample_workspace, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    report = run_until_complete(_options(sample_workspace, state_dir))
    assert report["status"] == "complete"
    store = SnapshotStore(state_dir)
    state = store.load()
    assert state is not None
    assert state.recovery_point["fusion_version"] == "f2"
    with pytest.raises(SegmentationError) as exc:
        store.load(expected_config_hash="not-the-config-hash")
    assert exc.value.code == E_SNAPSHOT_STALE

    changed_config = json.loads(Path(sample_workspace["config"]).read_text())
    changed_config["classification_weights"]["0"]["bias"] = 0.33
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed_config))
    strict = RunOptions(
        index_path=sample_workspace["index"],
        config_path=changed_path,
        fusion_versions_path=sample_workspace["fusion"],
        state_dir=state_dir,
        report_path=tmp_path / "strict-report.json",
        strict_recovery=True,
        max_batches=100,
    )
    strict_report = run_until_complete(strict)
    assert strict_report["status"] == "complete"
    assert "E_SNAPSHOT_STALE" in strict_report["codes"]
    assert strict_report["snapshot_stale"]["code"] == E_SNAPSHOT_STALE


def test_partial_generation_is_never_current(sample_workspace, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    run_until_complete(_options(sample_workspace, state_dir))
    store = SnapshotStore(state_dir)
    before = store.load().recovery_point
    partial = store.generations / ".building-interrupted"
    partial.mkdir()
    (partial / "manifest.json").write_text('{"complete": false}')
    after = store.load().recovery_point
    assert before == after
    assert store.current_path.read_text().strip().startswith("{")


def test_sigterm_finishes_inflight_work_and_exits_zero(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    data = tmp_path / "data"
    generate_sample_workspace(
        data,
        width=64,
        height=64,
        rows=6,
        cols=6,
        overlap=8,
        seed=99,
        patch_batches=1,
    )
    config = json.loads((root / "configs" / "segmentation.json").read_text())
    config["max_workers"] = 1
    config["recompute_budget"] = 1000
    config_path = tmp_path / "segmentation.json"
    config_path.write_text(json.dumps(config))
    state_dir = tmp_path / "state"
    report_path = tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "remote_seg",
        "run",
        "--index",
        str(data / "index.json"),
        "--config",
        str(config_path),
        "--fusion-versions",
        str(data / "fusion_versions.json"),
        "--state-dir",
        str(state_dir),
        "--report",
        str(report_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.25)
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "graceful_stop"
    assert report["snapshot"]["global_version"] >= 1
    recovered = SnapshotStore(state_dir).load()
    assert recovered is not None
    assert recovered.recovery_point["fusion_version"] == "f2"
