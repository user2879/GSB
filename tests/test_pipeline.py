"""业务要求 4、8：交替优化确定性、原子持久化与强制终止恢复。"""

from __future__ import annotations

from pathlib import Path

from photometric_altopt.faults import FaultConfig
from photometric_altopt.persistence import (
    cleanup_temp_files,
    directory_manifest,
    read_json,
)
from photometric_altopt.pipeline import (
    CONVERGED_MAX_ITER,
    RunPaths,
    _round_paths,
    resume_run,
    run_alternating,
)


def test_run_writes_committed_artifacts(observation_bundle, tmp_path):
    bundle = observation_bundle
    cfg = bundle["config"]
    run_root = tmp_path / "run"
    summary = run_alternating(
        cfg, bundle["observations"], bundle["views"], run_root
    )
    assert summary["rounds_committed"] >= 1
    paths = RunPaths(run_root)
    for round_id in range(summary["rounds_committed"]):
        rp = _round_paths(paths, round_id)
        for key in ("record", "conflict", "normal", "albedo", "images"):
            assert rp[key].exists()
            assert Path(str(rp[key]) + ".committed").exists()


def test_deterministic_content_hash(observation_bundle, tmp_path):
    bundle = observation_bundle
    cfg = bundle["config"]
    run_a = run_alternating(cfg, bundle["observations"], bundle["views"], tmp_path / "a")
    run_b = run_alternating(cfg, bundle["observations"], bundle["views"], tmp_path / "b")
    assert run_a["convergence_reason"] == run_b["convergence_reason"]
    manifest_a = directory_manifest(tmp_path / "a")
    manifest_b = directory_manifest(tmp_path / "b")
    assert manifest_a == manifest_b


def test_crash_and_resume_byte_identical(observation_bundle, tmp_path):
    bundle = observation_bundle
    cfg = bundle["config"]
    cfg.opt.max_iterations = 5

    baseline_root = tmp_path / "baseline"
    run_alternating(cfg, bundle["observations"], bundle["views"], baseline_root)

    crash_root = tmp_path / "crash"
    crashed = run_alternating(
        cfg, bundle["observations"], bundle["views"], crash_root, crash_after_round=1
    )
    assert crashed["crashed"] is True
    assert read_json(crash_root / "status.json")["state"] == "crashed"

    resumed = resume_run(cfg, bundle["observations"], bundle["views"], crash_root)
    assert resumed["rounds_committed"] == 5
    assert read_json(crash_root / "status.json")["state"] == "completed"

    assert directory_manifest(baseline_root) == directory_manifest(crash_root)


def test_uncommitted_temp_files_are_ignored(observation_bundle, tmp_path):
    bundle = observation_bundle
    cfg = bundle["config"]
    run_root = tmp_path / "run"
    run_alternating(cfg, bundle["observations"], bundle["views"], run_root)
    # 伪造一个未提交的临时文件与无主标记，清理后不得影响已提交状态
    snap = RunPaths(run_root).snapshot_dir
    ghost = snap / ".round_999_normals.npy.tmp"
    ghost.write_bytes(b"partial")
    orphan = snap / "round_999_albedo.npy.committed"
    orphan.write_text("{}", encoding="utf-8")
    removed = cleanup_temp_files(run_root)
    assert any("round_999" in name for name in removed)
    assert not ghost.exists()
    assert not orphan.exists()


def test_max_iteration_convergence_reason(observation_bundle, tmp_path):
    bundle = observation_bundle
    cfg = bundle["config"]
    cfg.opt.max_iterations = 3
    cfg.opt.loss_delta_tol = 0.0  # 禁止损失阈值提前收敛
    summary = run_alternating(cfg, bundle["observations"], bundle["views"], tmp_path / "run")
    assert summary["convergence_reason"] == CONVERGED_MAX_ITER
    assert summary["rounds_committed"] == 3


def test_fault_config_serialized_with_run(observation_bundle, tmp_path):
    bundle = observation_bundle
    cfg = bundle["config"]
    fault = FaultConfig(skipped_views=[4])
    run_root = tmp_path / "run"
    run_alternating(cfg, bundle["observations"], bundle["views"], run_root, fault=fault)
    manifest = read_json(run_root / "run_manifest.json")
    assert manifest["fault"]["skipped_views"] == [4]
