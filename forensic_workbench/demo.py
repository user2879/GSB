"""End-to-end demonstration and crash-recovery evidence collection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import AnalysisWindow, PipelineConfig, ScopedRect
from .errors import ForensicError
from .pipeline import execute
from .runcontrol import inspect_run, record_recovery
from .source import SourceSpec, ensure_source, generate_source
from .state import SnapshotStore
from .util import atomic_write_json, sha256_file


def _run_step(
    workspace: Path,
    config: PipelineConfig,
    *,
    step_name: str,
    run_root: Path,
    verify_full: bool = True,
) -> dict[str, Any]:
    run_dir = Path(tempfile.mkdtemp(prefix=f"{step_name}-", dir=str(run_root)))
    started = time.perf_counter()
    result = execute(
        workspace,
        config,
        verify_full=verify_full,
        run_dir=run_dir,
        run_id=f"demo-{step_name}-v{len(list(run_root.glob('*'))) + 1:03d}",
    )
    elapsed = time.perf_counter() - started
    return {
        "step": step_name,
        "run_dir": str(run_dir),
        "elapsed_seconds": round(elapsed, 4),
        "snapshot_version": result.commit.version if result.commit else None,
        "snapshot_hash": result.commit.snapshot_hash if result.commit else None,
        "conclusion_hash": result.conclusion_hash,
        "integrity_status": result.graph.nodes["conclusion"].payload["integrity_status"],
        "overall_suspicion": result.graph.nodes["conclusion"].payload["overall_suspicion"],
        "changed_parameters": result.audit["changed_parameters"],
        "recomputed_nodes": result.audit["recomputed_nodes"],
        "recomputed_count": len(result.audit["recomputed_nodes"]),
        "reused_count": len(result.audit["reused_nodes"]),
        "deactivated_count": len(result.audit["deactivated_nodes"]),
        "reuse_counts": result.audit["reuse_counts"],
        "full_verified": result.full_verified,
        "run_control": inspect_run(run_dir),
        "config": config.to_dict(),
    }


def _determinism_probe(workspace: Path) -> dict[str, Any]:
    probe_root = workspace / "reports" / "determinism-probe"
    first = probe_root / "first"
    second = probe_root / "second"
    spec = SourceSpec(width=160, height=90, fps=30, duration_seconds=1, seed=20261119, gop_size=10)
    source_a = generate_source(first, spec, overwrite=True)
    source_b = generate_source(second, spec, overwrite=True)
    video_a = sha256_file(source_a.video_path)
    video_b = sha256_file(source_b.video_path)
    truth_a = sha256_file(source_a.truth_path)
    truth_b = sha256_file(source_b.truth_path)
    manifest_a = sha256_file(first / "source_manifest.json")
    manifest_b = sha256_file(second / "source_manifest.json")
    return {
        "spec": spec.to_dict(),
        "frame_count": source_a.frame_count,
        "video_sha256": video_a,
        "truth_sha256": truth_a,
        "manifest_sha256": manifest_a,
        "pixel_content_sha256": source_a.content_sha256,
        "video_bytes_identical": video_a == video_b,
        "truth_bytes_identical": truth_a == truth_b,
        "manifest_bytes_identical": manifest_a == manifest_b,
        "all_bytes_identical": video_a == video_b and truth_a == truth_b and manifest_a == manifest_b,
    }


def _crash_child(
    workspace: Path,
    run_dir: Path,
    config: PipelineConfig,
    *,
    expected_base_version: int,
    expected_base_hash: str,
) -> None:
    execute(
        workspace,
        config,
        expected_base_version=expected_base_version,
        expected_base_hash=expected_base_hash,
        verify_full=False,
        run_dir=run_dir,
        run_id=f"crash-child-v{expected_base_version + 1:06d}",
        crash_after_stage="after_snapshot_file",
    )
    raise ForensicError(
        "E_SNAPSHOT_VERSION",
        "crash simulation unexpectedly completed without forced termination",
        {"run_dir": str(run_dir), "workspace": str(workspace)},
    )


def run_demo(
    workspace: Path,
    *,
    spec: SourceSpec | None = None,
    skip_crash_recovery: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    run_root = workspace / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    reports = workspace / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    spec = spec or SourceSpec()
    started = time.perf_counter()
    source = ensure_source(workspace / "source", spec, verify_hashes=False)
    generation_seconds = time.perf_counter() - started
    store = SnapshotStore(workspace)
    existing = store.load()
    if existing is not None:
        raise ForensicError(
            "E_SNAPSHOT_VERSION",
            "demo requires a fresh workspace with no committed snapshot",
            {
                "workspace": str(workspace),
                "current_version": existing.version,
                "current_snapshot_hash": existing.snapshot_hash,
                "hint": "choose a new --workspace directory for the demo",
            },
        )
    baseline_config = source.default_pipeline_config()
    determinism = _determinism_probe(workspace)
    steps: list[dict[str, Any]] = []
    baseline = _run_step(
        workspace,
        baseline_config,
        step_name="baseline",
        run_root=run_root,
        verify_full=True,
    )
    baseline_node_hashes = store.load().graph.node_hashes()  # type: ignore[union-attr]
    steps.append(baseline)

    target_gop = min(2, source.gop_count - 1)
    mask_start = target_gop * source.spec.gop_size
    mask_end = min(source.frame_count, mask_start + source.spec.gop_size)
    mask_variant = baseline_config.with_regions(
        mask=list(baseline_config.watermark_mask.regions)
        + [
            ScopedRect(
                x=min(5, source.spec.width - 8),
                y=min(5, source.spec.height - 8),
                width=min(8, source.spec.width),
                height=min(8, source.spec.height),
                frame_start=mask_start,
                frame_end=mask_end,
            )
        ]
    )
    mask_step = _run_step(
        workspace,
        mask_variant,
        step_name="mask-variant",
        run_root=run_root,
        verify_full=True,
    )
    steps.append(mask_step)

    sampling_gop = min(max(1, source.gop_count // 2), source.gop_count - 1)
    sampling_start = sampling_gop * source.spec.gop_size
    sampling_end = min(source.frame_count, sampling_start + source.spec.gop_size)
    sampling_regions = []
    if sampling_start > 0:
        sampling_regions.append(ScopedRect(0, 0, source.spec.width, source.spec.height, 0, sampling_start))
    sampling_regions.append(
        ScopedRect(
            x=min(2, source.spec.width - 8),
            y=min(2, source.spec.height - 8),
            width=max(4, source.spec.width - 8),
            height=max(4, source.spec.height - 8),
            frame_start=sampling_start,
            frame_end=sampling_end,
        )
    )
    if sampling_end < source.frame_count:
        sampling_regions.append(
            ScopedRect(
                0,
                0,
                source.spec.width,
                source.spec.height,
                sampling_end,
                source.frame_count,
            )
        )
    sampling_variant = mask_variant.with_regions(
        sampling=sampling_regions,
        window=AnalysisWindow(0, source.frame_count),
    )
    sampling_step = _run_step(
        workspace,
        sampling_variant,
        step_name="sampling-variant",
        run_root=run_root,
        verify_full=True,
    )
    steps.append(sampling_step)

    window_gop = min(max(1, source.gop_count // 3), source.gop_count - 1)
    window_end_gop = min(source.gop_count, window_gop + 2)
    window = AnalysisWindow(
        start_frame=min(source.frame_count - 1, window_gop * source.spec.gop_size + 3),
        end_frame=min(
            source.frame_count,
            max(window_gop * source.spec.gop_size + 4, window_end_gop * source.spec.gop_size - 2),
        ),
    )
    window_variant = sampling_variant.with_regions(window=window)
    window_step = _run_step(
        workspace,
        window_variant,
        step_name="window-variant",
        run_root=run_root,
        verify_full=True,
    )
    steps.append(window_step)

    committed = store.load()
    if committed is None:
        raise ForensicError("E_SNAPSHOT_VERSION", "demo lost its committed snapshot")
    restart_reload = SnapshotStore(workspace).load()
    if restart_reload is None:
        raise ForensicError("E_SNAPSHOT_VERSION", "restart reload returned no snapshot")
    restart_hashes_match = restart_reload.graph.node_hashes() == committed.graph.node_hashes()

    crash_recovery: dict[str, Any] | None = None
    if not skip_crash_recovery:
        before = committed
        crash_root = workspace / "recovery-runs"
        crash_root.mkdir(parents=True, exist_ok=True)
        crash_run_dir = Path(tempfile.mkdtemp(prefix="forced-termination-", dir=str(crash_root)))
        crash_config = window_variant.with_regions(
            mask=list(window_variant.watermark_mask.regions)
            + [
                ScopedRect(
                    x=min(1, source.spec.width - 6),
                    y=min(1, source.spec.height - 6),
                    width=min(6, source.spec.width),
                    height=min(6, source.spec.height),
                    frame_start=sampling_start,
                    frame_end=sampling_end,
                )
            ]
        )
        config_path = crash_run_dir / "candidate_config.json"
        atomic_write_json(config_path, crash_config.to_dict())
        command = [
            sys.executable,
            "-m",
            "forensic_workbench",
            "_crash-child",
            "--workspace",
            str(workspace),
            "--run-dir",
            str(crash_run_dir),
            "--config",
            str(config_path),
            "--expected-base-version",
            str(before.version),
            "--expected-base-hash",
            before.snapshot_hash,
        ]
        project_root = Path(__file__).resolve().parent.parent
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project_root) + os.pathsep + environment.get("PYTHONPATH", "")
        process = subprocess.run(
            command,
            cwd=str(project_root),
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if process.returncode != 17:
            raise ForensicError(
                "E_SNAPSHOT_VERSION",
                "forced-termination child did not exit at the injected crash point",
                {
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "run_dir": str(crash_run_dir),
                },
            )
        run_status = inspect_run(crash_run_dir)
        recovery = store.recover_orphans()
        recovered = SnapshotStore(workspace).load()
        if recovered is None:
            raise ForensicError("E_SNAPSHOT_VERSION", "recovery returned no committed snapshot")
        recovered_hash_matches = recovered.snapshot_hash == before.snapshot_hash
        recovered_node_hashes_match = recovered.graph.node_hashes() == before.graph.node_hashes()
        recovery_record = record_recovery(
            crash_run_dir,
            {
                "clean_exit": False,
                "reason": "forced termination before CURRENT pointer commit",
                "run_status": run_status,
                "state_recovery": recovery["report"],
                "state_recovery_report": recovery["report_path"],
                "restored_version": recovered.version,
                "restored_snapshot_hash": recovered.snapshot_hash,
                "node_hashes_match_committed": recovered_node_hashes_match,
            },
        )
        after_recovery_step = _run_step(
            workspace,
            window_variant,
            step_name="post-recovery-reuse",
            run_root=run_root,
            verify_full=False,
        )
        steps.append(after_recovery_step)
        crash_recovery = {
            "child_returncode": process.returncode,
            "child_stdout": process.stdout,
            "child_stderr": process.stderr,
            "run_dir": str(crash_run_dir),
            "run_status_before_recovery": run_status,
            "close_marker_written": (crash_run_dir / "close.json").exists(),
            "orphan_issue_codes": [
                issue["code"] for issue in recovery["report"]["issues"]
            ],
            "orphaned": recovery["report"]["orphaned"],
            "state_recovery_report": recovery["report_path"],
            "run_recovery_report": str(recovery_record),
            "current_snapshot_unchanged": recovered_hash_matches,
            "node_hashes_match_committed": recovered_node_hashes_match,
            "post_recovery_conclusion_matches": after_recovery_step["conclusion_hash"]
            == window_step["conclusion_hash"],
            "post_recovery_recomputed_count": after_recovery_step["recomputed_count"],
        }
    final_snapshot = SnapshotStore(workspace).load()
    if final_snapshot is None:
        raise ForensicError("E_SNAPSHOT_VERSION", "demo final snapshot is missing")
    report = {
        "format": "forensic-demo-report-v1",
        "workspace": str(workspace.resolve()),
        "source": {
            "spec": source.spec.to_dict(),
            "frame_count": source.frame_count,
            "gop_count": source.gop_count,
            "video_sha256": source.video_sha256,
            "truth_sha256": source.truth_sha256,
            "pixel_content_sha256": source.content_sha256,
            "manifest_sha256": source.manifest_digest,
            "video_bytes": source.video_path.stat().st_size,
            "generation_seconds": round(generation_seconds, 4),
            "shot_boundaries": sorted(source.shot_boundaries),
            "gop_boundaries": list(source.gop_boundaries),
        },
        "determinism_probe": determinism,
        "baseline_node_count": len(baseline_node_hashes),
        "restart_reload": {
            "version": restart_reload.version,
            "snapshot_hash": restart_reload.snapshot_hash,
            "node_hashes_match": restart_hashes_match,
            "conclusion_hash": restart_reload.graph.nodes["conclusion"].content_hash,
        },
        "steps": steps,
        "crash_recovery": crash_recovery,
        "final": {
            "version": final_snapshot.version,
            "snapshot_hash": final_snapshot.snapshot_hash,
            "node_count": len(final_snapshot.graph.nodes),
            "conclusion_hash": final_snapshot.graph.nodes["conclusion"].content_hash,
            "integrity_status": final_snapshot.graph.nodes["conclusion"].payload["integrity_status"],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    report_path = reports / "demo_report.json"
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def read_config_file(path: Path) -> PipelineConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForensicError(
            "E_FRAME_FORMAT",
            "config file cannot be read",
            {"path": str(path), "reason": str(exc)},
        ) from exc
    try:
        return PipelineConfig.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ForensicError(
            "E_FRAME_FORMAT",
            "config file has an invalid structure",
            {"path": str(path), "reason": str(exc)},
        ) from exc
