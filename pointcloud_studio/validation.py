"""End-to-end acceptance validation and reproducible run record."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import OpenGL

from .engine import StudioEngine
from .injections import run_fault_matrix
from .mesh import ReconstructionParams
from .persistence import AtomicStateStore
from .render import OffscreenViewport, result_to_ppm
from .runtime import ThreadedReconstructionService
from .sample import SampleConfig, edit_sequence, load_sample_bundle, write_sample_bundle
from .snapshots import SnapshotStore, scene_from_engine
from .util import atomic_write_json, sha256_file


def _compare_bundle_bytes(first: Path, second: Path) -> dict[str, Any]:
    names = ["points.bin", "normals.bin", "camera.json", "edits.json", "manifest.json"]
    equal = {name: (first / name).read_bytes() == (second / name).read_bytes() for name in names}
    return {"all_equal": all(equal.values()), "files": equal}


def _copy_current_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    current_store = AtomicStateStore(source)
    current = current_store._read_current()
    if current is None:
        raise RuntimeError("cannot copy a workspace without CURRENT")
    (destination / "states").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "CURRENT", destination / "CURRENT")
    shutil.copytree(
        source / "states" / str(current["state"]),
        destination / "states" / str(current["state"]),
    )
    if (source / "closed.json").exists():
        shutil.copy2(source / "closed.json", destination / "closed.json")


def run_validation(
    *,
    sample_dir: str | Path,
    workspace_dir: str | Path,
    report_path: str | Path,
    regenerate_sample: bool = True,
    operations_limit: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    sample = Path(sample_dir)
    workspace = Path(workspace_dir)
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    config = SampleConfig()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if regenerate_sample:
        manifest = write_sample_bundle(sample, config)
    else:
        manifest = load_sample_bundle(sample)[3]

    with tempfile.TemporaryDirectory(prefix="pcstudio-repeat-") as temporary:
        repeat = Path(temporary) / "sample"
        write_sample_bundle(repeat, config)
        determinism = _compare_bundle_bytes(sample, repeat)
    check("deterministic_sample_byte_identical", determinism["all_equal"], determinism)
    check("sample_point_count_120k", int(manifest["point_count"]) == 120_000, manifest["point_count"])

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    points, normals, confidence, manifest = load_sample_bundle(sample)
    engine = StudioEngine(points, normals, confidence, params=ReconstructionParams())
    store = AtomicStateStore(workspace)
    open_report = store.open()
    initial_commit = store.commit(engine, operation_seq=0)
    snapshots = SnapshotStore()
    service = ThreadedReconstructionService(engine, store, snapshots)
    initial_scene = service.start(commit_initial=False)
    viewport = OffscreenViewport(snapshots)
    initial_render = viewport.render(width=320, height=200, expected_version=initial_scene.version)
    check(
        "initial_snapshot_atomic_version_1",
        initial_scene.version == 1 and snapshots.get().content_sha256 == initial_scene.content_sha256,
        {
            "version": initial_scene.version,
            "scene_sha256": initial_scene.content_sha256,
            "point_count": initial_scene.point_count,
        },
    )
    check(
        "gl_context_status_is_structured",
        initial_render.ok or initial_render.code == "E_GL_CONTEXT",
        initial_render.to_dict(),
    )

    operations = edit_sequence(config)["operations"]
    if operations_limit is not None:
        operations = operations[:operations_limit]
    operation_summaries: list[dict[str, Any]] = []
    for operation in operations:
        result = service.submit(operation).result(timeout=300.0)
        record = result.operation
        visible = snapshots.get()
        check(
            f"snapshot_version_atomic_op_{record['seq']}",
            visible.version == result.published_snapshot_version == store._read_current()["version"],
            {
                "worker_base_version": result.base_snapshot_version,
                "published_version": result.published_snapshot_version,
                "visible_version": visible.version,
                "scene_sha256": visible.content_sha256,
            },
        )
        check(
            f"snapshot_immutable_op_{record['seq']}",
            not visible.positions.flags.writeable
            and not visible.normals.flags.writeable
            and not visible.mesh.vertices.flags.writeable,
            {"version": visible.version},
        )
        invalidation = record["invalidation"]
        comparison = record["consistency_check"]
        check(
            f"invalidation_closure_exact_op_{record['seq']}",
            invalidation["closure_equals_recomputed"]
            and invalidation["closure_patch_keys"] == invalidation["recomputed_patch_keys"],
            {
                "recomputed": invalidation["recomputed_patch_count"],
                "reused": invalidation["reused_patch_count"],
                "closure_equal": invalidation["closure_equals_recomputed"],
            },
        )
        check(
            f"incremental_full_consistency_op_{record['seq']}",
            comparison["ok"] and comparison["maximum_position_deviation"] <= comparison["tolerance"],
            {
                "maximum_position_deviation": comparison["maximum_position_deviation"],
                "criteria": comparison["criteria"],
                "first_divergence": comparison["first_divergence"],
            },
        )
        if record["normal_audit"] is not None:
            check(
                f"normal_outside_region_unchanged_op_{record['seq']}",
                record["normal_audit"]["outside_selection_normals_bitwise_unchanged"],
                record["normal_audit"],
            )
        operation_summaries.append(
            {
                "seq": record["seq"],
                "type": record["operation_type"],
                "selected_point_count": record["selected_point_count"],
                "point_count_after": record["point_count_after"],
                "production_rebuild_mode": comparison["production_rebuild_mode"],
                "recomputed_patch_count": invalidation["recomputed_patch_count"],
                "reused_patch_count": invalidation["reused_patch_count"],
                "whole_model_rebuild": invalidation["whole_model_rebuild"],
                "maximum_position_deviation": comparison["maximum_position_deviation"],
                "deviation_distribution": comparison["deviation_distribution"],
                "comparison_ok": comparison["ok"],
                "snapshot_version": result.published_snapshot_version,
                "scene_sha256": result.published_scene_sha256,
                "mesh_sha256": record["mesh_content_sha256"],
                "ledger_sha256": record["ledger_sha256"],
                "normal_audit": record["normal_audit"],
            }
        )

    latest_snapshot = snapshots.get()
    latest_render = viewport.render(
        width=320,
        height=200,
        expected_version=latest_snapshot.version,
    )
    check(
        "latest_gl_or_cpu_fallback",
        latest_render.ok or latest_render.code == "E_GL_CONTEXT",
        latest_render.to_dict(),
    )
    if latest_render.ok:
        result_to_ppm(latest_render, report_file.parent / "latest_snapshot.ppm")
    independent = store.independent_verify_current()
    check(
        "independent_ledger_recompute",
        independent["ok"] and independent["maximum_numeric_difference"] == 0.0,
        independent,
    )
    viewport.close()
    service.close(clean_shutdown=True)
    final_version = engine.version

    clean_store = AtomicStateStore(workspace)
    clean_open = clean_store.open()
    clean_load = clean_store.load_current(independent_ledger_check=True)
    check(
        "normal_close_marker_and_clean_reopen",
        clean_open.clean_previous_exit is True and clean_load.version == final_version,
        {
            "clean_previous_exit": clean_open.clean_previous_exit,
            "version": clean_load.version,
            "orphans": clean_open.recovered_orphans,
        },
    )
    clean_store.close(clean_load.engine)

    crash_workspace = workspace.parent / f"{workspace.name}_crash_recovery"
    _copy_current_workspace(workspace, crash_workspace)
    child_script = Path(__file__).resolve().parent.parent / "scripts" / "crash_child.py"
    child = subprocess.run(
        [sys.executable, str(child_script), str(crash_workspace)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    crash_store = AtomicStateStore(crash_workspace)
    crash_open = crash_store.open()
    crash_load = crash_store.load_current(independent_ledger_check=False)
    crash_verify = crash_store.independent_verify_current()
    check(
        "forced_termination_recovery",
        child.returncode == 23
        and crash_open.clean_previous_exit is False
        and crash_load.version == final_version
        and len(crash_open.recovered_orphans) >= 1
        and crash_verify["ok"],
        {
            "child_exit_code": child.returncode,
            "clean_previous_exit": crash_open.clean_previous_exit,
            "recovered_version": crash_load.version,
            "orphans": crash_open.recovered_orphans,
            "recovery_report_path": crash_open.recovery_report_path,
            "fake_exit_report_written": False,
            "independent_verify": crash_verify,
        },
    )
    crash_store.close(crash_load.engine)

    fault_report = run_fault_matrix(
        sample_dir=sample,
        workspace=workspace,
        output_dir=workspace / "fault_matrix",
        loaded_engine=AtomicStateStore(workspace).load_current().engine,
    )
    check("all_business_error_codes", fault_report["all_matched"], fault_report)

    report = {
        "schema": "pointcloud-studio.validation-report/1",
        "generated_at_unix": round(time.time(), 6),
        "duration_seconds": round(time.time() - started, 3),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "virtual_env": sys.prefix,
            "platform": platform.platform(),
            "os_name": os.name,
            "numpy": np.__version__,
            "pyopengl": OpenGL.__version__,
            "sample_dir": str(sample),
            "workspace_dir": str(workspace),
        },
        "sample": {
            "manifest_sha256": sha256_file(sample / "manifest.json"),
            "points_sha256": manifest["files"]["points.bin"]["sha256"],
            "normals_sha256": manifest["files"]["normals.bin"]["sha256"],
            "point_count": manifest["point_count"],
            "determinism": determinism,
        },
        "workspace_open": {
            "clean_previous_exit": open_report.clean_previous_exit,
            "initial_commit": initial_commit,
        },
        "render": {
            "initial": initial_render.to_dict(),
            "latest": latest_render.to_dict(),
            "cpu_mesh_ledger_path_retained": independent["ok"],
        },
        "operations": operation_summaries,
        "independent_ledger_verification": independent,
        "clean_reopen": {
            "clean_previous_exit": clean_open.clean_previous_exit,
            "version": clean_load.version,
        },
        "fault_matrix": fault_report,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "final_committed_version": final_version,
        "final_snapshot_sha256": latest_snapshot.content_sha256,
    }
    atomic_write_json(report_file, report)
    return report
