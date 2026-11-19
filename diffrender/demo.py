"""Reproducible end-to-end demonstration for the incremental renderer."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .errors import DiffRenderError
from .hashing import canonical_json_bytes, canonical_json_hash
from .persistence import StateStore
from .pipeline import IncrementalPipeline, ToleranceConfig
from .renderer import (
    compare_gradient_to_finite_difference,
    render_full,
    tile_layout,
)
from .scene import default_scene, target_scene_from


OUTPUT_SCHEMA = "diffrender-demo-v1"


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(value)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return canonical_json_hash(value)


def _safe_reset(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden:
        raise DiffRenderError("E_SCENE_INVALID", f"refusing to reset unsafe path {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _clamp_parameter(key: str, value: float) -> float:
    field = key.rsplit(".", 1)[-1]
    if field == "radius":
        return min(2.0, max(0.05, value))
    if field == "albedo":
        return min(1.0, max(0.0, value))
    if field == "roughness":
        return min(2.0, max(0.0, value))
    if field == "ambient":
        return min(1.0, max(0.0, value))
    if key.startswith("light."):
        return min(4.0, max(-4.0, value))
    if field == "focal":
        return min(160.0, max(8.0, value))
    if key.startswith("camera."):
        if field in ("ex", "ey", "ez"):
            return min(3.0, max(-3.0, value))
        return min(1.2, max(-1.2, value))
    return min(4.0, max(-4.0, value))


def _learning_rate(key: str) -> float:
    field = key.rsplit(".", 1)[-1]
    if field in ("albedo", "roughness", "ambient"):
        return 18.0
    if key.startswith("light."):
        return 12.0
    if key.startswith("sphere."):
        return 3.0
    if field in ("yaw", "pitch", "roll"):
        return 0.20
    if field == "focal":
        return 0.20
    if key.startswith("camera."):
        return 4.0
    return 1.0


def _entry_summary(entry: dict[str, Any], comparison: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "sequence": entry["sequence"],
        "key": entry["key"],
        "meaningful": entry["meaningful"],
        "reason": entry["reason"],
        "old_value_hex": entry["old_value_hex"],
        "new_value_hex": entry["new_value_hex"],
        "delta": entry["delta"],
        "tolerance": entry["tolerance"],
        "closure_size": entry["closure_size"],
        "closure": entry["closure"],
        "dirty_raster_tiles": entry["dirty_raster_tiles"],
        "dirty_shade_tiles": entry["dirty_shade_tiles"],
        "recomputed_count": len(entry["recomputed"]),
        "reused_count": entry["reused_count"],
        "reused_manifest_hash": entry["reused_manifest_hash"],
        "image_hash": entry.get("image_hash"),
        "loss": entry.get("loss"),
        "comparison": comparison,
    }


def _compare(pipeline: IncrementalPipeline) -> dict[str, Any]:
    report = pipeline.compare_with_full(tolerance=1.0e-11)
    return {
        "image_hashes_equal": report["incremental_image_hash"] == report["full_image_hash"],
        "gradient_hashes_equal": report["incremental_gradient_hash"] == report["full_gradient_hash"],
        "image_hash": report["incremental_image_hash"],
        "gradient_hash": report["incremental_gradient_hash"],
        "gradient_semantic_hash": report["gradient"]["semantic_hash"],
        "gradient_max_abs_error": report["gradient"]["max_abs_error"],
        "first_divergent_tile": None,
        "first_divergent_param": None,
        "loss": report["loss"],
    }


def _run_optimization(
    pipeline: IncrementalPipeline,
    store: StateStore,
    schedule: Sequence[str],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for step_index, key in enumerate(schedule, start=1):
        gradient = pipeline.gradient_result()
        before = _compare(pipeline)
        loss_before = gradient.loss
        scalar = gradient.scalar_gradients[key]
        proposed = pipeline.scene.value(key) - _learning_rate(key) * scalar
        new_value = _clamp_parameter(key, proposed)
        entry = pipeline.set_parameter(key, new_value)
        after = _compare(pipeline)
        pipeline.commit(store, closed=False)
        steps.append(
            {
                "step": step_index,
                "key": key,
                "loss_before": loss_before,
                "loss_after": pipeline.loss,
                "scalar_gradient": scalar,
                "proposed_value": proposed,
                "applied_value_hex": pipeline.scene.value(key).hex(),
                "closure_size": entry["closure_size"],
                "meaningful": entry["meaningful"],
                "image_hash": after["image_hash"],
                "gradient_hash": after["gradient_hash"],
                "before_comparison": before,
                "after_comparison": after,
            }
        )
    return steps


def _crash_child(workspace: Path) -> None:
    store = StateStore(workspace)
    loaded = store.read_snapshot()
    if loaded is None:
        raise DiffRenderError("E_SNAPSHOT_VERSION", "crash child found no committed snapshot")
    state, _pointer = loaded
    pipeline = IncrementalPipeline.from_state_dict(state)
    radius_key = "sphere.0.radius"
    pipeline.set_parameter(radius_key, pipeline.scene.value(radius_key) + 0.013)
    # This is intentionally written to staging only.  The child exits without a
    # Python unwind, no CLOSED marker, and no exit report.
    store.write_staging(
        "uncommitted-edit.json",
        {
            "schema": "diffrender-uncommitted-v1",
            "key": radius_key,
            "new_value_hex": pipeline.scene.value(radius_key).hex(),
            "image_hash": pipeline.render_result().image_hash,
            "gradient_hash": pipeline.gradient_result().gradient_hash,
        },
    )
    os._exit(97)


def run_demo(workspace: Path, output_dir: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    _safe_reset(workspace)
    _safe_reset(output_dir)
    store = StateStore(workspace)
    base_scene = default_scene()
    ground_truth_scene = target_scene_from(base_scene)
    target_render = render_full(ground_truth_scene)
    target = target_render.image

    # Determinism gate before any cache is involved.
    first_render = render_full(base_scene)
    second_render = render_full(base_scene)
    deterministic = (
        first_render.image_hash == second_render.image_hash
        and first_render.raster_hash == second_render.raster_hash
        and first_render.shade_hash == second_render.shade_hash
        and first_render.render_hash == second_render.render_hash
    )
    if not deterministic:
        raise DiffRenderError("E_TILE_HASH", "two identical full renders produced different hashes")

    gradient_fd = compare_gradient_to_finite_difference(
        base_scene, target, tolerance=2.0e-4, step=1.0e-8
    )
    max_fd_error = max(value["abs_error"] for value in gradient_fd.values())

    pipeline = IncrementalPipeline(base_scene, target, ToleranceConfig())
    pipeline.commit(store, closed=False)
    initial = _compare(pipeline)
    edits: list[dict[str, Any]] = []

    edit_sequence = (
        ("sphere.0.radius", base_scene.value("sphere.0.radius") + 5.0e-10),
        ("sphere.0.radius", base_scene.value("sphere.0.radius") + 0.045),
        ("light.dy", base_scene.value("light.dy") - 0.060),
        ("sphere.1.albedo", base_scene.value("sphere.1.albedo") + 0.040),
        ("camera.yaw", base_scene.value("camera.yaw") + 0.012),
        ("light.dy", base_scene.value("light.dy") - 0.060 + 5.0e-10),
    )
    for key, value in edit_sequence:
        entry = pipeline.set_parameter(key, value)
        comparison = _compare(pipeline)
        pipeline.commit(store, closed=False)
        edits.append(_entry_summary(entry, comparison))

    optimization_schedule = (
        "sphere.0.radius",
        "sphere.1.cx",
        "light.dy",
        "sphere.1.albedo",
        "sphere.2.radius",
        "light.dx",
        "camera.yaw",
        "sphere.0.cz",
        "sphere.2.albedo",
    )
    optimization = _run_optimization(pipeline, store, optimization_schedule)
    final_optimization_comparison = _compare(pipeline)
    pipeline.commit(store, closed=False)

    stable_manifest = pipeline.node_hash_manifest()
    stable_stamp_manifest = pipeline.node_stamp_manifest()
    stable_version = pipeline.snapshot_version
    stable_state, stable_pointer = store.read_snapshot(expected_snapshot_version=stable_version)
    assert stable_state is not None and stable_pointer is not None
    stable_graph_hash = canonical_json_hash(stable_state["graph"])

    child_command = [
        sys.executable,
        "-m",
        "diffrender.demo",
        "--crash-child",
        "--workspace",
        str(workspace),
    ]
    child = subprocess.run(
        child_command,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if child.returncode != 97:
        raise DiffRenderError(
            "E_SNAPSHOT_VERSION",
            "force-termination child did not exit with the expected status",
            {"returncode": child.returncode, "stderr": child.stderr.decode("utf-8", "replace")},
        )

    recovered_state, recovery = store.recover(expected_snapshot_version=stable_version)
    if recovered_state is None or recovery is None:
        raise DiffRenderError("E_SNAPSHOT_VERSION", "recovery did not find the committed snapshot")
    recovered_pipeline = IncrementalPipeline.from_state_dict(recovered_state)
    recovered_manifest = recovered_pipeline.node_hash_manifest()
    recovered_stamps = recovered_pipeline.node_stamp_manifest()
    if recovered_manifest != stable_manifest:
        first = next(
            key for key in sorted(set(stable_manifest) | set(recovered_manifest))
            if stable_manifest.get(key) != recovered_manifest.get(key)
        )
        raise DiffRenderError(
            "E_DEPENDENCY_STALE",
            "recovered node hash differs from committed state",
            {"node": first, "committed": stable_manifest.get(first), "recovered": recovered_manifest.get(first)},
        )
    if recovered_stamps != stable_stamp_manifest:
        raise DiffRenderError("E_DEPENDENCY_STALE", "recovered version stamps changed")
    if recovered_state.get("exit_report") is not None:
        raise DiffRenderError("E_SNAPSHOT_VERSION", "forced termination state contains a fabricated exit report")
    if not recovery.forced_termination_detected or recovery.clean_shutdown:
        raise DiffRenderError("E_SNAPSHOT_VERSION", "forced termination was not detected")
    if not recovery.orphans:
        raise DiffRenderError("E_SNAPSHOT_VERSION", "no orphaned staging cache was recovered")

    continuation_key = "sphere.2.albedo"
    continuation = recovered_pipeline.set_parameter(
        continuation_key, recovered_pipeline.scene.value(continuation_key) + 0.015
    )
    continuation_comparison = _compare(recovered_pipeline)
    recovered_pipeline.commit(store, closed=False)

    closed_version = recovered_pipeline.commit(store, closed=True)
    closed_state, closed_pointer = store.read_snapshot(expected_snapshot_version=closed_version)
    assert closed_state is not None and closed_pointer is not None
    if not closed_state.get("closed") or not closed_state.get("clean_shutdown"):
        raise DiffRenderError("E_SNAPSHOT_VERSION", "controlled close did not write a complete manifest")
    if not closed_state.get("exit_report") or "node_hashes" not in closed_state["exit_report"]:
        raise DiffRenderError("E_SNAPSHOT_VERSION", "controlled close did not write an exit report")
    if not (store.committed_dir / "CLOSED.json").exists():
        raise DiffRenderError("E_SNAPSHOT_VERSION", "controlled close did not write the closing marker")
    closed_recovery_state, closed_recovery = store.recover()
    if closed_recovery is None or not closed_recovery.clean_shutdown:
        raise DiffRenderError("E_SNAPSHOT_VERSION", "clean snapshot was not recognized after restart")

    final_comparison = _compare(recovered_pipeline)
    final_manifest = recovered_pipeline.node_hash_manifest()
    final_stamps = recovered_pipeline.node_stamp_manifest()
    hash_manifest = {
        "schema": OUTPUT_SCHEMA,
        "deterministic_full_render": deterministic,
        "base_parameter_hash": base_scene.parameter_hash(),
        "target_image_hash": target_render.image_hash,
        "base_raster_hash": first_render.raster_hash,
        "base_shade_hash": first_render.shade_hash,
        "base_render_hash": first_render.render_hash,
        "initial_incremental_vs_full": initial,
        "finite_difference_max_abs_error": max_fd_error,
        "finite_difference_tolerance": 2.0e-4,
        "final_image_hash": final_comparison["image_hash"],
        "final_gradient_hash": final_comparison["gradient_hash"],
        "final_loss": recovered_pipeline.loss,
        "final_parameter_hash": recovered_pipeline.scene.parameter_hash(),
        "final_parameter_versions": recovered_pipeline.param_versions,
        "final_node_hashes": final_manifest,
        "final_node_stamps": final_stamps,
        "final_node_manifest_hash": canonical_json_hash(final_manifest),
        "stable_snapshot_version": stable_version,
        "stable_snapshot_graph_hash": stable_graph_hash,
        "recovered_snapshot_version": recovery.recovered_snapshot_version,
        "recovered_generation": recovery.recovered_generation,
        "recovered_node_manifest_hash": canonical_json_hash(recovered_manifest),
        "controlled_close_snapshot_version": closed_version,
        "controlled_close_exit_report_hash": closed_state["exit_report_hash"],
        "ledger_entries": len(recovered_pipeline.ledger),
    }

    ledger_document = {
        "schema": OUTPUT_SCHEMA,
        "tolerances": recovered_pipeline.tolerances.to_dict(),
        "entries": list(recovered_pipeline.ledger),
        "final_entry_count": len(recovered_pipeline.ledger),
    }
    recovery_document = {
        "schema": OUTPUT_SCHEMA,
        "recovery": recovery.to_dict(),
        "clean_close_recovery": closed_recovery.to_dict() if closed_recovery else None,
        "stable_node_manifest_hash": canonical_json_hash(stable_manifest),
        "recovered_node_manifest_hash": canonical_json_hash(recovered_manifest),
        "node_hashes_match": recovered_manifest == stable_manifest,
        "version_stamps_match": recovered_stamps == stable_stamp_manifest,
        "fabricated_exit_report": bool(recovered_state.get("exit_report")),
        "continuation": _entry_summary(continuation, continuation_comparison),
        "child_returncode": child.returncode,
    }
    summary = {
        "schema": OUTPUT_SCHEMA,
        "deterministic_full_render": deterministic,
        "tile_grid": {
            "width": base_scene.width,
            "height": base_scene.height,
            "tile_size": base_scene.tile_size,
            "tiles": len(tile_layout(base_scene)),
        },
        "edits": edits,
        "optimization": optimization,
        "final_optimization_comparison": final_optimization_comparison,
        "final_comparison": final_comparison,
        "max_finite_difference_error": max_fd_error,
        "forced_termination_detected": recovery.forced_termination_detected,
        "orphans_recovered": len(recovery.orphans),
        "clean_close_written": bool(closed_state.get("closed")),
        "two_path_hashes_equal": all(
            item.get("image_hashes_equal") and item.get("gradient_hashes_equal")
            for item in [initial]
            + [entry["comparison"] for entry in edits]
            + [step["before_comparison"] for step in optimization]
            + [step["after_comparison"] for step in optimization]
            + [continuation_comparison, final_comparison]
        ),
    }

    output_hashes = {
        "hash_manifest": _write_json(output_dir / "hash_manifest.json", hash_manifest),
        "invalidation_ledger": _write_json(output_dir / "invalidation_ledger.json", ledger_document),
        "recovery_report": _write_json(output_dir / "recovery_report.json", recovery_document),
        "summary": _write_json(output_dir / "summary.json", summary),
    }
    index = {
        "schema": OUTPUT_SCHEMA,
        "output_hashes": output_hashes,
        "two_path_hashes_equal": summary["two_path_hashes_equal"],
        "final_image_hash": final_comparison["image_hash"],
        "final_gradient_hash": final_comparison["gradient_hash"],
        "ledger_entries": len(recovered_pipeline.ledger),
    }
    _write_json(output_dir / "index.json", index)
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("artifacts/demo_workspace"),
        help="persistent workspace (will be reset by the demo)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/demo"),
        help="deterministic JSON output directory",
    )
    parser.add_argument("--crash-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.crash_child:
        _crash_child(args.workspace)
        return 0
    try:
        index = run_demo(args.workspace, args.out)
    except DiffRenderError as exc:
        error = exc.to_dict()
        _write_json(args.out / "ERROR.json", error)
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(index).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
