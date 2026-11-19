"""Command-line entry point and end-to-end acceptance demo."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .canonical import atomic_write_json, canonical_json_bytes, with_content_hash
from .engine import CadApplication
from .errors import DomainError, ErrorCode
from .geometry import PatchGeometry
from .model import NURBSModel, generate_model


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _error_case(code: ErrorCode, callable_: Any) -> dict[str, Any]:
    try:
        callable_()
    except DomainError as exc:
        return {
            "expected": code.value,
            "actual": exc.code.value,
            "matched": exc.code == code,
            "message": exc.message,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "expected": code.value,
            "actual": type(exc).__name__,
            "matched": False,
            "message": str(exc),
        }
    return {"expected": code.value, "actual": None, "matched": False, "message": "no error raised"}


def _buffer_deviation(first: Any, second: Any) -> dict[str, Any]:
    delta = np.abs(first.rgb.astype(np.int16) - second.rgb.astype(np.int16))
    with np.errstate(invalid="ignore"):
        depth_delta = np.abs(first.depth - second.depth)
    finite_depth = depth_delta[np.isfinite(depth_delta)]
    return {
        "hash_equal": first.content_hash == second.content_hash,
        "pixels_changed": int(np.count_nonzero(np.any(delta != 0, axis=2))),
        "max_channel_deviation": int(np.max(delta)) if delta.size else 0,
        "max_depth_deviation": float(np.max(finite_depth)) if finite_depth.size else 0.0,
    }


def _state_deviation(first: Any, second: Any) -> dict[str, Any]:
    first_buffers = first.render_all()
    second_buffers = second.render_all()
    displays = {
        mode: _buffer_deviation(first_buffers[mode], second_buffers[mode])
        for mode in sorted(first_buffers)
    }
    point_deviation = 0.0
    normal_deviation = 0.0
    for key in sorted(first.geometries):
        first_geometry = first.geometries[key]
        second_geometry = second.geometries[key]
        point_deviation = max(
            point_deviation,
            float(np.max(np.abs(first_geometry.points - second_geometry.points))),
        )
        normal_deviation = max(
            normal_deviation,
            float(np.max(np.abs(first_geometry.normals - second_geometry.normals))),
        )
    return {
        "state_hash_equal": first.state_hash == second.state_hash,
        "geometry_hash_equal": first.geometry_hash == second.geometry_hash,
        "normals_hash_equal": first.normals_hash == second.normals_hash,
        "iso_hash_equal": first.hashes.iso_hash == second.hashes.iso_hash,
        "tessellation_hash_equal": first.hashes.tessellation_hash == second.hashes.tessellation_hash,
        "display_hashes_equal": first.display_hashes == second.display_hashes,
        "max_point_deviation": point_deviation,
        "max_normal_deviation": normal_deviation,
        "display": displays,
        "all_equal": (
            first.state_hash == second.state_hash
            and first.geometry_hash == second.geometry_hash
            and first.normals_hash == second.normals_hash
            and first.display_hashes == second.display_hashes
            and point_deviation == 0.0
            and normal_deviation == 0.0
            and all(item["hash_equal"] for item in displays.values())
        ),
    }


def _choose_edit_value(app: CadApplication, patch_key: str, i: int, j: int, dz: float) -> list[float]:
    point = app.current.model.patch(patch_key).control_points[i, j]
    return [float(point[0]), float(point[1]), float(point[2] + dz)]


def run_demo(
    *,
    demo_root: Path,
    seed: int,
    grid_resolution: int,
    max_undo_depth: int,
    report_path: Path | None = None,
) -> dict[str, Any]:
    demo_root = Path(demo_root)
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True, exist_ok=True)
    state_a_root = demo_root / "state_a"
    state_b_root = demo_root / "state_b"
    screenshots = demo_root / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)

    first_model = generate_model(seed)
    second_model = generate_model(seed)
    first_bytes = first_model.to_bytes()
    second_bytes = second_model.to_bytes()
    generation = {
        "seed": seed,
        "model_hash": first_model.content_hash(),
        "byte_identical": first_bytes == second_bytes,
        "bytes": len(first_bytes),
        "patch_count": len(first_model.patches),
        "knots_valid": all(
            len(patch.knots_u) == patch.control_shape[0] + patch.degree_u + 1
            and len(patch.knots_v) == patch.control_shape[1] + patch.degree_v + 1
            and bool(np.all(patch.knots_u[1:] >= patch.knots_u[:-1]))
            and bool(np.all(patch.knots_v[1:] >= patch.knots_v[:-1]))
            for patch in first_model.patches
        ),
    }

    app_a = CadApplication.create(
        state_a_root,
        seed=seed,
        max_undo_depth=max_undo_depth,
        grid_resolution=grid_resolution,
    )
    app_b = CadApplication.create(
        state_b_root,
        seed=seed,
        max_undo_depth=max_undo_depth,
        grid_resolution=grid_resolution,
    )
    if app_a.current.state_hash != app_b.current.state_hash:
        raise DomainError(ErrorCode.GEOM_STALE, "two identically seeded applications disagree")

    base_summary = app_a.state_summary(0)
    before_edit_geometry = app_a.current.geometries["p00"]
    edit_one_value = _choose_edit_value(app_a, "p00", 4, 4, 0.16)
    state_one = app_a.edit_control_point("p00", 4, 4, edit_one_value)
    state_one_summary = app_a.state_summary()
    after_edit_geometry = state_one.geometries["p00"]
    closure = state_one.last_closure
    assert closure is not None
    affected = np.zeros(after_edit_geometry.vertex_count, dtype=bool)
    affected[np.asarray(closure.vertices, dtype=np.int64)] = True
    unaffected = ~affected
    closure_report = {
        "patch_keys": list(closure.patch_keys),
        "vertex_count": len(closure.vertices),
        "iso_sample_count": len(closure.iso_samples),
        "triangle_count": len(closure.triangles),
        "total_vertices": after_edit_geometry.vertex_count,
        "local_only": len(closure.vertices) < after_edit_geometry.vertex_count,
        "actual_matches_closure": (
            tuple(after_edit_geometry.recompute_trace["actual_recomputed"]["vertices"]) == closure.vertices
            and tuple(after_edit_geometry.recompute_trace["actual_recomputed"]["triangles"]) == closure.triangles
            and tuple(after_edit_geometry.recompute_trace["actual_recomputed"]["iso_samples"])
            == closure.iso_samples
        ),
        "outside_vertices_bitwise_equal": all(
            old.reshape(after_edit_geometry.vertex_count, 3)[unaffected].tobytes()
            == new.reshape(after_edit_geometry.vertex_count, 3)[unaffected].tobytes()
            for old, new in (
                (before_edit_geometry.points, after_edit_geometry.points),
                (before_edit_geometry.du, after_edit_geometry.du),
                (before_edit_geometry.dv, after_edit_geometry.dv),
                (before_edit_geometry.normals, after_edit_geometry.normals),
            )
        ),
        "other_patches_unchanged": all(
            app_a.history.states[0].geometries[key].geometry_hash
            == state_one.geometries[key].geometry_hash
            == state_one.geometries[key].geometry_hash
            for key in ("p01", "p02")
        ),
    }

    edit_two_before = app_a.current.model.patch("p01").control_points[3, 3].copy()
    state_two = app_a.edit_control_point(
        "p01",
        3,
        3,
        [float(edit_two_before[0]), float(edit_two_before[1]), float(edit_two_before[2] + 0.13)],
    )
    state_two_summary = app_a.state_summary()

    undo_one = app_a.undo()
    undo_one_summary = app_a.state_summary()
    undo_restore = _state_deviation(state_one, undo_one)
    undo_two = app_a.undo()
    undo_to_base = _state_deviation(app_a.history.states[0], undo_two)
    redo_one = app_a.redo()
    redo_one_report = _state_deviation(state_one, redo_one)
    redo_two = app_a.redo()
    redo_two_report = _state_deviation(state_two, redo_two)

    app_a.undo()
    branch_value = _choose_edit_value(app_a, "p02", 2, 2, -0.19)
    branch_state = app_a.edit_control_point("p02", 2, 2, branch_value)
    branch_report = {
        "new_version": branch_state.version,
        "parent_version": branch_state.parent_version,
        "redo_depth_after_new_edit": len(app_a.history.redo_stack),
        "redo_branch_truncated": len(app_a.history.redo_stack) == 0,
        "parent_not_reused_from_redo_branch": branch_state.parent_version != state_two.version,
        "discarded_redo_version": state_two.version,
    }
    branch_summary = app_a.state_summary()

    reconstruction = app_a.reconstruct_patch_at_version(state_one.version, "p00").to_dict()

    # Build the same branch in the reference workspace, without interruption.
    for key, i, j, value in (
        ("p00", 4, 4, edit_one_value),
        (
            "p01",
            3,
            3,
            [float(edit_two_before[0]), float(edit_two_before[1]), float(edit_two_before[2] + 0.13)],
        ),
    ):
        app_b.edit_control_point(key, i, j, value)
    app_b.undo()
    app_b.edit_control_point("p02", 2, 2, branch_value)
    if app_a.current.state_hash != app_b.current.state_hash:
        raise DomainError(ErrorCode.GEOM_STALE, "reference branch did not reproduce the demo branch")

    pre_crash_world = app_a.history.states[branch_state.version]
    pre_crash_summary = app_a.state_summary()
    crash_value = _choose_edit_value(app_a, "p02", 2, 1, 0.27)
    marker_path = demo_root / "crash_marker.json"
    worker_env = dict(os.environ)
    source_path = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = worker_env.get("PYTHONPATH")
    worker_env["PYTHONPATH"] = source_path if not existing_pythonpath else source_path + os.pathsep + existing_pythonpath
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "run.py"),
        "_crash-worker",
        "--root",
        str(state_a_root),
        "--patch",
        "p02",
        "--u",
        "2",
        "--v",
        "1",
        "--value",
        str(crash_value[0]),
        str(crash_value[1]),
        str(crash_value[2]),
        "--marker",
        str(marker_path),
    ]
    worker = subprocess.run(command, env=worker_env, capture_output=True, text=True, check=False)
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.exists() else {}
    crash_report = {
        "worker_exit_code": worker.returncode,
        "forced_exit_simulated": worker.returncode == 23,
        "staging_was_written": bool(marker.get("staging_path")),
        "marker": marker,
        "worker_stderr": worker.stderr.strip(),
    }
    app_a = CadApplication.open(state_a_root, max_undo_depth=max_undo_depth)
    recovered_state = app_a.current
    recovery_report = app_a.recovery.to_dict() if app_a.recovery is not None else {}
    recovery_deviation = _state_deviation(pre_crash_world, recovered_state)
    recovery_report.update(
        {
            "expected_version": pre_crash_summary["version"],
            "expected_state_hash": pre_crash_summary["state_hash"],
            "actual_version": recovered_state.version,
            "actual_state_hash": recovered_state.state_hash,
            "orphan_count": len(recovery_report.get("orphan_paths", [])),
            "deviation": recovery_deviation,
        }
    )

    continuation_a = _choose_edit_value(app_a, "p01", 2, 4, -0.21)
    continuation_b = _choose_edit_value(app_b, "p01", 2, 4, -0.21)
    app_a.edit_control_point("p01", 2, 4, continuation_a)
    app_b.edit_control_point("p01", 2, 4, continuation_b)
    continuation_report = _state_deviation(app_a.current, app_b.current)
    continuation_report.update(
        {
            "recovered_version": app_a.current.version,
            "uninterrupted_version": app_b.current.version,
            "versions_equal": app_a.current.version == app_b.current.version,
            "parents_equal": app_a.current.parent_version == app_b.current.parent_version,
        }
    )

    for mode in ("shaded_surface", "control_net", "isoparametric"):
        app_a.export_ppm(0, mode, screenshots / f"v0_{mode}.ppm")
        app_a.export_ppm(app_a.current.version, mode, screenshots / f"v{app_a.current.version}_{mode}.ppm")

    closed_a = app_a.close()
    controlled_reopen = CadApplication.open(state_a_root, max_undo_depth=max_undo_depth)
    controlled_report = {
        "close": closed_a,
        "reopen_status": controlled_reopen.recovery.status if controlled_reopen.recovery else None,
        "no_fake_forced_termination": bool(controlled_reopen.recovery and controlled_reopen.recovery.status == "CONTROLLED_EXIT"),
        "state_equal": controlled_reopen.current.state_hash == app_a.current.state_hash,
    }
    controlled_reopen.close()
    app_b.close()

    # Exception-chain coverage on isolated copies so committed demo state is never modified.
    errors_root = demo_root / "error_cases"
    errors_root.mkdir(parents=True, exist_ok=True)
    malformed_model = copy.deepcopy(first_model.to_dict())
    malformed_model["patches"][0]["degree_u"] = 0
    bad_knots_model = copy.deepcopy(first_model.to_dict())
    bad_knots_model["patches"][0]["knots_u"][2] = 0.75
    bad_knots_model["patches"][0]["knots_u"][3] = 0.0

    undo_empty_root = errors_root / "undo_empty"
    undo_empty_app = CadApplication.create(undo_empty_root, seed=seed, max_undo_depth=4, grid_resolution=8)
    undo_empty_case = _error_case(ErrorCode.UNDO_EMPTY, undo_empty_app.undo)
    undo_empty_app.close()

    cache_copy = errors_root / "cache_hash"
    shutil.copytree(state_a_root, cache_copy)
    current_manifest = json.loads((cache_copy / "MANIFEST.json").read_text(encoding="utf-8"))
    cache_version = int(current_manifest["current_version"])
    cache_file = cache_copy / "commits" / f"v{cache_version:06d}" / "patches" / "p00.json"
    cache_file.write_bytes(cache_file.read_bytes() + b" ")
    cache_case = _error_case(ErrorCode.CACHE_HASH, lambda: CadApplication.open(cache_copy, max_undo_depth=4))

    stale_copy = errors_root / "snapshot_stale"
    shutil.copytree(state_a_root, stale_copy)
    stale_manifest_path = stale_copy / "MANIFEST.json"
    stale_manifest = json.loads(stale_manifest_path.read_text(encoding="utf-8"))
    stale_manifest["current_state_hash"] = "0" * 64
    stale_manifest_path.write_bytes(canonical_json_bytes(stale_manifest))
    stale_case = _error_case(ErrorCode.SNAPSHOT_STALE, lambda: CadApplication.open(stale_copy, max_undo_depth=4))

    missing_copy = errors_root / "manifest_missing"
    shutil.copytree(state_a_root, missing_copy)
    (missing_copy / "MANIFEST.json").unlink()
    missing_case = _error_case(ErrorCode.SNAPSHOT_STALE, lambda: CadApplication.open(missing_copy, max_undo_depth=4))

    tampered_closure = closure.__class__(
        patch_keys=closure.patch_keys,
        vertices=closure.vertices[1:],
        iso_samples=closure.iso_samples[1:],
        triangles=closure.triangles,
    )
    closure_case = _error_case(
        ErrorCode.CLOSURE_MISMATCH,
        lambda: after_edit_geometry.verify_closure(before_edit_geometry, tampered_closure),
    )
    geometry_stale_case = _error_case(
        ErrorCode.GEOM_STALE,
        lambda: app_a.current.model.patch("p00").evaluate(2.0, 0.5),
    )
    schema_case = _error_case(ErrorCode.SCHEMA_VERSION, lambda: NURBSModel.from_dict(malformed_model))
    knot_case = _error_case(ErrorCode.KNOT_MONOTONIC, lambda: NURBSModel.from_dict(bad_knots_model))

    error_cases = {
        "degree_control_mismatch": schema_case,
        "nonmonotonic_knots": knot_case,
        "undo_empty": undo_empty_case,
        "cache_hash_mismatch": cache_case,
        "snapshot_stale": stale_case,
        "manifest_missing_after_termination": missing_case,
        "closure_mismatch": closure_case,
        "geometry_stale": geometry_stale_case,
    }

    checks = {
        "deterministic_generation": generation["byte_identical"] and generation["knots_valid"],
        "closure_exact": closure_report["actual_matches_closure"] and closure_report["outside_vertices_bitwise_equal"] and closure_report["other_patches_unchanged"],
        "historical_reconstruction": bool(reconstruction["match"]) and not reconstruction["full_model_rebuild"],
        "undo_restores_hashes": undo_restore["all_equal"],
        "redo_restores_hashes": redo_one_report["all_equal"] and redo_two_report["all_equal"],
        "branch_truncates_redo": branch_report["redo_branch_truncated"] and branch_report["parent_not_reused_from_redo_branch"],
        "forced_termination_recovery": crash_report["forced_exit_simulated"] and recovery_deviation["all_equal"] and recovery_report["orphan_count"] > 0,
        "continuation_matches_uninterrupted": continuation_report["all_equal"] and continuation_report["versions_equal"] and continuation_report["parents_equal"],
        "controlled_exit_distinguished": controlled_report["no_fake_forced_termination"] and controlled_report["state_equal"],
        "all_error_codes_observed": all(item["matched"] for item in error_cases.values()),
    }
    report = {
        "schema_version": "cad-surface.demo-report.v1",
        "command": "python run.py demo",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "display_path": "deterministic software rasterizer",
            "report_root": str(demo_root),
        },
        "sample_generation": generation,
        "initial_state": base_summary,
        "edit_one_state": state_one_summary,
        "edit_two_state": state_two_summary,
        "closure": closure_report,
        "undo_redo": {
            "undo_one_to_v0_exact": undo_restore["all_equal"],
            "undo_two_to_v0_exact": undo_to_base["all_equal"],
            "redo_one_exact": redo_one_report["all_equal"],
            "redo_two_exact": redo_two_report["all_equal"],
            "undo_one_deviation": undo_restore,
            "redo_one_deviation": redo_one_report,
            "undo_empty_after_reaching_base": undo_to_base["all_equal"],
        },
        "branch": branch_report,
        "branch_state": branch_summary,
        "historical_reconstruction": reconstruction,
        "forced_termination": crash_report,
        "recovery": recovery_report,
        "recovery_deviation": recovery_deviation,
        "continuation_vs_uninterrupted": continuation_report,
        "controlled_exit": controlled_report,
        "error_cases": error_cases,
        "screenshots": sorted(str(path.relative_to(demo_root)) for path in screenshots.glob("*.ppm")),
        "checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }
    destination = Path(report_path) if report_path is not None else demo_root / "report.json"
    atomic_write_json(destination, with_content_hash(report))
    return report


def _load_app(args: argparse.Namespace) -> CadApplication:
    return CadApplication.open(Path(args.root), max_undo_depth=args.max_depth)


def _parse_value(values: Sequence[str]) -> list[float]:
    if len(values) != 3:
        raise DomainError(ErrorCode.SCHEMA_VERSION, "--value requires exactly three floats")
    return [float(item) for item in values]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cad-surface", description="Versioned deterministic NURBS CAD demo")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="run deterministic generation, editing, crash recovery, and hashes")
    demo.add_argument("--demo-root", default="artifacts/cad_demo")
    demo.add_argument("--seed", type=int, default=20261119)
    demo.add_argument("--grid", type=int, default=16)
    demo.add_argument("--max-depth", type=int, default=16)
    demo.add_argument("--report", default=None)

    generate = subparsers.add_parser("generate", help="generate the deterministic model and print its hash")
    generate.add_argument("--seed", type=int, default=20261119)

    create = subparsers.add_parser("create", help="create a persistent workspace")
    create.add_argument("root")
    create.add_argument("--seed", type=int, default=20261119)
    create.add_argument("--grid", type=int, default=16)
    create.add_argument("--max-depth", type=int, default=32)

    edit = subparsers.add_parser("edit", help="commit one control point edit")
    edit.add_argument("--root", required=True)
    edit.add_argument("--patch", required=True)
    edit.add_argument("--u", type=int, required=True)
    edit.add_argument("--v", type=int, required=True)
    edit.add_argument("--value", nargs=3, required=True)
    edit.add_argument("--max-depth", type=int, default=32)

    undo = subparsers.add_parser("undo")
    undo.add_argument("--root", required=True)
    undo.add_argument("--max-depth", type=int, default=32)

    redo = subparsers.add_parser("redo")
    redo.add_argument("--root", required=True)
    redo.add_argument("--max-depth", type=int, default=32)

    status = subparsers.add_parser("status")
    status.add_argument("--root", required=True)
    status.add_argument("--max-depth", type=int, default=32)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--root", required=True)
    compare.add_argument("--first", type=int, required=True)
    compare.add_argument("--second", type=int, required=True)
    compare.add_argument("--max-depth", type=int, default=32)

    export = subparsers.add_parser("export")
    export.add_argument("--root", required=True)
    export.add_argument("--version", type=int, required=True)
    export.add_argument("--mode", choices=("control_net", "shaded_surface", "isoparametric"), required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--max-depth", type=int, default=32)

    close = subparsers.add_parser("close")
    close.add_argument("--root", required=True)
    close.add_argument("--max-depth", type=int, default=32)

    worker = subparsers.add_parser("_crash-worker", help=argparse.SUPPRESS)
    worker.add_argument("--root", required=True)
    worker.add_argument("--patch", required=True)
    worker.add_argument("--u", type=int, required=True)
    worker.add_argument("--v", type=int, required=True)
    worker.add_argument("--value", nargs=3, required=True)
    worker.add_argument("--marker", required=True)
    worker.add_argument("--max-depth", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list:
        argv_list = ["demo"]
    parser = _build_parser()
    args = parser.parse_args(argv_list)
    try:
        if args.command == "demo":
            report = run_demo(
                demo_root=Path(args.demo_root),
                seed=args.seed,
                grid_resolution=args.grid,
                max_undo_depth=args.max_depth,
                report_path=Path(args.report) if args.report else None,
            )
            _json_print(
                {
                    "result": report["result"],
                    "report": str(Path(args.report) if args.report else Path(args.demo_root) / "report.json"),
                    "checks": report["checks"],
                    "model_hash": report["sample_generation"]["model_hash"],
                    "final_geometry_hash": report["branch_state"]["geometry_hash"],
                    "recovery_state_hash": report["recovery"]["actual_state_hash"],
                }
            )
            return 0 if report["result"] == "PASS" else 1
        if args.command == "generate":
            first = generate_model(args.seed)
            second = generate_model(args.seed)
            _json_print(
                {
                    "seed": args.seed,
                    "content_hash": first.content_hash(),
                    "byte_identical": first.to_bytes() == second.to_bytes(),
                    "patches": [patch.key for patch in first.patches],
                }
            )
            return 0
        if args.command == "create":
            app = CadApplication.create(
                Path(args.root),
                seed=args.seed,
                max_undo_depth=args.max_depth,
                grid_resolution=args.grid,
            )
            _json_print(app.state_summary())
            app.close()
            return 0
        if args.command == "edit":
            app = _load_app(args)
            app.edit_control_point(args.patch, args.u, args.v, _parse_value(args.value))
            _json_print(app.state_summary())
            app.close()
            return 0
        if args.command == "undo":
            app = _load_app(args)
            app.undo()
            _json_print(app.state_summary())
            app.close()
            return 0
        if args.command == "redo":
            app = _load_app(args)
            app.redo()
            _json_print(app.state_summary())
            app.close()
            return 0
        if args.command == "status":
            app = _load_app(args)
            _json_print({"recovery": app.recovery.to_dict() if app.recovery else None, **app.state_summary()})
            app.close()
            return 0
        if args.command == "compare":
            app = _load_app(args)
            _json_print(app.compare_versions(args.first, args.second))
            app.close()
            return 0
        if args.command == "export":
            app = _load_app(args)
            path = app.export_ppm(args.version, args.mode, Path(args.output))
            _json_print({"output": str(path), **app.state_summary(args.version)})
            app.close()
            return 0
        if args.command == "close":
            app = _load_app(args)
            _json_print(app.close())
            return 0
        if args.command == "_crash-worker":
            app = CadApplication.open(Path(args.root), max_undo_depth=args.max_depth)
            staging = app.stage_crash_probe(args.patch, args.u, args.v, _parse_value(args.value))
            marker = with_content_hash(
                {
                    "schema_version": "cad-surface.crash-marker.v1",
                    **staging,
                    "pid": os.getpid(),
                    "exit_code": 23,
                    "closed_flag_written": False,
                }
            )
            atomic_write_json(Path(args.marker), marker)
            os._exit(23)
        parser.error("unknown command")
        return 2
    except DomainError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(
            json.dumps(
                {
                    "code": ErrorCode.SNAPSHOT_STALE.value,
                    "message": "workspace manifest is missing",
                    "details": {"path": str(exc)},
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
