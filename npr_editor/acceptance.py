"""Reproducible end-to-end acceptance run with machine-readable evidence."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .binary_format import decode_binary, encode_binary
from .compare import ComparisonStore, VersionTrace, compare_versions
from .editing import Editor
from .errors import NprError
from .geometry import GeometryEngine, flatten_shape
from .migration import assert_idempotent, migrate_document
from .model import canonical_hash, project_from_dict
from .png import encode_png
from .render import RenderConfig, RenderEngine, verify_render_hashes
from .samples import build_sample_project, legacy_v1_document, legacy_v2_document, write_samples
from .snapshots import SnapshotStore, atomic_write_bytes

CONFIG = RenderConfig(width=240, height=160, supersample=1)


def _project_vertex_map(project) -> dict[str, np.ndarray]:
    engine = GeometryEngine(project)
    return {shape.id: engine.ensure(shape.id).vertices.copy() for shape in project.shapes}


def _capture_error(code: str, action: Callable[[], Any]) -> dict[str, Any]:
    try:
        action()
    except NprError as exc:
        return {
            "expected_code": code,
            "actual_code": exc.code,
            "passed": exc.code == code,
            "message": exc.message,
            "details": exc.details,
        }
    except Exception as exc:  # deliberately reported rather than hidden
        return {
            "expected_code": code,
            "actual_code": type(exc).__name__,
            "passed": False,
            "message": str(exc),
            "details": {},
        }
    return {
        "expected_code": code,
        "actual_code": None,
        "passed": False,
        "message": "no error raised",
        "details": {},
    }


def _write_png(path: Path, image: np.ndarray) -> dict[str, str]:
    data = encode_png(image)
    atomic_write_bytes(path, data)
    return {
        "file": path.name,
        "pixel_sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
        "png_sha256": hashlib.sha256(data).hexdigest(),
    }


def run_acceptance(workspace: str | Path, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    out = Path(out_dir).resolve() if out_dir is not None else workspace / "artifacts" / "acceptance"
    out.mkdir(parents=True, exist_ok=True)
    sample_dir = workspace / "samples"
    sample_manifest = write_samples(sample_dir)
    project = build_sample_project()
    checks: list[dict[str, Any]] = []

    roundtrip_checks = []
    for byte_order in ("little", "big"):
        for index_width in (2, 4):
            encoded = encode_binary(
                project, byte_order=byte_order, index_width=index_width
            )
            decoded = decode_binary(encoded.data)
            expected_vertices = _project_vertex_map(project)
            actual_vertices = _project_vertex_map(decoded.project)
            max_error = max(
                float(np.abs(expected_vertices[shape_id] - actual_vertices[shape_id]).max())
                for shape_id in expected_vertices
            )
            passed = (
                decoded.project_hash == project.content_hash()
                and max_error == 0.0
                and decoded.parsed.byte_order == byte_order
                and decoded.parsed.index_width == index_width
            )
            roundtrip_checks.append(
                {
                    "byte_order": byte_order,
                    "index_width": index_width,
                    "passed": passed,
                    "json_hash": f"json://sha256:{project.content_hash()}",
                    "binary_hash": f"bin://sha256:{encoded.content_hash}",
                    "roundtrip_project_hash": f"json://sha256:{decoded.project_hash}",
                    "max_vertex_error": max_error,
                }
            )
    legacy_width = encode_binary(project, format_version=1, byte_order="big", index_width=2)
    legacy_parsed = decode_binary(legacy_width.data)
    roundtrip_checks.append(
        {
            "byte_order": "big",
            "index_width": 2,
            "format_version": 1,
            "width_encoding": legacy_parsed.parsed.width_encoding,
            "passed": legacy_parsed.parsed.width_encoding == 0,
            "json_hash": f"json://sha256:{project.content_hash()}",
            "binary_hash": f"bin://sha256:{legacy_width.content_hash}",
            "roundtrip_project_hash": f"json://sha256:{legacy_parsed.project_hash}",
            "max_vertex_error": 0.0,
        }
    )
    checks.append(
        {
            "name": "format_roundtrip_all_combinations",
            "passed": all(item["passed"] for item in roundtrip_checks),
            "variants": roundtrip_checks,
        }
    )

    migrations = {}
    for name, document in (
        ("v1", legacy_v1_document(project)),
        ("v2", legacy_v2_document(project)),
        ("v3", project.to_dict()),
    ):
        result = migrate_document(document)
        migrations[name] = {
            "source_version": result.source_version,
            "changed": result.changed,
            "actions": [action.to_dict() for action in result.actions],
            "project_hash": f"json://sha256:{result.content_hash_after}",
            "idempotent": assert_idempotent(document),
        }
    migrated_v1 = migrate_document(legacy_v1_document(project)).project
    migrated_v2 = migrate_document(legacy_v2_document(project)).project
    canonical_render = RenderEngine(project, CONFIG).render("lineart")
    v1_render = RenderEngine(migrated_v1, CONFIG).render("lineart")
    v2_render = RenderEngine(migrated_v2, CONFIG).render("lineart")
    migration_pass = (
        canonical_render.pixel_sha256 == v1_render.pixel_sha256 == v2_render.pixel_sha256
        and all(item["idempotent"] for item in migrations.values())
    )
    checks.append(
        {
            "name": "schema_migration_and_lineart_equivalence",
            "passed": migration_pass,
            "migrations": migrations,
            "lineart_pixel_sha256": canonical_render.pixel_sha256,
            "v1_lineart_pixel_sha256": v1_render.pixel_sha256,
            "v2_lineart_pixel_sha256": v2_render.pixel_sha256,
        }
    )

    engine = RenderEngine(project, CONFIG)
    mode_results: dict[str, Any] = {}
    for mode in ("lineart", "sketch", "toon"):
        engine.geometry.reset_stats()
        engine.reset_stats()
        result = engine.render(mode)  # type: ignore[arg-type]
        mode_results[mode] = {
            "pixel_sha256": result.pixel_sha256,
            "png_sha256": result.png_sha256,
            "flatten_builds": result.flatten_builds,
            "skeleton_builds": result.skeleton_builds,
            "render_builds": result.render_builds,
            "image": _write_png(out / f"preview_{mode}.png", result.image),
        }
    mode_pass = all(
        item["flatten_builds"] == 0 and item["skeleton_builds"] == 0
        for mode in ("sketch", "toon")
        for item in [mode_results[mode]]
    )
    checks.append(
        {
            "name": "mode_switch_reuses_geometry_and_skeleton",
            "passed": mode_pass,
            "modes": mode_results,
            "geometry_hash": project.geometry_hash(),
        }
    )

    editor = Editor(project, config=CONFIG)
    before_hashes = editor.geometry_hashes()
    before_render = editor.render("lineart")
    before_arrays = {
        shape.id: editor.geometry.flattened(shape.id).vertices.copy()
        for shape in project.shapes
    }
    editor.edit_control_point("tree_crown", 0, 0, 1, 151.0, 83.0)
    after_hashes = editor.geometry_hashes()
    after_render = editor.render("lineart")
    changed_geometry = sorted(
        shape_id for shape_id in before_hashes if before_hashes[shape_id] != after_hashes[shape_id]
    )
    changed_render = sorted(
        shape_id
        for shape_id in before_render.shape_hashes
        if before_render.shape_hashes[shape_id] != after_render.shape_hashes[shape_id]
    )
    unaffected_equal = all(
        np.array_equal(before_arrays[shape_id], editor.geometry.flattened(shape_id).vertices)
        for shape_id in before_arrays
        if shape_id != "tree_crown"
    )
    undo = editor.undo()
    undo_render = editor.render("lineart")
    undo_pass = (
        undo.project_hash == project.content_hash()
        and editor.geometry_hashes() == before_hashes
        and undo_render.shape_hashes == before_render.shape_hashes
        and undo_render.pixel_sha256 == before_render.pixel_sha256
    )
    undo_image = _write_png(out / "undo_restored.png", undo_render.image)
    checks.append(
        {
            "name": "precision_edit_undo_and_render_restore",
            "passed": changed_geometry == ["tree_crown"]
            and changed_render == ["tree_crown"]
            and unaffected_equal
            and undo_pass,
            "changed_geometry": changed_geometry,
            "changed_render_hashes": changed_render,
            "unaffected_vertices_equal": unaffected_equal,
            "before_geometry_hashes": dict(sorted(before_hashes.items())),
            "after_geometry_hashes": dict(sorted(after_hashes.items())),
            "before_render_hashes": dict(sorted(before_render.shape_hashes.items())),
            "after_render_hashes": dict(sorted(after_render.shape_hashes.items())),
            "undo": {
                "passed": undo_pass,
                "restored_stamp": undo.restored_version_stamp,
                "restored_project_hash": undo.project_hash,
                "restored_geometry_hashes": dict(sorted(editor.geometry_hashes().items())),
                "restored_render_hashes": dict(sorted(undo_render.shape_hashes.items())),
                "restored_pixel_sha256": undo_render.pixel_sha256,
                "image": undo_image,
            },
        }
    )

    snapshot_workspace = out / "snapshot_workspace"
    if snapshot_workspace.exists():
        shutil.rmtree(snapshot_workspace)
    store = SnapshotStore(snapshot_workspace)
    snapshot_engine = RenderEngine(project, CONFIG)
    snapshot_render = snapshot_engine.render("lineart")
    first_snapshot = store.commit(
        project, mode="lineart", result=snapshot_render, engine=snapshot_engine
    )

    worker = workspace / "scripts" / "forced_termination_worker.py"
    child = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--workspace",
            str(snapshot_workspace),
            "--stamp",
            "000000000002",
        ],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=60,
    )
    recovery = store.recover()
    snapshot_pass = (
        child.returncode == 23
        and recovery.latest is not None
        and recovery.latest.snapshot_hash == first_snapshot.snapshot_hash
        and bool(recovery.reclaimed_orphans)
    )
    recovery_summary = recovery.to_dict()
    recovery_summary["reclaimed_orphan_count"] = len(recovery_summary.pop("reclaimed_orphans"))

    second_editor = Editor(project, config=CONFIG)
    second_editor.edit_control_point("tree_crown", 0, 0, 1, 151.0, 83.0)
    second_render = second_editor.render("lineart")
    second_snapshot = store.commit(
        second_editor.project,
        mode="lineart",
        result=second_render,
        engine=second_editor.renderer,
    )
    restored_engine = RenderEngine(store.load_project(second_snapshot), CONFIG)
    restored_shapes = store.restore_render_cache(second_snapshot, restored_engine, "lineart")
    restored_render = restored_engine.render("lineart")
    cache_pass = (
        restored_render.pixel_sha256 == second_render.pixel_sha256
        and restored_render.render_builds == 0
        and len(restored_shapes) == len(project.shapes)
    )
    checks.append(
        {
            "name": "snapshot_atomic_commit_recovery_and_cache_reuse",
            "passed": snapshot_pass and cache_pass,
            "first_snapshot": {
                "version_stamp": first_snapshot.version_stamp,
                "snapshot_hash": first_snapshot.snapshot_hash,
            },
            "forced_termination_exit_code": child.returncode,
            "recovery": recovery_summary,
            "second_snapshot": {
                "version_stamp": second_snapshot.version_stamp,
                "snapshot_hash": second_snapshot.snapshot_hash,
            },
            "restored_shapes": sorted(restored_shapes),
            "restored_render": {
                "pixel_sha256": restored_render.pixel_sha256,
                "render_builds": restored_render.render_builds,
                "cache_hits": restored_render.render_cache_hits,
            },
        }
    )

    compare_left = migrate_document(legacy_v1_document(project)).project
    left_trace = VersionTrace(
        version_stamp=compare_left.version_stamp,
        source_schema_version=1,
        migration_actions=[a.to_dict() for a in migrate_document(legacy_v1_document(project)).actions],
    )
    right_trace = VersionTrace(
        version_stamp=second_editor.project.version_stamp,
        source_schema_version=3,
        edited_shapes=("tree_crown",),
        edit_actions=[{"shape": "tree_crown", "field": "control_point"}],
    )
    compare_dir = out / "comparison_workspace"
    if compare_dir.exists():
        shutil.rmtree(compare_dir)
    comparison_store = ComparisonStore(compare_dir)
    comparison_result, comparison_path = comparison_store.save(
        compare_left,
        second_editor.project,
        left_trace,
        right_trace,
        config=CONFIG,
    )
    recomputed = comparison_store.recompute(
        compare_left.version_stamp, second_editor.project.version_stamp
    )
    comparison_pass = (
        recomputed == comparison_result
        and not comparison_result["suspicious_items"]
        and any(item["kind"] == "format_version" for item in comparison_result["differences"])
        and any(item["kind"] == "control_point" for item in comparison_result["differences"])
    )
    atomic_write_bytes(
        out / "comparison_side_by_side.png",
        (comparison_path / "side_by_side.png").read_bytes(),
    )
    checks.append(
        {
            "name": "version_compare_attribution_and_recompute",
            "passed": comparison_pass,
            "comparison_hash": comparison_result["comparison_hash"],
            "difference_kinds": sorted({item["kind"] for item in comparison_result["differences"]}),
            "sources": sorted(
                {source for item in comparison_result["differences"] for source in item["sources"]}
            ),
            "suspicious_items": comparison_result["suspicious_items"],
            "side_by_side_png_sha256": comparison_result["side_by_side_png_sha256"],
            "recomputed_equal": recomputed == comparison_result,
        }
    )

    current_doc = project.to_dict()
    bad_endian = bytearray(encode_binary(project).data)
    bad_endian[4:8] = b"\xde\xad\xbe\xef"
    bad_index = bytearray(encode_binary(project, byte_order="little", index_width=4).data)
    import struct

    struct.pack_into("<I", bad_index, len(bad_index) - 4, 0xFFFF)
    invalid_sample = json.loads(json.dumps(current_doc))
    invalid_sample["stroke_sample"]["curve_steps"] = 0
    out_of_range_schema = json.loads(json.dumps(current_doc))
    out_of_range_schema["schema_version"] = 99
    empty_editor = Editor(project, config=CONFIG)
    older_doc = project.to_dict()
    older_doc["version_stamp"] = "000000000001"
    error_matrix = {
        "E_SCHEMA_VERSION": _capture_error(
            "E_SCHEMA_VERSION", lambda: migrate_document(out_of_range_schema)
        ),
        "E_FORMAT_ENDIAN": _capture_error(
            "E_FORMAT_ENDIAN", lambda: decode_binary(bytes(bad_endian))
        ),
        "E_FORMAT_INDEX": _capture_error(
            "E_FORMAT_INDEX", lambda: decode_binary(bytes(bad_index))
        ),
        "E_STROKE_SAMPLE": _capture_error(
            "E_STROKE_SAMPLE", lambda: project_from_dict(invalid_sample)
        ),
        "E_UNDO_EMPTY": _capture_error("E_UNDO_EMPTY", lambda: empty_editor.undo()),
        "E_SNAPSHOT_STALE": _capture_error(
            "E_SNAPSHOT_STALE",
            lambda: store.commit(
                project,
                mode="lineart",
                result=snapshot_render,
                engine=snapshot_engine,
            ),
        ),
        "E_RENDER_MISMATCH": _capture_error(
            "E_RENDER_MISMATCH",
            lambda: verify_render_hashes({"tree_crown": "expected"}, {"tree_crown": "actual"}),
        ),
    }
    checks.append(
        {
            "name": "business_error_code_injection",
            "passed": all(item["passed"] for item in error_matrix.values()),
            "errors": error_matrix,
        }
    )

    error_report = {
        "format": "npr-error-report-v1",
        "checks": error_matrix,
        "all_passed": all(item["passed"] for item in error_matrix.values()),
    }
    error_report["report_hash"] = canonical_hash(error_report)
    hash_report = {
        "format": "npr-consistent-hash-report-v1",
        "project_hash": f"json://sha256:{project.content_hash()}",
        "geometry_hash": project.geometry_hash(),
        "format_roundtrip": roundtrip_checks,
        "migration": migrations,
        "render_modes": {
            mode: {
                "pixel_sha256": entry["pixel_sha256"],
                "png_sha256": entry["png_sha256"],
            }
            for mode, entry in mode_results.items()
        },
        "editing": {
            "before_geometry_hashes": dict(sorted(before_hashes.items())),
            "after_geometry_hashes": dict(sorted(after_hashes.items())),
            "before_render_hashes": dict(sorted(before_render.shape_hashes.items())),
            "after_render_hashes": dict(sorted(after_render.shape_hashes.items())),
            "undo_restored_pixel_sha256": undo_render.pixel_sha256,
        },
        "snapshots": {
            "first": {
                "version_stamp": first_snapshot.version_stamp,
                "snapshot_hash": first_snapshot.snapshot_hash,
            },
            "second": {
                "version_stamp": second_snapshot.version_stamp,
                "snapshot_hash": second_snapshot.snapshot_hash,
            },
            "restored_pixel_sha256": restored_render.pixel_sha256,
        },
        "comparison": {
            "comparison_hash": comparison_result["comparison_hash"],
            "side_by_side_png_sha256": comparison_result["side_by_side_png_sha256"],
        },
    }
    hash_report["report_hash"] = canonical_hash(hash_report)
    atomic_write_bytes(
        out / "error_report.json",
        (json.dumps(error_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    atomic_write_bytes(
        out / "hash_report.json",
        (json.dumps(hash_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    report = {
        "format": "npr-acceptance-v1",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
            "numpy": __import__("numpy").__version__,
            "target_platform": "Windows 10/11 with Python 3.11 or 3.12",
            "container_used": False,
        },
        "samples": {
            "directory": str(sample_dir.relative_to(workspace)),
            "manifest_hash": canonical_hash(sample_manifest),
            "project_hash": f"json://sha256:{project.content_hash()}",
            "project_geometry_hash": project.geometry_hash(),
        },
        "checks": checks,
        "supplemental_reports": {
            "error_report.json": error_report["report_hash"],
            "hash_report.json": hash_report["report_hash"],
        },
        "summary": {
            "passed": all(check["passed"] for check in checks),
            "check_count": len(checks),
            "error_codes_verified": sorted(error_matrix),
        },
    }
    report["report_hash"] = canonical_hash(report)
    atomic_write_bytes(
        out / "acceptance_report.json",
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run deterministic NPR acceptance")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = run_acceptance(args.workspace, out_dir=args.out)
    print(
        json.dumps(
            {
                "passed": report["summary"]["passed"],
                "check_count": report["summary"]["check_count"],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
