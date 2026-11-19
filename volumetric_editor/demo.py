"""Repeatable end-to-end demonstration with hashes and fixed injected failures."""

from __future__ import annotations

import csv
import hashlib
import locale
import math
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__
from .errors import BusinessError, ErrorCode
from .offscreen import probe_offscreen, write_backend_log
from .rendering import VOLUME_FIELDS, plan_tiles, trace_sha256
from .scene import canonical_json_bytes, generate_scene, save_scene, scene_bytes, scene_sha256
from .scheduling import estimate_task_cost, estimate_task_memory
from .session import IncrementalSession


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> bytes:
    payload_bytes = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload_bytes)
    return payload_bytes


def _write_frame(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.ascontiguousarray(image, dtype="<f4").tobytes(order="C")
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _trim_tile_reports(report: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(report)
    trimmed["tile_reports"] = [
        {
            key: item[key]
            for key in (
                "tile_key",
                "tx",
                "ty",
                "bounds",
                "status",
                "estimated_memory_bytes",
                "estimated_cost_units",
                "image_sha256",
                "volume_trace_sha256",
                "combined_sha256",
            )
        }
        for item in report.get("tile_reports", [])
    ]
    return trimmed


def _capture_error(label: str, expected: ErrorCode, action: Callable[[], Any]) -> dict[str, Any]:
    try:
        action()
    except BusinessError as exc:
        return {
            "label": label,
            "expected_code": str(expected),
            "actual_code": str(exc.code),
            "matched": exc.code == expected,
            "message": exc.message,
            "details": exc.details,
        }
    except Exception as exc:  # defensive: the demo must not crash on injections
        return {
            "label": label,
            "expected_code": str(expected),
            "actual_code": "E_INTERNAL",
            "matched": False,
            "message": f"{type(exc).__name__}: {exc}",
            "details": {},
        }
    return {
        "label": label,
        "expected_code": str(expected),
        "actual_code": "E_NONE",
        "matched": False,
        "message": "operation unexpectedly succeeded",
        "details": {},
    }


def _volume_csv(session: IncrementalSession, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    line_count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "tile_key",
                "x",
                "y",
                "step",
                *VOLUME_FIELDS,
            ]
        )
        for key in sorted(session.tiles):
            tile = session.tiles[key]
            active = np.argwhere(tile.volume_inside.astype(bool))
            for local_y, local_x, step in active:
                values = tile.volume_trace[int(local_y), int(local_x), int(step)]
                x = int(tile.x0 + local_x)
                y = int(tile.y0 + local_y)
                row = [key, x, y, int(step), *(format(float(value), ".9g") for value in values)]
                writer.writerow(row)
                line_count += 1
                if len(samples) < 8:
                    samples.append(
                        {
                            "tile_key": key,
                            "x": x,
                            "y": y,
                            "step": int(step),
                            **{name: float(value) for name, value in zip(VOLUME_FIELDS, values, strict=True)},
                        }
                    )
    digest = _sha256_bytes(path.read_bytes())
    return {
        "path": path,
        "line_count": line_count,
        "sha256": digest,
        "sample_records": samples,
    }


def _write_scatter_closure(path: Path, report: dict[str, Any]) -> str:
    payload = {
        "edit": report["edit"],
        "closure": report["closure"],
        "reuse": report["reuse"],
        "actual_recomputed_tiles": report["actual_recomputed_tiles"],
        "actual_matches_closure": report["actual_matches_closure"],
    }
    content = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return _sha256_bytes(content)


def run_demo(
    root: str | Path = ".",
    seed: int = 20261119,
    width: int = 96,
    height: int = 64,
    backend: str = "auto",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    data_dir = root_path / "data"
    work_dir = root_path / "work"
    frames_dir = work_dir / "frames"
    data_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene_one = generate_scene(seed=seed, width=width, height=height)
    scene_two = generate_scene(seed=seed, width=width, height=height)
    scene_deterministic = scene_bytes(scene_one) == scene_bytes(scene_two)
    scene_path = data_dir / "scene.json"
    save_scene(scene_one, scene_path)

    backend_report = probe_offscreen(backend)
    backend_log = work_dir / "backend.log"
    write_backend_log(backend_report, backend_log)

    matrix_specs = [(1, 16), (2, 16), (4, 16), (2, 24), (3, 24)]
    matrix_items: list[dict[str, Any]] = []
    matrix_images: list[str] = []
    matrix_traces: list[str] = []
    for threads, tile_size in matrix_specs:
        session = IncrementalSession(scene_one, tile_size=tile_size)
        result = session.render_full(threads=threads)
        snapshot = session.snapshot()
        frame_path = frames_dir / f"matrix_t{tile_size:02d}_w{threads:02d}.rgba"
        frame_hash = _write_frame(frame_path, session.image)
        matrix_images.append(snapshot.image_sha256)
        matrix_traces.append(snapshot.trace_sha256)
        matrix_items.append(
            {
                "threads": threads,
                "tile_size": tile_size,
                "effective_threads": result["scheduler"]["effective_threads"],
                "serialized": result["scheduler"]["serialized"],
                "degradation_events": result["scheduler"]["degradation_events"],
                "canonical_image_sha256": snapshot.image_sha256,
                "volume_trace_sha256": snapshot.trace_sha256,
                "frame_file": _relative(frame_path, root_path),
                "frame_file_sha256": frame_hash,
                "tile_hashes": snapshot.tile_hashes,
            }
        )

    matrix_image_invariant = len(set(matrix_images)) == 1
    matrix_trace_invariant = len(set(matrix_traces)) == 1

    budget_session = IncrementalSession(scene_one, tile_size=16)
    specs = plan_tiles(width, height, 16)
    largest_memory = max(estimate_task_memory(scene_one, spec) for spec in specs)
    largest_cost = max(estimate_task_cost(scene_one, spec) for spec in specs)
    degradation_memory = math.ceil(largest_memory * 3.10)
    degradation_compute = math.ceil(largest_cost * 2.50)
    degradation_result = budget_session.render_full(
        threads=4,
        queue_memory_budget_bytes=degradation_memory,
        compute_budget_units=degradation_compute,
    )
    scheduler_report = degradation_result["scheduler"]
    all_budget_tiles_completed = (
        scheduler_report["completed_task_count"] == scheduler_report["task_count"]
        and len(scheduler_report["completed_tiles"]) == scheduler_report["task_count"]
    )
    degradation_frame_path = frames_dir / "budget_degraded.rgba"
    _write_frame(degradation_frame_path, budget_session.image)
    budget_section = {
        "requested_threads": 4,
        "queue_memory_budget_bytes": degradation_memory,
        "compute_budget_units": degradation_compute,
        "effective_threads": scheduler_report["effective_threads"],
        "serialized": scheduler_report["serialized"],
        "degradation_events": scheduler_report["degradation_events"],
        "dispatch_waves": scheduler_report["dispatch_waves"],
        "wave_count": scheduler_report["wave_count"],
        "completed_task_count": scheduler_report["completed_task_count"],
        "task_count": scheduler_report["task_count"],
        "all_tiles_completed": all_budget_tiles_completed,
        "frame_file": _relative(degradation_frame_path, root_path),
        "canonical_image_sha256": budget_session.snapshot().image_sha256,
    }

    memory_zero_error = _capture_error(
        "zero_queue_memory_budget",
        ErrorCode.BUDGET_DENIED,
        lambda: IncrementalSession(scene_one, 16).render_full(
            threads=2, queue_memory_budget_bytes=0
        ),
    )
    denied_task_error = _capture_error(
        "task_admission_with_one_byte_queue_memory",
        ErrorCode.BUDGET_DENIED,
        lambda: IncrementalSession(scene_one, 16).render_full(
            threads=2, queue_memory_budget_bytes=1
        ),
    )

    incremental = IncrementalSession(scene_one, tile_size=16)
    baseline_render = incremental.render_full(threads=4)
    baseline = incremental.snapshot()
    ior_old = incremental.scene.surface_by_id("lens_left").ior
    ior_new = round(ior_old + 0.17, 6)
    ior_report = incremental.edit_ior("lens_left", ior_new, threads=3)
    scatter_old = incremental.scene.light_cone.scatter
    scatter_new = round(scatter_old + 0.14, 6)
    scatter_report = incremental.edit_scatter(scatter_new, threads=3)
    scatter_closure_path = work_dir / "scatter_closure.json"
    scatter_closure_file_hash = _write_scatter_closure(scatter_closure_path, scatter_report)
    scatter_rows = scatter_report["closure"].pop("volume_segment_rows")
    scatter_report["closure"]["volume_segment_rows_file"] = _relative(scatter_closure_path, root_path)
    scatter_report["closure"]["volume_segment_rows_file_sha256"] = scatter_closure_file_hash
    scatter_report["closure"]["volume_segment_rows_sample"] = scatter_rows[:6]
    scatter_report["closure"]["volume_segment_rows_last"] = scatter_rows[-3:]
    retile_report = incremental.edit_tile_size(24)

    final_frame_path = frames_dir / "incremental_final.rgba"
    final_frame_hash = _write_frame(final_frame_path, incremental.image)
    trace_path = data_dir / "volume_trace.npy"
    with trace_path.open("wb") as handle:
        np.save(handle, incremental.trace, allow_pickle=False)
    trace_file_hash = _sha256_bytes(trace_path.read_bytes())
    trace_array_hash = trace_sha256(incremental.trace)
    volume_csv_path = work_dir / "volume_segments.csv"
    volume_csv = _volume_csv(incremental, volume_csv_path)
    active_segments = int(np.sum(incremental.inside, dtype=np.int64))
    volume_report = {
        "fields": list(VOLUME_FIELDS),
        "shape": list(incremental.trace.shape),
        "dtype": "float32",
        "total_segments": int(incremental.trace.shape[0] * incremental.trace.shape[1] * incremental.trace.shape[2]),
        "active_segments": active_segments,
        "trace_npy_file": _relative(trace_path, root_path),
        "trace_npy_sha256": trace_file_hash,
        "trace_array_sha256": trace_array_hash,
        "segments_csv_file": _relative(volume_csv_path, root_path),
        "segments_csv_sha256": volume_csv["sha256"],
        "segments_csv_line_count": volume_csv["line_count"],
        "sample_segments": volume_csv["sample_records"],
    }

    error_session = IncrementalSession(scene_one, tile_size=16)
    error_session.render_full(threads=2)
    errors = [
        _capture_error(
            "invalid_tile_size",
            ErrorCode.TILE_SIZE,
            lambda: error_session.render_full(threads=2, tile_size=0),
        ),
        _capture_error(
            "threads_out_of_bounds",
            ErrorCode.THREAD_BOUNDS,
            lambda: IncrementalSession(scene_one, 16).render_full(threads=0),
        ),
        _capture_error(
            "ior_out_of_range",
            ErrorCode.IOR_RANGE,
            lambda: error_session.edit_ior("lens_left", 9.0),
        ),
        _capture_error(
            "negative_cone_scatter",
            ErrorCode.MEDIUM_COEFF,
            lambda: error_session.edit_scatter(-1.0),
        ),
        memory_zero_error,
        denied_task_error,
    ]

    checks = {
        "scene_serialization_byte_identical": scene_deterministic,
        "parallel_image_byte_invariant": matrix_image_invariant,
        "parallel_volume_trace_byte_invariant": matrix_trace_invariant,
        "budget_serial_or_reduced_all_tiles_completed": all_budget_tiles_completed,
        "ior_closure_equals_actual": ior_report["actual_matches_closure"],
        "ior_reused_tiles_unchanged": ior_report["reuse"]["tile_content_hashes_unchanged"],
        "scatter_closure_equals_actual": scatter_report["actual_matches_closure"],
        "scatter_outside_segments_unchanged": scatter_report["outside_volume_trace_preserved"],
        "scatter_reused_tiles_unchanged": scatter_report["reuse"]["tile_content_hashes_unchanged"],
        "retile_recomputed_nothing": retile_report["actual_recomputed_tiles"] == [],
        "retile_canonical_image_unchanged": retile_report["reused"]["canonical_image_sha256_unchanged"],
        "retile_canonical_trace_unchanged": retile_report["reused"]["canonical_volume_trace_sha256_unchanged"],
        "all_injected_error_codes_matched": all(item["matched"] for item in errors),
    }
    checks["all_pass"] = all(checks.values())

    report = {
        "schema_version": 1,
        "application": "volumetric-editor",
        "application_version": __version__,
        "command": "python -m volumetric_editor demo",
        "environment": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "sys_default_encoding": sys.getdefaultencoding(),
            "preferred_encoding": locale.getpreferredencoding(False),
            "deterministic_utf8_io": True,
        },
        "config": {
            "root": ".",
            "seed": seed,
            "width": width,
            "height": height,
            "requested_backend": backend,
        },
        "backend": backend_report,
        "scene": {
            "file": _relative(scene_path, root_path),
            "sha256": scene_sha256(scene_one),
            "generated_twice_byte_identical": scene_deterministic,
            "surface_count": len(scene_one.surfaces),
            "surfaces": [
                {
                    "surface_id": surface.surface_id,
                    "kind": surface.kind,
                    "normal": list(surface.normal),
                    "ior": surface.ior,
                    "faces": surface.to_dict()["faces"],
                }
                for surface in scene_one.surfaces
            ],
            "light_cone": scene_one.light_cone.to_dict(),
            "occluders": [item.to_dict() for item in scene_one.occluders],
        },
        "parallel_matrix": {
            "items": matrix_items,
            "unique_image_hashes": sorted(set(matrix_images)),
            "unique_volume_trace_hashes": sorted(set(matrix_traces)),
            "image_byte_invariant": matrix_image_invariant,
            "volume_trace_byte_invariant": matrix_trace_invariant,
        },
        "budget": budget_section,
        "incremental": {
            "baseline": baseline.to_dict(),
            "baseline_render": _trim_tile_reports(baseline_render["scheduler"]),
            "ior": ior_report,
            "scatter": scatter_report,
            "tile_size": retile_report,
            "final_frame_file": _relative(final_frame_path, root_path),
            "final_frame_file_sha256": final_frame_hash,
            "final_frame": incremental.snapshot().to_dict(),
        },
        "volume_report": volume_report,
        "errors": errors,
        "checks": checks,
    }

    report_path = work_dir / "demo_report.json"
    payload = _write_json(report_path, report)
    (work_dir / "demo_report.sha256").write_text(_sha256_bytes(payload) + "\n", encoding="ascii")
    return {
        "report_file": _relative(report_path, root_path),
        "report_sha256": _sha256_bytes(payload),
        "checks": checks,
        "scene_sha256": report["scene"]["sha256"],
        "canonical_image_sha256": matrix_images[0],
        "volume_trace_sha256": matrix_traces[0],
        "backend": backend_report["api"],
        "software_fallback": backend_report["software_fallback"],
        "injected_errors": [
            {"label": item["label"], "expected": item["expected_code"], "actual": item["actual_code"]}
            for item in errors
        ],
    }


def run_generate(root: str | Path = ".", seed: int = 20261119, width: int = 96, height: int = 64) -> dict[str, Any]:
    root_path = Path(root).resolve()
    data_dir = root_path / "data"
    work_dir = root_path / "work"
    data_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    scene = generate_scene(seed=seed, width=width, height=height)
    path = data_dir / "scene.json"
    digest = save_scene(scene, path)
    return {
        "scene_file": _relative(path, root_path),
        "scene_sha256": digest,
        "surface_count": len(scene.surfaces),
    }


def run_render(
    root: str | Path = ".",
    seed: int = 20261119,
    width: int = 96,
    height: int = 64,
    threads: int = 4,
    tile_size: int = 16,
    backend: str = "auto",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    generated = run_generate(root_path, seed=seed, width=width, height=height)
    scene = generate_scene(seed=seed, width=width, height=height)
    backend_report = probe_offscreen(backend)
    work_dir = root_path / "work"
    write_backend_log(backend_report, work_dir / "backend.log")
    session = IncrementalSession(scene, tile_size=tile_size)
    result = session.render_full(threads=threads)
    frame_path = work_dir / "frame.rgba"
    frame_hash = _write_frame(frame_path, session.image)
    report = {
        "scene": generated,
        "backend": backend_report,
        "threads": threads,
        "tile_size": tile_size,
        "scheduler": _trim_tile_reports(result["scheduler"]),
        "frame": session.snapshot().to_dict(),
        "frame_file": _relative(frame_path, root_path),
        "frame_file_sha256": frame_hash,
    }
    report_path = work_dir / "render_report.json"
    payload = _write_json(report_path, report)
    return {
        "report_file": _relative(report_path, root_path),
        "report_sha256": _sha256_bytes(payload),
        "frame_file": report["frame_file"],
        "frame_file_sha256": frame_hash,
        "canonical_image_sha256": session.snapshot().image_sha256,
        "volume_trace_sha256": session.snapshot().trace_sha256,
    }
