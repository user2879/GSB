"""Serializable fault injection matrix for the six business errors."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .consistency import compare_meshes
from .engine import StudioEngine
from .errors import ErrorCode, StudioError
from .persistence import AtomicStateStore
from .render import OffscreenViewport
from .sample import load_sample_bundle
from .snapshots import SnapshotStore, scene_from_engine
from .util import atomic_write_json


def _capture(action: Any) -> dict[str, Any]:
    try:
        value = action()
        return {"ok": True, "value": value}
    except StudioError as exc:
        return {"ok": False, **exc.to_dict()}
    except Exception as exc:
        return {
            "ok": False,
            "code": None,
            "message": "unexpected non-business exception",
            "context": {"exception_type": type(exc).__name__, "reason": str(exc)},
        }


def _copy_current_state(source_workspace: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    source_store = AtomicStateStore(source_workspace)
    current = source_store._read_current()
    if current is None:
        raise RuntimeError("fault matrix requires a committed CURRENT state")
    (destination / "states").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_workspace / "CURRENT", destination / "CURRENT")
    shutil.copytree(
        source_workspace / "states" / str(current["state"]),
        destination / "states" / str(current["state"]),
    )
    closed = source_workspace / "closed.json"
    if closed.exists():
        shutil.copy2(closed, destination / "closed.json")


def run_fault_matrix(
    *,
    sample_dir: str | Path,
    workspace: str | Path,
    output_dir: str | Path,
    loaded_engine: StudioEngine | None = None,
) -> dict[str, Any]:
    sample = Path(sample_dir)
    workspace_path = Path(workspace)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if loaded_engine is None:
        loaded_engine = AtomicStateStore(workspace_path).load_current().engine

    injection_config: dict[str, Any] = {
        "schema": "pointcloud-studio.fault-injection/1",
        "note": "All faults are induced through copied inputs or serialized parameters; no source mutation is used.",
        "cases": [
            {
                "id": "point_cloud_format",
                "expected_code": ErrorCode.E_POINT_CLOUD_FORMAT.value,
                "injection": {"target": "sample_copy/points.bin", "operation": "xor_byte", "offset": 0, "mask": 255},
            },
            {
                "id": "neighbor_radius",
                "expected_code": ErrorCode.E_NEIGHBOR_RADIUS.value,
                "injection": {
                    "operation_type": "reestimate_normals",
                    "neighbor_radius": 0.001,
                    "minimum": 0.05,
                },
            },
            {
                "id": "snapshot_stale",
                "expected_code": ErrorCode.E_SNAPSHOT_STALE.value,
                "injection": {"expected_version": "current_version - 1"},
            },
            {
                "id": "mesh_mismatch",
                "expected_code": ErrorCode.E_MESH_MISMATCH.value,
                "injection": {
                    "injected_vertex_offset": [0.01, 0.0, 0.0],
                    "tolerance": 1.0e-9,
                },
            },
            {
                "id": "cache_hash",
                "expected_code": ErrorCode.E_CACHE_HASH.value,
                "injection": {"target": "copied_current_state/patches/first.pch", "operation": "xor_byte", "offset": 40, "mask": 1},
            },
            {
                "id": "gl_context",
                "expected_code": ErrorCode.E_GL_CONTEXT.value,
                "injection": {
                    "forced_unavailable_reason": "serialized test parameter disables the GL context"
                },
            },
        ],
    }
    atomic_write_json(output / "fault_injection_config.json", injection_config)
    results: list[dict[str, Any]] = []

    sample_copy = output / "sample_format_corrupt"
    if sample_copy.exists():
        shutil.rmtree(sample_copy)
    shutil.copytree(sample, sample_copy)
    points_payload = bytearray((sample_copy / "points.bin").read_bytes())
    points_payload[0] ^= 0xFF
    (sample_copy / "points.bin").write_bytes(points_payload)
    result = _capture(lambda: load_sample_bundle(sample_copy))
    results.append({"id": "point_cloud_format", "expected": ErrorCode.E_POINT_CLOUD_FORMAT.value, **result})

    bad_radius_operation = {
        "seq": 9001,
        "type": "reestimate_normals",
        "params": {
            "region": {"kind": "sphere", "center": [0.0, 0.0, 0.0], "radius": 0.2},
            "neighbor_radius": 0.001,
        },
    }
    result = _capture(lambda: loaded_engine.apply_operation(bad_radius_operation))
    results.append({"id": "neighbor_radius", "expected": ErrorCode.E_NEIGHBOR_RADIUS.value, **result})

    snapshot_store = SnapshotStore()
    current_scene = snapshot_store.publish(scene_from_engine(loaded_engine))
    stale_version = current_scene.version - 1
    result = _capture(lambda: snapshot_store.get(stale_version))
    results.append({"id": "snapshot_stale", "expected": ErrorCode.E_SNAPSHOT_STALE.value, **result})

    result = _capture(
        lambda: compare_meshes(
            loaded_engine.mesh,
            loaded_engine.mesh,
            loaded_engine.patches,
            tolerance=1.0e-9,
            edit_seq=9004,
            injected_vertex_offset=(0.01, 0.0, 0.0),
            raise_on_mismatch=True,
        )
    )
    results.append({"id": "mesh_mismatch", "expected": ErrorCode.E_MESH_MISMATCH.value, **result})

    cache_case = output / "cache_hash_corrupt"
    _copy_current_state(workspace_path, cache_case)
    cache_files = sorted((cache_case / "states").rglob("*.pch"))
    if not cache_files:
        raise RuntimeError("fault matrix did not find a patch cache to corrupt")
    cache_payload = bytearray(cache_files[0].read_bytes())
    index = min(40, len(cache_payload) - 1)
    cache_payload[index] ^= 1
    cache_files[0].write_bytes(cache_payload)
    cache_store = AtomicStateStore(cache_case)
    cache_store.open()
    result = _capture(cache_store.load_current)
    results.append({"id": "cache_hash", "expected": ErrorCode.E_CACHE_HASH.value, **result})

    forced_viewport = OffscreenViewport(
        SnapshotStore(),
        forced_unavailable_reason="serialized test parameter disables the GL context",
    )
    render_result = forced_viewport.render(width=64, height=64)
    if render_result.ok:
        result = {"ok": True, "value": render_result.to_dict()}
    else:
        result = {
            "ok": False,
            "code": render_result.code,
            "message": render_result.message,
            "context": render_result.context,
        }
    results.append({"id": "gl_context", "expected": ErrorCode.E_GL_CONTEXT.value, **result})

    for item in results:
        item["matched_expected_code"] = item.get("code") == item["expected"] and item.get("ok") is False
    report = {
        "schema": "pointcloud-studio.fault-matrix-report/1",
        "injections": injection_config,
        "results": results,
        "all_matched": all(item["matched_expected_code"] for item in results),
    }
    atomic_write_json(output / "fault_matrix_report.json", report)
    return report

