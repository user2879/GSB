"""Deterministic artifact export for the CLI example and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .engine import SceneRender
from .errors import CoverageError, ErrorCode
from .hashing import file_sha256, pretty_json_text
from .ledger import ledger_hash
from .scene import scene_hash, scene_json_text


ARRAY_SPECS = {
    "coverage_nonzero": ("<f4", "coverage_nonzero.f32"),
    "coverage_evenodd": ("<f4", "coverage_evenodd.f32"),
    "coverage_selected": ("<f4", "coverage_selected.f32"),
    "coverage_difference": ("<f4", "coverage_difference.f32"),
    "dependency_edges": ("<i4", "dependency_edges.i32"),
    "gray": ("<u1", "gray.u8"),
    "quant_error": ("<f4", "quant_error.f32"),
    "reconstruction_error": ("<f4", "reconstruction_error.f32"),
    "layer_bgra": ("<u1", "layer.bgra"),
}


def _write_json(path: Path, payload: Any) -> str:
    text = pretty_json_text(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return file_sha256(path)


def _write_binary(path: Path, array: np.ndarray, dtype: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalised = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    path.write_bytes(normalised.tobytes(order="C"))
    return {
        "file": path.as_posix(),
        "dtype": dtype,
        "shape": list(normalised.shape),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def export_render(render: SceneRender, out_dir: str | Path) -> dict[str, Any]:
    """Write JSON, raw binary buffers and a hash manifest.

    The manifest deliberately contains no timestamps, host paths or random
    identifiers, so repeated exports are byte-identical.
    """

    base = Path(out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    scene_path = base / "scene.json"
    scene_path.write_text(scene_json_text(render.scene, pretty=True), encoding="utf-8", newline="\n")
    (base / "scene.sha256").write_text(scene_hash(render.scene) + "\n", encoding="utf-8", newline="\n")

    ledger_path = base / "ledger.json"
    ledger_file_hash = _write_json(ledger_path, render.ledger)
    (base / "ledger.sha256").write_text(ledger_hash(render.ledger) + "\n", encoding="utf-8", newline="\n")

    composite_path = base / "composite.bgra"
    composite_info = _write_binary(composite_path, render.buffer, "<u1")
    composite_info["file"] = _relative(composite_path, base)

    region_manifest = []
    for region_id in render.region_ids:
        state = render.regions[region_id]
        region_dir = base / "regions" / f"region_{region_id:04d}"
        region_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "coverage_nonzero": state.dual.nonzero.coverage,
            "coverage_evenodd": state.dual.evenodd.coverage,
            "coverage_selected": state.selected_coverage,
            "coverage_difference": state.dual.difference,
            "dependency_edges": state.dual.nonzero.dependency_edge_counts,
            "gray": state.quantization.gray,
            "quant_error": state.quantization.quant_error,
            "reconstruction_error": state.quantization.reconstruction_error,
            "layer_bgra": state.layer.bgra,
        }
        file_info: dict[str, Any] = {}
        for key, array in arrays.items():
            dtype, filename = ARRAY_SPECS[key]
            info = _write_binary(region_dir / filename, array, dtype)
            info["file"] = _relative(region_dir / filename, base)
            file_info[key] = info
        region_manifest.append(
            {
                "region_id": region_id,
                "category": state.region.get("category"),
                "layer_hash": state.layer_hash,
                "layer_content_hash": state.layer.layer_content_hash,
                "files": file_info,
            }
        )

    manifest = {
        "schema": "vgcover.artifact_manifest.v1",
        "scene": {
            "file": "scene.json",
            "sha256": scene_hash(render.scene),
            "file_sha256": file_sha256(scene_path),
        },
        "ledger": {
            "file": "ledger.json",
            "sha256": ledger_hash(render.ledger),
            "file_sha256": ledger_file_hash,
        },
        "composite": {
            "file": "composite.bgra",
            "dtype": "<u1",
            "shape": list(render.buffer.shape),
            "sha256": render.composite.buffer_hash,
            "file_sha256": composite_info["sha256"],
        },
        "regions": region_manifest,
    }
    _write_json(base / "manifest.json", manifest)
    _write_json(base / "summary.json", render.summary())
    return manifest


def verify_export_manifest(out_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    base = Path(out_dir).resolve()
    if manifest is None:
        manifest_path = base / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    checked = 0
    file_entries = [manifest["scene"], manifest["ledger"], manifest["composite"]]
    for region in manifest.get("regions", []):
        file_entries.extend(region.get("files", {}).values())
    for entry in file_entries:
        checked += 1
        path = base / entry["file"]
        if not path.is_file():
            errors.append({"file": entry["file"], "reason": "missing"})
            continue
        actual = file_sha256(path)
        expected = entry.get("file_sha256", entry.get("sha256"))
        if actual != expected:
            errors.append(
                {
                    "file": entry["file"],
                    "reason": "sha256_mismatch",
                    "expected": expected,
                    "actual": actual,
                }
            )
    if errors:
        raise CoverageError(
            ErrorCode.LEDGER_MISMATCH,
            "exported artifact hash verification failed",
            details={"errors": errors},
        )
    return {"ok": True, "checked_files": checked}
