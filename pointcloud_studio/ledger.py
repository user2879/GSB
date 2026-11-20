"""Recomputable geometric error ledger for reconstruction patches."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .errors import ErrorCode, StudioError
from .formats import encode_normals, encode_points
from .mesh import (
    CombinedMesh,
    PatchMesh,
    ReconstructionParams,
    build_all_patches,
    combine_mesh,
    patch_cache_name,
)
from .util import canonical_json, sha256_bytes

LEDGER_SCHEMA = "pointcloud-studio.error-ledger/1"
FORMULA = "position_bound = quantization_bound + 2*radius*sin(normal_angle_bound) + chord_height"


def _entry(patch: PatchMesh) -> dict[str, Any]:
    return {
        "cache_file": f"patches/{patch_cache_name(patch.key)}",
        "content_sha256": patch.content_sha256,
        "source_points_sha256": patch.source_points_sha256,
        "source_point_count": patch.point_count,
        "vertex_count": int(patch.vertices.shape[0]),
        "triangle_count": int(patch.triangles.shape[0]),
        "quantization_bound": patch.quantization_bound,
        "normal_fit_residual": patch.normal_fit_residual,
        "normal_angle_bound_radians": patch.normal_angle_bound,
        "normal_position_bound": patch.normal_position_bound,
        "chord_height": patch.chord_height,
        "position_error_upper_bound": patch.position_error_upper_bound,
    }


def build_ledger(
    points: np.ndarray,
    normals: np.ndarray,
    confidence: np.ndarray,
    patches: Mapping[str, PatchMesh],
    mesh: CombinedMesh,
    params: ReconstructionParams,
    neighbor_radius: float,
) -> dict[str, Any]:
    entries = {
        key: _entry(patch)
        for key, patch in sorted(patches.items(), key=lambda item: item[0].encode("ascii"))
    }
    if entries:
        worst_key = max(entries, key=lambda key: entries[key]["position_error_upper_bound"])
        network_bound = float(entries[worst_key]["position_error_upper_bound"])
    else:
        worst_key = None
        network_bound = 0.0
    points_hash = sha256_bytes(encode_points(points))
    normals_hash = sha256_bytes(encode_normals(normals, confidence))
    core = {
        "schema": LEDGER_SCHEMA,
        "neighbor_radius": float(neighbor_radius),
        "parameters": {
            **params.to_dict(),
            "points_content_sha256": points_hash,
            "normals_content_sha256": normals_hash,
            "triangulation_cell_origin": [float(item) for item in params.layout_origin],
        },
        "metric": {
            "formula": FORMULA,
            "quantization": "max_i norm(0.5*spacing(float32 coordinate))",
            "normal_fit_residual": "sqrt(min_eigenvalue(covariance(points_in_patch)))",
            "normal_angle_bound": "max(initial_bound, atan2(residual+quantization, 0.5*neighbor_radius))",
            "normal_position_bound": "2*neighbor_radius*sin(normal_angle_bound)",
            "chord_height": "max perpendicular deviation of source points from incident-triangle local planes",
            "combination": "pointwise Euclidean upper bounds added by triangle inequality",
        },
        "patch_count": len(entries),
        "entries": entries,
        "network": {
            "position_error_upper_bound": network_bound,
            "worst_patch": worst_key,
            "aggregation": "max over patch upper bounds",
        },
        "mesh": {
            "content_sha256": mesh.content_sha256,
            "patch_keys": list(mesh.patch_keys),
            "vertex_count": mesh.vertex_count,
            "triangle_count": mesh.triangle_count,
        },
    }
    core["ledger_sha256"] = sha256_bytes(canonical_json(core).encode("utf-8"))
    return core


def _compare_number(name: str, expected: float, actual: float, key: str, tolerances: dict[str, float]) -> float:
    difference = abs(float(expected) - float(actual))
    tolerance = tolerances.get(name, 1.0e-12)
    if difference > tolerance:
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "independent ledger recomputation disagrees",
            context={
                "patch_key": key,
                "field": name,
                "ledger_value": float(expected),
                "recomputed_value": float(actual),
                "absolute_difference": difference,
                "tolerance": tolerance,
            },
        )
    return difference


def verify_ledger(
    points: np.ndarray,
    normals: np.ndarray,
    confidence: np.ndarray,
    ledger: Mapping[str, Any],
    *,
    check_content_hashes: bool = True,
) -> dict[str, Any]:
    """Recompute every ledger quantity from point/normal data and parameters."""

    verify_ledger_digest(ledger)
    try:
        parameters = ledger["parameters"]
        params = ReconstructionParams.from_dict(parameters)
        radius = float(ledger["neighbor_radius"])
        entries = ledger["entries"]
    except (KeyError, TypeError, ValueError) as exc:
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "ledger parameters are incomplete",
            context={"reason": str(exc)},
        ) from exc

    expected_points_hash = sha256_bytes(encode_points(points))
    if parameters.get("points_content_sha256") != expected_points_hash:
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "ledger was built for a different point cloud",
            context={
                "ledger_points_sha256": parameters.get("points_content_sha256"),
                "actual_points_sha256": expected_points_hash,
            },
        )
    expected_normals_hash = sha256_bytes(encode_normals(normals, confidence))
    if parameters.get("normals_content_sha256") != expected_normals_hash:
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "ledger was built for a different normal file",
            context={
                "ledger_normals_sha256": parameters.get("normals_content_sha256"),
                "actual_normals_sha256": expected_normals_hash,
            },
        )
    recomputed_patches, _, _ = build_all_patches(points, normals, params, radius)
    recomputed_mesh = combine_mesh(recomputed_patches)

    ledger_keys = set(entries)
    recomputed_keys = set(recomputed_patches)
    if ledger_keys != recomputed_keys:
        missing = sorted(recomputed_keys - ledger_keys, key=lambda item: item.encode("ascii"))
        extra = sorted(ledger_keys - recomputed_keys, key=lambda item: item.encode("ascii"))
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "ledger patch set differs from independently reconstructed patches",
            context={"missing_from_ledger": missing[:10], "extra_in_ledger": extra[:10]},
        )

    number_tolerances = {
        "quantization_bound": 1.0e-12,
        "normal_fit_residual": 1.0e-12,
        "normal_angle_bound_radians": 1.0e-12,
        "normal_position_bound": 1.0e-12,
        "chord_height": 1.0e-12,
        "position_error_upper_bound": 1.0e-12,
    }
    maximum_difference = 0.0
    for key in sorted(entries, key=lambda item: item.encode("ascii")):
        entry = entries[key]
        patch = recomputed_patches[key]
        if check_content_hashes and entry.get("content_sha256") != patch.content_sha256:
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "ledger content hash does not match recomputed patch",
                context={
                    "patch_key": key,
                    "ledger_hash": entry.get("content_sha256"),
                    "recomputed_hash": patch.content_sha256,
                },
            )
        if entry.get("source_points_sha256") != patch.source_points_sha256:
            raise StudioError(
                ErrorCode.E_MESH_MISMATCH,
                "ledger source-point hash does not match patch membership",
                context={"patch_key": key, "ledger": entry.get("source_points_sha256"), "actual": patch.source_points_sha256},
            )
        for field, value in (
            ("quantization_bound", patch.quantization_bound),
            ("normal_fit_residual", patch.normal_fit_residual),
            ("normal_angle_bound_radians", patch.normal_angle_bound),
            ("normal_position_bound", patch.normal_position_bound),
            ("chord_height", patch.chord_height),
            ("position_error_upper_bound", patch.position_error_upper_bound),
        ):
            maximum_difference = max(
                maximum_difference,
                _compare_number(field, float(entry[field]), float(value), key, number_tolerances),
            )

    if ledger.get("mesh", {}).get("content_sha256") != recomputed_mesh.content_sha256:
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "ledger mesh hash differs from independently rebuilt mesh",
            context={
                "ledger_hash": ledger.get("mesh", {}).get("content_sha256"),
                "recomputed_hash": recomputed_mesh.content_sha256,
            },
        )
    return {
        "ok": True,
        "patch_count": len(entries),
        "vertex_count": recomputed_mesh.vertex_count,
        "triangle_count": recomputed_mesh.triangle_count,
        "ledger_sha256": ledger.get("ledger_sha256"),
        "mesh_content_sha256": recomputed_mesh.content_sha256,
        "maximum_numeric_difference": maximum_difference,
        "network_position_error_upper_bound": float(ledger["network"]["position_error_upper_bound"]),
    }


def verify_ledger_files(
    points_path: str,
    normals_path: str,
    ledger_path: str,
) -> dict[str, Any]:
    """Third-party style verification from only the two binary files and ledger."""

    import json
    from pathlib import Path

    from .formats import load_normals, load_points

    points = load_points(points_path)
    normals, confidence = load_normals(normals_path)
    with Path(ledger_path).open("r", encoding="utf-8") as handle:
        ledger = json.load(handle)
    return verify_ledger(points, normals, confidence, ledger)


def ledger_digest(ledger: Mapping[str, Any]) -> str:
    without_digest = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    return sha256_bytes(canonical_json(without_digest).encode("utf-8"))


def verify_ledger_digest(ledger: Mapping[str, Any]) -> bool:
    expected = ledger.get("ledger_sha256")
    actual = ledger_digest(ledger)
    if expected != actual:
        raise StudioError(
            ErrorCode.E_MESH_MISMATCH,
            "ledger digest mismatch",
            context={"expected": expected, "actual": actual},
        )
    return True
