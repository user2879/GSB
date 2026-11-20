"""Incremental editing engine with explicit cache invalidation and auditing."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from .consistency import compare_meshes
from .errors import ErrorCode, StudioError
from .formats import encode_normals, encode_points
from .ledger import build_ledger
from .mesh import (
    CombinedMesh,
    PatchMesh,
    ReconstructionParams,
    assign_patches,
    boundary_neighbor_keys,
    build_all_patches,
    build_patches_for_keys,
    combine_mesh,
)
from .normals import reestimate_normals_in_region, validate_neighbor_radius
from .spatial import select_box, select_sphere
from .util import canonical_json, sha256_bytes
from .viewport import camera_pose_from_dict, select_viewport_box


@dataclass
class CloudData:
    positions: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    neighbor_counts: np.ndarray
    residuals: np.ndarray
    error_bounds: np.ndarray
    source_ids: np.ndarray
    normal_radius: float
    params: ReconstructionParams

    def clone_arrays(self) -> "CloudData":
        return CloudData(
            positions=self.positions.copy(),
            normals=self.normals.copy(),
            confidence=self.confidence.copy(),
            neighbor_counts=self.neighbor_counts.copy(),
            residuals=self.residuals.copy(),
            error_bounds=self.error_bounds.copy(),
            source_ids=self.source_ids.copy(),
            normal_radius=float(self.normal_radius),
            params=replace(self.params),
        )

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(encode_points(self.positions))
        digest.update(encode_normals(self.normals, self.confidence))
        digest.update(canonical_json(self.params.to_dict()).encode("utf-8"))
        digest.update(np.float64(self.normal_radius).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class EngineSnapshotData:
    positions: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    neighbor_counts: np.ndarray
    residuals: np.ndarray
    error_bounds: np.ndarray
    normal_radius: float
    params: ReconstructionParams
    patches: Mapping[str, PatchMesh]
    mesh: CombinedMesh
    ledger: dict[str, Any]
    version: int
    cloud_sha256: str
    operation_count: int


def _region_from_operation(operation_type: str, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if operation_type == "remove_sphere":
        region = {
            "kind": "sphere",
            "center": [float(item) for item in params["center"]],
            "radius": float(params["radius"]),
        }
    elif operation_type == "remove_box":
        region = {
            "kind": "box",
            "bounds_min": [float(item) for item in params["bounds_min"]],
            "bounds_max": [float(item) for item in params["bounds_max"]],
        }
    elif operation_type == "reestimate_normals":
        region = dict(params["region"])
        if region.get("kind") == "sphere":
            region["center"] = [float(item) for item in region["center"]]
            region["radius"] = float(region["radius"])
        elif region.get("kind") == "box":
            region["bounds_min"] = [float(item) for item in region["bounds_min"]]
            region["bounds_max"] = [float(item) for item in region["bounds_max"]]
        else:
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "unsupported edit region kind",
                context={"kind": region.get("kind")},
            )
    else:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "operation has no local edit region",
            context={"operation_type": operation_type},
        )
    if region["kind"] == "sphere":
        if len(region["center"]) != 3 or not all(math.isfinite(item) for item in region["center"]):
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "sphere center must contain three finite coordinates",
                context={"center": region["center"]},
            )
        if not math.isfinite(region["radius"]) or region["radius"] <= 0.0:
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "edit sphere radius must be positive and finite",
                context={"radius": region["radius"]},
            )
    else:
        lower = np.asarray(region["bounds_min"], dtype=np.float64)
        upper = np.asarray(region["bounds_max"], dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "box bounds must contain three finite coordinates",
                context={"bounds_min": region["bounds_min"], "bounds_max": region["bounds_max"]},
            )
        if np.any(lower > upper):
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "box minimum bound exceeds maximum bound",
                context={"bounds_min": region["bounds_min"], "bounds_max": region["bounds_max"]},
            )
    return region["kind"], region


def _point_key_map(keys: list[str], groups: list[np.ndarray], count: int) -> np.ndarray:
    mapping = np.empty(count, dtype=object)
    for key, group in zip(keys, groups, strict=True):
        mapping[group] = key
    return mapping


def _sphere_patch_keys(patches: Mapping[str, PatchMesh], center: np.ndarray, radius: float, support: float) -> set[str]:
    threshold = float(radius + support)
    result: set[str] = set()
    for key, patch in patches.items():
        if patch.vertices.shape[0] == 0:
            continue
        closest = np.clip(center, patch.aabb_min, patch.aabb_max)
        if float(np.linalg.norm(closest - center)) <= threshold:
            result.add(key)
    return result


def _box_patch_keys(
    patches: Mapping[str, PatchMesh],
    lower: np.ndarray,
    upper: np.ndarray,
    support: float,
) -> set[str]:
    expanded_lower = lower - support
    expanded_upper = upper + support
    result: set[str] = set()
    for key, patch in patches.items():
        if patch.vertices.shape[0] == 0:
            continue
        if np.all(patch.aabb_max >= expanded_lower) and np.all(patch.aabb_min <= expanded_upper):
            result.add(key)
    return result


class StudioEngine:
    """Owns mutable CPU state; callers receive immutable snapshots only."""

    def __init__(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        confidence: np.ndarray,
        *,
        params: ReconstructionParams | None = None,
        neighbor_radius: float = 0.35,
        source_ids: np.ndarray | None = None,
    ) -> None:
        params = params or ReconstructionParams()
        params.validate()
        radius = validate_neighbor_radius(neighbor_radius)
        count = int(positions.shape[0])
        if positions.shape != (count, 3) or normals.shape != (count, 3) or confidence.shape != (count,):
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "cloud array shapes are inconsistent",
                context={
                    "positions": list(positions.shape),
                    "normals": list(normals.shape),
                    "confidence": list(confidence.shape),
                },
            )
        if source_ids is None:
            source_ids = np.arange(count, dtype=np.int64)
        cloud = CloudData(
            positions=np.asarray(positions, dtype=np.float32).copy(),
            normals=np.asarray(normals, dtype=np.float32).copy(),
            confidence=np.asarray(confidence, dtype=np.float32).copy(),
            neighbor_counts=np.full(count, -1, dtype=np.int32),
            residuals=np.full(count, np.nan, dtype=np.float32),
            error_bounds=np.full(count, params.initial_normal_error_bound_radians, dtype=np.float32),
            source_ids=np.asarray(source_ids, dtype=np.int64).copy(),
            normal_radius=radius,
            params=params,
        )
        patches, keys, groups = build_all_patches(
            cloud.positions, cloud.normals, cloud.params, cloud.normal_radius
        )
        self.cloud = cloud
        self.patches: dict[str, PatchMesh] = patches
        self.mesh = combine_mesh(patches)
        self.ledger = self._build_ledger(self.cloud, self.patches, self.mesh)
        self.version = 1
        self.history: list[dict[str, Any]] = []
        self.last_invalidation: dict[str, Any] | None = None
        self.last_comparison: dict[str, Any] | None = None
        self.last_normal_audit: dict[str, Any] | None = None
        self.assignment_keys = keys
        self.assignment_groups = groups

    @classmethod
    def restore(
        cls,
        positions: np.ndarray,
        normals: np.ndarray,
        confidence: np.ndarray,
        neighbor_counts: np.ndarray,
        residuals: np.ndarray,
        error_bounds: np.ndarray,
        source_ids: np.ndarray,
        *,
        normal_radius: float,
        params: ReconstructionParams,
        patches: Mapping[str, PatchMesh],
        mesh: CombinedMesh,
        ledger: dict[str, Any],
        version: int,
        history: list[dict[str, Any]],
    ) -> "StudioEngine":
        """Restore a committed state without rebuilding its cached patches."""

        params.validate()
        radius = validate_neighbor_radius(normal_radius)
        count = int(positions.shape[0])
        engine = cls.__new__(cls)
        engine.cloud = CloudData(
            positions=np.asarray(positions, dtype=np.float32).copy(),
            normals=np.asarray(normals, dtype=np.float32).copy(),
            confidence=np.asarray(confidence, dtype=np.float32).copy(),
            neighbor_counts=np.asarray(neighbor_counts, dtype=np.int32).copy(),
            residuals=np.asarray(residuals, dtype=np.float32).copy(),
            error_bounds=np.asarray(error_bounds, dtype=np.float32).copy(),
            source_ids=np.asarray(source_ids, dtype=np.int64).copy(),
            normal_radius=radius,
            params=params,
        )
        if engine.cloud.positions.shape[0] != count:
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "restored cloud array shapes are inconsistent",
                context={"point_count": count},
            )
        engine.patches = dict(patches)
        engine.mesh = mesh
        engine.ledger = dict(ledger)
        engine.version = int(version)
        engine.history = list(history)
        engine.last_invalidation = None
        engine.last_comparison = None
        engine.last_normal_audit = None
        engine.assignment_keys, engine.assignment_groups = assign_patches(
            engine.cloud.positions, engine.cloud.normals, engine.cloud.params
        )
        return engine

    def _build_ledger(
        self,
        cloud: CloudData,
        patches: Mapping[str, PatchMesh],
        mesh: CombinedMesh,
    ) -> dict[str, Any]:
        return build_ledger(
            cloud.positions,
            cloud.normals,
            cloud.confidence,
            patches,
            mesh,
            cloud.params,
            cloud.normal_radius,
        )

    def _build_candidate(
        self,
        cloud: CloudData,
        *,
        roots: set[str],
        forced_keys: set[str] | None = None,
        reason: str,
        operation: Mapping[str, Any],
        tolerance_override: float | None,
        injected_vertex_offset: tuple[float, float, float] | None,
    ) -> tuple[CloudData, dict[str, PatchMesh], CombinedMesh, dict[str, Any], dict[str, Any], dict[str, Any]]:
        keys, groups = assign_patches(cloud.positions, cloud.normals, cloud.params)
        group_map = dict(zip(keys, groups, strict=True))
        universe = set(self.patches) | set(keys) | roots
        closure = set(roots)
        for root in sorted(roots, key=lambda item: item.encode("ascii")):
            closure |= boundary_neighbor_keys(root, universe)
        if forced_keys is not None:
            closure |= set(forced_keys)

        recomputed_attempts = sorted(closure, key=lambda item: item.encode("ascii"))
        built = build_patches_for_keys(
            recomputed_attempts,
            group_map,
            cloud.positions,
            cloud.normals,
            cloud.params,
            cloud.normal_radius,
        )
        candidate_patches = {
            key: patch for key, patch in self.patches.items() if key not in closure
        }
        candidate_patches.update(built)
        reused_keys = sorted(set(self.patches) - closure, key=lambda item: item.encode("ascii"))
        for key in reused_keys:
            if candidate_patches.get(key) is not self.patches[key]:
                raise StudioError(
                    ErrorCode.E_MESH_MISMATCH,
                    "unaffected patch cache object was replaced",
                    context={"patch_key": key},
                )

        candidate_mesh = combine_mesh(candidate_patches)
        candidate_ledger = self._build_ledger(cloud, candidate_patches, candidate_mesh)

        full_patches, _, _ = build_all_patches(
            cloud.positions, cloud.normals, cloud.params, cloud.normal_radius
        )
        full_mesh = combine_mesh(full_patches)
        tolerance = (
            float(tolerance_override)
            if tolerance_override is not None
            else cloud.params.position_tolerance
        )
        comparison = compare_meshes(
            candidate_mesh,
            full_mesh,
            candidate_patches,
            tolerance=tolerance,
            edit_seq=int(operation.get("seq", len(self.history) + 1)),
            injected_vertex_offset=injected_vertex_offset,
            raise_on_mismatch=True,
        )
        invalidation = {
            "reason": reason,
            "support_radius": cloud.params.support_radius,
            "root_patch_keys": sorted(roots, key=lambda item: item.encode("ascii")),
            "closure_patch_keys": recomputed_attempts,
            "recomputed_patch_keys": recomputed_attempts,
            "reused_patch_keys": reused_keys,
            "closure_equals_recomputed": recomputed_attempts == sorted(
                closure, key=lambda item: item.encode("ascii")
            ),
            "whole_model_rebuild": False,
            "recomputed_patch_count": len(recomputed_attempts),
            "reused_patch_count": len(reused_keys),
            "nonempty_patch_count": len(candidate_patches),
            "unaffected_vertices_bitwise_reused": True,
        }
        mesh_comparison = {
            **comparison,
            "production_rebuild_mode": "incremental-local-patches",
            "full_rebuild_used_for_production_result": False,
            "full_rebuild_used_only_for_consistency_check": True,
            "full_patch_count": len(full_patches),
        }
        return cloud, candidate_patches, candidate_mesh, candidate_ledger, invalidation, mesh_comparison

    def _commit(
        self,
        operation: dict[str, Any],
        cloud: CloudData,
        patches: dict[str, PatchMesh],
        mesh: CombinedMesh,
        ledger: dict[str, Any],
        invalidation: dict[str, Any],
        comparison: dict[str, Any],
        *,
        normal_audit: dict[str, Any] | None,
        selected_count: int,
        point_count_before: int,
        point_count_after: int,
    ) -> dict[str, Any]:
        before_version = self.version
        before_hash = self.cloud.content_sha256()
        self.cloud = cloud
        self.patches = patches
        self.mesh = mesh
        self.ledger = ledger
        self.version += 1
        self.last_invalidation = invalidation
        self.last_comparison = comparison
        self.last_normal_audit = normal_audit
        record = {
            "seq": int(operation.get("seq", len(self.history) + 1)),
            "operation_type": str(operation["type"]),
            "parameters": operation.get("params", {}),
            "status": "committed",
            "before_version": before_version,
            "after_version": self.version,
            "before_cloud_sha256": before_hash,
            "after_cloud_sha256": cloud.content_sha256(),
            "point_count_before": point_count_before,
            "point_count_after": point_count_after,
            "selected_point_count": int(selected_count),
            "normal_radius": float(cloud.normal_radius),
            "normal_audit": normal_audit,
            "invalidation": invalidation,
            "consistency_check": comparison,
            "mesh_content_sha256": mesh.content_sha256,
            "ledger_sha256": ledger["ledger_sha256"],
        }
        self.history.append(record)
        return record

    def apply_operation(
        self,
        operation: Mapping[str, Any],
        *,
        tolerance_override: float | None = None,
        injected_vertex_offset: tuple[float, float, float] | None = None,
    ) -> dict[str, Any]:
        operation_type = str(operation.get("type"))
        raw_params = dict(operation.get("params", {}))
        seq = int(operation.get("seq", len(self.history) + 1))
        normalized_operation = {"seq": seq, "type": operation_type, "params": raw_params}
        point_count_before = int(self.cloud.positions.shape[0])

        if operation_type in {"remove_sphere", "remove_box", "remove_viewport_box"}:
            if operation_type == "remove_viewport_box":
                kind = "viewport"
                region = {"kind": "viewport"}
                try:
                    camera = camera_pose_from_dict(raw_params["camera"])
                    width = int(raw_params["width"])
                    height = int(raw_params["height"])
                    rect = [float(item) for item in raw_params["rect"]]
                except (KeyError, TypeError, ValueError) as exc:
                    raise StudioError(
                        ErrorCode.E_POINT_CLOUD_FORMAT,
                        "viewport box operation parameters are malformed",
                        context={"reason": str(exc), "params": raw_params},
                    ) from exc
                selected = select_viewport_box(
                    self.cloud.positions, camera, width, height, rect
                )
                # Screen-space selection already identifies the exact point
                # set.  Do not expand it into a world AABB, which would wrongly
                # include unrelated points along the camera ray.
                spatial_roots: set[str] = set()
            else:
                kind, region = _region_from_operation(operation_type, raw_params)
            old_keys, old_groups = assign_patches(
                self.cloud.positions, self.cloud.normals, self.cloud.params
            )
            old_map = _point_key_map(old_keys, old_groups, point_count_before)
            if kind == "viewport":
                pass
            elif kind == "sphere":
                center = np.asarray(region["center"], dtype=np.float64)
                selected = select_sphere(self.cloud.positions, center, region["radius"])
                spatial_roots = _sphere_patch_keys(
                    self.patches, center, region["radius"], self.cloud.params.support_radius
                )
            else:
                lower = np.asarray(region["bounds_min"], dtype=np.float64)
                upper = np.asarray(region["bounds_max"], dtype=np.float64)
                selected = select_box(self.cloud.positions, lower, upper)
                spatial_roots = _box_patch_keys(
                    self.patches, lower, upper, self.cloud.params.support_radius
                )
            roots = set(spatial_roots)
            if selected.size:
                roots |= {str(key) for key in old_map[selected] if key is not None}
            keep = np.ones(point_count_before, dtype=bool)
            keep[selected] = False
            candidate = CloudData(
                positions=self.cloud.positions[keep].copy(),
                normals=self.cloud.normals[keep].copy(),
                confidence=self.cloud.confidence[keep].copy(),
                neighbor_counts=self.cloud.neighbor_counts[keep].copy(),
                residuals=self.cloud.residuals[keep].copy(),
                error_bounds=self.cloud.error_bounds[keep].copy(),
                source_ids=self.cloud.source_ids[keep].copy(),
                normal_radius=self.cloud.normal_radius,
                params=replace(self.cloud.params),
            )
            (
                candidate,
                patches,
                mesh,
                ledger,
                invalidation,
                comparison,
            ) = self._build_candidate(
                candidate,
                roots=roots,
                reason="local_point_removal",
                operation=normalized_operation,
                tolerance_override=tolerance_override,
                injected_vertex_offset=injected_vertex_offset,
            )
            record = self._commit(
                normalized_operation,
                candidate,
                patches,
                mesh,
                ledger,
                invalidation,
                comparison,
                normal_audit=None,
                selected_count=int(selected.size),
                point_count_before=point_count_before,
                point_count_after=int(candidate.positions.shape[0]),
            )
            self.assignment_keys, self.assignment_groups = assign_patches(
                candidate.positions, candidate.normals, candidate.params
            )
            return record

        if operation_type == "reestimate_normals":
            kind, region = _region_from_operation(operation_type, raw_params)
            neighbor_radius = validate_neighbor_radius(raw_params["neighbor_radius"])
            old_keys, old_groups = assign_patches(
                self.cloud.positions, self.cloud.normals, self.cloud.params
            )
            old_map = _point_key_map(old_keys, old_groups, point_count_before)
            if kind == "sphere":
                center = np.asarray(region["center"], dtype=np.float64)
                selected = select_sphere(self.cloud.positions, center, region["radius"])
                spatial_roots = _sphere_patch_keys(
                    self.patches, center, region["radius"], self.cloud.params.support_radius
                )
            else:
                lower = np.asarray(region["bounds_min"], dtype=np.float64)
                upper = np.asarray(region["bounds_max"], dtype=np.float64)
                selected = select_box(self.cloud.positions, lower, upper)
                spatial_roots = _box_patch_keys(
                    self.patches, lower, upper, self.cloud.params.support_radius
                )
            before_normals = self.cloud.normals.copy()
            before_confidence = self.cloud.confidence.copy()
            normal_result = reestimate_normals_in_region(
                self.cloud.positions,
                self.cloud.normals,
                self.cloud.confidence,
                self.cloud.error_bounds,
                selected,
                neighbor_radius,
                previous_neighbor_counts=self.cloud.neighbor_counts,
                previous_residuals=self.cloud.residuals,
            )
            outside = np.ones(point_count_before, dtype=bool)
            outside[selected] = False
            if not np.array_equal(normal_result.normals[outside], before_normals[outside]) or not np.array_equal(
                normal_result.confidence[outside], before_confidence[outside]
            ):
                raise StudioError(
                    ErrorCode.E_MESH_MISMATCH,
                    "normal re-estimation modified points outside its selected region",
                    context={"selected_count": int(selected.size)},
                )
            candidate = CloudData(
                positions=self.cloud.positions.copy(),
                normals=normal_result.normals,
                confidence=normal_result.confidence,
                neighbor_counts=normal_result.neighbor_counts,
                residuals=normal_result.residuals,
                error_bounds=normal_result.error_bounds,
                source_ids=self.cloud.source_ids.copy(),
                normal_radius=neighbor_radius,
                params=replace(self.cloud.params),
            )
            new_keys, new_groups = assign_patches(candidate.positions, candidate.normals, candidate.params)
            new_map = _point_key_map(new_keys, new_groups, point_count_before)
            roots = set(spatial_roots)
            if selected.size:
                roots |= {str(key) for key in old_map[selected] if key is not None}
                roots |= {str(key) for key in new_map[selected] if key is not None}
            (
                candidate,
                patches,
                mesh,
                ledger,
                invalidation,
                comparison,
            ) = self._build_candidate(
                candidate,
                roots=roots,
                reason="local_normal_reestimate",
                operation=normalized_operation,
                tolerance_override=tolerance_override,
                injected_vertex_offset=injected_vertex_offset,
            )
            audit = {
                **normal_result.audit(),
                "outside_selection_normals_bitwise_unchanged": True,
                "outside_selection_confidence_bitwise_unchanged": True,
            }
            record = self._commit(
                normalized_operation,
                candidate,
                patches,
                mesh,
                ledger,
                invalidation,
                comparison,
                normal_audit=audit,
                selected_count=int(selected.size),
                point_count_before=point_count_before,
                point_count_after=point_count_before,
            )
            self.assignment_keys, self.assignment_groups = new_keys, new_groups
            return record

        if operation_type == "set_recon_param":
            name = str(raw_params.get("name"))
            value = float(raw_params.get("value"))
            allowed = {
                "triangulation_cell_size",
                "patch_size",
                "support_radius",
                "position_tolerance",
            }
            if name not in allowed or not math.isfinite(value) or value <= 0.0:
                raise StudioError(
                    ErrorCode.E_POINT_CLOUD_FORMAT,
                    "invalid reconstruction parameter adjustment",
                    context={"name": name, "value": value, "allowed": sorted(allowed)},
                )
            params = replace(self.cloud.params, **{name: value})
            params.validate()
            candidate = self.cloud.clone_arrays()
            candidate.params = params
            old_keys = set(self.patches)
            new_keys, groups = assign_patches(candidate.positions, candidate.normals, candidate.params)
            group_map = dict(zip(new_keys, groups, strict=True))
            closure = old_keys | set(new_keys)
            attempts = sorted(closure, key=lambda item: item.encode("ascii"))
            built = build_patches_for_keys(
                attempts,
                group_map,
                candidate.positions,
                candidate.normals,
                candidate.params,
                candidate.normal_radius,
            )
            patches = dict(built)
            mesh = combine_mesh(patches)
            ledger = self._build_ledger(candidate, patches, mesh)
            full_patches, _, _ = build_all_patches(
                candidate.positions, candidate.normals, candidate.params, candidate.normal_radius
            )
            full_mesh = combine_mesh(full_patches)
            tolerance = (
                float(tolerance_override)
                if tolerance_override is not None
                else candidate.params.position_tolerance
            )
            comparison = {
                **compare_meshes(
                    mesh,
                    full_mesh,
                    patches,
                    tolerance=tolerance,
                    edit_seq=seq,
                    injected_vertex_offset=injected_vertex_offset,
                    raise_on_mismatch=True,
                ),
                "production_rebuild_mode": "explicit-global-parameter-replacement",
                "full_rebuild_used_for_production_result": True,
                "full_rebuild_used_only_for_consistency_check": False,
                "full_patch_count": len(full_patches),
            }
            invalidation = {
                "reason": "global_reconstruction_parameter_change",
                "support_radius": candidate.params.support_radius,
                "root_patch_keys": sorted(old_keys, key=lambda item: item.encode("ascii")),
                "closure_patch_keys": attempts,
                "recomputed_patch_keys": attempts,
                "reused_patch_keys": [],
                "closure_equals_recomputed": True,
                "whole_model_rebuild": True,
                "recomputed_patch_count": len(attempts),
                "reused_patch_count": 0,
                "nonempty_patch_count": len(patches),
                "unaffected_vertices_bitwise_reused": True,
            }
            record = self._commit(
                normalized_operation,
                candidate,
                patches,
                mesh,
                ledger,
                invalidation,
                comparison,
                normal_audit=None,
                selected_count=0,
                point_count_before=point_count_before,
                point_count_after=point_count_before,
            )
            self.assignment_keys, self.assignment_groups = new_keys, groups
            return record

        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "unsupported operation type",
            context={
                "operation_type": operation_type,
                "known": [
                    "remove_sphere",
                    "remove_box",
                    "remove_viewport_box",
                    "reestimate_normals",
                    "set_recon_param",
                ],
            },
        )

    def snapshot_data(self) -> EngineSnapshotData:
        positions = self.cloud.positions.copy()
        normals = self.cloud.normals.copy()
        confidence = self.cloud.confidence.copy()
        counts = self.cloud.neighbor_counts.copy()
        residuals = self.cloud.residuals.copy()
        bounds = self.cloud.error_bounds.copy()
        for array in (positions, normals, confidence, counts, residuals, bounds):
            array.setflags(write=False)
        return EngineSnapshotData(
            positions=positions,
            normals=normals,
            confidence=confidence,
            neighbor_counts=counts,
            residuals=residuals,
            error_bounds=bounds,
            normal_radius=float(self.cloud.normal_radius),
            params=replace(self.cloud.params),
            patches=dict(self.patches),
            mesh=self.mesh,
            ledger=self.ledger,
            version=self.version,
            cloud_sha256=self.cloud.content_sha256(),
            operation_count=len(self.history),
        )

    def checkpoint(self) -> dict[str, Any]:
        """Capture mutable engine state for commit-failure rollback."""

        return {
            "cloud": self.cloud.clone_arrays(),
            "patches": dict(self.patches),
            "mesh": self.mesh,
            "ledger": self.ledger,
            "version": self.version,
            "history": list(self.history),
            "last_invalidation": self.last_invalidation,
            "last_comparison": self.last_comparison,
            "last_normal_audit": self.last_normal_audit,
            "assignment_keys": list(self.assignment_keys),
            "assignment_groups": [group.copy() for group in self.assignment_groups],
        }

    def rollback(self, checkpoint: Mapping[str, Any]) -> None:
        self.cloud = checkpoint["cloud"].clone_arrays()
        self.patches = dict(checkpoint["patches"])
        self.mesh = checkpoint["mesh"]
        self.ledger = checkpoint["ledger"]
        self.version = int(checkpoint["version"])
        self.history = list(checkpoint["history"])
        self.last_invalidation = checkpoint["last_invalidation"]
        self.last_comparison = checkpoint["last_comparison"]
        self.last_normal_audit = checkpoint["last_normal_audit"]
        self.assignment_keys = list(checkpoint["assignment_keys"])
        self.assignment_groups = [group.copy() for group in checkpoint["assignment_groups"]]
