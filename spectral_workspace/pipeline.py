"""Stage-level dependency graph and incremental pipeline execution."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical import qfloat, sha256_obj
from .color import adapt_xyz, spectral_to_xyz, xyz_to_target
from .errors import DomainError, E_INJECT_CONFIG, E_STAGE_STALE
from .model import Scene
from .standards import INTEGRATION_STEP_NM, WAVELENGTH_GRID_HASH, cmf_hash


PIPELINE_VERSION = "spectral-pipeline/v1"
STAGE_SCHEMA = "spectral-stage/v1"

STAGE_KIND_ORDER = {
    "spectral_xyz": 0,
    "adapted_xyz": 1,
    "material_color": 2,
}


def spectral_stage_id(material_id: str, light_id: str) -> str:
    return f"spectral_xyz:{material_id}:{light_id}"


def adapted_stage_id(material_id: str, light_id: str) -> str:
    return f"adapted_xyz:{material_id}:{light_id}"


def material_stage_id(material_id: str) -> str:
    return f"material_color:{material_id}"


def stage_content_hash(record: Mapping[str, Any]) -> str:
    return sha256_obj(
        {
            "schema": STAGE_SCHEMA,
            "stage_id": record.get("stage_id"),
            "kind": record.get("kind"),
            "material_id": record.get("material_id"),
            "light_id": record.get("light_id"),
            "dependency_signature": record.get("dependency_signature"),
            "product": record.get("product"),
        }
    )


def make_stage_record(
    *,
    stage_id: str,
    kind: str,
    material_id: str,
    light_id: str | None,
    dependency_signature: Mapping[str, Any],
    product: Mapping[str, Any],
    revision: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": STAGE_SCHEMA,
        "stage_id": stage_id,
        "kind": kind,
        "material_id": material_id,
        "light_id": light_id,
        "dependency_signature": copy.deepcopy(dependency_signature),
        "dependency_hash": sha256_obj(dependency_signature),
        "computed_revision": revision,
        "product": copy.deepcopy(product),
    }
    record["content_hash"] = stage_content_hash(record)
    return record


@dataclass
class PipelineResult:
    stages: dict[str, dict[str, Any]]
    recomputed: list[dict[str, Any]]
    reused: list[dict[str, Any]]
    removed: list[str]
    dirty_closure: list[str]

    def delta(self, *, revision: int, operation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "revision": revision,
            "operation": operation.get("op"),
            "operation_hash": sha256_obj(operation),
            "dirty_closure": list(self.dirty_closure),
            "recomputed": copy.deepcopy(self.recomputed),
            "reused": copy.deepcopy(self.reused),
            "removed_stage_ids": list(self.removed),
            "recomputed_count": len(self.recomputed),
            "reused_count": len(self.reused),
        }


def _stage_sort_key(stage_id: str) -> tuple[int, str]:
    kind = stage_id.split(":", 1)[0]
    return (STAGE_KIND_ORDER.get(kind, 99), stage_id)


def _stage_reuse_entry(stage_id: str, record: Mapping[str, Any], *, reused: bool) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "source_version": int(record["computed_revision"]),
        "content_hash": str(record["content_hash"]),
        "reused": reused,
    }


def _stage_record_is_valid(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("content_hash"), str)
        and record.get("schema") == STAGE_SCHEMA
        and stage_content_hash(record) == record.get("content_hash")
    )


def _camera_white_hash(scene: Scene) -> str:
    return sha256_obj(list(scene.config.camera_white_point_xyz))


def build_pipeline(
    scene: Scene,
    old_stages: Mapping[str, Any] | None,
    *,
    revision: int,
    injection: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Build or incrementally reuse the stage graph for ``scene``."""

    old = old_stages if old_stages is not None else {}
    stages: dict[str, dict[str, Any]] = {}
    recomputed: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []

    def reuse_or_compute(
        *,
        stage_id: str,
        kind: str,
        material_id: str,
        light_id: str | None,
        signature: Mapping[str, Any],
        compute,
    ) -> dict[str, Any]:
        cached = old.get(stage_id)
        if (
            _stage_record_is_valid(cached)
            and cached.get("dependency_signature") == signature
        ):
            stages[stage_id] = cached
            reused.append(_stage_reuse_entry(stage_id, cached, reused=True))
            return cached
        product = compute()
        record = make_stage_record(
            stage_id=stage_id,
            kind=kind,
            material_id=material_id,
            light_id=light_id,
            dependency_signature=signature,
            product=product,
            revision=revision,
        )
        stages[stage_id] = record
        recomputed.append(_stage_reuse_entry(stage_id, record, reused=False))
        return record

    camera_hash = _camera_white_hash(scene)
    adapted_hashes: dict[str, dict[str, str]] = {}

    for material_id in sorted(scene.materials):
        material = scene.materials[material_id]
        adapted_hashes[material_id] = {}
        for light_id in sorted(scene.lights):
            light = scene.lights[light_id]
            spectral_id = spectral_stage_id(material_id, light_id)
            spectral_signature = {
                "pipeline_version": PIPELINE_VERSION,
                "stage_kind": "spectral_xyz",
                "material_id": material_id,
                "light_id": light_id,
                "material_spectrum_hash": material.spectrum_hash(),
                "light_spd_hash": light.spd_hash(),
                "observer": scene.config.observer,
                "cmf_hash": cmf_hash(scene.config.observer),
                "wavelength_grid_hash": WAVELENGTH_GRID_HASH,
                "integration_step_nm": INTEGRATION_STEP_NM,
                "normalization": "normalize_light_Y_to_100",
            }
            spectral = reuse_or_compute(
                stage_id=spectral_id,
                kind="spectral_xyz",
                material_id=material_id,
                light_id=light_id,
                signature=spectral_signature,
                compute=lambda m=material, l=light: spectral_to_xyz(
                    m, l, scene.config.observer
                ),
            )

            adapted_id = adapted_stage_id(material_id, light_id)
            adapted_signature = {
                "pipeline_version": PIPELINE_VERSION,
                "stage_kind": "adapted_xyz",
                "material_id": material_id,
                "light_id": light_id,
                "spectral_content_hash": spectral["content_hash"],
                "adaptation_method": scene.config.adaptation_method,
                "camera_white_point_hash": camera_hash,
            }
            adapted = reuse_or_compute(
                stage_id=adapted_id,
                kind="adapted_xyz",
                material_id=material_id,
                light_id=light_id,
                signature=adapted_signature,
                compute=lambda sp=spectral["product"]: adapt_xyz(
                    sp,
                    scene.config.camera_white_point_xyz,
                    scene.config.adaptation_method,
                ),
            )
            adapted_hashes[material_id][light_id] = adapted["content_hash"]

        final_id = material_stage_id(material_id)
        final_signature = {
            "pipeline_version": PIPELINE_VERSION,
            "stage_kind": "material_color",
            "material_id": material_id,
            "target_color_space": scene.config.target_color_space,
            "material_appearance_hash": material.appearance_hash(),
            "adapted_hashes": adapted_hashes[material_id],
            "light_weights": {
                light_id: scene.lights[light_id].weight
                for light_id in sorted(scene.lights)
            },
            "aggregation": "weighted_mean_of_adapted_xyz",
        }

        def compute_material_color(material=material) -> dict[str, Any]:
            contributions: list[dict[str, Any]] = []
            total_weight = 0.0
            aggregate = [0.0, 0.0, 0.0]
            for light_id in sorted(scene.lights):
                light = scene.lights[light_id]
                adapted = stages[adapted_stage_id(material_id, light_id)]
                xyz = adapted["product"]["xyz"]
                total_weight += float(light.weight)
                for index in range(3):
                    aggregate[index] += float(light.weight) * float(xyz[index])
                contributions.append(
                    {
                        "light_id": light_id,
                        "weight": qfloat(light.weight),
                        "adapted_xyz": list(xyz),
                        "adapted_stage_hash": adapted["content_hash"],
                    }
                )
            if total_weight <= 0.0:
                raise DomainError(
                    "E_SCENE_INVALID",
                    "scene light weights must sum to a positive value",
                )
            aggregate = [value / total_weight for value in aggregate]
            return {
                "material_id": material.id,
                "aggregation": "weighted_mean_of_adapted_xyz",
                "light_contributions": contributions,
                "aggregate_xyz": [qfloat(value) for value in aggregate],
                "surface": {
                    "roughness": material.roughness,
                    "metallic": material.metallic,
                    "semantics": "retained surface metadata; colour XYZ remains colourimetric",
                },
                "color": xyz_to_target(aggregate, scene.config.target_color_space),
            }

        reuse_or_compute(
            stage_id=final_id,
            kind="material_color",
            material_id=material_id,
            light_id=None,
            signature=final_signature,
            compute=compute_material_color,
        )

    removed = sorted(set(old) - set(stages), key=_stage_sort_key)

    if injection is not None:
        _apply_stage_failure_injection(
            stages, injection=injection, revision=revision
        )

    return PipelineResult(
        stages={key: stages[key] for key in sorted(stages, key=_stage_sort_key)},
        recomputed=recomputed,
        reused=reused,
        removed=removed,
        dirty_closure=sorted((entry["stage_id"] for entry in recomputed), key=_stage_sort_key),
    )


def _apply_stage_failure_injection(
    stages: dict[str, dict[str, Any]],
    *,
    injection: Mapping[str, Any],
    revision: int,
) -> None:
    stage_failure = injection.get("stage_failure")
    if not stage_failure:
        return
    if not isinstance(stage_failure, Mapping):
        raise DomainError(
            E_INJECT_CONFIG,
            "stage_failure injection must be an object",
            pointer="/stage_failure",
        )
    operation_version = stage_failure.get("operation_version")
    if operation_version != revision:
        return

    kind = stage_failure.get("stage_kind")
    material_id = stage_failure.get("material_id")
    light_id = stage_failure.get("light_id")
    if kind == "spectral_xyz":
        stage_id = spectral_stage_id(str(material_id), str(light_id))
    elif kind == "adapted_xyz":
        stage_id = adapted_stage_id(str(material_id), str(light_id))
    elif kind == "material_color":
        stage_id = material_stage_id(str(material_id))
    else:
        raise DomainError(
            E_INJECT_CONFIG,
            "stage_failure has an unknown stage_kind",
            pointer="/stage_failure/stage_kind",
            details={"raw": kind},
        )

    record = stages.get(stage_id)
    if record is None:
        raise DomainError(
            E_INJECT_CONFIG,
            "stage_failure references a stage that does not exist at this revision",
            pointer="/stage_failure",
            details={"stage_id": stage_id, "revision": revision},
        )
    delta = stage_failure.get("delta_xyz", [0.25, 0.0, 0.0])
    if not isinstance(delta, list) or len(delta) != 3:
        raise DomainError(
            E_INJECT_CONFIG,
            "stage_failure.delta_xyz must be a three-element array",
            pointer="/stage_failure/delta_xyz",
            details={"raw": delta},
        )
    product = copy.deepcopy(record["product"])
    if kind in {"spectral_xyz", "adapted_xyz"}:
        target = product["xyz"]
    else:
        target = product["aggregate_xyz"]
        product["color"]["xyz"] = list(target)
    for index in range(3):
        target[index] = qfloat(float(target[index]) + float(delta[index]))
    if kind == "material_color":
        product["color"] = xyz_to_target(
            product["aggregate_xyz"], product["color"].get("space", "sRGB")
        )
    record["product"] = product
    record["content_hash"] = stage_content_hash(record)


def _compare_values(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
) -> tuple[bool, float, dict[str, Any] | None]:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            return False, math.inf, {
                "field": path,
                "incremental": left,
                "full": right,
            }
        difference = abs(float(left) - float(right))
        tolerance = atol + rtol * max(abs(float(left)), abs(float(right)))
        if difference > tolerance:
            return False, difference, {
                "field": path,
                "incremental": left,
                "full": right,
                "abs_diff": difference,
                "tolerance": tolerance,
            }
        return True, difference, None
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing = sorted(left_keys ^ right_keys)
            return False, math.inf, {
                "field": path,
                "key_difference": missing,
            }
        maximum = 0.0
        for key in sorted(left):
            ok, difference, mismatch = _compare_values(
                left[key],
                right[key],
                path=f"{path}/{key}" if path else f"/{key}",
                atol=atol,
                rtol=rtol,
            )
            maximum = max(maximum, difference)
            if not ok:
                return False, maximum, mismatch
        return True, maximum, None
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(
        right, Sequence
    ) and not isinstance(right, (str, bytes)):
        if len(left) != len(right):
            return False, math.inf, {
                "field": path,
                "incremental_length": len(left),
                "full_length": len(right),
            }
        maximum = 0.0
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            ok, difference, mismatch = _compare_values(
                left_item,
                right_item,
                path=f"{path}/{index}",
                atol=atol,
                rtol=rtol,
            )
            maximum = max(maximum, difference)
            if not ok:
                return False, maximum, mismatch
        return True, maximum, None
    if left != right:
        return False, math.inf, {
            "field": path,
            "incremental": left,
            "full": right,
        }
    return True, 0.0, None


def verify_against_full(
    scene: Scene,
    stages: Mapping[str, dict[str, Any]],
    *,
    revision: int,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> dict[str, Any]:
    """Compare incrementally selected stages against a cache-free rebuild."""

    full = build_pipeline(scene, {}, revision=revision, injection=None)
    ordered_ids = sorted(
        set(stages) | set(full.stages),
        key=_stage_sort_key,
    )
    maximum = 0.0
    for stage_id in ordered_ids:
        incremental_record = stages.get(stage_id)
        full_record = full.stages.get(stage_id)
        if incremental_record is None or full_record is None:
            raise DomainError(
                E_STAGE_STALE,
                "incremental stage set diverges from a full rebuild",
                pointer=f"/stages/{stage_id}",
                details={
                    "stage_id": stage_id,
                    "field": "stage_presence",
                    "incremental_present": incremental_record is not None,
                    "full_present": full_record is not None,
                },
            )
        ok, difference, mismatch = _compare_values(
            incremental_record.get("product"),
            full_record.get("product"),
            path="/product",
            atol=atol,
            rtol=rtol,
        )
        maximum = max(maximum, difference)
        if not ok:
            raise DomainError(
                E_STAGE_STALE,
                "incremental stage product diverges from a full rebuild",
                pointer=f"/stages/{stage_id}/product",
                details={
                    "stage_id": stage_id,
                    "stage_kind": incremental_record.get("kind"),
                    "first_divergence": mismatch,
                    "comparison_tolerance": {"atol": atol, "rtol": rtol},
                },
            )
        if incremental_record.get("content_hash") != stage_content_hash(
            incremental_record
        ):
            raise DomainError(
                E_STAGE_STALE,
                "stage content hash does not match its persisted product",
                pointer=f"/stages/{stage_id}",
                details={"stage_id": stage_id, "field": "content_hash"},
            )
    return {
        "status": "ok",
        "stage_count": len(ordered_ids),
        "max_abs_diff": qfloat(maximum),
        "tolerance": {"atol": atol, "rtol": rtol},
    }
