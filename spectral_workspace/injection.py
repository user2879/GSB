"""Serializable, source-free fault injection configuration."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from .canonical import sha256_obj
from .errors import DomainError, E_INJECT_CONFIG
from .model import parse_scene


INJECTION_SCHEMA = "spectral-injection/v1"
_ALLOWED_KEYS = {
    "schema",
    "name",
    "reflectance_out_of_range",
    "white_point_zero",
    "observer_invalid",
    "stage_failure",
}
_STAGE_KINDS = {"spectral_xyz", "adapted_xyz", "material_color"}


def validate_injection_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {"schema": INJECTION_SCHEMA}
    if not isinstance(config, Mapping):
        raise DomainError(
            E_INJECT_CONFIG,
            "injection configuration must be an object",
            pointer="",
            details={"raw_type": type(config).__name__},
        )
    unknown = sorted(set(config) - _ALLOWED_KEYS)
    if unknown:
        raise DomainError(
            E_INJECT_CONFIG,
            "injection configuration contains unknown keys",
            pointer="",
            details={"unknown": unknown, "allowed": sorted(_ALLOWED_KEYS)},
        )
    schema = config.get("schema", INJECTION_SCHEMA)
    if schema != INJECTION_SCHEMA:
        raise DomainError(
            E_INJECT_CONFIG,
            "unsupported injection schema",
            pointer="/schema",
            details={"raw": schema, "expected": INJECTION_SCHEMA},
        )
    result: dict[str, Any] = {"schema": INJECTION_SCHEMA}
    if "name" in config:
        if not isinstance(config["name"], str):
            raise DomainError(
                E_INJECT_CONFIG, "injection name must be a string", pointer="/name"
            )
        result["name"] = config["name"]

    if "reflectance_out_of_range" in config:
        item = config["reflectance_out_of_range"]
        if not isinstance(item, Mapping):
            raise DomainError(
                E_INJECT_CONFIG,
                "reflectance_out_of_range must be an object",
                pointer="/reflectance_out_of_range",
            )
        material_id = item.get("material_id")
        sample_index = item.get("sample_index")
        value = item.get("value")
        if not isinstance(material_id, str):
            raise DomainError(
                E_INJECT_CONFIG,
                "material_id must be a string",
                pointer="/reflectance_out_of_range/material_id",
            )
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < 81
        ):
            raise DomainError(
                E_INJECT_CONFIG,
                "sample_index must be an integer in [0, 80]",
                pointer="/reflectance_out_of_range/sample_index",
                details={"raw": sample_index},
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DomainError(
                E_INJECT_CONFIG,
                "value must be a finite number",
                pointer="/reflectance_out_of_range/value",
                details={"raw": value},
            )
        result["reflectance_out_of_range"] = {
            "material_id": material_id,
            "sample_index": sample_index,
            "value": float(value),
        }

    if "white_point_zero" in config:
        item = config["white_point_zero"]
        if not isinstance(item, Mapping):
            raise DomainError(
                E_INJECT_CONFIG,
                "white_point_zero must be an object",
                pointer="/white_point_zero",
            )
        value = item.get("value")
        if not isinstance(value, list) or len(value) != 3:
            raise DomainError(
                E_INJECT_CONFIG,
                "white_point_zero.value must be a three-element array",
                pointer="/white_point_zero/value",
                details={"raw": value},
            )
        cleaned: list[float] = []
        for index, component in enumerate(value):
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(float(component))
            ):
                raise DomainError(
                    E_INJECT_CONFIG,
                    "white point components must be finite numbers",
                    pointer=f"/white_point_zero/value/{index}",
                    details={"raw": component},
                )
            cleaned.append(float(component))
        result["white_point_zero"] = {"value": cleaned}

    if "observer_invalid" in config:
        item = config["observer_invalid"]
        if not isinstance(item, Mapping) or not isinstance(item.get("value"), str):
            raise DomainError(
                E_INJECT_CONFIG,
                "observer_invalid.value must be a string",
                pointer="/observer_invalid/value",
            )
        result["observer_invalid"] = {"value": item["value"]}

    if "stage_failure" in config:
        item = config["stage_failure"]
        if not isinstance(item, Mapping):
            raise DomainError(
                E_INJECT_CONFIG,
                "stage_failure must be an object",
                pointer="/stage_failure",
            )
        version = item.get("operation_version")
        kind = item.get("stage_kind")
        material_id = item.get("material_id")
        light_id = item.get("light_id")
        delta = item.get("delta_xyz", [0.25, 0.0, 0.0])
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise DomainError(
                E_INJECT_CONFIG,
                "stage_failure.operation_version must be a positive integer",
                pointer="/stage_failure/operation_version",
                details={"raw": version},
            )
        if kind not in _STAGE_KINDS:
            raise DomainError(
                E_INJECT_CONFIG,
                "stage_failure.stage_kind is invalid",
                pointer="/stage_failure/stage_kind",
                details={"raw": kind, "allowed": sorted(_STAGE_KINDS)},
            )
        if not isinstance(material_id, str):
            raise DomainError(
                E_INJECT_CONFIG,
                "stage_failure.material_id must be a string",
                pointer="/stage_failure/material_id",
            )
        if kind in {"spectral_xyz", "adapted_xyz"} and not isinstance(light_id, str):
            raise DomainError(
                E_INJECT_CONFIG,
                "stage_failure.light_id is required for a pair stage",
                pointer="/stage_failure/light_id",
            )
        if not isinstance(delta, list) or len(delta) != 3:
            raise DomainError(
                E_INJECT_CONFIG,
                "stage_failure.delta_xyz must be a three-element array",
                pointer="/stage_failure/delta_xyz",
            )
        cleaned_delta: list[float] = []
        for index, component in enumerate(delta):
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(float(component))
            ):
                raise DomainError(
                    E_INJECT_CONFIG,
                    "delta_xyz components must be finite numbers",
                    pointer=f"/stage_failure/delta_xyz/{index}",
                    details={"raw": component},
                )
            cleaned_delta.append(float(component))
        result["stage_failure"] = {
            "operation_version": version,
            "stage_kind": kind,
            "material_id": material_id,
            "light_id": light_id if isinstance(light_id, str) else None,
            "delta_xyz": cleaned_delta,
        }

    return result


def injection_hash(config: Mapping[str, Any] | None) -> str:
    return sha256_obj(validate_injection_config(config))


def _material_container(scene_payload: dict[str, Any], material_id: str) -> dict[str, Any]:
    materials = scene_payload.get("materials")
    if isinstance(materials, dict) and isinstance(materials.get(material_id), dict):
        return materials[material_id]
    if isinstance(materials, list):
        for item in materials:
            if isinstance(item, dict) and item.get("id") == material_id:
                return item
    raise DomainError(
        E_INJECT_CONFIG,
        "reflectance injection references an unknown material",
        pointer="/reflectance_out_of_range/material_id",
        details={"material_id": material_id},
    )


def apply_parameter_injections(
    scene_payload: Mapping[str, Any], config: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Copy and mutate raw scene input according to parameter-level injections."""

    cleaned = validate_injection_config(config)
    payload = copy.deepcopy(dict(scene_payload))
    scene_config = payload.get("scene")
    if not isinstance(scene_config, dict):
        scene_config = {}
        payload["scene"] = scene_config

    if "observer_invalid" in cleaned:
        scene_config["observer"] = cleaned["observer_invalid"]["value"]
    if "white_point_zero" in cleaned:
        scene_config["camera_white_point_xyz"] = cleaned["white_point_zero"]["value"]
    if "reflectance_out_of_range" in cleaned:
        item = cleaned["reflectance_out_of_range"]
        material = _material_container(payload, item["material_id"])
        reflectance = material.get("reflectance")
        if not isinstance(reflectance, list):
            raise DomainError(
                E_INJECT_CONFIG,
                "reflectance injection target has no reflectance array",
                pointer=f"/materials/{item['material_id']}/reflectance",
            )
        reflectance[item["sample_index"]] = item["value"]
    return payload


def run_parameter_injection_case(
    scene_payload: Mapping[str, Any], config: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply injections and validation, returning either success or a business error."""

    try:
        injected = apply_parameter_injections(scene_payload, config)
        parse_scene(injected, source="<injection-case>")
        return {"ok": True, "injection_hash": injection_hash(config)}
    except DomainError as error:
        try:
            config_hash = injection_hash(config)
        except DomainError:
            config_hash = sha256_obj(config)
        return {
            "ok": False,
            "injection_hash": config_hash,
            "error": error.to_dict()["error"],
        }
