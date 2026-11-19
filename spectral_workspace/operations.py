"""Versioned operations that evolve an immutable scene state."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .errors import DomainError, E_OPERATION_INVALID
from .model import (
    Light,
    Material,
    Scene,
    light_from_dict,
    material_from_dict,
    parse_scene,
    scene_config_from_dict,
)


SUPPORTED_OPERATIONS = (
    "initialize",
    "add_material",
    "edit_material",
    "remove_material",
    "add_light",
    "replace_light",
    "remove_light",
    "set_light_weight",
    "set_observer",
    "set_adaptation_method",
    "set_camera_white_point",
    "set_target_color_space",
)


def validate_operation(operation: Any, *, expected_version: int | None = None) -> None:
    if not isinstance(operation, Mapping):
        raise DomainError(
            E_OPERATION_INVALID,
            "operation must be a JSON object",
            details={"raw_type": type(operation).__name__},
        )
    version = operation.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise DomainError(
            E_OPERATION_INVALID,
            "operation version must be a positive integer",
            pointer="/version",
            details={"raw": version},
        )
    if expected_version is not None and version != expected_version:
        raise DomainError(
            E_OPERATION_INVALID,
            "operation version is not the next sequential revision",
            pointer="/version",
            details={"expected": expected_version, "actual": version},
        )
    name = operation.get("op")
    if name not in SUPPORTED_OPERATIONS:
        raise DomainError(
            E_OPERATION_INVALID,
            "unsupported operation",
            pointer="/op",
            details={"raw": name, "supported": list(SUPPORTED_OPERATIONS)},
        )
    if not isinstance(operation.get("payload", {}), Mapping):
        raise DomainError(
            E_OPERATION_INVALID,
            "operation payload must be an object",
            pointer="/payload",
        )


def _payload(operation: Mapping[str, Any]) -> Mapping[str, Any]:
    return operation.get("payload", {})


def _require_scene(scene: Scene | None) -> Scene:
    if scene is None:
        raise DomainError(
            E_OPERATION_INVALID,
            "workspace must be initialised before this operation",
        )
    return scene


def _parse_from_parts(
    scene: Scene,
    *,
    config_data: Mapping[str, Any] | None = None,
    materials: Mapping[str, Material] | None = None,
    lights: Mapping[str, Light] | None = None,
) -> Scene:
    data = scene.to_dict()
    if config_data is not None:
        data["scene"] = dict(config_data)
    if materials is not None:
        data["materials"] = {
            key: value.to_dict() for key, value in sorted(materials.items())
        }
    if lights is not None:
        data["lights"] = {key: value.to_dict() for key, value in sorted(lights.items())}
    return parse_scene(data, source="<operation-result>")


def apply_operation(scene: Scene | None, operation: Mapping[str, Any]) -> Scene:
    """Return the new scene after applying one already version-checked operation."""

    name = operation["op"]
    payload = _payload(operation)

    if name == "initialize":
        if scene is not None:
            raise DomainError(
                E_OPERATION_INVALID,
                "initialize is only valid for an empty workspace",
                pointer="/op",
            )
        raw_scene = payload.get("scene")
        if not isinstance(raw_scene, Mapping):
            raise DomainError(
                E_OPERATION_INVALID,
                "initialize payload must contain a scene object",
                pointer="/payload/scene",
            )
        return parse_scene(raw_scene, source="<initialize>")

    current = _require_scene(scene)

    if name == "add_material":
        raw = payload.get("material")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise DomainError(
                E_OPERATION_INVALID,
                "add_material requires payload.material with an id",
                pointer="/payload/material",
            )
        material_id = raw["id"]
        if material_id in current.materials:
            raise DomainError(
                E_OPERATION_INVALID,
                "material already exists",
                pointer="/payload/material/id",
                details={"material_id": material_id},
            )
        material = material_from_dict(
            raw, pointer="/payload/material", expected_id=material_id
        )
        materials = dict(current.materials)
        materials[material_id] = material
        return _parse_from_parts(current, materials=materials)

    if name == "edit_material":
        material_id = payload.get("material_id")
        if not isinstance(material_id, str) or material_id not in current.materials:
            raise DomainError(
                E_OPERATION_INVALID,
                "edit_material references an unknown material",
                pointer="/payload/material_id",
                details={"material_id": material_id},
            )
        changes = payload.get("changes")
        if not isinstance(changes, Mapping) or not changes:
            raise DomainError(
                E_OPERATION_INVALID,
                "edit_material requires a non-empty changes object",
                pointer="/payload/changes",
            )
        allowed = {"reflectance", "roughness", "metallic"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise DomainError(
                E_OPERATION_INVALID,
                "edit_material contains unsupported changes",
                pointer="/payload/changes",
                details={"unknown": unknown, "allowed": sorted(allowed)},
            )
        updated = current.materials[material_id].replace(**dict(changes))
        materials = dict(current.materials)
        materials[material_id] = updated
        return _parse_from_parts(current, materials=materials)

    if name == "remove_material":
        material_id = payload.get("material_id")
        if not isinstance(material_id, str) or material_id not in current.materials:
            raise DomainError(
                E_OPERATION_INVALID,
                "remove_material references an unknown material",
                pointer="/payload/material_id",
                details={"material_id": material_id},
            )
        if len(current.materials) <= 1:
            raise DomainError(
                E_OPERATION_INVALID,
                "scene must retain at least one material",
                pointer="/payload/material_id",
            )
        materials = dict(current.materials)
        materials.pop(material_id)
        return _parse_from_parts(current, materials=materials)

    if name == "add_light":
        raw = payload.get("light")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise DomainError(
                E_OPERATION_INVALID,
                "add_light requires payload.light with an id",
                pointer="/payload/light",
            )
        light_id = raw["id"]
        if light_id in current.lights:
            raise DomainError(
                E_OPERATION_INVALID,
                "light already exists",
                pointer="/payload/light/id",
                details={"light_id": light_id},
            )
        light = light_from_dict(raw, pointer="/payload/light", expected_id=light_id)
        lights = dict(current.lights)
        lights[light_id] = light
        return _parse_from_parts(current, lights=lights)

    if name == "replace_light":
        raw = payload.get("light")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise DomainError(
                E_OPERATION_INVALID,
                "replace_light requires the complete replacement light object",
                pointer="/payload/light",
            )
        light_id = raw["id"]
        if light_id not in current.lights:
            raise DomainError(
                E_OPERATION_INVALID,
                "replace_light references an unknown light",
                pointer="/payload/light/id",
                details={"light_id": light_id},
            )
        light = light_from_dict(raw, pointer="/payload/light", expected_id=light_id)
        lights = dict(current.lights)
        lights[light_id] = light
        return _parse_from_parts(current, lights=lights)

    if name == "remove_light":
        light_id = payload.get("light_id")
        if not isinstance(light_id, str) or light_id not in current.lights:
            raise DomainError(
                E_OPERATION_INVALID,
                "remove_light references an unknown light",
                pointer="/payload/light_id",
                details={"light_id": light_id},
            )
        if len(current.lights) <= 1:
            raise DomainError(
                E_OPERATION_INVALID,
                "scene must retain at least one light",
                pointer="/payload/light_id",
            )
        lights = dict(current.lights)
        lights.pop(light_id)
        return _parse_from_parts(current, lights=lights)

    if name == "set_light_weight":
        light_id = payload.get("light_id")
        if not isinstance(light_id, str) or light_id not in current.lights:
            raise DomainError(
                E_OPERATION_INVALID,
                "set_light_weight references an unknown light",
                pointer="/payload/light_id",
                details={"light_id": light_id},
            )
        updated = current.lights[light_id].replace(weight=payload.get("weight"))
        lights = dict(current.lights)
        lights[light_id] = updated
        return _parse_from_parts(current, lights=lights)

    config = current.config.to_dict()
    config_pointer = "/payload"
    if name == "set_observer":
        config["observer"] = payload.get("observer")
        config_pointer = "/payload/observer"
    elif name == "set_adaptation_method":
        config["adaptation_method"] = payload.get("method")
        config_pointer = "/payload/method"
    elif name == "set_camera_white_point":
        config["camera_white_point_xyz"] = payload.get("camera_white_point_xyz")
        config_pointer = "/payload/camera_white_point_xyz"
    elif name == "set_target_color_space":
        config["target_color_space"] = payload.get("target_color_space")
        config_pointer = "/payload/target_color_space"
    else:  # pragma: no cover - validate_operation protects this branch
        raise DomainError(E_OPERATION_INVALID, "unsupported operation", pointer="/op")

    try:
        new_config = scene_config_from_dict(config, pointer="/scene")
    except DomainError as error:
        if error.pointer and error.pointer.startswith("/scene"):
            error.pointer = config_pointer
        raise
    data = current.to_dict()
    data["scene"] = new_config.to_dict()
    return parse_scene(data, source="<operation-result>")


def copy_operation(operation: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(operation))
