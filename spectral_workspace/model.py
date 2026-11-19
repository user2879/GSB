"""Spectral material/light/scene data model and JSON validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_json, qfloat, read_json, sha256_obj
from .errors import (
    DomainError,
    E_INPUT_JSON,
    E_SCENE_INVALID,
    E_SPECTRUM_RANGE,
    E_WHITE_POINT,
    join_pointer,
)
from .standards import (
    D65_WHITE_XYZ,
    SAMPLE_COUNT,
    WAVELENGTHS_NM,
    normalize_adaptation_method,
    normalize_color_space,
    normalize_observer,
)


SCENE_SCHEMA = "spectral-material-scene/v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _require_finite_number(value: Any, *, pointer: str, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(
            E_SCENE_INVALID,
            f"{what} must be a finite number",
            pointer=pointer,
            details={"raw": value},
        )
    number = float(value)
    if not math.isfinite(number):
        raise DomainError(
            E_SCENE_INVALID,
            f"{what} must be finite",
            pointer=pointer,
            details={"raw": value},
        )
    return number


def _validate_id(value: Any, *, pointer: str, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise DomainError(
            E_SCENE_INVALID,
            f"{what} must match {_ID_RE.pattern}",
            pointer=pointer,
            details={"raw": value},
        )
    return value


def _validate_wavelengths(
    value: Any, *, pointer: str
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DomainError(
            E_SPECTRUM_RANGE,
            "wavelengths_nm must be a JSON array",
            pointer=pointer,
            details={"raw_type": type(value).__name__},
        )
    if len(value) != SAMPLE_COUNT:
        raise DomainError(
            E_SPECTRUM_RANGE,
            "spectral sampling is incomplete",
            pointer=pointer,
            details={"expected_count": SAMPLE_COUNT, "actual_count": len(value)},
        )
    wavelengths: list[int] = []
    previous: int | None = None
    for index, raw in enumerate(value):
        item_pointer = join_pointer(pointer, index)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DomainError(
                E_SPECTRUM_RANGE,
                "wavelength must be a numeric value",
                pointer=item_pointer,
                details={"raw": raw, "expected_nm": WAVELENGTHS_NM[index]},
            )
        numeric = float(raw)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise DomainError(
                E_SPECTRUM_RANGE,
                "wavelength must be a finite integer nanometre value",
                pointer=item_pointer,
                details={"raw": raw, "expected_nm": WAVELENGTHS_NM[index]},
            )
        current = int(numeric)
        if previous is not None and current <= previous:
            raise DomainError(
                E_SPECTRUM_RANGE,
                "wavelengths must be strictly increasing",
                pointer=item_pointer,
                details={"previous_nm": previous, "raw_nm": current},
            )
        if current != WAVELENGTHS_NM[index]:
            raise DomainError(
                E_SPECTRUM_RANGE,
                "wavelength grid must be exactly 380..780 nm at 5 nm steps",
                pointer=item_pointer,
                details={"expected_nm": WAVELENGTHS_NM[index], "raw_nm": current},
            )
        wavelengths.append(current)
        previous = current
    return tuple(wavelengths)


@dataclass(frozen=True)
class Material:
    id: str
    wavelengths_nm: tuple[int, ...]
    reflectance: tuple[float, ...]
    roughness: float
    metallic: float

    def spectrum_hash(self) -> str:
        return sha256_obj(
            {
                "wavelengths_nm": list(self.wavelengths_nm),
                "reflectance": list(self.reflectance),
            }
        )

    def appearance_hash(self) -> str:
        return sha256_obj(
            {"roughness": self.roughness, "metallic": self.metallic}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "wavelengths_nm": list(self.wavelengths_nm),
            "reflectance": list(self.reflectance),
            "roughness": self.roughness,
            "metallic": self.metallic,
        }

    def replace(self, **changes: Any) -> "Material":
        data = self.to_dict()
        data.update(changes)
        return material_from_dict(data, pointer=f"/materials/{self.id}", expected_id=self.id)


def material_from_dict(
    data: Any,
    *,
    pointer: str,
    expected_id: str | None = None,
) -> Material:
    if not isinstance(data, Mapping):
        raise DomainError(
            E_SCENE_INVALID,
            "material must be a JSON object",
            pointer=pointer,
            details={"raw_type": type(data).__name__},
        )

    material_id = data.get("id", expected_id)
    material_id = _validate_id(
        material_id, pointer=join_pointer(pointer, "id"), what="material id"
    )
    if expected_id is not None and material_id != expected_id:
        raise DomainError(
            E_SCENE_INVALID,
            "material id does not match its object key",
            pointer=join_pointer(pointer, "id"),
            details={"key": expected_id, "id": material_id},
        )

    wavelengths = _validate_wavelengths(
        data.get("wavelengths_nm"), pointer=join_pointer(pointer, "wavelengths_nm")
    )

    raw_reflectance = data.get("reflectance")
    if not isinstance(raw_reflectance, list):
        raise DomainError(
            E_SPECTRUM_RANGE,
            "reflectance must be a JSON array",
            pointer=join_pointer(pointer, "reflectance"),
            details={"raw_type": type(raw_reflectance).__name__},
        )
    if len(raw_reflectance) != SAMPLE_COUNT:
        raise DomainError(
            E_SPECTRUM_RANGE,
            "reflectance sampling is incomplete",
            pointer=join_pointer(pointer, "reflectance"),
            details={
                "expected_count": SAMPLE_COUNT,
                "actual_count": len(raw_reflectance),
            },
        )

    reflectance: list[float] = []
    for index, raw in enumerate(raw_reflectance):
        item_pointer = join_pointer(pointer, "reflectance", index)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DomainError(
                E_SPECTRUM_RANGE,
                "reflectance sample must be numeric",
                pointer=item_pointer,
                details={"raw": raw, "wavelength_nm": wavelengths[index]},
            )
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise DomainError(
                E_SPECTRUM_RANGE,
                "reflectance sample must be finite",
                pointer=item_pointer,
                details={"raw": raw, "wavelength_nm": wavelengths[index]},
            )
        if numeric < 0.0 or numeric > 1.0:
            raise DomainError(
                E_SPECTRUM_RANGE,
                "reflectance must lie in [0, 1]",
                pointer=item_pointer,
                details={
                    "raw": raw,
                    "wavelength_nm": wavelengths[index],
                    "min": 0.0,
                    "max": 1.0,
                },
            )
        reflectance.append(qfloat(numeric))

    roughness = _require_finite_number(
        data.get("roughness"),
        pointer=join_pointer(pointer, "roughness"),
        what="roughness",
    )
    metallic = _require_finite_number(
        data.get("metallic"),
        pointer=join_pointer(pointer, "metallic"),
        what="metallic",
    )
    for name, number in (("roughness", roughness), ("metallic", metallic)):
        if number < 0.0 or number > 1.0:
            raise DomainError(
                E_SCENE_INVALID,
                f"{name} must lie in [0, 1]",
                pointer=join_pointer(pointer, name),
                details={"raw": number},
            )

    return Material(
        id=material_id,
        wavelengths_nm=wavelengths,
        reflectance=tuple(reflectance),
        roughness=qfloat(roughness),
        metallic=qfloat(metallic),
    )


@dataclass(frozen=True)
class Light:
    id: str
    wavelengths_nm: tuple[int, ...]
    spd: tuple[float, ...]
    weight: float = 1.0

    def spd_hash(self) -> str:
        return sha256_obj(
            {"wavelengths_nm": list(self.wavelengths_nm), "spd": list(self.spd)}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "wavelengths_nm": list(self.wavelengths_nm),
            "spd": list(self.spd),
            "weight": self.weight,
        }

    def replace(self, **changes: Any) -> "Light":
        data = self.to_dict()
        data.update(changes)
        return light_from_dict(data, pointer=f"/lights/{self.id}", expected_id=self.id)


def light_from_dict(
    data: Any,
    *,
    pointer: str,
    expected_id: str | None = None,
) -> Light:
    if not isinstance(data, Mapping):
        raise DomainError(
            E_SCENE_INVALID,
            "light must be a JSON object",
            pointer=pointer,
            details={"raw_type": type(data).__name__},
        )
    light_id = data.get("id", expected_id)
    light_id = _validate_id(light_id, pointer=join_pointer(pointer, "id"), what="light id")
    if expected_id is not None and light_id != expected_id:
        raise DomainError(
            E_SCENE_INVALID,
            "light id does not match its object key",
            pointer=join_pointer(pointer, "id"),
            details={"key": expected_id, "id": light_id},
        )

    wavelengths = _validate_wavelengths(
        data.get("wavelengths_nm"), pointer=join_pointer(pointer, "wavelengths_nm")
    )
    raw_spd = data.get("spd")
    if not isinstance(raw_spd, list):
        raise DomainError(
            E_SPECTRUM_RANGE,
            "spd must be a JSON array",
            pointer=join_pointer(pointer, "spd"),
            details={"raw_type": type(raw_spd).__name__},
        )
    if len(raw_spd) != SAMPLE_COUNT:
        raise DomainError(
            E_SPECTRUM_RANGE,
            "spectral power distribution sampling is incomplete",
            pointer=join_pointer(pointer, "spd"),
            details={"expected_count": SAMPLE_COUNT, "actual_count": len(raw_spd)},
        )
    spd: list[float] = []
    for index, raw in enumerate(raw_spd):
        item_pointer = join_pointer(pointer, "spd", index)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DomainError(
                E_SPECTRUM_RANGE,
                "SPD sample must be numeric",
                pointer=item_pointer,
                details={"raw": raw, "wavelength_nm": wavelengths[index]},
            )
        numeric = float(raw)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise DomainError(
                E_SPECTRUM_RANGE,
                "SPD sample must be finite and non-negative",
                pointer=item_pointer,
                details={"raw": raw, "wavelength_nm": wavelengths[index]},
            )
        spd.append(qfloat(numeric))

    weight = _require_finite_number(
        data.get("weight", 1.0),
        pointer=join_pointer(pointer, "weight"),
        what="light weight",
    )
    if weight <= 0.0:
        raise DomainError(
            E_SCENE_INVALID,
            "light weight must be greater than zero",
            pointer=join_pointer(pointer, "weight"),
            details={"raw": weight},
        )
    return Light(
        id=light_id,
        wavelengths_nm=wavelengths,
        spd=tuple(spd),
        weight=qfloat(weight),
    )


def validate_white_point(
    value: Any, *, pointer: str = "/scene/camera_white_point_xyz"
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise DomainError(
            E_WHITE_POINT,
            "white point must be an XYZ array with three components",
            pointer=pointer,
            details={"raw": value},
        )
    components: list[float] = []
    for index, raw in enumerate(value):
        item_pointer = join_pointer(pointer, index)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DomainError(
                E_WHITE_POINT,
                "white point component must be numeric",
                pointer=item_pointer,
                details={"raw": raw},
            )
        number = float(raw)
        if not math.isfinite(number) or number <= 0.0:
            raise DomainError(
                E_WHITE_POINT,
                "white point components must be finite and strictly positive",
                pointer=item_pointer,
                details={"raw": raw},
            )
        components.append(qfloat(number))
    y = components[1]
    return (
        qfloat(components[0] / y),
        1.0,
        qfloat(components[2] / y),
    )


@dataclass(frozen=True)
class SceneConfig:
    observer: str
    camera_white_point_xyz: tuple[float, float, float]
    adaptation_method: str
    target_color_space: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "observer": self.observer,
            "camera_white_point_xyz": list(self.camera_white_point_xyz),
            "adaptation_method": self.adaptation_method,
            "target_color_space": self.target_color_space,
        }


def scene_config_from_dict(
    data: Any, *, pointer: str = "/scene"
) -> SceneConfig:
    if not isinstance(data, Mapping):
        raise DomainError(
            E_SCENE_INVALID,
            "scene configuration must be a JSON object",
            pointer=pointer,
            details={"raw_type": type(data).__name__},
        )
    observer = normalize_observer(
        data.get("observer"), pointer=join_pointer(pointer, "observer")
    )
    white_point = validate_white_point(
        data.get("camera_white_point_xyz", list(D65_WHITE_XYZ)),
        pointer=join_pointer(pointer, "camera_white_point_xyz"),
    )
    adaptation = normalize_adaptation_method(
        data.get("adaptation_method", "bradford"),
        pointer=join_pointer(pointer, "adaptation_method"),
    )
    color_space = normalize_color_space(
        data.get("target_color_space", "sRGB"),
        pointer=join_pointer(pointer, "target_color_space"),
    )
    return SceneConfig(
        observer=observer,
        camera_white_point_xyz=white_point,
        adaptation_method=adaptation,
        target_color_space=color_space,
    )


@dataclass(frozen=True)
class Scene:
    config: SceneConfig
    materials: Mapping[str, Material]
    lights: Mapping[str, Light]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENE_SCHEMA,
            "scene": self.config.to_dict(),
            "materials": {
                key: self.materials[key].to_dict() for key in sorted(self.materials)
            },
            "lights": {key: self.lights[key].to_dict() for key in sorted(self.lights)},
        }


def _iter_named_records(
    value: Any, *, pointer: str, what: str
) -> list[tuple[str, Any, str]]:
    records: list[tuple[str, Any, str]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            if not isinstance(key, str):
                raise DomainError(
                    E_SCENE_INVALID,
                    f"{what} keys must be strings",
                    pointer=pointer,
                    details={"raw_key": repr(key)},
                )
            records.append((key, value[key], join_pointer(pointer, key)))
        return records
    if isinstance(value, list):
        seen: set[str] = set()
        for index, item in enumerate(value):
            item_pointer = join_pointer(pointer, index)
            if not isinstance(item, Mapping) or "id" not in item:
                raise DomainError(
                    E_SCENE_INVALID,
                    f"{what} list items must be objects with an id",
                    pointer=item_pointer,
                )
            item_id = item["id"]
            if not isinstance(item_id, str):
                raise DomainError(
                    E_SCENE_INVALID,
                    f"{what} id must be a string",
                    pointer=join_pointer(item_pointer, "id"),
                    details={"raw": item_id},
                )
            if item_id in seen:
                raise DomainError(
                    E_SCENE_INVALID,
                    f"duplicate {what} id",
                    pointer=join_pointer(item_pointer, "id"),
                    details={"id": item_id},
                )
            seen.add(item_id)
            records.append((item_id, item, item_pointer))
        return records
    raise DomainError(
        E_SCENE_INVALID,
        f"{what} must be an object or array",
        pointer=pointer,
        details={"raw_type": type(value).__name__},
    )


def parse_scene(payload: Any, *, source: str = "<input>") -> Scene:
    if not isinstance(payload, Mapping):
        raise DomainError(
            E_SCENE_INVALID,
            "scene input must be a JSON object",
            pointer="",
            details={"source": source, "raw_type": type(payload).__name__},
        )

    if "materials" not in payload:
        raise DomainError(
            E_SCENE_INVALID, "scene input is missing materials", pointer="/materials"
        )
    if "lights" not in payload:
        raise DomainError(
            E_SCENE_INVALID, "scene input is missing lights", pointer="/lights"
        )

    materials: dict[str, Material] = {}
    for key, raw, item_pointer in _iter_named_records(
        payload["materials"], pointer="/materials", what="material"
    ):
        material = material_from_dict(raw, pointer=item_pointer, expected_id=key)
        materials[material.id] = material

    lights: dict[str, Light] = {}
    for key, raw, item_pointer in _iter_named_records(
        payload["lights"], pointer="/lights", what="light"
    ):
        light = light_from_dict(raw, pointer=item_pointer, expected_id=key)
        lights[light.id] = light

    if not materials:
        raise DomainError(
            E_SCENE_INVALID, "scene must contain at least one material", pointer="/materials"
        )
    if not lights:
        raise DomainError(
            E_SCENE_INVALID, "scene must contain at least one light", pointer="/lights"
        )

    config = scene_config_from_dict(payload.get("scene"), pointer="/scene")
    return Scene(config=config, materials=materials, lights=lights)


def scene_from_state(state_scene: Any) -> Scene:
    """Parse the scene portion persisted in a stable snapshot."""

    return parse_scene(state_scene, source="<stable-state>")


def public_scene(scene: Scene) -> dict[str, Any]:
    """Return a deterministic, pretty-print friendly scene document."""

    return scene.to_dict()


def load_scene_json(path: str | Path) -> Scene:
    source = str(path)
    try:
        payload = read_json(path)
    except Exception as error:  # noqa: BLE001 - mapped to a stable business error
        raise DomainError(
            E_INPUT_JSON,
            "scene JSON could not be read",
            pointer=source,
            details={"error": str(error)},
        ) from error
    return parse_scene(payload, source=source)


def write_scene_json(
    path: str | Path, scene: Scene, *, pretty: bool = True
) -> bytes:
    return atomic_write_json(path, scene.to_dict(), pretty=pretty)
