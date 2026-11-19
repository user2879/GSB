"""Deterministic local synthesis of spectral materials and lights."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, qfloat
from .model import SCENE_SCHEMA
from .standards import (
    D65_WHITE_XYZ,
    INTEGRATION_STEP_NM,
    WAVELENGTHS_NM,
)


DEFAULT_SEED = 20261119


class SplitMix64:
    """Tiny fixed PRNG so sample bytes do not depend on a Python RNG version."""

    def __init__(self, seed: int) -> None:
        self.state = seed & ((1 << 64) - 1)

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)

    def uniform(self, low: float, high: float) -> float:
        fraction = self.next_u64() / float(1 << 64)
        return low + (high - low) * fraction


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _reflectance_curve(material_index: int, rng: SplitMix64) -> list[float]:
    phases = [rng.uniform(0.0, math.tau) for _ in range(4)]
    amplitudes = [rng.uniform(0.015, 0.070) for _ in range(4)]
    feature_center = rng.uniform(430.0, 720.0)
    feature_width = rng.uniform(18.0, 52.0)
    feature_height = rng.uniform(-0.16, 0.18)
    baseline = 0.055 + 0.035 * material_index
    values: list[float] = []
    for wavelength in WAVELENGTHS_NM:
        position = (wavelength - 380.0) / 400.0
        value = (
            baseline
            + 0.42 * (position ** (0.75 + 0.15 * material_index))
            + amplitudes[0] * math.sin(math.tau * 1.1 * position + phases[0])
            + amplitudes[1] * math.sin(math.tau * 2.7 * position + phases[1])
            + amplitudes[2] * math.sin(math.tau * 5.1 * position + phases[2])
            + amplitudes[3] * math.cos(math.tau * 0.6 * position + phases[3])
            + feature_height
            * math.exp(-0.5 * ((wavelength - feature_center) / feature_width) ** 2)
        )
        values.append(qfloat(_clamp(value, 0.015, 0.985), 9))
    return values


def _planck_spd(temperature_k: float) -> list[float]:
    """Relative black-body spectral power at the canonical sample grid."""

    values: list[float] = []
    for wavelength_nm in WAVELENGTHS_NM:
        wavelength_m = wavelength_nm * 1e-9
        exponent = 1.438776877e-2 / (wavelength_m * temperature_k)
        if exponent > 700.0:
            radiance = 0.0
        else:
            radiance = 1.0 / (
                (wavelength_m**5) * (math.exp(exponent) - 1.0)
            )
        values.append(radiance)
    maximum = max(values)
    return [qfloat(value / maximum, 9) for value in values]


def generate_sample_scene(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    rng = SplitMix64(seed)
    material_specs = [
        ("mat_oxide", 0.36, 0.05),
        ("mat_polymer", 0.68, 0.02),
        ("mat_metal", 0.24, 0.91),
    ]
    materials: dict[str, Any] = {}
    for index, (material_id, roughness, metallic) in enumerate(material_specs):
        materials[material_id] = {
            "id": material_id,
            "wavelengths_nm": list(WAVELENGTHS_NM),
            "reflectance": _reflectance_curve(index, rng),
            "roughness": roughness,
            "metallic": metallic,
        }

    lights = {
        "light_day": {
            "id": "light_day",
            "wavelengths_nm": list(WAVELENGTHS_NM),
            "spd": _planck_spd(6500.0),
            "weight": 1.0,
        },
        "light_warm": {
            "id": "light_warm",
            "wavelengths_nm": list(WAVELENGTHS_NM),
            "spd": _planck_spd(3000.0),
            "weight": 0.65,
        },
    }

    return {
        "schema": SCENE_SCHEMA,
        "scene": {
            "observer": "CIE1931_2",
            "camera_white_point_xyz": list(D65_WHITE_XYZ),
            "adaptation_method": "bradford",
            "target_color_space": "sRGB",
        },
        "materials": materials,
        "lights": lights,
    }


def write_sample_scene(
    path: str | Path, seed: int = DEFAULT_SEED, *, replace: bool = True
) -> dict[str, Any]:
    sample = generate_sample_scene(seed)
    destination = Path(path)
    if destination.exists() and not replace:
        raise FileExistsError(destination)
    atomic_write_json(destination, sample, pretty=True)
    return sample


def replace_light_variant(
    light: dict[str, Any], *, temperature_k: float, weight: float
) -> dict[str, Any]:
    replacement = dict(light)
    replacement["spd"] = _planck_spd(temperature_k)
    replacement["weight"] = weight
    return replacement


def edit_reflectance_variant(
    material: dict[str, Any], *, start_index: int, end_index: int, amount: float
) -> dict[str, Any]:
    replacement = dict(material)
    reflectance = list(material["reflectance"])
    for index in range(start_index, min(end_index, len(reflectance))):
        wavelength = WAVELENGTHS_NM[index]
        modulation = math.sin((wavelength - 380.0) * math.tau / 180.0)
        reflectance[index] = qfloat(
            _clamp(reflectance[index] + amount * modulation, 0.0, 1.0), 9
        )
    replacement["reflectance"] = reflectance
    return replacement


def demo_operations(scene_payload: dict[str, Any], seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    materials = scene_payload["materials"]
    lights = scene_payload["lights"]
    edited_oxide = edit_reflectance_variant(
        materials["mat_oxide"], start_index=14, end_index=34, amount=0.055
    )
    warm_replacement = replace_light_variant(
        lights["light_warm"], temperature_k=3400.0, weight=0.75
    )
    return [
        {
            "version": 1,
            "op": "initialize",
            "payload": {"scene": scene_payload},
        },
        {
            "version": 2,
            "op": "edit_material",
            "payload": {
                "material_id": "mat_oxide",
                "changes": {"reflectance": edited_oxide["reflectance"]},
            },
        },
        {
            "version": 3,
            "op": "edit_material",
            "payload": {
                "material_id": "mat_polymer",
                "changes": {"roughness": 0.61, "metallic": 0.07},
            },
        },
        {
            "version": 4,
            "op": "replace_light",
            "payload": {"light": warm_replacement},
        },
        {
            "version": 5,
            "op": "set_observer",
            "payload": {"observer": "CIE1964_10"},
        },
        {
            "version": 6,
            "op": "set_adaptation_method",
            "payload": {"method": "von_kries"},
        },
        {
            "version": 7,
            "op": "edit_material",
            "payload": {
                "material_id": "mat_metal",
                "changes": {"metallic": 0.97, "roughness": 0.19},
            },
        },
    ]
