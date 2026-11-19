"""Spectral-to-XYZ, chromatic adaptation and target colour conversion."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .canonical import qfloat, sha256_obj
from .errors import DomainError, E_WHITE_POINT
from .model import Light, Material, validate_white_point
from .standards import (
    ADAPTATION_BASES,
    CMF_DATA,
    CMF_TABLE_VERSION,
    COLOR_SPACES,
    INTEGRATION_STEP_NM,
    OBSERVER_DISPLAY_NAMES,
    WAVELENGTHS_NM,
    cmf_hash,
    normalize_adaptation_method,
    normalize_color_space,
    normalize_observer,
)


def _clean_sequence(values: Sequence[float]) -> list[float]:
    return [qfloat(float(value)) for value in values]


@lru_cache(maxsize=4)
def _observer_array(observer: str) -> np.ndarray:
    canonical = normalize_observer(observer)
    return np.asarray(CMF_DATA[canonical], dtype=np.float64)


def _check_source_white(raw_xyz: Sequence[float], *, light_id: str) -> None:
    for index, value in enumerate(raw_xyz):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise DomainError(
                E_WHITE_POINT,
                "light SPD produces an invalid source white point",
                pointer=f"/lights/{light_id}/spd",
                details={"component": index, "raw": float(value)},
            )


def spectral_to_xyz(
    material: Material, light: Light, observer: str
) -> dict[str, Any]:
    """Integrate a material/light pair under one standard observer.

    Both the illuminant-only term and the reflectance-weighted term are kept in
    the returned product.  The illuminant term is normalised so a perfect
    diffuse reflector has Y=100, which makes the scale explicit and repeatable.
    """

    canonical_observer = normalize_observer(observer)
    if material.wavelengths_nm != WAVELENGTHS_NM or light.wavelengths_nm != WAVELENGTHS_NM:
        raise DomainError(
            "E_SPECTRUM_RANGE",
            "material and light must use the canonical 380..780 nm / 5 nm grid",
            details={"material_id": material.id, "light_id": light.id},
        )

    spd = np.asarray(light.spd, dtype=np.float64)
    reflectance = np.asarray(material.reflectance, dtype=np.float64)
    cmf = _observer_array(canonical_observer)

    light_integral = np.sum(spd[:, None] * cmf, axis=0) * float(INTEGRATION_STEP_NM)
    reflectance_integral = (
        np.sum((spd * reflectance)[:, None] * cmf, axis=0)
        * float(INTEGRATION_STEP_NM)
    )
    denominator = float(light_integral[1])
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise DomainError(
            E_WHITE_POINT,
            "light SPD has zero luminance under the selected observer",
            pointer=f"/lights/{light.id}/spd",
            details={"observer": canonical_observer, "light_y_integral": denominator},
        )
    k = 100.0 / denominator
    xyz = k * reflectance_integral
    source_white_raw = k * light_integral
    _check_source_white(source_white_raw, light_id=light.id)
    source_white = source_white_raw / source_white_raw[1]

    return {
        "material_id": material.id,
        "light_id": light.id,
        "observer": canonical_observer,
        "observer_name": OBSERVER_DISPLAY_NAMES[canonical_observer],
        "cmf_version": CMF_TABLE_VERSION,
        "cmf_hash": cmf_hash(canonical_observer),
        "wavelength_grid_hash": sha256_obj(list(WAVELENGTHS_NM)),
        "integration_step_nm": INTEGRATION_STEP_NM,
        "normalization": {
            "method": "normalize_light_Y_to_100",
            "k": qfloat(k),
            "light_Y_integral": qfloat(denominator),
            "scale": "Y=100 for a perfect diffuse reflector under this source",
        },
        "terms": {
            "light_integral_xyz": _clean_sequence(light_integral),
            "reflectance_x_light_integral_xyz": _clean_sequence(reflectance_integral),
        },
        "source_white_raw_xyz": _clean_sequence(source_white_raw),
        "source_white_xyz_y1": _clean_sequence(source_white),
        "xyz": _clean_sequence(xyz),
    }


def _mat3_vec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    return [
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    ]


def _mat3_mul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(float(left[row][k]) * float(right[k][col]) for k in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def adaptation_matrix(
    method: str,
    source_white_xyz: Sequence[float],
    camera_white_xyz: Sequence[float],
) -> dict[str, Any]:
    canonical_method = normalize_adaptation_method(method)
    source = normalize_white_point_for_adaptation(source_white_xyz)
    camera = normalize_white_point_for_adaptation(camera_white_xyz)

    if canonical_method == "none":
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        return {
            "method": "none",
            "applied": False,
            "source_white_xyz_y1": _clean_sequence(source),
            "camera_white_point_xyz_y1": _clean_sequence(camera),
            "matrix": identity,
            "basis": {"name": "identity", "matrix": identity, "inverse": identity},
            "cone_source": _clean_sequence(source),
            "cone_camera": _clean_sequence(camera),
            "cone_scale": [1.0, 1.0, 1.0],
        }

    basis = ADAPTATION_BASES[canonical_method]
    cone_source = _mat3_vec(basis.matrix, source)
    cone_camera = _mat3_vec(basis.matrix, camera)
    for index, value in enumerate(cone_source):
        if not math.isfinite(float(value)) or float(value) == 0.0:
            raise DomainError(
                E_WHITE_POINT,
                "source white point cannot be transformed by the adaptation basis",
                details={"component": index, "cone_value": value},
            )
    for index, value in enumerate(cone_camera):
        if not math.isfinite(float(value)) or float(value) == 0.0:
            raise DomainError(
                E_WHITE_POINT,
                "camera white point cannot be transformed by the adaptation basis",
                details={"component": index, "cone_value": value},
            )

    applied = any(
        abs(source[index] - camera[index]) > 1e-12 for index in range(3)
    )
    if not applied:
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        scale = [1.0, 1.0, 1.0]
    else:
        scale = [cone_camera[index] / cone_source[index] for index in range(3)]
        diag = [
            [scale[0], 0.0, 0.0],
            [0.0, scale[1], 0.0],
            [0.0, 0.0, scale[2]],
        ]
        matrix = _mat3_mul(_mat3_mul(basis.inverse, diag), basis.matrix)

    return {
        "method": canonical_method,
        "applied": applied,
        "source_white_xyz_y1": _clean_sequence(source),
        "camera_white_point_xyz_y1": _clean_sequence(camera),
        "matrix": [[qfloat(value) for value in row] for row in matrix],
        "basis": {
            "name": basis.name,
            "matrix": [[qfloat(value) for value in row] for row in basis.matrix],
            "inverse": [[qfloat(value) for value in row] for row in basis.inverse],
        },
        "cone_source": _clean_sequence(cone_source),
        "cone_camera": _clean_sequence(cone_camera),
        "cone_scale": _clean_sequence(scale),
    }


def normalize_white_point_for_adaptation(
    value: Sequence[float],
) -> tuple[float, float, float]:
    if len(value) != 3:
        raise DomainError(E_WHITE_POINT, "white point must have three XYZ components")
    components = [float(value[index]) for index in range(3)]
    for index, number in enumerate(components):
        if not math.isfinite(number) or number <= 0.0:
            raise DomainError(
                E_WHITE_POINT,
                "white point components must be finite and strictly positive",
                details={"component": index, "raw": components[index]},
            )
    return (components[0] / components[1], 1.0, components[2] / components[1])


def adapt_xyz(
    spectral_product: dict[str, Any],
    camera_white_point_xyz: Sequence[float],
    method: str,
) -> dict[str, Any]:
    camera = validate_white_point(list(camera_white_point_xyz))
    source_raw = spectral_product.get("source_white_xyz_y1")
    if not isinstance(source_raw, list) or len(source_raw) != 3:
        raise DomainError(
            E_WHITE_POINT,
            "spectral stage did not provide a source white point",
            details={"stage": "spectral_xyz"},
        )
    matrix_info = adaptation_matrix(method, source_raw, camera)
    source_xyz = spectral_product["xyz"]
    adapted = _mat3_vec(matrix_info["matrix"], source_xyz)
    return {
        "material_id": spectral_product["material_id"],
        "light_id": spectral_product["light_id"],
        "method": matrix_info["method"],
        "applied": matrix_info["applied"],
        "source_white_xyz_y1": matrix_info["source_white_xyz_y1"],
        "camera_white_point_xyz_y1": matrix_info["camera_white_point_xyz_y1"],
        "matrix": matrix_info["matrix"],
        "basis": matrix_info["basis"],
        "cone_source": matrix_info["cone_source"],
        "cone_camera": matrix_info["cone_camera"],
        "cone_scale": matrix_info["cone_scale"],
        "xyz": _clean_sequence(adapted),
    }


def _srgb_encode(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def xyz_to_target(
    adapted_xyz: Sequence[float], target_color_space: str
) -> dict[str, Any]:
    space_name = normalize_color_space(target_color_space)
    space = COLOR_SPACES[space_name]
    xyz = [float(value) for value in adapted_xyz]

    if space.matrix_xyz_to_linear_rgb is None:
        return {
            "space": space.name,
            "transfer": space.transfer,
            "reference_white_xyz_y1": list(space.white_xyz),
            "xyz": _clean_sequence(xyz),
            "linear_rgb": None,
            "rgb_unclipped": None,
            "rgb": None,
            "in_gamut": True,
        }

    scaled_xyz = [value / 100.0 for value in xyz]
    linear = _mat3_vec(space.matrix_xyz_to_linear_rgb, scaled_xyz)
    encoded_unclipped = [_srgb_encode(value) for value in linear]
    clipped = [min(1.0, max(0.0, value)) for value in encoded_unclipped]
    in_gamut = all(-1e-12 <= value <= 1.0 + 1e-12 for value in linear)
    return {
        "space": space.name,
        "transfer": space.transfer,
        "reference_white_xyz_y1": list(space.white_xyz),
        "xyz": _clean_sequence(xyz),
        "linear_rgb": _clean_sequence(linear),
        "rgb_unclipped": _clean_sequence(encoded_unclipped),
        "rgb": _clean_sequence(clipped),
        "in_gamut": bool(in_gamut),
    }
