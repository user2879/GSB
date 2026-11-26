"""Integer-rational DLT triangulation with fixed-point quantization records."""

from __future__ import annotations

from typing import Any

from .constants import ENGINE_VERSION
from .fixed import determinant3, div_round_nearest, quantize, stable_hash
from .geometry import project_fixed
from .matching import pair_identifier
from .models import Camera, ReconstructionConfig


def triangle_identifier(pair: dict[str, Any], inlier: dict[str, Any]) -> str:
    first, second = pair["view_ids"]
    left_x, left_y = inlier["left"]
    right_x, right_y = inlier["right"]
    return (
        f"t:{min(first, second):04d}:{max(first, second):04d}:"
        f"{left_x:04d}:{left_y:04d}:{right_x:04d}:{right_y:04d}"
    )


def triangle_key(pair: dict[str, Any], inlier: dict[str, Any], config: ReconstructionConfig) -> str:
    return stable_hash(
        {
            "engine_version": ENGINE_VERSION,
            "stage": "triangulation",
            "pair_key": pair["key"],
            "inlier": inlier,
            "bits": config.bits,
            "quant_level": config.quant_level,
            "min_disparity": config.min_disparity,
        }
    )


def _signed_ratio_fixed(numerator: int, denominator: int, scale: int) -> tuple[int, str, int]:
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    return div_round_nearest(numerator * scale, denominator)


def _replace_column(matrix: list[list[int]], column: int, vector: list[int]) -> list[list[int]]:
    return [
        [vector[row] if index == column else matrix[row][index] for index in range(3)]
        for row in range(3)
    ]


def _dlt_rows(camera: Camera, point: list[int]) -> list[list[int]]:
    projection = camera.projection_numerator()
    u, v = point
    row_u = [u * projection[2][index] - projection[0][index] for index in range(4)]
    row_v = [v * projection[2][index] - projection[1][index] for index in range(4)]
    return [row_u, row_v]


def _reprojection_residual(
    cameras: dict[int, Camera],
    pair: dict[str, Any],
    inlier: dict[str, Any],
    q_point: tuple[int, int, int],
    scale: int,
) -> tuple[int, dict[str, Any]]:
    homogeneous = (q_point[0], q_point[1], q_point[2], scale)
    residual = 0
    directions: dict[str, list[Any]] = {}
    for side, view_id in (("left", pair["view_ids"][0]), ("right", pair["view_ids"][1])):
        camera = cameras[view_id]
        projection = camera.projection_numerator()
        rows = [
            sum(projection[row][index] * homogeneous[index] for index in range(4))
            for row in range(3)
        ]
        coordinate = inlier[side]
        if rows[2] == 0:
            return -1, {"error": "zero_projective_depth"}
        fixed_errors = []
        for axis, expected in (("u", coordinate[0]), ("v", coordinate[1])):
            numerator = rows[0] - expected * rows[2] if axis == "u" else rows[1] - expected * rows[2]
            error_fixed, direction, remainder = _signed_ratio_fixed(
                abs(numerator), abs(rows[2]), scale
            )
            fixed_errors.append(error_fixed)
            directions[f"{side}.{axis}"] = {
                "fixed": error_fixed,
                "direction": direction,
                "remainder": remainder,
            }
        residual += sum(fixed_errors)
    return residual, directions


def triangulate_inlier(
    view_ids: tuple[int, int],
    cameras: dict[int, Camera],
    pair: dict[str, Any],
    inlier: dict[str, Any],
    config: ReconstructionConfig,
) -> tuple[dict[str, Any] | None, str]:
    rows = _dlt_rows(cameras[view_ids[0]], inlier["left"])
    rows.extend(_dlt_rows(cameras[view_ids[1]], inlier["right"]))
    matrix = [row[:3] for row in rows]
    rhs = [row[3] for row in rows]
    determinant = determinant3(matrix)
    if determinant == 0:
        return None, "parallel_or_coplanar_degenerate"
    scale = config.scale
    numerators = []
    for column in range(3):
        numerators.append(
            determinant3(_replace_column(matrix, column, [-value for value in rhs]))
        )
    raw_values = []
    division_rounding = []
    for numerator in numerators:
        value, direction, remainder = _signed_ratio_fixed(numerator, determinant, scale)
        raw_values.append(value)
        division_rounding.append(
            {"direction": direction, "remainder": remainder}
        )
    quantized_values = []
    quantization = []
    for value in raw_values:
        q_value, direction, remainder = quantize(value, config.quant_step)
        quantized_values.append(q_value)
        quantization.append(
                {"step": config.quant_step, "direction": direction, "remainder": remainder}
        )
    q_point = tuple(quantized_values)
    residual, residual_rounding = _reprojection_residual(
        cameras, pair, inlier, q_point, scale
    )
    if residual < 0:
        return None, residual_rounding.get("error", "reprojection_failure")
    identifier = triangle_identifier(pair, inlier)
    key = triangle_key(pair, inlier, config)
    record = {
        "id": identifier,
        "key": key,
        "pair_id": pair_identifier(*view_ids),
        "pair_key": pair["key"],
        "view_ids": list(view_ids),
        "left_id": inlier["left_id"],
        "right_id": inlier["right_id"],
        "left": inlier["left"],
        "right": inlier["right"],
        "determinant": determinant,
        "xyz_fixed": [raw_values[0], raw_values[1], raw_values[2]],
        "xyz_q": [q_point[0], q_point[1], q_point[2]],
        "scale": scale,
        "division_rounding": [
            {"value": raw_values[index], **division_rounding[index]} for index in range(3)
        ],
        "quantization": [
            {
                "value": q_point[index],
                "raw": raw_values[index],
                **quantization[index],
            }
            for index in range(3)
        ],
        "residual_fixed": residual,
        "residual_rounding": residual_rounding,
    }
    return record, ""


def project_triangle(
    triangle: dict[str, Any],
    camera: Camera,
    config: ReconstructionConfig,
) -> tuple[int, int, int] | None:
    return project_fixed(
        camera,
        triangle["xyz_q"][0],
        triangle["xyz_q"][1],
        triangle["xyz_q"][2],
        config.scale,
    )
