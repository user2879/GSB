"""Integer fixed-point projection, triangulation and quantization."""

from __future__ import annotations

from typing import Any, Iterable

from .fixed import digest, quantize_xyz, round_div


def projection(
    xyz_fixed: Iterable[int],
    intrinsics: list[list[int]],
    extrinsics: list[list[int]],
    fixed_bits: int,
) -> dict[str, Any]:
    x, y, z = (int(value) for value in xyz_fixed)
    scale = 1 << fixed_bits
    rotation = [row[:3] for row in extrinsics]
    translation = [row[3] for row in extrinsics]
    camera = [
        rotation[i][0] * x + rotation[i][1] * y + rotation[i][2] * z + translation[i] * scale
        for i in range(3)
    ]
    u_num = intrinsics[0][0] * camera[0] + intrinsics[0][1] * camera[1] + intrinsics[0][2] * camera[2]
    v_num = intrinsics[1][0] * camera[0] + intrinsics[1][1] * camera[1] + intrinsics[1][2] * camera[2]
    w = intrinsics[2][0] * camera[0] + intrinsics[2][1] * camera[1] + intrinsics[2][2] * camera[2]
    if w == 0:
        return {
            "valid": False,
            "u": None,
            "v": None,
            "depth": None,
            "rounding": {"u": "invalid", "v": "invalid"},
            "camera": camera,
        }
    u, u_dir = round_div(u_num, w)
    v, v_dir = round_div(v_num, w)
    return {
        "valid": True,
        "u": u,
        "v": v,
        "depth": w,
        "camera": camera,
        "rounding": {"u": u_dir, "v": v_dir},
    }


def _det3(matrix: list[list[int]]) -> int:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _solve3(matrix: list[list[int]], rhs: list[int]) -> tuple[list[int], list[str], int]:
    determinant = _det3(matrix)
    if determinant == 0:
        regularized = [row[:] for row in matrix]
        for index in range(3):
            regularized[index][index] += 1
        determinant = _det3(regularized)
        if determinant == 0:
            raise ZeroDivisionError("singular triangulation normal matrix")
        matrix = regularized
    values: list[int] = []
    directions: list[str] = []
    for column in range(3):
        replaced = [row[:] for row in matrix]
        for row in range(3):
            replaced[row][column] = rhs[row]
        numerator = _det3(replaced)
        value, direction = round_div(numerator, determinant)
        values.append(value)
        directions.append(direction)
    return values, directions, determinant


def triangulate_observation(
    left_coord: list[int],
    right_coord: list[int],
    left_camera: dict[str, Any],
    right_camera: dict[str, Any],
    fixed_bits: int,
    quantum: int,
) -> dict[str, Any]:
    scale = 1 << fixed_bits
    matrix_rows: list[list[int]] = []
    rhs: list[int] = []
    observations: list[dict[str, Any]] = []
    for camera, observed, role in (
        (left_camera, left_coord, "left"),
        (right_camera, right_coord, "right"),
    ):
        intrinsics = camera["intrinsics"]
        extrinsics = camera["extrinsics"]
        rotation = [row[:3] for row in extrinsics]
        translation = [row[3] for row in extrinsics]
        for axis, pixel in (("u", int(observed[0])), ("v", int(observed[1]))):
            k0 = intrinsics[0] if axis == "u" else intrinsics[1]
            k2 = intrinsics[2]
            combined = [k2[i] * pixel - k0[i] for i in range(3)]
            coefficients = [sum(combined[i] * rotation[i][j] for i in range(3)) for j in range(3)]
            row_rhs = -sum(combined[i] * translation[i] for i in range(3)) * scale
            matrix_rows.append(coefficients)
            rhs.append(row_rhs)
            observations.append(
                {
                    "role": role,
                    "view": camera["id"],
                    "axis": axis,
                    "pixel": pixel,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                }
            )

    normal = [[0, 0, 0] for _ in range(3)]
    normal_rhs = [0, 0, 0]
    for row, row_rhs in zip(matrix_rows, rhs):
        for i in range(3):
            normal_rhs[i] += row[i] * row_rhs
            for j in range(3):
                normal[i][j] += row[i] * row[j]
    raw_xyz, solve_directions, determinant = _solve3(normal, normal_rhs)
    quantized_xyz, quantization_directions = quantize_xyz(raw_xyz, quantum)

    reprojections: list[dict[str, Any]] = []
    residual = 0
    for observation in observations:
        projected = projection(
            quantized_xyz,
            observation["intrinsics"],
            observation["extrinsics"],
            fixed_bits,
        )
        expected = observation["pixel"]
        error = int(projected[observation["axis"]]) - expected if projected["valid"] else 0
        residual += error * error
        reprojections.append(
            {
                "view": observation["view"],
                "axis": observation["axis"],
                "expected": expected,
                "projected": projected[observation["axis"]],
                "error": error,
            }
        )
    return {
        "raw_xyz": raw_xyz,
        "xyz": quantized_xyz,
        "residual": residual,
        "solve_rounding": {axis: direction for axis, direction in zip(("x", "y", "z"), solve_directions)},
        "quantization_rounding": quantization_directions,
        "determinant": determinant,
        "reprojections": reprojections,
    }


def point_stable_key(pair_identifier: str, left_feature: str, right_feature: str) -> str:
    return f"{pair_identifier}:{left_feature}:{right_feature}"


def point_cache_payload(
    pair: dict[str, Any],
    left_camera: dict[str, Any],
    right_camera: dict[str, Any],
    stable_key: str,
    *,
    fixed_bits: int,
    quantum: int,
    quantization_level: int,
) -> dict[str, Any]:
    return {
        "stage": "point",
        "stable_key": stable_key,
        "pair_cache_key": pair["cache_key"],
        "left_camera_key": left_camera["camera_hash"],
        "right_camera_key": right_camera["camera_hash"],
        "fixed_bits": fixed_bits,
        "quantum": quantum,
        "quantization_level": quantization_level,
        "algorithm": "integer-normal-cramer-v1",
    }


def build_point_candidates(
    selected_pairs: list[dict[str, Any]],
    features_by_view: dict[str, dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    *,
    fixed_bits: int,
    quantum: int,
    quantization_level: int,
    residual_threshold: int,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for pair in sorted(selected_pairs, key=lambda item: item["pair_id"]):
        left_view = pair["left"]
        right_view = pair["right"]
        left_features = {feature["id"]: feature for feature in features_by_view[left_view]["features"]}
        right_features = {feature["id"]: feature for feature in features_by_view[right_view]["features"]}
        for match in sorted(
            pair["matches"],
            key=lambda item: (item["left_coord"][0], item["left_coord"][1], item["right_coord"], item["right_feature"]),
        ):
            left_feature = left_features[match["left_feature"]]
            right_feature = right_features[match["right_feature"]]
            stable_key = point_stable_key(pair["pair_id"], left_feature["id"], right_feature["id"])
            triangulated = triangulate_observation(
                left_feature["coord"],
                right_feature["coord"],
                cameras[left_view],
                cameras[right_view],
                fixed_bits,
                quantum,
            )
            residual_pass = triangulated["residual"] <= residual_threshold
            point_id = "p" + digest(stable_key)[:20]
            cache_payload = point_cache_payload(
                pair,
                cameras[left_view],
                cameras[right_view],
                stable_key,
                fixed_bits=fixed_bits,
                quantum=quantum,
                quantization_level=quantization_level,
            )
            points.append(
                {
                    "id": point_id,
                    "stable_key": stable_key,
                    "sort_key": stable_key,
                    "pair_id": pair["pair_id"],
                    "left_view": left_view,
                    "right_view": right_view,
                    "left_feature": left_feature["id"],
                    "right_feature": right_feature["id"],
                    "left_coord": left_feature["coord"],
                    "right_coord": right_feature["coord"],
                    "raw_xyz": triangulated["raw_xyz"],
                    "xyz": triangulated["xyz"],
                    "residual": triangulated["residual"],
                    "residual_pass": residual_pass,
                    "solve_rounding": triangulated["solve_rounding"],
                    "quantization_rounding": triangulated["quantization_rounding"],
                    "determinant": triangulated["determinant"],
                    "reprojections": triangulated["reprojections"],
                    "cache_key": digest(cache_payload),
                    "cache_basis": cache_payload,
                    "budget_accepted": False,
                    "budget_owner_views": [left_view, right_view],
                }
            )
    points.sort(key=lambda point: point["sort_key"])
    return points
