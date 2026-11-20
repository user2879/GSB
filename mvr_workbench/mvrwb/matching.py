"""跨视图描述子匹配、对极几何内点筛选与线性三角化。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import ReconstructionConfig
from .errors import E_PARALLAX_OUT_OF_RANGE, StageError
from .util import content_fingerprint, fingerprint_many, quantize_float, quantize_scalar, view_pair_id

EPIPOLAR_TOL_PIXELS = 2.5
REPROJECTION_TOL_PIXELS = 2.0


def _match_distance(left: list[int], right: list[int]) -> float:
    """归一化互相关距离（1-NCC），量化描述子反量化后计算，跨视图对亮度稳健。"""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 2.0
    ncc = float(np.dot(a, b) / denom)
    return 0.0 if abs(1.0 - ncc) < 1e-12 else 1.0 - ncc


def _distance_matrix(features_a: list[dict], features_b: list[dict]) -> np.ndarray:
    """向量化 NCC 距离矩阵 (1-NCC)，与逐对 _match_distance 同值。"""
    matrix_a = np.asarray([f["descriptor"] for f in features_a], dtype=np.float64)
    matrix_b = np.asarray([f["descriptor"] for f in features_b], dtype=np.float64)
    centered_a = matrix_a - matrix_a.mean(axis=1, keepdims=True)
    centered_b = matrix_b - matrix_b.mean(axis=1, keepdims=True)
    norm_a = np.linalg.norm(centered_a, axis=1)
    norm_b = np.linalg.norm(centered_b, axis=1)
    denom = np.outer(norm_a, norm_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = (centered_a @ centered_b.T) / np.where(denom > 0.0, denom, 1.0)
    distance = 1.0 - ncc
    distance[denom <= 0.0] = 2.0
    # 负零归一，保证跨运行字节级稳定
    return np.where(np.abs(distance) < 1e-12, 0.0, distance)


def _mutual_nn(features_a: list[dict], features_b: list[dict]) -> list[tuple[int, int, float]]:
    if not features_a or not features_b:
        return []
    distance_matrix = _distance_matrix(features_a, features_b)
    # 距离升序、并列取目标索引最小：argmin 天然返回首个最小索引
    a_to_b = np.argmin(distance_matrix, axis=1)
    b_to_a = np.argmin(distance_matrix, axis=0)
    mutual: list[tuple[int, int, float]] = []
    for index_a in range(len(features_a)):
        index_b = int(a_to_b[index_a])
        if int(b_to_a[index_b]) == index_a:
            mutual.append((index_a, index_b, float(distance_matrix[index_a, index_b])))
    mutual.sort(key=lambda item: (item[2], item[0], item[1]))
    return mutual


def _epipolar_inliers(
    matches: list[tuple[int, int, int]],
    features_a: list[dict],
    features_b: list[dict],
    camera_a,
    camera_b,
) -> list[tuple[int, int, int]]:
    # 本质矩阵 E = [t]_x R，其中相机 B 相对相机 A 的位姿
    rel_rotation = camera_b.rotation @ camera_a.rotation.T
    rel_translation = camera_b.translation - rel_rotation @ camera_a.translation
    tx, ty, tz = rel_translation
    skew = np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]])
    essential = skew @ rel_rotation
    inv_ka = np.linalg.inv(camera_a.intrinsics)
    inv_kb = np.linalg.inv(camera_b.intrinsics)

    inliers: list[tuple[int, int, int]] = []
    for index_a, index_b, distance in matches:
        pa = np.array([features_a[index_a]["x"], features_a[index_a]["y"], 1.0])
        pb = np.array([features_b[index_b]["x"], features_b[index_b]["y"], 1.0])
        ray_a = inv_ka @ pa
        ray_b = inv_kb @ pb
        # Sampson 一阶几何误差近似
        fpb = essential @ ray_b
        e_t_pa = essential.T @ ray_a
        denom = fpb[0] ** 2 + fpb[1] ** 2 + e_t_pa[0] ** 2 + e_t_pa[1] ** 2
        if denom <= 0.0:
            continue
        numerator = ray_a @ essential @ ray_b
        sampson = (numerator * numerator) / denom
        if sampson <= EPIPOLAR_TOL_PIXELS ** 2:
            inliers.append((index_a, index_b, distance))
    return inliers


def _triangulate_dlt(ray_a: np.ndarray, ray_b: np.ndarray,
                     camera_a, camera_b) -> np.ndarray:
    """标准线性三角化。

    输入 ray_* 为内参逆变换后的归一化相机坐标，因此投影矩阵取外参
    [R|t]（不含 K）；由归一化坐标叉乘投影行构造齐次方程组后 SVD 求解。
    """
    projection_a = np.hstack(
        [camera_a.rotation, camera_a.translation.reshape(3, 1)]
    )
    projection_b = np.hstack(
        [camera_b.rotation, camera_b.translation.reshape(3, 1)]
    )
    xa, ya = ray_a[0] / ray_a[2], ray_a[1] / ray_a[2]
    xb, yb = ray_b[0] / ray_b[2], ray_b[1] / ray_b[2]
    rows = [
        xa * projection_a[2] - projection_a[0],
        ya * projection_a[2] - projection_a[1],
        xb * projection_b[2] - projection_b[0],
        yb * projection_b[2] - projection_b[1],
    ]
    matrix_a = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(matrix_a)
    homogeneous = vh[-1]
    return homogeneous[:3] / homogeneous[3]


def _parallax_deg(point: np.ndarray, camera_a, camera_b) -> float:
    # 相机中心在世界系下为 -R^T t（translation 是相机系量，不能直接相减）
    center_a = -camera_a.rotation.T @ camera_a.translation
    center_b = -camera_b.rotation.T @ camera_b.translation
    ray_a = point - center_a
    ray_b = point - center_b
    cosine = float(np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b)))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def match_and_triangulate(
    view_a: int,
    view_b: int,
    feat_a: dict,
    feat_b: dict,
    camera_a,
    camera_b,
    cfg: ReconstructionConfig,
    *,
    force_parallax_fault: bool = False,
) -> dict[str, Any]:
    pair_id = view_pair_id(view_a, view_b)
    features_a = feat_a["content"]["features"]
    features_b = feat_b["content"]["features"]
    mutual = _mutual_nn(features_a, features_b)
    inliers = _epipolar_inliers(mutual, features_a, features_b, camera_a, camera_b)

    if len(inliers) < cfg.match_min_inliers:
        content = {
            "pair_id": pair_id,
            "view_a": view_a,
            "view_b": view_b,
            "accepted": False,
            "inlier_count": len(inliers),
            "match_min_inliers": cfg.match_min_inliers,
            "points": [],
            "matches": [],
        }
        return {
            "artifact_id": f"match-{pair_id}",
            "stage": "matching",
            "pair_id": pair_id,
            "view_ids": [view_a, view_b],
            "content": content,
            "content_fingerprint": content_fingerprint(content),
        }

    points: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    frac_bits = cfg.fixed_point_frac_bits
    for match_rank, (index_a, index_b, descriptor_distance) in enumerate(inliers):
        feature_a = features_a[index_a]
        feature_b = features_b[index_b]
        ray_a = np.linalg.inv(camera_a.intrinsics) @ np.array(
            [feature_a["x"], feature_a["y"], 1.0]
        )
        ray_b = np.linalg.inv(camera_b.intrinsics) @ np.array(
            [feature_b["x"], feature_b["y"], 1.0]
        )
        point = _triangulate_dlt(ray_a, ray_b, camera_a, camera_b)
        if not np.isfinite(point).all():
            continue
        pixels_a, depth_a, _ = camera_a.project(point.reshape(1, 3))
        pixels_b, depth_b, _ = camera_b.project(point.reshape(1, 3))
        if depth_a[0] <= 0.0 or depth_b[0] <= 0.0:
            continue
        error_a = float(np.linalg.norm(pixels_a[0] - np.array([feature_a["x"], feature_a["y"]])))
        error_b = float(np.linalg.norm(pixels_b[0] - np.array([feature_b["x"], feature_b["y"]])))
        if error_a > REPROJECTION_TOL_PIXELS or error_b > REPROJECTION_TOL_PIXELS:
            continue
        parallax = _parallax_deg(point, camera_a, camera_b)
        if force_parallax_fault:
            raise StageError(
                E_PARALLAX_OUT_OF_RANGE,
                f"{pair_id} 注入视差越界故障: 视差 {parallax:.4f} 度被判定非法",
                stage="matching", pair_id=pair_id,
            )
        if parallax < cfg.triang_min_parallax_deg:
            continue

        point_key = "pt-{}-{}-{}".format(
            pair_id, feature_a["feature_id"], feature_b["feature_id"]
        )
        confidence = (1.0 - min(1.0, descriptor_distance / 4096.0)) * (
            parallax / 90.0
        )
        points.append(
            {
                "point_key": point_key,
                "pair_id": pair_id,
                "owner_view_id": max(view_a, view_b),
                "source_view_ids": [view_a, view_b],
                "xyz_q": [quantize_scalar(v, frac_bits) for v in point.tolist()],
                "parallax_q": quantize_scalar(parallax, frac_bits),
                "confidence_q": quantize_scalar(confidence, frac_bits),
            }
        )
        matches.append(
            {
                "point_key": point_key,
                "feature_id_a": feature_a["feature_id"],
                "feature_id_b": feature_b["feature_id"],
                "pixel_a": [feature_a["x"], feature_a["y"]],
                "pixel_b": [feature_b["x"], feature_b["y"]],
                "descriptor_distance": descriptor_distance,
                "reproj_error_q": [
                    quantize_scalar(error_a, frac_bits),
                    quantize_scalar(error_b, frac_bits),
                ],
            }
        )

    # 三维点按 point_key 稳定排序，匹配随点顺序
    order = sorted(range(len(points)), key=lambda idx: points[idx]["point_key"])
    points = [points[idx] for idx in order]
    matches = [matches[idx] for idx in order]
    content = {
        "pair_id": pair_id,
        "view_a": view_a,
        "view_b": view_b,
        "accepted": True,
        "inlier_count": len(inliers),
        "match_min_inliers": cfg.match_min_inliers,
        "triang_min_parallax_q": quantize_scalar(cfg.triang_min_parallax_deg, frac_bits),
        "points": points,
        "matches": matches,
    }
    return {
        "artifact_id": f"match-{pair_id}",
        "stage": "matching",
        "pair_id": pair_id,
        "view_ids": [view_a, view_b],
        "content": content,
        "content_fingerprint": content_fingerprint(content),
    }


def match_artifact_key(pair_id: str, fp_a: str, fp_b: str, cam_fp_a: str,
                       cam_fp_b: str, param_fp: str) -> str:
    return fingerprint_many(
        "matching", pair_id, fp_a, fp_b, cam_fp_a, cam_fp_b, param_fp
    )


def dequantize_point(point: dict, frac_bits: int) -> np.ndarray:
    return np.asarray(
        [quantize_float(v / float(1 << frac_bits), frac_bits) for v in point["xyz_q"]],
        dtype=np.float64,
    )
