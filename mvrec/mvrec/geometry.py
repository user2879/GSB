"""Cross-view matching, triangulation, depth maps and voxel merging."""

import numpy as np

from .errors import (
    E_MATCH_TOO_FEW_INLIERS,
    E_PARALLAX_INVALID,
    E_VOXEL_INVALID,
    PipelineError,
)

MATCH_DISTANCE_THRESHOLD = 0.25


def _projection_matrix(view):
    K, R, t = view["K"], view["R"], view["t"]
    return K @ np.hstack([R, t.reshape(3, 1)])


def match_views(view_a, features_a, view_b, features_b):
    """Mutual-nearest-neighbour descriptor matching with a distance gate."""
    if not features_a or not features_b:
        return []
    desc_a = np.stack([f["descriptor"] for f in features_a])
    desc_b = np.stack([f["descriptor"] for f in features_b])
    distances = np.sqrt(
        np.maximum(
            0.0,
            np.sum(desc_a ** 2, axis=1)[:, None]
            + np.sum(desc_b ** 2, axis=1)[None, :]
            - 2.0 * desc_a @ desc_b.T,
        )
    )
    matches = []
    for index_a, feature_a in enumerate(features_a):
        index_b = int(np.argmin(distances[index_a]))
        reverse_index = int(np.argmin(distances[:, index_b]))
        if reverse_index != index_a:
            continue
        distance = float(distances[index_a, index_b])
        if distance > MATCH_DISTANCE_THRESHOLD:
            continue
        feature_b = features_b[index_b]
        matches.append({
            "view_a": view_a["view_id"],
            "view_b": view_b["view_id"],
            "feature_a": feature_a["feature_id"],
            "feature_b": feature_b["feature_id"],
            "key_a": feature_a["stable_key"],
            "key_b": feature_b["stable_key"],
            "distance": distance,
            "stable_key": (
                min(view_a["view_id"], view_b["view_id"]),
                max(view_a["view_id"], view_b["view_id"]),
                min(feature_a["stable_key"], feature_b["stable_key"]),
                max(feature_a["stable_key"], feature_b["stable_key"]),
            ),
        })
    matches.sort(key=lambda item: item["stable_key"])
    return matches


def _triangulate_point(P_a, point_a, P_b, point_b):
    xa, ya = point_a
    xb, yb = point_b
    A = np.array([
        xa * P_a[2] - P_a[0],
        ya * P_a[2] - P_a[1],
        xb * P_b[2] - P_b[0],
        yb * P_b[2] - P_b[1],
    ])
    _, _, vt = np.linalg.svd(A)
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    point3d = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(point3d)):
        return None
    return point3d


def _bearing(view, point2d):
    K_inv = np.linalg.inv(view["K"])
    ray_cam = K_inv @ np.array([point2d[0], point2d[1], 1.0])
    ray_world = view["R"].T @ ray_cam
    return ray_world / np.linalg.norm(ray_world)


def triangulate_matches(view_a, features_a, view_b, features_b, matches,
                        min_parallax_deg, injections=None):
    """Triangulate matches with parallax, cheirality and reprojection checks."""
    inject = (injections or {}).get(str(view_a["view_id"]),
                                    (injections or {}).get(view_a["view_id"], {}))
    P_a, P_b = _projection_matrix(view_a), _projection_matrix(view_b)
    feat_a_by_id = {f["feature_id"]: f for f in features_a}
    feat_b_by_id = {f["feature_id"]: f for f in features_b}
    points = []
    for match in matches:
        fa = feat_a_by_id[match["feature_a"]]
        fb = feat_b_by_id[match["feature_b"]]
        point3d = _triangulate_point(P_a, (fa["x"], fa["y"]), P_b, (fb["x"], fb["y"]))
        if point3d is None:
            continue
        depth_a = float(view_a["R"][2] @ point3d + view_a["t"][2])
        depth_b = float(view_b["R"][2] @ point3d + view_b["t"][2])
        if depth_a <= 0 or depth_b <= 0:
            continue
        bearing_a = _bearing(view_a, (fa["x"], fa["y"]))
        bearing_b = _bearing(view_b, (fb["x"], fb["y"]))
        cos_angle = float(np.clip(np.dot(bearing_a, bearing_b), -1.0, 1.0))
        parallax_deg = float(np.degrees(np.arccos(cos_angle)))
        if inject.get("parallax_violation"):
            parallax_deg = 0.0
        if parallax_deg < min_parallax_deg:
            continue
        repro_a = P_a @ np.append(point3d, 1.0)
        repro_b = P_b @ np.append(point3d, 1.0)
        if abs(repro_a[2]) < 1e-12 or abs(repro_b[2]) < 1e-12:
            continue
        err_a = np.hypot(repro_a[0] / repro_a[2] - fa["x"], repro_a[1] / repro_a[2] - fa["y"])
        err_b = np.hypot(repro_b[0] / repro_b[2] - fb["x"], repro_b[1] / repro_b[2] - fb["y"])
        confidence = float(
            np.clip(1.0 - 0.5 * (err_a + err_b) - 0.01 * match["distance"], 0.0, 1.0)
        )
        points.append({
            "xyz": point3d,
            "source_views": [view_a["view_id"], view_b["view_id"]],
            "pair_id": "%d-%d" % tuple(sorted((view_a["view_id"], view_b["view_id"]))),
            "feature_a": fa["feature_id"],
            "feature_b": fb["feature_id"],
            "parallax_deg": parallax_deg,
            "confidence": confidence,
            "stable_key": (
                min(view_a["view_id"], view_b["view_id"]),
                max(view_a["view_id"], view_b["view_id"]),
                min(fa["stable_key"], fb["stable_key"]),
                max(fa["stable_key"], fb["stable_key"]),
            ),
        })
    points.sort(key=lambda item: item["stable_key"])
    return points


def match_and_triangulate_pair(view_a, features_a, view_b, features_b,
                               min_inliers, min_parallax_deg, injections=None):
    matches = match_views(view_a, features_a, view_b, features_b)
    if len(matches) < min_inliers:
        raise PipelineError(
            E_MATCH_TOO_FEW_INLIERS,
            "only %d inlier matches between views %d and %d"
            % (len(matches), view_a["view_id"], view_b["view_id"]),
            {"view_a": view_a["view_id"], "view_b": view_b["view_id"],
             "inliers": len(matches)},
        )
    points = triangulate_matches(
        view_a, features_a, view_b, features_b, matches, min_parallax_deg, injections
    )
    if not points:
        raise PipelineError(
            E_PARALLAX_INVALID,
            "no triangulated point satisfied the parallax constraint",
            {"view_a": view_a["view_id"], "view_b": view_b["view_id"]},
        )
    return matches, points


def build_depth_map(view, points3d, config):
    """Project points into a view, quantize depth and apply box smoothing."""
    width, height = config["depth_width"], config["depth_height"]
    levels, passes = config["depth_levels"], config["depth_smooth_passes"]
    P = _projection_matrix(view)
    depth = np.zeros((height, width), dtype=np.float64)
    for point in points3d:
        projected = P @ np.append(point["xyz"], 1.0)
        if projected[2] <= 0:
            continue
        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        col = int(np.floor(u * width / view["width"]))
        row = int(np.floor(v * height / view["height"]))
        if not (0 <= row < height and 0 <= col < width):
            continue
        z_value = float(projected[2])
        if depth[row, col] == 0.0 or z_value < depth[row, col]:
            depth[row, col] = z_value
    finite = depth[depth > 0]
    if finite.size:
        d_min, d_max = float(finite.min()), float(finite.max())
        span = (d_max - d_min) if d_max > d_min else 1.0
        quantized = np.zeros_like(depth)
        occupied = depth > 0
        quantized[occupied] = np.clip(
            np.floor((depth[occupied] - d_min) / span * (levels - 1)) + 1, 1, levels
        )
    else:
        quantized = depth.copy()
    smoothed = quantized.copy()
    for _ in range(passes):
        updated = smoothed.copy()
        for row in range(height):
            for col in range(width):
                if smoothed[row, col] == 0:
                    continue
                values = [smoothed[row, col]]
                for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + drow, col + dcol
                    if 0 <= nr < height and 0 <= nc < width and smoothed[nr, nc] > 0:
                        values.append(smoothed[nr, nc])
                updated[row, col] = round(float(sum(values)) / len(values), 6)
        smoothed = updated
    occupied = int(np.count_nonzero(smoothed))
    summary = {
        "view_id": view["view_id"],
        "occupied_cells": occupied,
        "depth_sum": round(float(smoothed.sum()), 6),
        "depth_max": round(float(smoothed.max()), 6),
        "stable_key": (view["view_id"],),
    }
    return smoothed, summary


def voxel_key(xyz, voxel_size):
    return tuple(int(np.floor(coord / voxel_size)) for coord in xyz)


def merge_point_cloud(point_records, voxel_size):
    """Merge points into voxels; representative chosen by stable coordinate order."""
    if voxel_size <= 0:
        raise PipelineError(E_VOXEL_INVALID, "voxel size must be positive")
    buckets = {}
    for record in point_records:
        key = voxel_key(record["xyz"], voxel_size)
        buckets.setdefault(key, []).append(record)
    merged = []
    for key in sorted(buckets):
        members = sorted(
            buckets[key],
            key=lambda item: (
                tuple(round(float(v), 9) for v in item["xyz"]),
                item["stable_key"],
            ),
        )
        representative = members[0]
        xyz = np.stack([item["xyz"] for item in members]).mean(axis=0)
        merged.append({
            "xyz": xyz,
            "voxel_key": key,
            "source_views": sorted({v for item in members for v in item["source_views"]}),
            "pair_id": representative["pair_id"],
            "confidence": round(float(np.mean([item["confidence"] for item in members])), 9),
            "member_count": len(members),
            "stable_key": key,
        })
    merged.sort(key=lambda item: item["stable_key"])
    return merged
