"""三角化、深度图融合与体素级点云合并。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import RebuildConfig
from .errors import BizCode, WorkbenchError
from .features import Feature, MatchRecord, feature_lookup
from .loader import Camera
from .utils import depth_quantize, fixed_point_quantize, voxel_key


def _dtype_for(precision: str) -> np.dtype:
    return np.dtype(np.float32 if precision == "float32" else np.float64)


def _projection_matrix(camera: Camera, dtype: np.dtype) -> np.ndarray:
    extrinsic = np.hstack(
        [np.asarray(camera.rotation, dtype=dtype),
         np.asarray(camera.translation, dtype=dtype).reshape(3, 1)]
    )
    return np.asarray(camera.intrinsic, dtype=dtype) @ extrinsic


@dataclass(frozen=True)
class TriPoint:
    point_id: str
    match_id: str
    view_a: int
    view_b: int
    x: float
    y: float
    z: float
    disparity: float
    confidence: float
    boundary: bool
    kept: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id, "match_id": self.match_id,
            "view_a": self.view_a, "view_b": self.view_b,
            "x": self.x, "y": self.y, "z": self.z,
            "disparity": self.disparity, "confidence": self.confidence,
            "boundary": self.boundary, "kept": self.kept,
        }


def triangulate_pair(matches: Sequence[MatchRecord],
                     features_a: Sequence[Feature],
                     features_b: Sequence[Feature],
                     camera_a: Camera, camera_b: Camera,
                     config: RebuildConfig, precision: str = "float64",
                     disparity_override: float | None = None
                     ) -> List[TriPoint]:
    """线性 DLT 三角化，最小视差门限（恰好相等保留并标记边界）。"""
    dtype = _dtype_for(precision)
    pa = _projection_matrix(camera_a, dtype)
    pb = _projection_matrix(camera_b, dtype)
    lookup_a = feature_lookup(features_a)
    lookup_b = feature_lookup(features_b)
    points: List[TriPoint] = []

    for order, match in enumerate(matches):
        fa = lookup_a[match.feature_a]
        fb = lookup_b[match.feature_b]
        xa, ya = np.array(fa.x, dtype=dtype), np.array(fa.y, dtype=dtype)
        xb, yb = np.array(fb.x, dtype=dtype), np.array(fb.y, dtype=dtype)
        a_mat = np.vstack([
            xa * pa[2] - pa[0],
            ya * pa[2] - pa[1],
            xb * pb[2] - pb[0],
            yb * pb[2] - pb[1],
        ])
        # SVD 求齐次最小奇异向量
        _, _, vh = np.linalg.svd(a_mat, full_matrices=False)
        homo = vh[-1]
        if abs(float(homo[3])) < 1e-15:
            continue
        coord = homo[:3] / homo[3]
        if not np.isfinite(coord).all():
            continue

        disparity = float(np.hypot(float(xa - xb), float(ya - yb)))
        if disparity_override is not None:
            disparity = float(disparity_override)
        boundary = np.isclose(disparity, config.min_disparity,
                              rtol=0.0, atol=1e-9)
        # 固定规则：>= 门限保留，< 门限丢弃；恰好相等标记边界样本
        kept = disparity >= config.min_disparity
        if not kept:
            continue

        # 置信度：两视图重投影误差的确定性映射
        proj_a = pa @ np.append(coord, 1.0)
        proj_b = pb @ np.append(coord, 1.0)
        rep_a = np.hypot(float(proj_a[0] / proj_a[2] - xa),
                         float(proj_a[1] / proj_a[2] - ya))
        rep_b = np.hypot(float(proj_b[0] / proj_b[2] - xb),
                         float(proj_b[1] / proj_b[2] - yb))
        confidence = float(1.0 / (1.0 + rep_a + rep_b))

        q = fixed_point_quantize(coord, config.fixed_point_bits).astype(dtype)
        points.append(
            TriPoint(
                point_id=f"p_{match.view_a:02d}_{match.view_b:02d}_{order:03d}",
                match_id=match.match_id, view_a=match.view_a,
                view_b=match.view_b,
                x=float(q[0]), y=float(q[1]), z=float(q[2]),
                disparity=disparity,
                confidence=float(fixed_point_quantize(
                    np.array(confidence, dtype=dtype),
                    config.fixed_point_bits)),
                boundary=bool(boundary), kept=True,
            )
        )
    return points


@dataclass
class DepthMap:
    view: int
    width: int
    height: int
    levels: np.ndarray  # int64，-1 表示空像素
    depth_values: np.ndarray  # 量化后的实际深度，空像素为 nan
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view": self.view, "width": self.width, "height": self.height,
            "levels": self.levels.tolist(),
            "depth_values": np.where(
                np.isnan(self.depth_values), None, self.depth_values
            ).tolist(),
            "summary": self.summary,
        }


def build_depth_map(points: Sequence[TriPoint], camera: Camera,
                    config: RebuildConfig, precision: str = "float64"
                    ) -> DepthMap:
    """把重建点投影到深度网格，量化档位 + 固定次数邻域平滑。"""
    dtype = _dtype_for(precision)
    width, height = config.depth_width, config.depth_height
    accum = np.full((height, width), np.nan, dtype=np.float64)
    best_conf = np.full((height, width), -np.inf, dtype=np.float64)
    intrinsic = np.asarray(camera.intrinsic, dtype=dtype)
    rotation = np.asarray(camera.rotation, dtype=dtype)
    translation = np.asarray(camera.translation, dtype=dtype)

    # 稳定处理顺序：按 point_id 排序，保证同像素冲突解决可复现
    for point in sorted(points, key=lambda p: p.point_id):
        world = np.array([point.x, point.y, point.z], dtype=dtype)
        cam = rotation @ world + translation
        depth = float(cam[2])
        if depth <= 0:
            continue
        pixel = intrinsic @ cam
        px = pixel[0] / pixel[2]
        py = pixel[1] / pixel[2]
        gx = int(np.clip(np.floor(px / camera.width * width), 0, width - 1))
        gy = int(np.clip(np.floor(py / camera.height * height), 0, height - 1))
        # 同像素多点：保留置信度高者，同置信度保留 point_id 字典序靠前者
        if point.confidence > best_conf[gy, gx]:
            accum[gy, gx] = depth
            best_conf[gy, gx] = point.confidence

    # 确定性平滑：邻域有效深度的固定权重均值，迭代固定次数
    smoothed = accum.copy()
    for _ in range(config.smooth_iterations):
        updated = smoothed.copy()
        for y in range(height):
            for x in range(width):
                if not np.isfinite(smoothed[y, x]):
                    continue
                neighbors = []
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and np.isfinite(
                        smoothed[ny, nx]
                    ):
                        neighbors.append(smoothed[ny, nx])
                if neighbors:
                    mean = float(np.mean(neighbors))
                    updated[y, x] = (
                        (1.0 - config.smooth_weight) * smoothed[y, x]
                        + config.smooth_weight * mean
                    )
        smoothed = updated

    levels = depth_quantize(smoothed, config.depth_levels)
    quantized = fixed_point_quantize(smoothed, config.fixed_point_bits)
    quantized = np.where(np.isfinite(quantized), quantized, np.nan)
    summary = _depth_summary(levels, quantized)
    return DepthMap(view=camera.view, width=width, height=height,
                    levels=levels, depth_values=quantized, summary=summary)


@dataclass(frozen=True)
class CloudPoint:
    point_id: str
    x: float
    y: float
    z: float
    voxel: Tuple[int, int, int]
    source_views: Tuple[int, ...]
    source_matches: Tuple[str, ...]
    confidence: float
    member_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id, "x": self.x, "y": self.y, "z": self.z,
            "voxel": list(self.voxel), "source_views": list(self.source_views),
            "source_matches": list(self.source_matches),
            "confidence": self.confidence, "member_count": self.member_count,
        }


def merge_point_cloud(all_points: Sequence[TriPoint], config: RebuildConfig,
                      precision: str = "float64") -> List[CloudPoint]:
    """体素级合并：同体素按坐标稳定排序取代表（置信度最高，再字典序）。"""
    if config.voxel_size <= 0:
        raise WorkbenchError(
            "voxel_size 为 0，无法进行体素合并",
            stage="point_cloud", code=BizCode.E_VOXEL_ZERO,
        )
    dtype = _dtype_for(precision)
    # 输入必须按视图序号 + point_id 稳定排序，保证合并顺序与串行一致
    ordered = sorted(
        all_points, key=lambda p: (p.view_a, p.view_b, p.point_id)
    )
    buckets: Dict[Tuple[int, int, int], List[TriPoint]] = {}
    for point in ordered:
        key = voxel_key(np.array([point.x, point.y, point.z], dtype=dtype),
                        config.voxel_size)
        buckets.setdefault(key, []).append(point)

    cloud: List[CloudPoint] = []
    for voxel in sorted(buckets):
        members = buckets[voxel]
        coords = np.array(
            [[p.x, p.y, p.z] for p in members], dtype=np.float64
        )
        order = np.lexsort(
            (coords[:, 2], coords[:, 1], coords[:, 0],
             np.array([-p.confidence for p in members]))
        )
        # 置信度优先（负置信度参与 lexsort），再坐标字典序
        rep_index = int(order[0])
        rep = members[rep_index]
        avg = coords.mean(axis=0)
        qavg = fixed_point_quantize(avg, config.fixed_point_bits)
        views = tuple(sorted({p.view_a for p in members} |
                             {p.view_b for p in members}))
        matches = tuple(sorted({p.match_id for p in members}))
        cloud.append(
            CloudPoint(
                point_id=f"c_{voxel[0]:+04d}_{voxel[1]:+04d}_{voxel[2]:+04d}",
                x=float(qavg[0]), y=float(qavg[1]), z=float(qavg[2]),
                voxel=voxel, source_views=views, source_matches=matches,
                confidence=rep.confidence, member_count=len(members),
            )
        )
    return cloud


def _depth_summary(levels: np.ndarray, values: np.ndarray) -> str:
    import hashlib

    finite_mask = np.isfinite(values)
    payload = {
        "levels": levels.tolist(),
        "finite_count": int(finite_mask.sum()),
        "mean": float(np.nanmean(values)) if finite_mask.any() else None,
    }
    return hashlib.sha256(
        repr(payload).encode("utf-8")
    ).hexdigest()[:16]
