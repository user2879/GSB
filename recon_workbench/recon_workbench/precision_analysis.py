"""float32 主路径与 float64 串行参考路径的逐级误差归因。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RebuildConfig
from .geometry import CloudPoint, DepthMap, TriPoint
from .utils import voxel_key

STAGES = ("feature", "match", "triangulation", "depth", "point_cloud")


@dataclass
class StageError:
    stage: str
    max_abs_error: float
    max_rel_error: float
    item_count: int
    first_breach: bool = False
    breach_item: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage, "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "item_count": self.item_count,
            "first_breach": self.first_breach,
            "breach_item": self.breach_item,
        }


@dataclass
class BoundarySample:
    sample_id: str
    kind: str  # threshold | voxel_divergence
    source: str  # triangulation | quantization_rounding | match_threshold
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"sample_id": self.sample_id, "kind": self.kind,
                "source": self.source, "detail": self.detail}


def _pairwise_errors(values_32: np.ndarray, values_64: np.ndarray
                     ) -> Tuple[float, float]:
    a = np.asarray(values_32, dtype=np.float64)
    b = np.asarray(values_64, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0
    abs_err = np.abs(a - b)
    # 相对误差仅在参考值显著非零（>1e-6）的分量上计算，
    # 零/近零分量只参与绝对误差，避免兜底分母放大。
    significant = np.abs(b) > 1e-6
    if significant.any():
        rel_err = abs_err[significant] / np.abs(b[significant])
        max_rel = float(rel_err.max())
    else:
        max_rel = 0.0
    return float(abs_err.max()), max_rel


def feature_stage_error(features_32: Sequence[Any],
                        features_64: Sequence[Any]) -> StageError:
    ids_32 = [f.feature_id for f in features_32]
    ids_64 = [f.feature_id for f in features_64]
    common = [fid for fid in ids_64 if fid in set(ids_32)]
    map32 = {f.feature_id: f for f in features_32}
    map64 = {f.feature_id: f for f in features_64}
    a = np.array([[map32[i].x, map32[i].y, map32[i].response] for i in common])
    b = np.array([[map64[i].x, map64[i].y, map64[i].response] for i in common])
    max_abs, max_rel = _pairwise_errors(a, b)
    return StageError("feature", max_abs, max_rel, len(common))


def match_stage_error(matches_32: Sequence[Any],
                      matches_64: Sequence[Any]) -> StageError:
    map32 = {m.match_id: m for m in matches_32}
    map64 = {m.match_id: m for m in matches_64}
    common = [mid for mid in map64 if mid in map32]
    a = np.array([[map32[i].distance] for i in common])
    b = np.array([[map64[i].distance] for i in common])
    max_abs, max_rel = _pairwise_errors(a, b)
    return StageError("match", max_abs, max_rel, len(common))


def triangulation_stage_error(points_32: Sequence[TriPoint],
                              points_64: Sequence[TriPoint]
                              ) -> StageError:
    map32 = {p.point_id: p for p in points_32}
    map64 = {p.point_id: p for p in points_64}
    common = [pid for pid in map64 if pid in map32]
    a = np.array([[map32[i].x, map32[i].y, map32[i].z] for i in common])
    b = np.array([[map64[i].x, map64[i].y, map64[i].z] for i in common])
    max_abs, max_rel = _pairwise_errors(a, b)
    return StageError("triangulation", max_abs, max_rel, len(common))


def depth_stage_error(depth_32: Sequence[DepthMap],
                      depth_64: Sequence[DepthMap]) -> StageError:
    map32 = {d.view: d for d in depth_32}
    map64 = {d.view: d for d in depth_64}
    a_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []
    for view in sorted(set(map32) & set(map64)):
        va = map32[view].depth_values
        vb = map64[view].depth_values
        mask = np.isfinite(va) & np.isfinite(vb)
        a_list.append(va[mask].reshape(-1))
        b_list.append(vb[mask].reshape(-1))
    if a_list:
        max_abs, max_rel = _pairwise_errors(
            np.concatenate(a_list), np.concatenate(b_list)
        )
        count = int(sum(arr.size for arr in a_list))
    else:
        max_abs, max_rel, count = 0.0, 0.0, 0
    return StageError("depth", max_abs, max_rel, count)


def point_cloud_stage_error(cloud_32: Sequence[CloudPoint],
                            cloud_64: Sequence[CloudPoint]) -> StageError:
    map32 = {c.point_id: c for c in cloud_32}
    map64 = {c.point_id: c for c in cloud_64}
    common = [cid for cid in map64 if cid in map32]
    a = np.array([[map32[i].x, map32[i].y, map32[i].z] for i in common])
    b = np.array([[map64[i].x, map64[i].y, map64[i].z] for i in common])
    max_abs, max_rel = _pairwise_errors(a, b)
    return StageError("point_cloud", max_abs, max_rel, len(common))


def _voxel_distance_to_boundary(point: np.ndarray, voxel: Tuple[int, int],
                                voxel_size: float) -> float:
    """点到其所属体素任一边界面的最小归一化距离。"""
    normalized = np.asarray(point, dtype=np.float64) / voxel_size
    margins = [min(normalized[k] - voxel[k], voxel[k] + 1 - normalized[k])
               for k in range(3)]
    return float(min(margins))


def classify_voxel_divergence(point_32: TriPoint, point_64: TriPoint,
                              config: RebuildConfig) -> str:
    """体素归属分歧归因：三角化漂移 vs 量化舍入。"""
    coord_32 = np.array([point_32.x, point_32.y, point_32.z], dtype=np.float64)
    coord_64 = np.array([point_64.x, point_64.y, point_64.z], dtype=np.float64)
    voxel_64 = voxel_key(coord_64, config.voxel_size)
    drift = float(np.linalg.norm(coord_32 - coord_64))
    boundary_margin = _voxel_distance_to_boundary(
        coord_64, voxel_64, config.voxel_size
    )
    # 漂移量大于参考点到体素边界的距离 => 本不会越界，主因三角化数值漂移；
    # 否则参考点本身贴近边界 => 量化舍入主导。
    if drift > boundary_margin * config.voxel_size:
        return "triangulation"
    return "quantization_rounding"


def analyze(precision_results: Dict[str, Dict[str, Any]],
            config: RebuildConfig) -> Dict[str, Any]:
    """汇总四个环节误差、首超容差环节与边界样本。"""
    r32 = precision_results["float32"]
    r64 = precision_results["float64"]

    stages = [
        feature_stage_error(r32["features_flat"], r64["features_flat"]),
        match_stage_error(r32["matches_flat"], r64["matches_flat"]),
        triangulation_stage_error(r32["tripoints_flat"],
                                  r64["tripoints_flat"]),
        depth_stage_error(r32["depth_maps"], r64["depth_maps"]),
        point_cloud_stage_error(r32["cloud"], r64["cloud"]),
    ]

    first_breach_stage: Optional[str] = None
    for stage in stages:
        breached = (
            stage.max_abs_error > config.abs_tolerance
            or stage.max_rel_error > config.rel_tolerance
        )
        if breached and first_breach_stage is None:
            stage.first_breach = True
            first_breach_stage = stage.stage

    boundary_samples: List[BoundarySample] = []
    # 阈值边界：匹配距离与视差门限
    for match in r64["matches_flat"]:
        if match.boundary:
            boundary_samples.append(
                BoundarySample(
                    sample_id=match.match_id, kind="threshold",
                    source="match_threshold",
                    detail={"distance": match.distance,
                            "threshold": config.match_distance_threshold},
                )
            )
    point64_map = {p.point_id: p for p in r64["tripoints_flat"]}
    for point in r64["tripoints_flat"]:
        if point.boundary:
            boundary_samples.append(
                BoundarySample(
                    sample_id=point.point_id, kind="threshold",
                    source="triangulation",
                    detail={"disparity": point.disparity,
                            "min_disparity": config.min_disparity},
                )
            )
    # 体素归属分歧
    point32_map = {p.point_id: p for p in r32["tripoints_flat"]}
    for pid, p64 in point64_map.items():
        p32 = point32_map.get(pid)
        if p32 is None:
            continue
        coord32 = np.array([p32.x, p32.y, p32.z])
        coord64 = np.array([p64.x, p64.y, p64.z])
        # 非有限点或越界巨值（极端 float32 漂移，floor 后超 int64）
        # 跳过体素归属比较，单独归因为三角化漂移。
        int64_limit = np.iinfo(np.int64).max
        if config.voxel_size <= 0:
            boundary_samples.append(
                BoundarySample(
                    sample_id=pid, kind="voxel_divergence",
                    source="quantization_rounding",
                    detail={"reason": "zero_voxel_size"},
                )
            )
            continue
        scaled = np.concatenate([coord32, coord64]) / config.voxel_size
        if (not (np.isfinite(coord32).all() and np.isfinite(coord64).all())
                or np.any(np.abs(scaled) > int64_limit)):
            boundary_samples.append(
                BoundarySample(
                    sample_id=pid, kind="voxel_divergence",
                    source="triangulation",
                    detail={"reason": "non_finite_or_overflow_coordinate"},
                )
            )
            continue
        v32 = voxel_key(coord32, config.voxel_size)
        v64 = voxel_key(coord64, config.voxel_size)
        if v32 != v64:
            source = classify_voxel_divergence(p32, p64, config)
            boundary_samples.append(
                BoundarySample(
                    sample_id=pid, kind="voxel_divergence", source=source,
                    detail={"voxel_float32": list(v32),
                            "voxel_float64": list(v64)},
                )
            )

    return {
        "tolerance": {"abs": config.abs_tolerance, "rel": config.rel_tolerance},
        "first_breach_stage": first_breach_stage,
        "stage_errors": [s.to_dict() for s in stages],
        "boundary_samples": [b.to_dict() for b in boundary_samples],
    }
