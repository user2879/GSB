"""确定性单视图角点提取与跨视图匹配。

Harris 角点响应 + 固定邻域灰度描述子，全流程无随机操作，
同输入同参数（同精度档位）逐字段可复现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import RebuildConfig
from .errors import BizCode, WorkbenchError
from .utils import fixed_point_quantize

HARRIS_K = 0.04


@dataclass(frozen=True)
class Feature:
    feature_id: str
    view: int
    x: float
    y: float
    response: float

    def to_dict(self) -> Dict[str, Any]:
        return {"feature_id": self.feature_id, "view": self.view,
                "x": self.x, "y": self.y, "response": self.response}


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    view_a: int
    view_b: int
    feature_a: str
    feature_b: str
    distance: float
    boundary: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id, "view_a": self.view_a,
            "view_b": self.view_b, "feature_a": self.feature_a,
            "feature_b": self.feature_b, "distance": self.distance,
            "boundary": self.boundary,
        }


def _dtype_for(precision: str) -> np.dtype:
    return np.dtype(np.float32 if precision == "float32" else np.float64)


def _structure_tensor(gray: np.ndarray, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = gray.astype(dtype) / 255.0
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy
    # 3x3 盒式窗（确定权重，无随机）
    kernel = np.ones((3, 3), dtype=dtype) / 9.0

    def box_convolve(src: np.ndarray) -> np.ndarray:
        padded = np.pad(src, 1, mode="edge")
        out = np.zeros_like(src)
        for dy in range(3):
            for dx in range(3):
                out += kernel[dy, dx] * padded[dy:dy + src.shape[0],
                                                dx:dx + src.shape[1]]
        return out

    return box_convolve(ixx), box_convolve(iyy), box_convolve(ixy)


def extract_features(gray: np.ndarray, view: int, config: RebuildConfig,
                     precision: str = "float64",
                     nonfinite_injection: bool = False) -> List[Feature]:
    """按阈值提取角点，响应降序、同响应坐标字典序截断。

    角点定位发生在整数像素网格上，检测与排序统一在 float64 下计算，
    保证两精度路径的角点集合与顺序逐字段一致；精度差异由下游描述子、
    三角化与深度环节承担。响应值再按目标精度量化输出。
    """
    dtype = _dtype_for(precision)
    # 检测/排序使用 float64 参考响应，避免 float32 舍入制造伪并列
    ixx_ref, iyy_ref, ixy_ref = _structure_tensor(
        gray, np.dtype(np.float64)
    )
    trace_ref = ixx_ref + iyy_ref
    response_ref = (
        (ixx_ref * iyy_ref - ixy_ref * ixy_ref)
        - HARRIS_K * trace_ref * trace_ref
    )
    # 目标精度响应（用于按精度量化输出与非有限数注入判定）
    ixx, iyy, ixy = _structure_tensor(gray, dtype)
    trace = ixx + iyy
    response = (ixx * iyy - ixy * ixy) - HARRIS_K * trace * trace
    radius = config.descriptor_radius
    height, width = gray.shape

    candidates: List[Tuple[float, int, int]] = []
    # 先收集所有过阈像素；非极大值抑制改为基于全局排序的贪心选取，
    # 不依赖浮点相等比较，从而消除 float32/float64 的并列歧义。
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            value = float(response_ref[y, x])
            if value < config.corner_threshold:
                continue
            candidates.append((value, int(x), int(y)))

    # 响应降序（Python float 取负，避免 numpy 标量排序差异）；
    # 响应先量化到定点档位，使浮点级抖动归并为确定的同响应组，
    # 同响应再按坐标字典序 (y, x) 的固定决断规则。
    quantized_responses = {
        (x, y): float(
            fixed_point_quantize(
                np.array(value, dtype=np.float64),
                config.fixed_point_bits,
            )
        )
        for value, x, y in candidates
    }
    candidates.sort(
        key=lambda item: (
            -quantized_responses[(item[1], item[2])], item[2], item[1]
        )
    )
    selected: List[Tuple[float, int, int]] = []
    occupied: set[Tuple[int, int]] = set()
    for value, x, y in candidates:
        # 非极大值抑制：仅保留 3x3 邻域内尚未被更强角点占据的像素
        if any((x + dx, y + dy) in occupied
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        occupied.add((x, y))
        selected.append((value, x, y))
        if len(selected) >= config.max_features_per_view:
            break

    features: List[Feature] = []
    for index, (value, x, y) in enumerate(selected):
        value = quantized_responses[(x, y)]
        qx = float(fixed_point_quantize(np.array(x, dtype=dtype),
                                        config.fixed_point_bits))
        qy = float(fixed_point_quantize(np.array(y, dtype=dtype),
                                        config.fixed_point_bits))
        qr = float(fixed_point_quantize(np.array(value, dtype=dtype),
                                        config.fixed_point_bits))
        if nonfinite_injection and index == 0:
            qx = float("nan")
        if not np.isfinite(qx) or not np.isfinite(qy) or not np.isfinite(qr):
            raise WorkbenchError(
                f"视图 {view} 特征描述含非有限数",
                view=view, stage="feature",
                code=BizCode.E_FEATURE_NONFINITE,
                detail={"index": index},
            )
        features.append(
            Feature(
                feature_id=f"v{view:02d}f{index:03d}",
                view=view, x=qx, y=qy, response=qr,
            )
        )
    return features


def _descriptor(gray: np.ndarray, x: int, y: int, radius: int,
                dtype: np.dtype) -> np.ndarray:
    height, width = gray.shape
    patch = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=dtype)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sx = min(max(x + dx, 0), width - 1)
            sy = min(max(y + dy, 0), height - 1)
            patch[dy + radius, dx + radius] = gray[sy, sx]
    vec = patch.reshape(-1).astype(dtype) / 255.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def match_views(features_a: Sequence[Feature], features_b: Sequence[Feature],
                gray_a: np.ndarray, gray_b: np.ndarray,
                view_a: int, view_b: int, config: RebuildConfig,
                precision: str = "float64") -> Tuple[List[MatchRecord], bool]:
    """对称最近邻 + 距离阈值匹配，返回匹配记录与阈值边界命中标记。"""
    dtype = _dtype_for(precision)
    radius = config.descriptor_radius
    coords_a = [(int(round(f.x)), int(round(f.y))) for f in features_a]
    coords_b = [(int(round(f.x)), int(round(f.y))) for f in features_b]
    desc_a = [_descriptor(gray_a, x, y, radius, dtype) for x, y in coords_a]
    desc_b = [_descriptor(gray_b, x, y, radius, dtype) for x, y in coords_b]

    # 距离矩阵全量计算，距离在配置阈值边界相等时保留（>= 规则，boundary 记录）
    distances = np.zeros((len(desc_a), len(desc_b)), dtype=dtype)
    for i, da in enumerate(desc_a):
        for j, db in enumerate(desc_b):
            distances[i, j] = np.linalg.norm(da - db)

    boundary_hit = False
    raw: List[Tuple[int, int, float, bool]] = []
    if distances.size:
        near_threshold = np.isclose(
            distances, config.match_distance_threshold,
            rtol=0.0, atol=1e-7,
        )
        boundary_hit = bool(near_threshold.any())
        nearest_b = np.argmin(distances, axis=1)
        nearest_a = np.argmin(distances, axis=0)
        for i in range(len(desc_a)):
            j = int(nearest_b[i])
            if int(nearest_a[j]) != i:
                continue
            distance = float(distances[i, j])
            if distance > config.match_distance_threshold:
                continue
            is_boundary = bool(near_threshold[i, j])
            raw.append((i, j, distance, is_boundary))

    # 稳定顺序：距离升序，同距离按 (a 索引, b 索引) 字典序
    raw.sort(key=lambda item: (item[2], item[0], item[1]))
    records: List[MatchRecord] = []
    for i, j, distance, is_boundary in raw:
        match_id = f"m_{view_a:02d}_{view_b:02d}_{i:03d}_{j:03d}"
        records.append(
            MatchRecord(
                match_id=match_id, view_a=view_a, view_b=view_b,
                feature_a=features_a[i].feature_id,
                feature_b=features_b[j].feature_id,
                distance=float(distance), boundary=is_boundary,
            )
        )
    return records, boundary_hit


def feature_lookup(features: Sequence[Feature]) -> Dict[str, Feature]:
    return {f.feature_id: f for f in features}
