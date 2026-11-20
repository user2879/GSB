"""确定性角点特征提取。

样本为黑底亮方块纹理标记：以 7x7 窗口均值响应做检测，
3x3 邻域非极大值抑制，按 (响应降序, y 升序, x 升序) 稳定排序后
截断到单视图配额；描述向量量化为定点整数。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import ReconstructionConfig
from .errors import E_NON_FINITE_FEATURE, StageError
from .util import content_fingerprint, fingerprint_many, quantize_scalar

WINDOW_RADIUS = 3
NMS_RADIUS = 1
DESCRIPTOR_RADIUS = 3


def _integral_image(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(gray.astype(np.int64), axis=0), axis=1)
    return integral


def _window_mean(integral: np.ndarray, x: int, y: int, radius: int) -> float:
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return total / float((x1 - x0) * (y1 - y0))


def _response_grid(integral: np.ndarray, radius: int) -> np.ndarray:
    """积分图向量化计算每个像素的 (2r+1) 窗口均值，与逐像素 _window_mean 同值。"""
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    ys = np.arange(height)
    xs = np.arange(width)
    y0 = np.clip(ys - radius, 0, None)
    y1 = np.clip(ys + radius + 1, None, height)
    x0 = np.clip(xs - radius, 0, None)
    x1 = np.clip(xs + radius + 1, None, width)
    total = (
        integral[np.ix_(y1, x1)]
        - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)]
        + integral[np.ix_(y0, x0)]
    )
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return total / area


def extract_features(view_id: int, gray: np.ndarray, cfg: ReconstructionConfig) -> dict[str, Any]:
    height, width = gray.shape
    integral = _integral_image(gray)
    # 候选点同时满足响应窗口与描述子窗口均完整落在图像内
    border = max(WINDOW_RADIUS, DESCRIPTOR_RADIUS)
    response_grid = _response_grid(integral, WINDOW_RADIUS)
    candidate_y, candidate_x = np.nonzero(response_grid >= cfg.feature_threshold)
    responses: list[tuple[float, int, int]] = [
        (float(response_grid[y, x]), int(y), int(x))
        for y, x in zip(candidate_y.tolist(), candidate_x.tolist())
        if border <= y < height - border and border <= x < width - border
    ]

    # 3x3 非极大值抑制：同响应取坐标字典序最小者
    response_map = {(y, x): response for response, y, x in responses}
    kept: list[tuple[float, int, int]] = []
    for response, y, x in responses:
        local_max = True
        for dy in range(-NMS_RADIUS, NMS_RADIUS + 1):
            for dx in range(-NMS_RADIUS, NMS_RADIUS + 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = response_map.get((y + dy, x + dx))
                if neighbor is None:
                    continue
                if neighbor > response or (neighbor == response and (dy, dx) < (0, 0)):
                    local_max = False
                    break
            if not local_max:
                break
        if local_max:
            kept.append((response, y, x))

    # 响应降序；同响应按 (y, x) 字典序升序
    kept.sort(key=lambda item: (-item[0], item[1], item[2]))
    kept = kept[: cfg.per_view_quota]

    frac_bits = cfg.fixed_point_frac_bits
    features: list[dict[str, Any]] = []
    for index, (response, y, x) in enumerate(kept):
        patch = gray[
            max(0, y - DESCRIPTOR_RADIUS): min(height, y + DESCRIPTOR_RADIUS + 1),
            max(0, x - DESCRIPTOR_RADIUS): min(width, x + DESCRIPTOR_RADIUS + 1),
        ]
        # 描述子：7x7 局部水平/垂直梯度的确定性整数编码，
        # 对整体亮度偏移不敏感、对纹理位置敏感，跨视图稳定且区分度高
        patch_int = patch.astype(np.int16)
        gx = patch_int[:, 1:] - patch_int[:, :-1]
        gy = patch_int[1:, :] - patch_int[:-1, :]
        descriptor = [int(value) for value in gx.flatten().tolist()]
        descriptor += [int(value) for value in gy.flatten().tolist()]
        feature = {
            "feature_id": f"f-{view_id:04d}-{index:04d}",
            "view_id": view_id,
            "x": x,
            "y": y,
            "response_q": quantize_scalar(response, frac_bits),
            "descriptor": descriptor,
        }
        if not all(math.isfinite(v) for v in descriptor) or not math.isfinite(response):
            raise StageError(
                E_NON_FINITE_FEATURE,
                f"视图 {view_id} 特征 {feature['feature_id']} 出现非有限值",
                stage="feature",
                view_id=view_id,
            )
        features.append(feature)

    content = {
        "view_id": view_id,
        "feature_threshold_q": quantize_scalar(cfg.feature_threshold, frac_bits),
        "per_view_quota": cfg.per_view_quota,
        "frac_bits": frac_bits,
        "features": features,
    }
    return {
        "artifact_id": f"feat-{view_id:04d}",
        "stage": "feature",
        "view_id": view_id,
        "content": content,
        "content_fingerprint": content_fingerprint(content),
    }


def feature_artifact_key(view_id: int, image_fp: str, camera_fp: str, param_fp: str) -> str:
    return fingerprint_many("feature", view_id, image_fp, camera_fp, param_fp)
