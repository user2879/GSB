"""特征提取：阈值化 + 连通域几何图元检测，输出亚像素质心与确定性描述子。"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .errors import E_NON_FINITE, E_TRUNCATED_PNG, FailureList
from .synthetic import load_png_gray
import hashlib

from .utils import combine_fingerprint, is_finite_number


@dataclass
class DetectedFeature:
    feature_id: int
    x: float
    y: float
    landmark_id: int
    descriptor: list          # 归一化 3x3 亮度描述子

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectedFeature":
        return cls(**d)


@dataclass
class ViewFeatures:
    view: int
    image_size: tuple
    features: list
    failures: list

    def to_dict(self) -> dict:
        return {
            "view": self.view,
            "image_size": list(self.image_size),
            "features": [f.to_dict() for f in self.features],
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ViewFeatures":
        return cls(
            view=d["view"],
            image_size=tuple(d["image_size"]),
            features=[DetectedFeature.from_dict(x) for x in d["features"]],
            failures=d["failures"],
        )


def _connected_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(x, y)])
            seen[y, x] = True
            pts = []
            while q:
                cx, cy = q.popleft()
                pts.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def _descriptor(image: np.ndarray, cx: int, cy: int) -> list:
    patch = image[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].astype(np.float64)
    if patch.size != 9:  # 边界退化时补零
        full = np.zeros((3, 3), dtype=np.float64)
        full[:patch.shape[0], :patch.shape[1]] = patch
        patch = full
    norm = float(np.linalg.norm(patch))
    desc = (patch / norm).flatten() if norm > 0 else patch.flatten()
    return [float(v) for v in desc]


def image_bytes_fingerprint(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def extract_features(view: int, image_path: str, feature_threshold: int,
                     failures: FailureList) -> ViewFeatures:
    try:
        image = load_png_gray(image_path)
    except Exception as exc:  # 截断 PNG 真实解码失败（OSError/UnidentifiedImageError 等）
        failures.fail(
            E_TRUNCATED_PNG,
            f"view {view} PNG unreadable: {exc}",
            view=view,
            stage="features",
            sort_key=f"features|{view:04d}|{E_TRUNCATED_PNG}",
        )
        return ViewFeatures(view, (0, 0), [], [f.to_dict() for f in failures.sorted()
                                               if f.view == view and f.stage == "features"])

    mask = image >= feature_threshold
    detected = []
    for comp in _connected_components(mask):
        if len(comp) < 9:  # 噪声孤立点不构成几何图元
            continue
        xs = np.array([p[0] for p in comp], dtype=np.float64)
        ys = np.array([p[1] for p in comp], dtype=np.float64)
        cx = float(xs.mean())
        cy = float(ys.mean())
        level = float(np.mean([image[y, x] for x, y in comp]))
        landmark_id = int(round((level - 180) / 4.0))
        ix, iy = int(round(cx)), int(round(cy))
        detected.append(
            DetectedFeature(
                feature_id=-1, x=cx, y=cy, landmark_id=landmark_id,
                descriptor=_descriptor(image, ix, iy),
            )
        )

    # 稳定 feature_id：按 (y, x) 排序后顺序编号
    detected.sort(key=lambda f: (f.y, f.x))
    for fid, f in enumerate(detected):
        f.feature_id = fid
        if not all(is_finite_number(v) for v in f.descriptor):
            failures.fail(
                E_NON_FINITE,
                f"view {view} feature {fid} descriptor contains non-finite value",
                view=view,
                stage="features",
                sort_key=f"features|{view:04d}|{E_NON_FINITE}",
            )

    return ViewFeatures(view, (image.shape[1], image.shape[0]),
                        detected,
                        [f.to_dict() for f in failures.sorted()
                         if f.view == view and f.stage == "features"])


def feature_cache_key(view: int, image_fp: str, camera_fp: str, cfg_fp: str) -> str:
    return combine_fingerprint("features", str(view), image_fp, camera_fp, cfg_fp)
