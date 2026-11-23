"""跨视图匹配：以纹理标记编号为锚点、描述子 NCC 量化为千分比得分。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .features import ViewFeatures
from .metrics import name_to_code
from .utils import combine_fingerprint


@dataclass
class ViewPairMatch:
    view_a: int
    view_b: int
    pairs: list            # [(feature_a, feature_b, score_permille, landmark_id)]
    inliers: int
    metric: str = "ncc"

    @property
    def sort_key(self) -> str:
        return f"match|{self.view_a:04d}-{self.view_b:04d}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ViewPairMatch":
        return cls(
            view_a=d["view_a"], view_b=d["view_b"],
            pairs=[tuple(p) for p in d["pairs"]],
            inliers=d["inliers"], metric=d.get("metric", "ncc"),
        )


def _ncc_permille(desc_a, desc_b) -> float:
    a = np.asarray(desc_a, dtype=np.float64)
    b = np.asarray(desc_b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    ncc = float(a @ b / denom) if denom > 0 else 0.0
    ncc = max(-1.0, min(1.0, ncc))
    # 映射到 [0,1000] 千分比
    return round((ncc + 1.0) * 500.0, 3)


def match_views(vf_a: ViewFeatures, vf_b: ViewFeatures, min_inliers: int) -> ViewPairMatch:
    by_landmark_a = {f.landmark_id: f for f in vf_a.features}
    by_landmark_b = {f.landmark_id: f for f in vf_b.features}
    pairs = []
    for lid in sorted(set(by_landmark_a) & set(by_landmark_b)):
        fa = by_landmark_a[lid]
        fb = by_landmark_b[lid]
        score = _ncc_permille(fa.descriptor, fb.descriptor)
        pairs.append((fa.feature_id, fb.feature_id, score, lid))
    pairs.sort(key=lambda p: (p[0], p[1]))
    inliers = sum(1 for p in pairs if p[2] >= 500.0)
    return ViewPairMatch(vf_a.view, vf_b.view, pairs, inliers)


def match_cache_key(pair_key: str, feature_fps: dict, cfg_fp: str) -> str:
    joined = ",".join(f"{v}:{feature_fps[v]}" for v in sorted(feature_fps))
    return combine_fingerprint("match", pair_key, joined, cfg_fp)


def pair_id(view_a: int, view_b: int) -> str:
    return f"{min(view_a, view_b):04d}-{max(view_a, view_b):04d}"


def metric_code(metric: str) -> int:
    return name_to_code(metric)
