"""三角化：DLT 线性三角化 + 最小视差校验。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .camera import Camera
from .errors import E_DISPARITY_OUT_OF_RANGE, FailureList
from .utils import combine_fingerprint


@dataclass
class LandmarkPoint3D:
    landmark_id: int
    x: float
    y: float
    z: float
    observations: list   # [(view, feature_id, u, v, depth)]
    min_disparity: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LandmarkPoint3D":
        return cls(**d)


def _projection_matrix(cam: Camera) -> np.ndarray:
    return cam.matrix() @ np.hstack(
        [np.array(cam.rot, dtype=np.float64), np.array(cam.trans, dtype=np.float64).reshape(3, 1)]
    )


def dlt_triangulate(cam_a: Camera, uv_a, cam_b: Camera, uv_b) -> np.ndarray:
    pa = _projection_matrix(cam_a)
    pb = _projection_matrix(cam_b)
    ua, va = uv_a
    ub, vb = uv_b
    a_mat = np.array([
        ua * pa[2] - pa[0],
        va * pa[2] - pa[1],
        ub * pb[2] - pb[0],
        vb * pb[2] - pb[1],
    ], dtype=np.float64)
    _, _, vh = np.linalg.svd(a_mat)
    homog = vh[-1]
    return homog[:3] / homog[3]


def triangulate_landmarks(
    cameras: dict,
    view_features: dict,
    pair_matches: list,
    min_disparity: float,
    failures: FailureList,
) -> list:
    feat_index: dict = {}
    for view, vf in view_features.items():
        for f in vf.features:
            feat_index[(view, f.feature_id)] = f

    # landmark -> 观测集合（跳过缺失相机参数的视图）
    obs: dict = {}
    for pm in pair_matches:
        if pm.view_a not in cameras or pm.view_b not in cameras:
            continue
        for fa_id, fb_id, _score, lid in pm.pairs:
            fa = feat_index.get((pm.view_a, fa_id))
            fb = feat_index.get((pm.view_b, fb_id))
            if fa is None or fb is None:
                continue
            if pm.view_a not in cameras or pm.view_b not in cameras:
                continue
            obs.setdefault(lid, {})[pm.view_a] = fa
            obs.setdefault(lid, {})[pm.view_b] = fb

    points = []
    for lid in sorted(obs):
        views_here = sorted(obs[lid])
        if len(views_here) < 2:
            continue
        ref_view = views_here[0]
        other_view = views_here[1]
        fa, fb = obs[lid][ref_view], obs[lid][other_view]
        p3 = dlt_triangulate(
            cameras[ref_view], (fa.x, fa.y),
            cameras[other_view], (fb.x, fb.y),
        )
        # 视差：参考视图与其余视图投影像素距离的最大值
        ref_uv = np.array([fa.x, fa.y])
        max_disp = 0.0
        for v in views_here[1:]:
            uv, _ = cameras[v].project(p3)
            max_disp = max(max_disp, float(np.linalg.norm(uv - ref_uv)))
        if max_disp < min_disparity:
            failures.fail(
                E_DISPARITY_OUT_OF_RANGE,
                f"landmark {lid} disparity {max_disp:.3f} < min {min_disparity}",
                view=ref_view,
                stage="triangulate",
                sort_key=f"triangulate|{ref_view:04d}|{lid:06d}|{E_DISPARITY_OUT_OF_RANGE}",
            )
            continue
        observations = []
        for v in views_here:
            if v not in cameras:
                continue
            f = obs[lid][v]
            _, depth = cameras[v].project(p3)
            observations.append((v, f.feature_id, round(f.x, 6), round(f.y, 6), round(depth, 6)))
        points.append(
            LandmarkPoint3D(
                landmark_id=lid,
                x=round(float(p3[0]), 6), y=round(float(p3[1]), 6), z=round(float(p3[2]), 6),
                observations=observations, min_disparity=round(max_disp, 6),
            )
        )
    points.sort(key=lambda p: p.landmark_id)
    return points


def triangulate_cache_key(match_fp: str, camera_fp: str, cfg_fp: str) -> str:
    return combine_fingerprint("triangulate", match_fp, camera_fp, cfg_fp)
