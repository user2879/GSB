"""集成：在交替流程中人为拉开两阶段参数，验证冲突清单被确定性填充。"""

from __future__ import annotations

import numpy as np

from photometric_altopt.conflict import compare_stages
from photometric_altopt.differentiable import SurfaceParams, normalize_normals
from photometric_altopt.photometric import solve_photometric


def test_conflict_populated_when_stages_diverge(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    ps = solve_photometric(
        bundle["observations"], bundle["views"], cfg.observation, cfg.solve
    )

    # 构造一个被扰动的 DR 参数：若干像素法向旋转 25 度、反照率偏移 0.2
    dr = SurfaceParams(ps.normals.copy(), ps.albedo.copy())
    angle = np.radians(25.0)
    for r, c in [(2, 3), (10, 11), (20, 1)]:
        base = ps.normals[r, c]
        axis = np.array([-base[1], base[0], 0.0])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        dr.normals[r, c] = normalize_normals(
            np.cos(angle) * base + np.sin(angle) * axis
        )
        dr.albedo[r, c] = np.clip(dr.albedo[r, c] + 0.2, 0.0, 1.0)

    report = compare_stages(
        ps.normals,
        ps.albedo,
        normalize_normals(dr.normals),
        dr.albedo,
        cfg.opt,
        ps_update_normal=ps.normals - dr.normals,
        dr_grad_normal=np.zeros_like(dr.normals),
    )
    assert report.first_conflict == (2, 3)
    assert len(report.pixels) == 3
    coords = [(p["row"], p["col"]) for p in report.pixels]
    assert coords == sorted(coords)
    for pixel in report.pixels:
        assert pixel["normal_angle_deg"] > cfg.opt.conflict_angle_deg
        assert "gradient_direction_angle_deg" in pixel
