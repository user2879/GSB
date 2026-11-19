"""业务要求 5：阶段间语义对照与确定性冲突定位。"""

from __future__ import annotations

import numpy as np

from photometric_altopt.config import PipelineConfig
from photometric_altopt.conflict import compare_stages


def test_no_conflict_when_identical(small_config: PipelineConfig):
    cfg = small_config
    h = w = 8
    normals = np.zeros((h, w, 3))
    normals[..., 2] = 1.0
    albedo = np.full((h, w), 0.5)
    report = compare_stages(normals, albedo, normals, albedo, cfg.opt)
    assert report.first_conflict is None
    assert report.pixels == []


def test_conflict_detects_normal_and_albedo(small_config: PipelineConfig):
    cfg = small_config
    h = w = 6
    ps_n = np.zeros((h, w, 3))
    ps_n[..., 2] = 1.0
    dr_n = ps_n.copy()
    ps_rho = np.full((h, w), 0.5)
    dr_rho = ps_rho.copy()

    # 法向冲突像素 (1, 1)：绕 x 轴旋转 20 度
    angle = np.radians(20.0)
    dr_n[1, 1] = [0.0, -np.sin(angle), np.cos(angle)]
    # 反照率冲突像素 (2, 3)
    dr_rho[2, 3] = 0.9

    grad_ps = np.zeros((h, w, 3))
    grad_ps[1, 1] = [1.0, 0.0, 0.0]
    grad_dr = np.zeros((h, w, 3))
    grad_dr[1, 1] = [0.0, 1.0, 0.0]

    report = compare_stages(ps_n, ps_rho, dr_n, dr_rho, cfg.opt, grad_ps, grad_dr)
    assert report.first_conflict == (1, 1)
    coords = [(p["row"], p["col"]) for p in report.pixels]
    assert coords == sorted(coords)  # 行优先、列升序
    assert (2, 3) in coords
    entry = report.pixels[0]
    assert "normal_angle" in entry["reasons"]
    assert np.isclose(entry["gradient_direction_angle_deg"], 90.0)
    assert entry["ps_gradient_magnitude"] == 1.0


def test_conflict_not_decided_by_global_loss(small_config: PipelineConfig):
    # 即使两阶段整体非常接近，单个越界像素仍必须被定位
    cfg = small_config
    h = w = 4
    n = np.zeros((h, w, 3))
    n[..., 2] = 1.0
    rho = np.full((h, w), 0.4)
    n2 = n.copy()
    n2[3, 0] = [np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))]
    report = compare_stages(n, rho, n2, rho, cfg.opt)
    assert report.first_conflict == (3, 0)
