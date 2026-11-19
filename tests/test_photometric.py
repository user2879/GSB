"""业务要求 2 与 7：光度立体求解、病态标记、退化/非有限/零光源异常。"""

from __future__ import annotations

import numpy as np
import pytest

from photometric_altopt.config import PipelineConfig, SolveConfig
from photometric_altopt.conflict import normal_angle_deg
from photometric_altopt.errors import (
    E_DEGENERATE_VIEWS,
    E_NONFINITE_ALBEDO,
    E_ZERO_LIGHT,
    BusinessError,
)
from photometric_altopt.faults import FaultConfig, build_valid_mask, inject_nonfinite_albedo
from photometric_altopt.forward import build_view_table
from photometric_altopt.generator import load_truth
from photometric_altopt.photometric import solve_photometric


def test_photometric_recovers_truth(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    result = solve_photometric(
        bundle["observations"], bundle["views"], cfg.observation, cfg.solve
    )
    truth = load_truth(bundle["dir"])
    assert np.allclose(np.linalg.norm(result.normals, axis=-1), 1.0)
    assert np.all(result.albedo >= 0.0)
    assert np.mean(normal_angle_deg(result.normals, truth["normals"])) < 2.0
    assert np.max(np.abs(result.albedo - truth["albedo"])) < 0.05
    assert result.used_views.all()


def test_ill_conditioned_flag_and_error_bounds(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    # 把条件数阈值压到 1，强制所有像素越界（状态型，不抛异常）
    strict = SolveConfig(
        irls_iterations=cfg.solve.irls_iterations,
        irls_tuning=cfg.solve.irls_tuning,
        weight_floor=cfg.solve.weight_floor,
        condition_threshold=0.99,
    )
    result = solve_photometric(
        bundle["observations"], bundle["views"], cfg.observation, strict
    )
    assert result.ill_conditioned.all()
    assert np.all(result.error_bounds["normal_angle_rad_halfwidth"] >= 0.0)
    assert np.all(result.error_bounds["albedo_abs_halfwidth"] >= 0.0)


def test_degenerate_views_raises(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    h, w = bundle["observations"].shape[1:]
    # 12 视点中仅保留 2 个，低于 min_views=4
    mask = build_valid_mask(bundle["views"].size, h, w, list(range(2, 12)))
    with pytest.raises(BusinessError) as exc:
        solve_photometric(
            bundle["observations"], bundle["views"], cfg.observation, cfg.solve, valid_mask=mask
        )
    assert exc.value.code == E_DEGENERATE_VIEWS
    assert exc.value.details["min_valid_views"] < cfg.solve.min_views


def test_nonfinite_albedo_raises(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    polluted = inject_nonfinite_albedo(
        np.zeros((cfg.observation.size, cfg.observation.size)),
        FaultConfig(nonfinite_albedo=True, nonfinite_pixels=[[1, 2], [3, 4]]),
    )
    with pytest.raises(BusinessError) as exc:
        solve_photometric(
            bundle["observations"],
            bundle["views"],
            cfg.observation,
            cfg.solve,
            albedo_seed=polluted,
        )
    assert exc.value.code == E_NONFINITE_ALBEDO
    assert exc.value.details["count"] == 2


def test_zero_light_raises(small_config: PipelineConfig):
    cfg = small_config
    cfg.observation.light_directions[2] = [0.0, 0.0, 0.0]
    with pytest.raises(BusinessError) as exc:
        build_view_table(cfg.observation)
    assert exc.value.code == E_ZERO_LIGHT
    assert 2 in exc.value.details["light_index"]


def test_solver_deterministic(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    r1 = solve_photometric(bundle["observations"], bundle["views"], cfg.observation, cfg.solve)
    r2 = solve_photometric(bundle["observations"], bundle["views"], cfg.observation, cfg.solve)
    assert np.array_equal(r1.normals, r2.normals)
    assert np.array_equal(r1.albedo, r2.albedo)
