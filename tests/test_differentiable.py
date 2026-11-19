"""业务要求 3 与 6：解析梯度核对、增量/全量渲染一致性。"""

from __future__ import annotations

import numpy as np
import pytest

from photometric_altopt.differentiable import (
    SurfaceParams,
    compute_gradients,
    finite_difference_check,
    tile_ranges,
    verify_incremental_vs_full,
)
from photometric_altopt.errors import E_RENDER_DIVERGE, BusinessError
from photometric_altopt.photometric import solve_photometric


def _params(observation_bundle) -> SurfaceParams:
    bundle = observation_bundle
    ps = solve_photometric(
        bundle["observations"], bundle["views"], bundle["config"].observation, bundle["config"].solve
    )
    return SurfaceParams(ps.normals, ps.albedo)


def test_analytic_gradient_matches_finite_difference(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    params = _params(bundle)
    report = finite_difference_check(
        params, bundle["observations"], bundle["views"], cfg.observation, cfg.opt, sample_count=16
    )
    assert report["max_abs_error"] <= cfg.opt.gradcheck_tol


def test_gradient_covers_both_parameter_types(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    params = _params(bundle)
    out = compute_gradients(params, bundle["observations"], bundle["views"], cfg.observation)
    assert out.grad_normal.shape == params.normals.shape
    assert out.grad_albedo.shape == params.albedo.shape
    assert np.all(np.isfinite(out.grad_normal))
    assert np.all(np.isfinite(out.grad_albedo))


def test_incremental_matches_full_render(observation_bundle):
    bundle = observation_bundle
    cfg = bundle["config"]
    params = _params(bundle)
    h, w = params.albedo.shape
    tiles = tile_ranges(h, w, cfg.opt.tile_rows, cfg.opt.tile_cols)
    report = verify_incremental_vs_full(
        params, bundle["observations"], bundle["views"], cfg.observation, cfg.opt, tiles
    )
    assert report["ok"] is True
    assert report["tile_count"] == len(tiles)
    assert np.isclose(report["full_loss"], report["incremental_loss"], atol=1e-12)


def test_render_divergence_detected_and_located(observation_bundle, monkeypatch):
    bundle = observation_bundle
    cfg = bundle["config"]
    params = _params(bundle)
    h, w = params.albedo.shape
    tiles = tile_ranges(h, w, cfg.opt.tile_rows, cfg.opt.tile_cols)

    # 故意让增量路径返回错误瓦片（模拟缓存污染），核对必须报 E_RENDER_DIVERGE
    from photometric_altopt import differentiable as dr

    original = dr.render_tiles_incremental

    def corrupted(p, views, obs_cfg, opt, dirty_tiles):  # noqa: ANN001
        partial, targets = original(p, views, obs_cfg, opt, dirty_tiles)
        r0, r1, c0, c1 = tiles[0]
        partial[:, r0:r1, c0:c1] += 1.0
        return partial, targets

    monkeypatch.setattr(dr, "render_tiles_incremental", corrupted)
    with pytest.raises(BusinessError) as exc:
        verify_incremental_vs_full(
            params, bundle["observations"], bundle["views"], cfg.observation, cfg.opt, tiles
        )
    assert exc.value.code == E_RENDER_DIVERGE
    # 首个分歧瓦片按行优先/列升序定位到第一个瓦片
    assert exc.value.details["first_diverging_tile"] == list(tiles[0])
    assert "depends_on" in exc.value.details


def test_tile_order_row_major(observation_bundle):
    bundle = observation_bundle
    h, w = bundle["observations"].shape[1:]
    tiles = tile_ranges(h, w, 8, 8)
    rows = [t[0] for t in tiles]
    cols = [t[2] for t in tiles]
    assert rows == sorted(rows)
    # 同一行块内列严格升序
    for r0 in sorted(set(rows)):
        row_cols = [c for r, c in zip(rows, cols) if r == r0]
        assert row_cols == sorted(row_cols)
