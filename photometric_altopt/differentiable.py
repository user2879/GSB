"""可微渲染阶段（业务要求 3、6）。

与光度立体共用 :mod:`photometric_altopt.forward` 的前向语义，独立维护
参数副本与解析梯度，不共享任何求解缓存。

损失定义（全部有效像素、全部视点）::

    L = mean_v,p ( I_render(v,p) - I_obs(v,p) )^2

参数为逐像素世界系法向（每步后重新归一化）与反照率（裁剪非负）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ObservationConfig, OptConfig
from .errors import E_RENDER_DIVERGE, BusinessError
from .forward import ViewTable, render_all, render_and_grad


@dataclass
class SurfaceParams:
    """DR 阶段独立持有的表面参数副本。"""

    normals: np.ndarray  # (H, W, 3)
    albedo: np.ndarray  # (H, W)

    def copy(self) -> "SurfaceParams":
        return SurfaceParams(self.normals.copy(), self.albedo.copy())


@dataclass
class GradientBundle:
    grad_normal: np.ndarray  # (H, W, 3)
    grad_albedo: np.ndarray  # (H, W)
    grad_normal_norm: float
    grad_albedo_norm: float
    loss: float
    images: np.ndarray


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)


def mse_loss(images: np.ndarray, observations: np.ndarray, valid: np.ndarray | None = None) -> float:
    diff = images - observations
    if valid is None:
        return float(np.mean(diff * diff))
    count = max(int(valid.sum()), 1)
    return float(np.sum(valid * diff * diff) / count)


def compute_gradients(
    params: SurfaceParams,
    observations: np.ndarray,
    views: ViewTable,
    obs_cfg: ObservationConfig,
    valid: np.ndarray | None = None,
) -> GradientBundle:
    """解析梯度：先对单位法向求导，再经归一化雅可比映射回原始参数。"""
    normals_unit = normalize_normals(params.normals)
    images, d_img_d_n, d_img_d_rho = render_and_grad(normals_unit, params.albedo, views, obs_cfg)
    if valid is None:
        valid = np.ones_like(observations, dtype=bool)
    valid = valid.astype(np.float64)
    denom = max(float(valid.sum()), 1.0)
    residual = images - observations
    factor = (2.0 / denom) * valid * residual  # (M, H, W)

    grad_unit_n = np.sum(factor[..., None] * d_img_d_n, axis=0)  # (H,W,3)
    grad_rho = np.sum(factor * d_img_d_rho, axis=0)

    # 归一化 n_hat = n / |n| 的雅可比：J = (I - n_hat n_hat^T) / |n|
    n = params.normals
    n_norm = np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    proj = grad_unit_n - np.sum(grad_unit_n * normals_unit, axis=-1, keepdims=True) * normals_unit
    grad_n = proj / n_norm

    return GradientBundle(
        grad_normal=grad_n,
        grad_albedo=grad_rho,
        grad_normal_norm=float(np.linalg.norm(grad_n)),
        grad_albedo_norm=float(np.linalg.norm(grad_rho)),
        loss=mse_loss(images, observations, valid.astype(bool)),
        images=images,
    )


def apply_gradient_step(params: SurfaceParams, bundle: GradientBundle, opt: OptConfig) -> SurfaceParams:
    """增量参数更新：法向梯度下降后归一化，反照率下降后裁剪非负。"""
    updated = params.copy()
    updated.normals = normalize_normals(updated.normals - opt.lr_normal * bundle.grad_normal)
    updated.albedo = np.clip(updated.albedo - opt.lr_albedo * bundle.grad_albedo, 0.0, None)
    return updated


def finite_difference_check(
    params: SurfaceParams,
    observations: np.ndarray,
    views: ViewTable,
    obs_cfg: ObservationConfig,
    opt: OptConfig,
    sample_count: int = 32,
    seed: int = 777,
) -> dict[str, float]:
    """在确定性像素样本上逐分量比较解析梯度与中心有限差分。

    返回最大绝对/相对误差；调用方按 ``opt.gradcheck_tol`` 判定。
    """
    rng = np.random.default_rng(seed)
    h, w = params.albedo.shape
    rows = rng.integers(0, h, size=sample_count)
    cols = rng.integers(0, w, size=sample_count)
    eps = opt.finite_diff_eps
    bundle = compute_gradients(params, observations, views, obs_cfg)
    max_abs = 0.0
    max_rel = 0.0

    for r, c in zip(rows, cols):
        for axis in range(3):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus.normals[r, c, axis] += eps
            p_minus.normals[r, c, axis] -= eps
            l_plus = compute_loss_only(p_plus, observations, views, obs_cfg)
            l_minus = compute_loss_only(p_minus, observations, views, obs_cfg)
            fd = (l_plus - l_minus) / (2.0 * eps)
            analytic = bundle.grad_normal[r, c, axis]
            max_abs = max(max_abs, abs(fd - analytic))
            max_rel = max(max_rel, abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12))

        p_plus = params.copy()
        p_minus = params.copy()
        p_plus.albedo[r, c] += eps
        p_minus.albedo[r, c] -= eps
        l_plus = compute_loss_only(p_plus, observations, views, obs_cfg)
        l_minus = compute_loss_only(p_minus, observations, views, obs_cfg)
        fd = (l_plus - l_minus) / (2.0 * eps)
        analytic = bundle.grad_albedo[r, c]
        max_abs = max(max_abs, abs(fd - analytic))
        max_rel = max(max_rel, abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12))

    return {"max_abs_error": max_abs, "max_rel_error": max_rel, "sample_count": float(sample_count)}


def compute_loss_only(
    params: SurfaceParams,
    observations: np.ndarray,
    views: ViewTable,
    obs_cfg: ObservationConfig,
) -> float:
    images = render_all(normalize_normals(params.normals), params.albedo, views, obs_cfg)
    return mse_loss(images, observations)


# ---- 瓦片化增量渲染与全量核对（业务要求 6） -------------------------------


def tile_ranges(rows: int, cols: int, tile_rows: int, tile_cols: int) -> list[tuple[int, int, int, int]]:
    """行优先、列升序的瓦片范围列表 (r0, r1, c0, c1)。"""
    ranges: list[tuple[int, int, int, int]] = []
    for r0 in range(0, rows, tile_rows):
        for c0 in range(0, cols, tile_cols):
            ranges.append((r0, min(r0 + tile_rows, rows), c0, min(c0 + tile_cols, cols)))
    return ranges


def render_tiles_incremental(
    params: SurfaceParams,
    views: ViewTable,
    obs_cfg: ObservationConfig,
    opt: OptConfig,
    dirty_tiles: list[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """增量路径：仅重算声明的脏瓦片，其余位置为 0，由调用方与基线拼接/核对。

    返回 (部分图像栈, 实际重算瓦片范围)。``dirty_tiles=None`` 表示全量范围，
    此时与 :func:`render_all` 语义一致，可直接逐瓦片核对。
    """
    h, w = params.albedo.shape
    all_tiles = tile_ranges(h, w, opt.tile_rows, opt.tile_cols)
    targets = all_tiles if dirty_tiles is None else dirty_tiles
    partial = np.zeros((views.size, h, w), dtype=np.float64)
    normals_unit = normalize_normals(params.normals)
    for r0, r1, c0, c1 in targets:
        sub_n = normals_unit[r0:r1, c0:c1]
        sub_rho = params.albedo[r0:r1, c0:c1]
        for m in range(views.size):
            n_cam = sub_n @ views.rotations[m].T
            n_dot_l = np.clip(n_cam @ views.lights[m], 0.0, None)
            n_dot_h = np.clip(n_cam @ views.half_vectors[m], 0.0, None)
            partial[m, r0:r1, c0:c1] = np.clip(
                sub_rho * n_dot_l + obs_cfg.specular_gain * np.power(n_dot_h, obs_cfg.shininess),
                0.0,
                None,
            )
    return partial, targets


def verify_incremental_vs_full(
    params: SurfaceParams,
    observations: np.ndarray,
    views: ViewTable,
    obs_cfg: ObservationConfig,
    opt: OptConfig,
    dirty_tiles: list[tuple[int, int, int, int]],
) -> dict[str, object]:
    """核对增量重渲染与当前参数全量重渲染，分歧即 ``E_RENDER_DIVERGE``。

    核对失败时按瓦片行优先、列升序定位首个分歧瓦片，并列出其依赖参数范围；
    不会通过放宽容差掩盖分歧。
    """
    full = render_all(normalize_normals(params.normals), params.albedo, views, obs_cfg)
    partial, recomputed = render_tiles_incremental(params, views, obs_cfg, opt, dirty_tiles)
    ranges = tile_ranges(*params.albedo.shape, opt.tile_rows, opt.tile_cols)
    recomputed_set = set(recomputed)

    for tile in ranges:
        r0, r1, c0, c1 = tile
        if tile not in recomputed_set:
            continue
        full_tile = full[:, r0:r1, c0:c1]
        inc_tile = partial[:, r0:r1, c0:c1]
        if not np.allclose(full_tile, inc_tile, atol=opt.render_check_atol, rtol=opt.render_check_rtol):
            diff = np.abs(full_tile - inc_tile)
            m_idx, rr, cc = np.unravel_index(int(np.argmax(diff)), diff.shape)
            raise BusinessError(
                E_RENDER_DIVERGE,
                "增量重渲染与全量重渲染在声明容差内不一致",
                layer="pipeline",
                details={
                    "first_diverging_tile": [r0, r1, c0, c1],
                    "tile_order": "row-major/col-ascending",
                    "max_abs_diff": float(diff.max()),
                    "atol": opt.render_check_atol,
                    "rtol": opt.render_check_rtol,
                    "worst_pixel": [int(m_idx), int(r0 + rr), int(c0 + cc)],
                    "depends_on": {
                        "normal_slice": [[r0, r1], [c0, c1]],
                        "albedo_slice": [[r0, r1], [c0, c1]],
                    },
                },
            )

    full_loss = mse_loss(full, observations)
    merged = full.copy()
    for r0, r1, c0, c1 in recomputed:
        merged[:, r0:r1, c0:c1] = partial[:, r0:r1, c0:c1]
    merged_loss = mse_loss(merged, observations)
    if abs(full_loss - merged_loss) > max(opt.render_check_atol, opt.render_check_rtol * abs(full_loss)):
        raise BusinessError(
            E_RENDER_DIVERGE,
            "增量路径损失与全量路径损失不一致",
            layer="pipeline",
            details={"full_loss": full_loss, "incremental_loss": merged_loss},
        )
    return {
        "ok": True,
        "recomputed_tiles": [list(t) for t in recomputed],
        "tile_count": len(recomputed),
        "full_loss": full_loss,
        "incremental_loss": merged_loss,
    }
