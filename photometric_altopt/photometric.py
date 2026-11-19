"""多视点光度立体求解（业务要求 2）。

每个像素独立地把全部可用视点的线性光观测堆叠成加权最小二乘问题::

    对每个视点 v:  I_v = rho * max(n.l_v, 0) + ks * max(n.h_v, 0)^sh + e_v

镜面参数（ks、sh）在前向模型中固定已知，因此每轮先按当前法向显式扣除
镜面预测，再对漫反射项做加权最小二乘；扣除后的残余偏差由 Huber IRLS
权重鲁棒处理（带权重下限，设计矩阵不会因降权而塌缩）。

实现要点：
* 全像素向量化：有效视点按像素收集到定长缓冲，批处理 SVD/最小二乘；
* Huber 权重仅依赖观测与当前拟合残差，不读取任何真值；
* 法向解出后 L2 归一化，反照率裁剪为非负；
* 设计矩阵条件数超过声明阈值的像素标记为病态，并给误差区间；
* 有效视点不足时抛 ``E_DEGENERATE_VIEWS``，非有限反照率输入抛
  ``E_NONFINITE_ALBEDO``（异常注入守卫，仅检查传入副本）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import ObservationConfig, SolveConfig
from .errors import (
    E_DEGENERATE_VIEWS,
    E_NONFINITE_ALBEDO,
    E_ZERO_LIGHT,
    BusinessError,
)
from .forward import ViewTable

if TYPE_CHECKING:
    from .differentiable import SurfaceParams


@dataclass
class PhotometricResult:
    normals: np.ndarray  # (H, W, 3) 单位法向（世界系）
    albedo: np.ndarray  # (H, W) 非负反照率
    residual: np.ndarray  # (H, W) 加权拟合 RMSE
    used_views: np.ndarray  # (H, W, M) bool，逐像素参与求解的视点
    condition_number: np.ndarray  # (H, W) 设计矩阵条件数
    ill_conditioned: np.ndarray  # (H, W) bool
    error_bounds: dict[str, np.ndarray]  # 法向夹角(弧度)/反照率半宽


def _finite_guard(observations: np.ndarray, albedo_seed: np.ndarray | None) -> None:
    if not np.all(np.isfinite(observations)):
        bad = np.argwhere(~np.isfinite(observations))
        raise BusinessError(
            E_NONFINITE_ALBEDO,
            "观测中出现非有限值（异常注入副本）",
            details={"pixel_or_view": bad[:16].tolist(), "count": int(bad.shape[0])},
        )
    if albedo_seed is not None and not np.all(np.isfinite(albedo_seed)):
        pixels = np.argwhere(~np.isfinite(albedo_seed))
        raise BusinessError(
            E_NONFINITE_ALBEDO,
            "反照率参数出现非有限值（NaN/Inf）",
            layer="parameter",
            details={"first_pixels": pixels[:16].tolist(), "count": int(pixels.shape[0])},
        )


def solve_photometric(
    observations: np.ndarray,
    views: ViewTable,
    obs_cfg: ObservationConfig,
    solve_cfg: SolveConfig,
    valid_mask: np.ndarray | None = None,
    albedo_seed: np.ndarray | None = None,
) -> PhotometricResult:
    """对全部像素执行多视点加权最小二乘光度立体。

    Args:
        observations: (M, H, W) 线性光观测（float32/64 均可）。
        views: 由 :func:`build_view_table` 构造的视点表。
        valid_mask: (M, H, W) bool，False 表示该视点该像素不参与（跳过注入）。
        albedo_seed: 可选反照率副本，仅用于非有限值守卫，不参与求解。
    """
    observations = np.asarray(observations, dtype=np.float64)
    _finite_guard(observations, albedo_seed)
    if np.any(np.linalg.norm(views.lights, axis=-1) <= 0.0):
        raise BusinessError(E_ZERO_LIGHT, "视点表中存在零向量光源", details={})

    m_total, h, w = observations.shape
    pixel_count = h * w
    if valid_mask is None:
        valid_mask = np.ones((m_total, h, w), dtype=bool)
    valid_mask = np.broadcast_to(valid_mask, (m_total, h, w)).copy()

    counts = valid_mask.sum(axis=0)
    if int(counts.min()) < solve_cfg.min_views:
        bad_pixels = np.argwhere(counts < solve_cfg.min_views)
        raise BusinessError(
            E_DEGENERATE_VIEWS,
            f"存在有效视点少于 {solve_cfg.min_views} 的像素集合",
            details={
                "min_valid_views": int(counts.min()),
                "required": solve_cfg.min_views,
                "first_pixels": bad_pixels[:16].tolist(),
                "count": int(bad_pixels.shape[0]),
            },
        )

    # 逐像素把有效视点收集到定长缓冲 K（不足处权重为 0）
    order = np.argsort(~valid_mask, axis=0, kind="stable")  # 有效视点排前
    flat_obs = observations.reshape(m_total, pixel_count)
    flat_mask = valid_mask.reshape(m_total, pixel_count)
    gathered_obs = np.take_along_axis(flat_obs, order.reshape(m_total, pixel_count), axis=0).T
    gathered_active = np.take_along_axis(flat_mask, order.reshape(m_total, pixel_count), axis=0).T

    # 相机系光源/半程向量按视点收集，并旋转回世界系得到世界系设计方向
    # v_cam = R v_world => v_world = R^T v_cam
    lights_world = np.einsum("mji,mj->mi", views.rotations, views.lights)
    half_world = np.einsum("mji,mj->mi", views.rotations, views.half_vectors)
    # order 形状 (M, P)，对广播后的 (M, P, 3) 沿视点轴收集，再转成 (P, K, 3)
    light_base = np.broadcast_to(lights_world[:, None, :], (m_total, pixel_count, 3))
    half_base = np.broadcast_to(half_world[:, None, :], (m_total, pixel_count, 3))
    order3 = np.broadcast_to(order.reshape(m_total, pixel_count, 1), (m_total, pixel_count, 3))
    light_gathered = np.take_along_axis(light_base, order3, axis=0).transpose(1, 0, 2)
    half_gathered = np.take_along_axis(half_base, order3, axis=0).transpose(1, 0, 2)
    active = gathered_active  # (P, K)
    y = gathered_obs  # (P, K)

    weights = active.astype(np.float64)
    params = np.zeros((pixel_count, 3), dtype=np.float64)
    rho = np.zeros(pixel_count, dtype=np.float64)
    condition = np.full(pixel_count, np.inf, dtype=np.float64)
    spec_pred = np.zeros_like(y)

    for _ in range(max(solve_cfg.irls_iterations, 1)):
        # 漫反射设计矩阵 A = n.l（逐像素 (K,3)），镜面项不进设计矩阵
        design = light_gathered
        y_diffuse = np.maximum(y - spec_pred, 0.0)
        w2 = weights * weights
        ata = np.einsum("pki,pk,pkj->pij", design, w2, design)
        aty = np.einsum("pki,pk,pk->pi", design, w2, y_diffuse)
        # ATA 为 3x3 对称（半正定）阵，用特征值分解比完整 SVD 更快
        eig_w, eig_v = np.linalg.eigh(ata)  # 特征值升序，列向量为特征向量
        # 填充槽位对应（近）零特征值，只统计超过奇异值下限的真实方向
        nonzero = eig_w >= solve_cfg.singular_floor
        last_idx = nonzero.sum(axis=1) - 1
        s_min = eig_w[np.arange(pixel_count), np.maximum(last_idx, 0)]
        lambda_max = eig_w[:, -1]
        # ATA 是设计矩阵 A 的 Gram 阵；设计矩阵条件数 = sqrt(lambda_max/lambda_min)
        condition = np.sqrt(lambda_max / np.maximum(s_min, solve_cfg.singular_floor))
        inv_w = np.where(nonzero, 1.0 / np.maximum(eig_w, solve_cfg.singular_floor), 0.0)
        # 未缩放向量 g = rho * n；对称阵解 = V diag(1/lambda) V^T b
        g = np.einsum("pij,pj,pjk,pk->pi", eig_v, inv_w, eig_v.transpose(0, 2, 1), aty)
        g_norm = np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        n_hat = g / g_norm
        # 设计矩阵固定为光源方向（不随迭代估计改写，避免自洽偏差）；
        # 单位法向确定后，反照率是带非负约束的一维加权最小二乘，裁剪只作用于投影
        n_dot_l = np.einsum("pki,pi->pk", design, n_hat)
        lit = n_dot_l > 0.0
        n_dot_l = np.where(lit, n_dot_l, 0.0)
        # 显式镜面预测：前向模型的 ks/光泽度已知，从观测中扣除后再做漫反射拟合
        n_dot_h = np.clip(np.einsum("pki,pi->pk", half_gathered, n_hat), 0.0, None)
        spec_pred = active * obs_cfg.specular_gain * np.power(n_dot_h, obs_cfg.shininess)
        denom = np.sum(weights * n_dot_l**2, axis=1)
        rho = np.clip(
            np.sum(weights * y_diffuse * n_dot_l, axis=1) / np.maximum(denom, 1e-12), 0.0, None
        )
        params = g

        # 残差 = 观测 - (漫反射 + 显式镜面)，残余偏差由 Huber 权重鲁棒处理
        prediction = rho[:, None] * n_dot_l + spec_pred
        residual = y - prediction
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.median(np.where(active, np.abs(residual), np.nan), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1e-6)
        standardized = residual / (solve_cfg.irls_tuning * scale[:, None])
        huber = np.where(
            active,
            np.where(np.abs(standardized) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(standardized), 1e-12)),
            0.0,
        )
        # 权重下限：镜面外点被持续降权，但任何有效视点都不会完全退出
        weights = active * np.maximum(huber, solve_cfg.weight_floor)

    # 终态整理
    norm = np.maximum(np.linalg.norm(params, axis=1, keepdims=True), 1e-12)
    normals = params / norm
    normals = normals.reshape(h, w, 3)
    rho = np.clip(rho, 0.0, None).reshape(h, w)

    final_pred = np.clip(np.einsum("pki,pi->pk", light_gathered, normals.reshape(pixel_count, 3)), 0.0, None)
    final_nh = np.clip(
        np.einsum("pki,pi->pk", half_gathered, normals.reshape(pixel_count, 3)), 0.0, None
    )
    final_spec = obs_cfg.specular_gain * np.power(final_nh, obs_cfg.shininess)
    final_pred = rho.reshape(pixel_count, 1) * final_pred + final_spec
    sq = weights * (y - final_pred) ** 2
    wsum = np.maximum(weights.sum(axis=1), 1e-12)
    rmse = np.sqrt(np.sum(sq, axis=1) / wsum).reshape(h, w)

    condition_map = condition.reshape(h, w)
    ill = condition_map > solve_cfg.condition_threshold
    used_views = valid_mask.transpose(1, 2, 0)  # (H, W, M)

    # 误差区间：用加权残差尺度 + 协方差最小奇异值估计
    sigma2 = (np.sum(sq, axis=1) / wsum).reshape(h, w)
    s_min = s_min.reshape(h, w)
    normal_angle_half = np.sqrt(sigma2) / np.maximum(np.sqrt(s_min), 1e-12) * solve_cfg.confidence_z
    rho_half = solve_cfg.confidence_z * np.sqrt(sigma2 / np.maximum(denom.reshape(h, w), 1e-12))

    return PhotometricResult(
        normals=normals,
        albedo=rho,
        residual=rmse,
        used_views=used_views,
        condition_number=condition_map,
        ill_conditioned=ill,
        error_bounds={
            "normal_angle_rad_halfwidth": np.clip(normal_angle_half, 0.0, np.pi),
            "albedo_abs_halfwidth": np.clip(rho_half, 0.0, None),
        },
    )


def photometric_blend(
    current: "SurfaceParams",
    ps_result: "PhotometricResult",
    blend: float,
) -> "PhotometricResult":
    """把一轮独立 PS 求解结果按 ``blend`` 融合到当前 DR 参数上。

    融合仅用于让 PS 阶段在交替流程中保持独立语义而不抹掉 DR 的增量改进；
    PS 的原始求解结果仍完整保留在返回值中，用于阶段间对照与冲突定位。
    ``blend=1.0`` 时等价于纯 PS 解（用于首轮初始化）。
    """
    from .differentiable import normalize_normals

    blended_n = normalize_normals(
        (1.0 - blend) * normalize_normals(current.normals) + blend * ps_result.normals
    )
    blended_rho = np.clip(
        (1.0 - blend) * current.albedo + blend * ps_result.albedo, 0.0, None
    )
    return PhotometricResult(
        normals=blended_n,
        albedo=blended_rho,
        residual=ps_result.residual.copy(),
        used_views=ps_result.used_views.copy(),
        condition_number=ps_result.condition_number.copy(),
        ill_conditioned=ps_result.ill_conditioned.copy(),
        error_bounds={k: v.copy() for k, v in ps_result.error_bounds.items()},
    )
