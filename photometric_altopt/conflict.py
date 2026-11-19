"""阶段间语义对照与冲突定位（业务要求 5）。

对同一轮的光度立体（PS）结果与可微渲染（DR）参数逐像素比较：

* 法向：单位向量夹角（度），超过 ``conflict_angle_deg`` 记冲突；
* 反照率：绝对偏差，超过 ``conflict_albedo_abs`` 记冲突；
* 同时记录两阶段更新/梯度方向的夹角与幅度，作为冲突诊断依据。

判定完全基于逐像素量，不使用整体损失接近作为一致性依据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OptConfig


@dataclass
class ConflictReport:
    conflict_mask: np.ndarray  # (H, W) bool
    angle_deg: np.ndarray  # (H, W)
    albedo_diff: np.ndarray  # (H, W)
    first_conflict: tuple[int, int] | None
    pixels: list[dict[str, object]]


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐像素单位法向夹角（度），数值裁剪到 arccos 合法域。"""
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """两个扁平向量的方向夹角（度）；零向量约定为 90 度（方向未定义）。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-15 or nb <= 1e-15:
        return 90.0
    cos = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compare_stages(
    ps_normals: np.ndarray,
    ps_albedo: np.ndarray,
    dr_normals: np.ndarray,
    dr_albedo: np.ndarray,
    opt: OptConfig,
    ps_update_normal: np.ndarray | None = None,
    dr_grad_normal: np.ndarray | None = None,
) -> ConflictReport:
    """逐像素对照两阶段结果，冲突像素按行优先、列升序排序。"""
    angle = normal_angle_deg(ps_normals, dr_normals)
    rho_diff = np.abs(ps_albedo - dr_albedo)
    mask = (angle > opt.conflict_angle_deg) | (rho_diff > opt.conflict_albedo_abs)

    coords = np.argwhere(mask)
    # np.argwhere 本身按行优先；显式稳定排序以固定契约
    order = np.lexsort((coords[:, 1], coords[:, 0])) if coords.size else np.empty(0, dtype=int)
    coords = coords[order]

    flat_ps_n = ps_normals.reshape(-1, 3)
    flat_dr_n = dr_normals.reshape(-1, 3)
    flat_ps_rho = ps_albedo.reshape(-1)
    flat_dr_rho = dr_albedo.reshape(-1)
    h, w = ps_albedo.shape

    pixels: list[dict[str, object]] = []
    for r, c in coords:
        flat_idx = int(r) * w + int(c)
        reasons = []
        if angle[r, c] > opt.conflict_angle_deg:
            reasons.append("normal_angle")
        if rho_diff[r, c] > opt.conflict_albedo_abs:
            reasons.append("albedo_abs")
        entry: dict[str, object] = {
            "row": int(r),
            "col": int(c),
            "reasons": reasons,
            "ps_normal": flat_ps_n[flat_idx].tolist(),
            "dr_normal": flat_dr_n[flat_idx].tolist(),
            "normal_angle_deg": float(angle[r, c]),
            "ps_albedo": float(flat_ps_rho[flat_idx]),
            "dr_albedo": float(flat_dr_rho[flat_idx]),
            "albedo_abs_diff": float(rho_diff[r, c]),
        }
        if ps_update_normal is not None and dr_grad_normal is not None:
            ps_vec = np.atleast_1d(ps_update_normal[r, c]).astype(np.float64)
            dr_vec = np.atleast_1d(dr_grad_normal[r, c]).astype(np.float64)
            entry["gradient_direction_angle_deg"] = _vector_angle_deg(ps_vec, dr_vec)
            entry["ps_gradient_magnitude"] = float(np.linalg.norm(ps_vec))
            entry["dr_gradient_magnitude"] = float(np.linalg.norm(dr_vec))
        pixels.append(entry)

    first = (int(coords[0][0]), int(coords[0][1])) if coords.size else None
    return ConflictReport(
        conflict_mask=mask,
        angle_deg=angle,
        albedo_diff=rho_diff,
        first_conflict=first,
        pixels=pixels,
    )
