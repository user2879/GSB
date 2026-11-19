"""步长与截断阈值抖动下的稳定性量化评估（业务要求 5）。

所有结论均由实际重渲数值计算，不使用任何预设方向：
* 逐像素最大偏差 ``max|a-b|`` 与均方偏差 ``mean((a-b)^2)``；
* 步长抖动按步长升序检查偏差是否单调不增（容差内）判定收敛；
* 阈值抖动按阈值升序（放宽截断）检查偏差是否不单调下降，判定发散/失稳。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .assembly import assemble_supersampled, downsample
from .renderer import render_tile
from .scheduler import build_tile_specs
from .scene import Scene


def _render_pixel_grid(scene: Scene) -> np.ndarray:
    """在当前场景配置下同步渲染整幅，返回最终像素分辨率的 float64 灰度。"""
    cfg = scene.config
    specs = build_tile_specs(scene)
    tiles = [render_tile(scene, s.index, s.x0, s.y0, s.width, s.height) for s in specs]
    grid, _, _, _ = assemble_supersampled(tiles, cfg)
    return downsample(grid, cfg.supersample)


def _metrics(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate - baseline
    return {
        "max_abs_pixel_deviation": float(np.max(np.abs(delta))),
        "mean_squared_pixel_deviation": float(np.mean(delta ** 2)),
        "mean_abs_pixel_deviation": float(np.mean(np.abs(delta))),
    }


def _monotone_nonincreasing(values: list[float], rel_tol: float = 1e-9) -> bool:
    if len(values) < 2:
        return True
    scale = max(1e-300, max(abs(v) for v in values))
    tol = rel_tol * scale
    return all(b <= a + tol for a, b in zip(values[:-1], values[1:]))


def _rank(values: list[float]) -> list[float]:
    """平均秩（处理并列），避免依赖 SciPy。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def evaluate_stability(scene: Scene, baseline_pixels: np.ndarray) -> dict[str, Any]:
    cfg = scene.config
    result: dict[str, Any] = {
        "baseline": {"step_length": cfg.step_length,
                     "transmittance_cutoff": cfg.transmittance_cutoff},
        "step_length_jitter": [],
        "cutoff_jitter": [],
        "conclusions": [],
    }

    # 步长抖动：固定阈值，只改步长。
    step_points: list[tuple[float, dict[str, float]]] = [(
        cfg.step_length,
        {"max_abs_pixel_deviation": 0.0,
         "mean_squared_pixel_deviation": 0.0,
         "mean_abs_pixel_deviation": 0.0,
         "is_baseline": True},
    )]
    for jitter in sorted(set(cfg.step_jitter)):
        if jitter == cfg.step_length:
            continue
        varied = replace(cfg, step_length=jitter)
        sub_scene = Scene(scene.volumes, scene.lights, scene.camera, varied,
                          scene.bbox_min, scene.bbox_max, scene.source_path)
        pixels = _render_pixel_grid(sub_scene)
        metrics = _metrics(baseline_pixels, pixels)
        result["step_length_jitter"].append(
            {"step_length": jitter, **metrics})
        step_points.append((jitter, metrics))

    if len(step_points) > 1:
        ordered = sorted(step_points, key=lambda p: p[0])
        steps_seq = [s0 for s0, _ in ordered]
        max_dev = [m["max_abs_pixel_deviation"] for _, m in ordered]
        mse = [m["mean_squared_pixel_deviation"] for _, m in ordered]

        def _monotone_nondecreasing(values: list[float], rel_tol: float = 1e-9) -> bool:
            if len(values) < 2:
                return True
            scale = max(1e-300, max(abs(v) for v in values))
            tol = rel_tol * scale
            return all(b >= a - tol for a, b in zip(values[:-1], values[1:]))

        # 基准取最细步长（其偏差为 0）：收敛 iff 偏差随步长增大单调不减，
        # 等价于细化步长时偏差单调回落；Spearman 接近 +1 佐证该趋势。
        rho = _spearman(steps_seq, max_dev)
        converging = _monotone_nondecreasing(max_dev) and             _monotone_nondecreasing(mse) and rho >= 0.9
        spread = float(max(max_dev) - min(max_dev))
        conclusion = (
            f"以最细基准步长 {cfg.step_length}（偏差 0）为参照，步长在 "
            f"{steps_seq} 上的逐像素最大偏差为 {[round(v, 3) for v in max_dev]}，"
            f"Spearman={rho:.3f}；"
            + ("步长增大偏差单调不减、细化步长偏差单调回落，呈数值收敛趋势。"
               if converging else
               "偏差未随步长单调变化，未观察到单调收敛（结论由上述实测值给出）。")
        )
        result["conclusions"].append(conclusion)
        result["step_trend"] = {
            "converging_when_refining": bool(converging),
            "monotone_nondecreasing_with_step": bool(
                _monotone_nondecreasing(max_dev)),
            "max_deviation_spread": spread,
            "spearman_step_vs_deviation": rho,
            "ordered_step_lengths": steps_seq,
            "ordered_max_deviations": max_dev,
        }

    # 截断阈值抖动：固定步长，只改阈值。阈值越大越早截断 -> 偏差倾向增大。
    cutoff_points: list[tuple[float, dict[str, float]]] = [(
        cfg.transmittance_cutoff,
        {"max_abs_pixel_deviation": 0.0,
         "mean_squared_pixel_deviation": 0.0,
         "mean_abs_pixel_deviation": 0.0,
         "is_baseline": True},
    )]
    for jitter in sorted(set(cfg.cutoff_jitter)):
        if jitter == cfg.transmittance_cutoff:
            continue
        varied = replace(cfg, transmittance_cutoff=jitter)
        sub_scene = Scene(scene.volumes, scene.lights, scene.camera, varied,
                          scene.bbox_min, scene.bbox_max, scene.source_path)
        pixels = _render_pixel_grid(sub_scene)
        metrics = _metrics(baseline_pixels, pixels)
        result["cutoff_jitter"].append(
            {"transmittance_cutoff": jitter, **metrics})
        cutoff_points.append((jitter, metrics))

    if len(cutoff_points) > 1:
        ordered = sorted(cutoff_points, key=lambda p: p[0])
        max_dev = [m["max_abs_pixel_deviation"] for _, m in ordered]
        nontrivial = any(v > 0.0 for v in max_dev)
        cuts = [c for c, _ in ordered]
        rho_all = _spearman(cuts, max_dev)
        strict = all(b >= a - 1e-9 * max(1.0, abs(a))
                     for a, b in zip(max_dev[:-1], max_dev[1:]))
        # 放宽侧：阈值不小于基准的点（含基准零点），平台不影响该侧的单调性判断。
        relax = [(c, m["max_abs_pixel_deviation"])
                 for c, m in ordered if c >= cfg.transmittance_cutoff]
        relax_cuts = [c for c, _ in relax]
        relax_dev = [v for _, v in relax]
        relax_strict = all(b >= a - 1e-9 * max(1.0, abs(a))
                           for a, b in zip(relax_dev[:-1], relax_dev[1:]))
        relax_rho = _spearman(relax_cuts, relax_dev)
        if not nontrivial:
            conclusion = (
                f"截断阈值在 {cuts} 范围内所有重渲与基准逐像素偏差均为 0："
                "没有任何光线在该阈值区间内提前截断，阈值不构成可观测影响（实测为零）。")
            diverging = False
        else:
            # 放宽侧严格单调不减即判发散；收紧侧受截断下限影响常出现平台，单独报告。
            diverging = relax_strict and relax_rho >= 0.9
            relation = ("放宽侧偏差随阈值放宽单调不减，呈发散/失稳趋势"
                        if diverging else "放宽侧未呈单调发散，阈值影响不稳定")
            conclusion = (
                f"以基准阈值 {cfg.transmittance_cutoff}（偏差 0）为参照，阈值在 {cuts} "
                f"上的逐像素最大偏差为 {[round(v, 4) for v in max_dev]}；"
                f"放宽侧 {relax_cuts} 的 Spearman={relax_rho:.3f}、严格单调={relax_strict}，"
                f"全区间 Spearman={rho_all:.3f}；实测为{relation}。")
        result["conclusions"].append(conclusion)
        result["cutoff_trend"] = {
            "diverging": bool(diverging),
            "strict_monotone_nondecreasing": bool(strict),
            "relax_side_strict_monotone": bool(relax_strict),
            "relax_side_spearman": relax_rho,
            "spearman_cutoff_vs_deviation": rho_all,
            "ordered_cutoffs": cuts,
            "ordered_max_deviations": max_dev,
        }

    if not result["conclusions"]:
        result["conclusions"].append("未提供任何抖动集合，跳过稳定性评估。")
    return result
