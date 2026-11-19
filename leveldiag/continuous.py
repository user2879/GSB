"""固定种子下的连续域候选布局求解。

输出全部以双精度保存：
  * 逐区域高程场 elevation 与密度场 density（按 cell 坐标索引）；
  * 逐图块连续地形权重抽样值 terrain_draw（均匀分布 -> 类别，类别本身离散，
    但抽样的连续 uniforms 一并留痕，供复算）；
  * 资源点候选（连续坐标量化到网格）。

确定性保证：区域按 id 排序、区域内 cell 按 (x,y) 字典序处理；
不使用线程；每个用途独立 PCG64 流。
"""

from __future__ import annotations

import numpy as np

from .seeding import build_seed_tree, derive_subseed, make_rng
from .spec import Spec
from .util import content_hash


def _sorted_cells(region: dict) -> list[tuple[int, int]]:
    return sorted((int(c[0]), int(c[1])) for c in region["cells"])


def solve_continuous(spec: Spec) -> dict:
    """求解连续域候选布局，返回可直接分片落盘的结构。"""
    raw = spec.raw
    master = spec.seed
    regions = sorted(raw["regions"], key=lambda r: r["id"])
    region_ids = [r["id"] for r in regions]
    seed_tree = build_seed_tree(master, region_ids)

    terrain_types = list(raw["terrain"]["types"])
    # 归一化权重：显式按名称排序累加，记录归一化误差
    weights = {k: float(v) for k, v in raw["terrain"]["weights"].items()}
    ordered = sorted(weights)
    weight_sum = 0.0
    for name in ordered:
        weight_sum += weights[name]
    normalized = {name: weights[name] / weight_sum for name in ordered}
    cumulative: list[float] = []
    acc = 0.0
    for name in ordered:
        acc += normalized[name]
        cumulative.append(acc)
    cumulative[-1] = 1.0  # 防止末端浮点漂移导致抽样越界
    p_array = np.array([normalized[n] for n in ordered], dtype=np.float64)

    terrain_rng = make_rng(derive_subseed(master, "terrain_choice"))
    width, height = spec.width, spec.height
    # 全局按 (y,x) 行主序抽样，顺序与区域划分无关
    uniforms = terrain_rng.random(width * height, dtype=np.float64)
    draw_flat = np.searchsorted(np.array(cumulative, dtype=np.float64),
                                uniforms, side="right")
    draw_flat = np.clip(draw_flat, 0, len(ordered) - 1)
    terrain_grid = [[ordered[int(draw_flat[y * width + x])]
                     for x in range(width)] for y in range(height)]
    terrain_uniforms = [[float(uniforms[y * width + x])
                         for x in range(width)] for y in range(height)]

    region_shards: dict[str, dict] = {}
    for idx, region in enumerate(regions):
        rid = region["id"]
        cells = _sorted_cells(region)
        elev_rng = make_rng(derive_subseed(master, "elevation_field", rid, idx))
        dens_rng = make_rng(derive_subseed(master, "density_field", rid, idx))
        n = len(cells)
        # 高程：截断到 [0,1] 的正态场；密度：均匀场，均为 float64
        elev = elev_rng.normal(loc=0.5, scale=0.2, size=n).astype(np.float64)
        elev = np.clip(elev, 0.0, 1.0)
        dens = dens_rng.random(n, dtype=np.float64)

        for anchor in raw.get("elevation_anchors", []) or []:
            if anchor.get("region_id") != rid:
                continue
            ax, ay = int(anchor["cell"][0]), int(anchor["cell"][1])
            target_value = float(anchor["continuous_elevation"])
            for j, (cx, cy) in enumerate(cells):
                if (cx, cy) == (ax, ay):
                    elev[j] = np.float64(target_value)

        for anchor in raw.get("density_anchors", []) or []:
            if anchor.get("region_id") != rid:
                continue
            ax, ay = int(anchor["cell"][0]), int(anchor["cell"][1])
            target_value = float(anchor["continuous_density"])
            for j, (cx, cy) in enumerate(cells):
                if (cx, cy) == (ax, ay):
                    dens[j] = np.float64(target_value)

        entries = []
        for j, (cx, cy) in enumerate(cells):
            entries.append({
                "cell": [cx, cy],
                "elevation": float(elev[j]),
                "density": float(dens[j]),
                "terrain": terrain_grid[cy][cx],
                "terrain_uniform": float(terrain_uniforms[cy][cx]),
            })
        shard = {
            "region_id": rid,
            "cells": entries,
            "field_shape": "per-cell rows ordered by (x,y)",
        }
        region_shards[rid] = shard

    # 资源点：在全部区域 cell 的并集中按行主序候选，用独立流抽样位置
    all_cells = sorted({(int(c[0]), int(c[1]))
                        for r in regions for c in r["cells"]})
    res_rng = make_rng(derive_subseed(master, "resource_placement"))
    rmin = int(raw["resources"]["count"]["min"])
    rmax = int(raw["resources"]["count"]["max"])
    target_count = int(res_rng.integers(rmin, rmax + 1))
    perm_index = res_rng.permutation(len(all_cells))
    chosen = [all_cells[int(i)] for i in perm_index[:target_count]]
    resources = [{
        "cell": [x, y],
        "continuous_position": [float(x) + 0.5, float(y) + 0.5],
    } for (x, y) in sorted(chosen)]

    normalization = {
        "declared_weights": {k: weights[k] for k in ordered},
        "sum_order": ordered,
        "weight_sum": float(weight_sum),
        "normalized_weights": {k: float(normalized[k]) for k in ordered},
        "normalization_error_bound": float(
            abs(weight_sum - 1.0) + len(ordered) * np.finfo(np.float64).eps),
        "cumulative": [float(v) for v in cumulative],
        "probability_array": [float(v) for v in p_array],
    }

    result = {
        "spec_hash": spec.spec_hash,
        "seed_tree": seed_tree,
        "terrain_grid": terrain_grid,
        "terrain_uniforms": terrain_uniforms,
        "region_shards": region_shards,
        "resources": resources,
        "weight_normalization": normalization,
    }
    result["continuous_hash"] = content_hash(
        {"terrain_grid": terrain_grid,
         "terrain_uniforms": terrain_uniforms,
         "region_shards": region_shards,
         "resources": resources,
         "normalization": normalization})
    return result
