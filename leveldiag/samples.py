"""确定性样本规格合成（不依赖网络与外部资源）。"""

from __future__ import annotations

import copy


def _base_spec() -> dict:
    spec = {
        "grid": {"width": 10, "height": 8},
        "quantization": {"elevation_step": 0.125, "density_step": 0.125,
                         "rounding": "nearest"},
        "terrain": {
            "types": ["grass", "rock", "water"],
            "weights": {"grass": 0.6, "rock": 0.3, "water": 0.1},
            "weight_normalization_tolerance": 1e-9,
        },
        "regions": [
            {"id": "r1", "cells": [[x, y] for x in range(0, 5)
                                   for y in range(0, 8)]},
            {"id": "r2", "cells": [[x, y] for x in range(5, 10)
                                   for y in range(0, 8)]},
        ],
        "connectivity": [
            {"id": "c1", "region_ids": ["r1", "r2"],
             "min_channel_width_ratio": 0.25, "tolerance": 1e-12}],
        "resources": {"count": {"min": 3, "max": 6},
                      "density_cap": 0.09, "tolerance": 1e-12},
        "terrain_rules": {
            "passable": ["grass", "rock"],
            "impassable": ["water"],
            "forbidden_adjacent": [["water", "spawn"]],
        },
        "points": {"spawn": [1, 1], "target": [8, 6]},
        "thresholds": {"elevation_pass": 0.5,
                       "cell_density_max": 0.875,
                       "cell_density_tolerance": 1e-12},
        "seed": 20241119,
    }
    # 确定性锚点：高程统一 0.75（量化后 0.75 > 0.5，BFS 连通不受地形阻断）；
    # 密度统一 0.5 步长（nearest 量化为 0，严格低于 cell_density_max）。
    # 地形仍按权重随机（含 water），覆盖归一化与禁止相邻逻辑。
    spec["elevation_anchors"] = [
        {"region_id": region["id"], "cell": [x, y],
         "continuous_elevation": 0.75}
        for region in spec["regions"]
        for x, y in region["cells"]
    ]
    spec["density_anchors"] = [
        {"region_id": region["id"], "cell": [x, y],
         "continuous_density": 0.5 * spec["quantization"]["density_step"]}
        for region in spec["regions"]
        for x, y in region["cells"]
    ]
    return spec


def normal_spec() -> dict:
    """正常规格：权重和精确为 1，全部约束确定通过。"""
    return _base_spec()


def weight_critical_spec() -> dict:
    """权重偏差临界样本：权重和偏差恰好等于声明容忍（边界相等，确定通过并在
    边界报告留痕，不静默四舍五入）。"""
    spec = _base_spec()
    eps = 1e-9
    # 单边加 eps：0.6+1e-9 + 0.3 + 0.1 在双精度下与 1 的差恰为 1e-9
    spec["terrain"]["weights"]["grass"] = 0.6 + eps
    # 容忍取双精度实际累加偏差（确定性可复算的临界值），使判定恰好落在边界
    spec["terrain"]["weight_normalization_tolerance"] = \
        abs((0.6 + eps) + 0.3 + 0.1 - 1.0)
    spec["seed"] = 20241120
    return spec


def boundary_spec(seed: int = 20241121) -> dict:
    """阈值落在量化边界样本：锚点单元连续密度恰为半步长，nearest 舍入到边界，
    阈值取该量化值，构造 UNCERTAIN（连续真值在误差区间两侧均可能违约）。"""
    spec = _base_spec()
    spec["seed"] = seed
    step = spec["quantization"]["density_step"]
    # 纯 grass 保证 BFS 连通不被地形类别干扰，聚焦量化边界判定本身
    spec["terrain"]["weights"] = {"grass": 1.0, "rock": 0.0, "water": 0.0}
    spec["terrain_rules"]["forbidden_adjacent"] = []
    # 两个锚点分别位于两个区域，连续值为 6.5 步长(0.8125)，nearest 舍入到
    # 7 步长(0.875)；阈值取量化值本身，观测值严格等于阈值且连续真值处于
    # 误差带内 -> UNCERTAIN（真值 0.8125 低于阈值 0.875，但差值未超误差上界）
    # 其余全部单元锚定到 0.5 步长（量化 0，严格低于阈值），仅两个锚点落在边界
    anchors = [
        {"region_id": "r1", "cell": [2, 3],
         "continuous_density": step * 6.5},
        {"region_id": "r2", "cell": [7, 4],
         "continuous_density": step * 6.5},
    ]
    boundary_cells = {("r1", 2, 3), ("r2", 7, 4)}
    for region in spec["regions"]:
        rid = region["id"]
        for x, y in region["cells"]:
            if (rid, x, y) not in boundary_cells:
                anchors.append({"region_id": rid, "cell": [x, y],
                                "continuous_density": step * 0.5})
    spec["density_anchors"] = anchors
    # 高程全部锚定到 0.75（量化后仍 0.75 > elevation_pass=0.5），
    # 使连通性确定通过，边界现象只由密度量化产生
    spec["elevation_anchors"] = [
        {"region_id": region["id"], "cell": [x, y],
         "continuous_elevation": 0.75}
        for region in spec["regions"]
        for x, y in region["cells"]
    ]
    spec["thresholds"]["cell_density_max"] = step * 7
    spec["thresholds"]["cell_density_tolerance"] = 0.0
    return spec


def overflow_spec() -> dict:
    """越界密度样本：密度上限极低，确定失败。"""
    spec = _base_spec()
    spec["seed"] = 20241122
    spec["resources"]["density_cap"] = 1e-6
    return spec


def all_samples() -> dict[str, dict]:
    return {
        "normal": normal_spec(),
        "weight_critical": weight_critical_spec(),
        "boundary": boundary_spec(),
        "overflow": overflow_spec(),
    }


def mutate(spec: dict, dotted_path: str, value) -> dict:
    """测试辅助：按点路径复制并修改规格。支持 a.b[0].c 形式。"""
    out = copy.deepcopy(spec)
    tokens = []
    for chunk in dotted_path.split("."):
        while "[" in chunk:
            head, rest = chunk.split("[", 1)
            if head:
                tokens.append(head)
            idx, chunk = rest.split("]", 1)
            tokens.append(int(idx))
        if chunk:
            tokens.append(chunk)
    cur = out
    for token in tokens[:-1]:
        cur = cur[token]
    cur[tokens[-1]] = value
    return out
