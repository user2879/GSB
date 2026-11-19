"""约束判定与误差区间合成。

每个判定条目记录：
  constraint_id / kind / 观测值 observed / 阈值 threshold /
  比较方向 direction / 比较容差 comparison_tolerance /
  量化误差上界 quant_error_bound / 归一化误差 normalization_error /
  传播合成总误差上界 propagated_error_bound /
  真值来源 truth_sources（连续场、映射表分片、网格、规格字段）/
  状态 status（PASS/FAIL/UNCERTAIN）/ 证据 evidence。

判定统一走 :func:`leveldiag.util.classify`，边界相等不静默四舍五入。
"""

from __future__ import annotations

from collections import deque

from .spec import Spec
from .util import interval_status, ulp


def _entry(base: dict) -> dict:
    status = interval_status(
        base["direction"], base["observed"], base["threshold"],
        base["comparison_tolerance"], base["propagated_error_bound"])
    base.update(status)
    return base


def _tile_index(grid: dict) -> dict[tuple[int, int], dict]:
    idx = {}
    for row in grid["tiles"]:
        for tile in row:
            idx[(tile["cell"][0], tile["cell"][1])] = tile
    return idx


def evaluate(spec: Spec, continuous: dict, mapping: dict, grid: dict) -> list[dict]:
    raw = spec.raw
    entries: list[dict] = []
    tile_at = _tile_index(grid)
    elev_bound = mapping["rules"]["elevation_error_bound"]
    dens_bound = mapping["rules"]["density_error_bound"]
    elev_step = mapping["rules"]["elevation_step"]
    dens_step = mapping["rules"]["density_step"]
    elev_threshold = float(raw["thresholds"]["elevation_pass"])

    region_members: dict[str, set[tuple[int, int]]] = {}
    for rid, shard in mapping["region_shards"].items():
        region_members[rid] = {tuple(e["cell"]) for e in shard["mappings"]}

    def blocked(cell: tuple[int, int]) -> bool:
        tile = tile_at[cell]
        # 连通性由量化高程与 elevation_pass 阈值决定（误差沿高程量化传播）；
        # 地形类别的通行/禁止语义在 forbidden_adjacent 条目独立判定
        return tile["elevation_q"] < elev_threshold

    # 1) 连通性 + 最小通道宽度比例
    for conn in raw["connectivity"]:
        cid = conn["id"]
        tol = float(conn.get("tolerance", 0.0))
        ratio_threshold = float(conn["min_channel_width_ratio"])
        members: set[tuple[int, int]] = set()
        for rid in conn["region_ids"]:
            members |= region_members[rid]
        start = min(members)
        seen = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cx + dx, cy + dy)
                if nb in members and nb not in seen and not blocked(nb):
                    seen.add(nb)
                    queue.append(nb)
        connected = members <= seen

        # 可通行边 / 区域并集内总相邻边
        total_edges = 0
        open_edges = 0
        min_margin = float("inf")
        margin_cell = None
        members_list = sorted(members)
        member_set = members
        for cell in members_list:
            cx, cy = cell
            for nb in ((cx + 1, cy), (cx, cy + 1)):
                if nb in member_set:
                    total_edges += 1
                    if not blocked(cell) and not blocked(nb):
                        open_edges += 1
            if not blocked(cell):
                margin = tile_at[cell]["elevation_q"] - elev_threshold
                if margin < min_margin:
                    min_margin, margin_cell = margin, cell
        observed_ratio = (open_edges / total_edges) if total_edges else 0.0
        # 比例由整数计数相除产生，误差只来自一次双精度除法
        ratio_arith_bound = ulp(1.0)
        chan_bound = elev_bound + ratio_arith_bound
        chan = _entry({
            "constraint_id": f"{cid}.channel_width_ratio",
            "kind": "channel_width_ratio",
            "observed": float(observed_ratio),
            "threshold": ratio_threshold,
            "direction": "ge",
            "comparison_tolerance": tol,
            "quant_error_bound": float(elev_bound),
            "normalization_error": 0.0,
            "arithmetic_error_bound": ratio_arith_bound,
            "propagated_error_bound": float(chan_bound),
            "truth_sources": [
                "mapping.region_shards.*.mappings[].elevation",
                "grid.tiles", "spec:connectivity.min_channel_width_ratio"],
            "evidence": {
                "open_edges": open_edges, "total_edges": total_edges,
                "connected_via_bfs": connected,
                "tightest_passable_cell": list(margin_cell) if margin_cell else None,
                "tightest_elevation_margin":
                    float(min_margin) if margin_cell is not None else None,
            },
        })
        entries.append(chan)

        # 连通性是离散结构量：BFS 基于已量化网格，观测值为确定的 0/1，
        # 误差带为 0（量化不确定性已在通道比例条目上承载）
        conn_entry = {
            "constraint_id": f"{cid}.connectivity",
            "kind": "connectivity",
            "observed": 1 if connected else 0,
            "threshold": 1,
            "direction": "ge",
            "comparison_tolerance": 0.0,
            "quant_error_bound": 0.0,
            "normalization_error": 0.0,
            "arithmetic_error_bound": 0.0,
            "propagated_error_bound": 0.0,
            "truth_sources": ["grid.tiles", "mapping.region_shards"],
            "evidence": {
                "reachable": sorted([list(c) for c in seen]),
                "required": sorted([list(c) for c in members]),
                "bfs_start": list(start),
            },
        }
        conn_entry["status"] = "PASS" if connected else "FAIL"
        conn_entry["total_band"] = 0.0
        conn_entry["on_exact_boundary"] = False
        conn_entry["plausible_low"] = float(conn_entry["observed"])
        conn_entry["plausible_high"] = float(conn_entry["observed"])
        entries.append(conn_entry)

    # 2) 资源数量区间（整数，精确判定）
    res_count = len(continuous["resources"])
    rmin = int(raw["resources"]["count"]["min"])
    rmax = int(raw["resources"]["count"]["max"])
    count_entry = {
        "constraint_id": "resources.count_interval",
        "kind": "resource_count_interval",
        "observed": int(res_count),
        "threshold": {"min": rmin, "max": rmax},
        "direction": "interval",
        "comparison_tolerance": 0.0,
        "quant_error_bound": 0.0,
        "normalization_error": 0.0,
        "arithmetic_error_bound": 0.0,
        "propagated_error_bound": 0.0,
        "total_band": 0.0,
        "plausible_low": res_count,
        "plausible_high": res_count,
        "on_exact_boundary": res_count in (rmin, rmax),
        "status": "PASS" if rmin <= res_count <= rmax else "FAIL",
        "truth_sources": ["continuous.resources", "spec:resources.count"],
        "evidence": {"resource_cells": grid["resource_cells"]},
    }
    entries.append(count_entry)

    # 3) 资源密度上限（资源数 / 区域并集单元数）
    total_cells = sum(len(s["mappings"]) for s in mapping["region_shards"].values())
    density = res_count / total_cells
    cap = float(raw["resources"]["density_cap"])
    cap_tol = float(raw["resources"].get("tolerance", 0.0))
    density_arith = ulp(1.0)
    entries.append(_entry({
        "constraint_id": "resources.density_cap",
        "kind": "resource_density_cap",
        "observed": float(density),
        "threshold": cap,
        "direction": "le",
        "comparison_tolerance": cap_tol,
        "quant_error_bound": 0.0,
        "normalization_error": 0.0,
        "arithmetic_error_bound": density_arith,
        "propagated_error_bound": density_arith,
        "truth_sources": ["continuous.resources", "grid.tiles",
                          "spec:resources.density_cap"],
        "evidence": {"resource_count": res_count, "region_cell_count": total_cells},
    }))

    # 4) 逐图块密度上限（量化密度 vs cell_density_max），承载量化边界判定
    cell_tol = float(raw["thresholds"]["cell_density_tolerance"])
    worst = None
    for rid in sorted(mapping["region_shards"]):
        for m in mapping["region_shards"][rid]["mappings"]:
            q = m["density"]["quantized"]
            if worst is None or q > worst["density"]["quantized"]:
                worst = m
    wq = worst["density"]["quantized"]
    cell_bound = dens_bound
    entries.append(_entry({
        "constraint_id": "thresholds.cell_density_max",
        "kind": "cell_density_max",
        "observed": float(wq),
        "threshold": float(raw["thresholds"]["cell_density_max"]),
        "direction": "le",
        "comparison_tolerance": cell_tol,
        "quant_error_bound": float(dens_bound),
        "normalization_error": 0.0,
        "arithmetic_error_bound": 0.0,
        "propagated_error_bound": float(cell_bound),
        "truth_sources": [
            "mapping.region_shards.*.mappings[].density",
            "spec:thresholds.cell_density_max"],
        "evidence": {
            "worst_cell": worst["cell"],
            "continuous": worst["density"]["continuous"],
            "quantized": worst["density"]["quantized"],
            "delta": worst["density"]["delta"],
            "density_step": dens_step,
        },
    }))

    # 5) 禁止相邻对（类别量，精确判定；地形抽样受归一化影响，登记归一化误差）
    norm_bound = float(continuous["weight_normalization"]["normalization_error_bound"])
    forbidden = [tuple(pair) for pair in
                 raw["terrain_rules"].get("forbidden_adjacent", [])]
    violations = []
    width, height = spec.width, spec.height
    for y in range(height):
        for x in range(width):
            tile = tile_at[(x, y)]
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    other = tile_at[(nx, ny)]
                    pair = (tile["terrain"], other["terrain"])
                    if pair in forbidden or tuple(reversed(pair)) in forbidden:
                        violations.append({"cells": [[x, y], [nx, ny]],
                                           "terrains": list(pair)})
    entries.append({
        "constraint_id": "terrain_rules.forbidden_adjacent",
        "kind": "forbidden_adjacent",
        "observed": len(violations),
        "threshold": 0,
        "direction": "eq",
        "comparison_tolerance": 0.0,
        "quant_error_bound": 0.0,
        "normalization_error": norm_bound,
        "arithmetic_error_bound": 0.0,
        # 类别量判定本身离散确定；归一化误差只作来源留痕，不进入比较带
        "propagated_error_bound": 0.0,
        "total_band": 0.0,
        "plausible_low": float(len(violations)),
        "plausible_high": float(len(violations)),
        "on_exact_boundary": len(violations) == 0,
        "status": "PASS" if not violations else "FAIL",
        "truth_sources": ["grid.tiles",
                          "continuous.weight_normalization",
                          "spec:terrain_rules.forbidden_adjacent"],
        "evidence": {"violations": violations,
                     "forbidden_pairs": [list(p) for p in forbidden]},
    })

    # 6) 地形权重归一化偏差（规格自检的运行时留痕）
    wsum = float(continuous["weight_normalization"]["weight_sum"])
    wtol = float(raw["terrain"]["weight_normalization_tolerance"])
    wdev = abs(wsum - 1.0)
    entries.append(_entry({
        "constraint_id": "terrain.weight_normalization",
        "kind": "weight_normalization",
        "observed": float(wdev),
        "threshold": wtol,
        "direction": "le",
        "comparison_tolerance": 0.0,
        "quant_error_bound": 0.0,
        "normalization_error": norm_bound,
        "arithmetic_error_bound": ulp(1.0),
        "propagated_error_bound": norm_bound + ulp(1.0),
        "truth_sources": ["continuous.weight_normalization",
                          "spec:terrain.weight_normalization_tolerance"],
        "evidence": {"weight_sum": wsum,
                     "sum_order": continuous["weight_normalization"]["sum_order"]},
    }))

    for e in entries:
        e.setdefault("quant_step_refs", {
            "elevation_step": elev_step, "density_step": dens_step})
    return entries
