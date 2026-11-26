"""栏内阅读路径串联、声明冲突裁决与跨栏栏序合并（并列全记录）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .errors import E_ADJUDICATION_BUDGET
from .hashing import cache_key, digest

INTRA_STAGE = "intra"
MERGE_STAGE = "merge"


@dataclass
class AdjudicationLedger:
    points_consumed: int = 0
    deductions: list[dict] = field(default_factory=list)
    pending_columns: list[int] = field(default_factory=list)


def assign_columns(regressions: list[dict], declaration: dict) -> dict[int, list[dict]]:
    bands = sorted(
        ((col["column_index"], col["x_interval"]) for col in declaration["columns"]),
        key=lambda item: item[1]["x_min"])
    grouped: dict[int, list[dict]] = {col: [] for col, _ in bands}
    for point in regressions:
        x = point["x"]
        for col_index, interval in bands:
            if interval["x_min"] <= x <= interval["x_max"]:
                grouped[col_index].append(point)
                break
    return grouped


def _path_cost(points: list[dict]) -> int:
    ordered = sorted(points, key=lambda item: item["y"])
    total = 0
    for prev, nxt in zip(ordered, ordered[1:]):
        total += abs(nxt["y"] - prev["y"]) + abs(nxt["x"] - prev["x"])
    return total


def build_intra_paths(regressions: list[dict], declaration: dict) -> dict:
    grouped = assign_columns(regressions, declaration)
    paths: dict[int, list[int]] = {}
    conflicts: list[dict] = []
    for col_index in sorted(grouped):
        points = sorted(grouped[col_index], key=lambda item: (item["y"], item["x"]))
        observed = [point["component_id"] for point in points]
        paths[col_index] = observed
    declared = sorted(declaration["candidate_paths"],
                      key=lambda item: (item["priority"], item["cost"]))
    observed_cost = sum(_path_cost(grouped[col]) for col in grouped)
    chosen_decl = declared[0]
    for candidate in declared[1:]:
        if candidate["path"] == chosen_decl["path"]:
            conflicts.append({
                "type": "duplicate_candidate_path",
                "declared": candidate["path"],
                "resolution": "keep_higher_priority",
            })
    expected_order = chosen_decl["path"]
    right_to_left = sorted(paths.keys())
    if expected_order != right_to_left:
        conflicts.append({
            "type": "declared_vs_right_to_left",
            "declared": expected_order,
            "observed_right_to_left": right_to_left,
            "resolution": "declared_priority",
            "declared_cost": chosen_decl["cost"],
            "observed_cost": observed_cost,
        })
    return {
        "paths": {str(col): paths[col] for col in sorted(paths)},
        "observed_column_order_right_to_left": right_to_left,
        "conflicts": conflicts,
        "chosen_declared_path": chosen_decl["path"],
        "fingerprint": digest([paths, conflicts]),
    }


def merge_columns(intra: dict, declaration: dict, cfg: Config,
                  ledger: AdjudicationLedger) -> dict:
    adjacency = {(edge["from"], edge["to"]) for edge in declaration["adjacency"]}
    adjacency |= {(edge["to"], edge["from"]) for edge in declaration["adjacency"]}
    right_to_left = sorted(int(col) for col in intra["paths"])
    declared_candidates = [
        tuple(path["path"]) for path in
        sorted(declaration["candidate_paths"], key=lambda item: (item["priority"], item["cost"]))]
    merged_order: list[int] = []
    tie_records: list[dict] = []
    pending: list[int] = []
    current = right_to_left[0]
    merged_order.append(current)
    remaining = right_to_left[1:]
    while remaining:
        feasible = [col for col in remaining if (current, col) in adjacency]
        if not feasible:
            feasible = list(remaining)
            tie_records.append({
                "type": "missing_adjacency_fallback",
                "from": current,
                "candidates": feasible,
            })
        # 声明存在多条候选路径时，声明层的备选也参与一次裁决点计费。
        declared_alive = [
            path for path in declared_candidates
            if current in path and any(col in path for col in feasible)]
        if len(declared_alive) > 1:
            cost = 1
            if ledger.points_consumed + cost > cfg.adjudication_budget:
                ledger.pending_columns = sorted(set(remaining))
                return {
                    "final_order": merged_order,
                    "tie_records": tie_records,
                    "pending_columns": ledger.pending_columns,
                    "business_code": E_ADJUDICATION_BUDGET,
                    "fingerprint": digest([merged_order, tie_records,
                                           ledger.pending_columns]),
                }
            ledger.points_consumed += cost
            ledger.deductions.append({
                "stage": MERGE_STAGE,
                "budget": "adjudication_point",
                "from": current,
                "candidate": "declared_path_tie",
                "cost": cost,
                "remaining": cfg.adjudication_budget - ledger.points_consumed,
            })
            tie_records.append({
                "type": "declared_candidate_tie",
                "from": current,
                "all_candidates": [list(path) for path in declared_alive],
                "chosen": list(declared_alive[0]),
                "suppressed": [
                    {"candidate": list(path),
                     "reason": "declared_priority_and_cost_tie_break",
                     "chosen": list(declared_alive[0]), "cost_paid": cost}
                    for path in declared_alive[1:]],
            })
        chosen = feasible[0]
        suppressed: list[dict] = []
        for candidate in feasible[1:]:
            cost = 1 + abs(candidate - chosen)
            if ledger.points_consumed + cost > cfg.adjudication_budget:
                pending.extend(c for c in feasible if c not in merged_order)
                ledger.pending_columns = sorted(set(pending))
                return {
                    "final_order": merged_order,
                    "tie_records": tie_records,
                    "pending_columns": ledger.pending_columns,
                    "business_code": E_ADJUDICATION_BUDGET,
                    "fingerprint": digest([merged_order, tie_records, ledger.pending_columns]),
                }
            ledger.points_consumed += cost
            ledger.deductions.append({
                "stage": MERGE_STAGE,
                "budget": "adjudication_point",
                "from": current,
                "candidate": candidate,
                "cost": cost,
                "remaining": cfg.adjudication_budget - ledger.points_consumed,
            })
            suppressed.append({
                "candidate": candidate,
                "reason": "stable_column_index_tie_break",
                "chosen": chosen,
                "cost_paid": cost,
            })
        if len(feasible) > 1:
            tie_records.append({
                "type": "parallel_feasible_tie",
                "from": current,
                "all_candidates": feasible,
                "chosen": chosen,
                "suppressed": suppressed,
            })
        current = chosen
        merged_order.append(current)
        remaining.remove(current)
    flat_reading_order: list[str] = []
    for col in merged_order:
        flat_reading_order.extend(intra["paths"][str(col)])
    return {
        "final_order": merged_order,
        "flat_reading_order": flat_reading_order,
        "tie_records": tie_records,
        "pending_columns": [],
        "business_code": None,
        "fingerprint": digest([merged_order, flat_reading_order, tie_records]),
    }


def intra_cache_key(sample, cfg: Config) -> str:
    return cache_key(
        INTRA_STAGE,
        ("fine", sample.artifacts["fine"]["cache_key"]),
        ("keypoint_version", sample.active_keypoint_fingerprint),
        ("quant", cfg.frac_bits),
    )


def merge_cache_key(sample, cfg: Config) -> str:
    return cache_key(
        MERGE_STAGE,
        ("intra", sample.artifacts["intra"]["cache_key"]),
        ("declaration", digest(sample.declaration)),
        ("adjudication_budget", cfg.adjudication_budget),
    )
