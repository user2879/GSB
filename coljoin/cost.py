"""整数定点代价模型与稳定裁决。

量化口径：
- 行计数类指标先乘以会话 ``scale`` 转成定点整数；
- 权重是无量纲整数，与定点指标相乘后除以 ``scale`` 反量化，再饱和到 63 位；
- 所有除法为 floor 截断，饱和上限为 :data:`coljoin.fixedpoint.SAT_MAX`；
- 每次 floor 除法向误差预算登记 1 个截断单位，合成对称误差区间。

裁决口径：
- 总代价是定点整数，胜负不使用任何浮点比较；
- 当差值绝对值落在两计划合成误差区间内时进入显式决胜：
  plan_id 字典序 -> 算子序列摘要字典序，并标注 error_influenced。
"""

from typing import Any, Dict, List, Tuple

from .fixedpoint import ErrorBudget, SAT_MAX, add, ceil_div, mul_fixed, quantize, sat
from .model import BlockGroupSpec, CostWeights, SessionSpec
from .statistics import (
    apply_filters,
    effective_rows,
    join_cardinality,
    ndv_of,
    stats_digest,
)


def _scaled(value: int, scale: int) -> int:
    return quantize(value, scale)


def _weighted(metric: int, weight: int, scale: int) -> int:
    return mul_fixed(metric, weight, scale)


class CostEstimator:
    def __init__(
        self,
        groups: List[BlockGroupSpec],
        weights: CostWeights,
        session: SessionSpec,
    ):
        self.groups = {group.bg_id: group for group in groups}
        self.weights = weights
        self.session = session
        self.scale = session.scale
        self.stats_version = stats_digest(groups)

    def _scan_rows(self, bg_id: str, filters) -> Tuple[int, int, int]:
        group = self.groups[bg_id]
        raw_rows = effective_rows(group)
        after = apply_filters(raw_rows, filters)
        return raw_rows, after, len(group.columns)

    def _scan_cost(self, bg_id: str, filters, budget: ErrorBudget) -> Dict[str, Any]:
        raw_rows, after_rows, col_count = self._scan_rows(bg_id, filters)
        blocks = ceil_div(raw_rows, self.session.block_granularity)
        budget.record_div("scan", "block_ceiling")
        access_q = _scaled(raw_rows * max(1, col_count), self.scale)
        block_q = _scaled(blocks, self.scale)
        scan_cost = add(
            _weighted(access_q, self.weights.scan, self.scale),
            _weighted(block_q, self.weights.scan, self.scale),
        )
        return {
            "op": "scan",
            "bg": bg_id,
            "raw_rows": raw_rows,
            "output_rows": after_rows,
            "blocks": blocks,
            "access_rows": raw_rows * max(1, col_count),
            "cost": scan_cost,
        }

    def _join_cost(
        self,
        step: Dict[str, Any],
        left_rows: int,
        right_rows: int,
        budget: ErrorBudget,
    ) -> Dict[str, Any]:
        algorithm = step["algorithm"]
        new_bg = step["new_bg"]
        predicates = step["predicates"]
        right_group = self.groups[new_bg]
        left_ndv = 1
        right_ndv = 1
        for pred in predicates:
            if pred["right_bg"] == new_bg:
                left_ndv = max(left_ndv, ndv_of(self.groups[pred["left_bg"]], pred["left_col"]))
                right_ndv = max(right_ndv, ndv_of(right_group, pred["right_col"]))
            else:
                left_ndv = max(left_ndv, ndv_of(self.groups[pred["right_bg"]], pred["right_col"]))
                right_ndv = max(right_ndv, ndv_of(right_group, pred["left_col"]))
        output_rows = join_cardinality(left_rows, right_rows, left_ndv, right_ndv)
        left_q = _scaled(left_rows, self.scale)
        right_q = _scaled(right_rows, self.scale)
        output_q = _scaled(output_rows, self.scale)
        if algorithm == "hash":
            build_rows = min(left_rows, right_rows)
            build_q = _scaled(build_rows, self.scale)
            compare_rows = max(left_rows, right_rows)
            compare_q = _scaled(compare_rows, self.scale)
            budget.record_div("hash_join", "probe_compare")
            memory_q = build_q
            cost = add(
                add(
                    _weighted(build_q, self.weights.build, self.scale),
                    _weighted(compare_q, self.weights.probe, self.scale),
                ),
                _weighted(memory_q, self.weights.memory, self.scale),
            )
            compares = compare_rows
        elif algorithm == "sort_merge":
            sort_left = left_rows * max(1, ceil_div(left_rows.bit_length(), 1))
            sort_right = right_rows * max(1, ceil_div(right_rows.bit_length(), 1))
            compares = sort_left + sort_right + left_rows + right_rows
            budget.record_div("sort_merge", "merge_compare")
            compare_q = _scaled(compares, self.scale)
            memory_q = add(left_q, right_q)
            cost = add(
                _weighted(compare_q, self.weights.compare, self.scale),
                _weighted(memory_q, self.weights.memory, self.scale),
            )
        else:
            compares = left_rows * right_rows
            if compares > SAT_MAX:
                compares = SAT_MAX
            compare_q = _scaled(sat(compares), self.scale)
            budget.record_div("nested_loop", "pair_compare")
            memory_q = _scaled(1, self.scale)
            cost = add(
                _weighted(compare_q, self.weights.compare, self.scale),
                _weighted(memory_q, self.weights.memory, self.scale),
            )
        peak_memory_q = output_q if algorithm == "hash" else (
            add(left_q, right_q) if algorithm == "sort_merge" else self.scale
        )
        return {
            "op": "join",
            "algorithm": algorithm,
            "new_bg": new_bg,
            "left_rows": left_rows,
            "right_rows": right_rows,
            "left_ndv": left_ndv,
            "right_ndv": right_ndv,
            "output_rows": output_rows,
            "compares": sat(compares),
            "peak_memory": peak_memory_q,
            "cost": sat(cost),
        }

    def _aggregate_cost(self, input_rows: int, budget: ErrorBudget) -> Dict[str, Any]:
        hash_groups = max(1, min(input_rows, 1024))
        budget.record_div("aggregate", "hash_group")
        input_q = _scaled(input_rows, self.scale)
        memory_q = _scaled(hash_groups, self.scale)
        cost = add(
            _weighted(input_q, self.weights.build, self.scale),
            _weighted(memory_q, self.weights.memory, self.scale),
        )
        return {
            "op": "aggregate",
            "input_rows": input_rows,
            "hash_groups": hash_groups,
            "peak_memory": memory_q,
            "cost": cost,
        }

    def evaluate(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        budget = ErrorBudget(self.scale)
        step_results: List[Dict[str, Any]] = []
        current_rows: Dict[str, int] = {}
        running_rows = 0
        total_cost = 0
        total_access = 0
        total_compares = 0
        peak_memory = 0
        for step in plan["body"]["steps"]:
            if step["op"] == "scan":
                result = self._scan_cost(step["bg"], step["filters"], budget)
                current_rows[step["bg"]] = result["output_rows"]
                if not step_results:
                    running_rows = result["output_rows"]
                total_access += result["access_rows"]
            elif step["op"] == "join":
                right_rows = current_rows[step["new_bg"]]
                result = self._join_cost(step, running_rows, right_rows, budget)
                running_rows = result["output_rows"]
                total_compares = add(total_compares, result["compares"])
                peak_memory = max(peak_memory, result["peak_memory"])
            else:
                result = self._aggregate_cost(running_rows, budget)
                peak_memory = max(peak_memory, result["peak_memory"])
            total_cost = add(total_cost, result["cost"])
            step_results.append(result)
        vector = {
            "total_cost": total_cost,
            "access_rows": sat(total_access),
            "compares": sat(total_compares),
            "peak_memory": peak_memory,
        }
        return {
            "plan_id": plan["plan_id"],
            "stats_version": self.stats_version,
            "session_stats_version": self.session.stats_version,
            "scale": self.scale,
            "saturation_limit": SAT_MAX,
            "truncation": "floor",
            "vector": vector,
            "error_interval": budget.interval(),
            "error_sources": budget.sources,
            "steps": step_results,
            "op_digest": plan["plan_id"],
        }


def choose_plan(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按总代价升序、plan_id 序裁决，并检测误差不可区分情形。"""
    ranked = sorted(
        evaluations,
        key=lambda item: (item["vector"]["total_cost"], item["plan_id"]),
    )
    chosen = ranked[0]
    ties: List[Dict[str, Any]] = []
    for candidate in ranked[1:]:
        delta = abs(
            candidate["vector"]["total_cost"] - chosen["vector"]["total_cost"]
        )
        tolerance = (
            chosen["error_interval"]["epsilon"]
            + candidate["error_interval"]["epsilon"]
        )
        if delta <= tolerance:
            ties.append(
                {
                    "plan_id": candidate["plan_id"],
                    "winner_id": chosen["plan_id"],
                    "delta": delta,
                    "tolerance": tolerance,
                    "winner_vector": chosen["vector"],
                    "candidate_vector": candidate["vector"],
                    "resolution": ["plan_id_order", "op_digest_order"],
                }
            )
    return {
        "chosen_plan_id": chosen["plan_id"],
        "error_influenced": bool(ties),
        "ties": ties,
        "ranking": [
            {
                "plan_id": item["plan_id"],
                "total_cost": item["vector"]["total_cost"],
                "error_epsilon": item["error_interval"]["epsilon"],
            }
            for item in ranked
        ],
    }
