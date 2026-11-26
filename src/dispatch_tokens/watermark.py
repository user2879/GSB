"""定点水位、量化误差界与可区分性裁决。

所有比例都用整数乘除完成，截断方向为向零截断。两个量是否“可区分”只看
整数误差界，绝不做浮点比较；落在界内时按稳定拓扑序 + 分片标识序决胜，
并显式标注受误差影响。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fixedpoint import TRUNC_TOWARD_ZERO, trunc_div, validate_scale


@dataclass(frozen=True)
class WaterLevel:
    node_id: str
    balance: int
    cap: int
    rate: int
    scale: int
    ratio_scaled: int
    quantization_error: int
    truncation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "balance": self.balance,
            "cap": self.cap,
            "rate": self.rate,
            "scale": self.scale,
            "ratio_scaled": self.ratio_scaled,
            "quantization_error": self.quantization_error,
            "truncation": self.truncation,
        }


def node_water_level(node_id: str, balance: int, cap: int, rate: int,
                     scale: int) -> WaterLevel:
    """水位比例（以 scale 为满刻度整数基）：balance * scale // cap。"""
    validate_scale(scale)
    if cap <= 0:
        ratio = 0
        error = scale
    else:
        ratio = trunc_div(balance * scale, cap)
        # 截断误差上界（同量纲整数单位）：余数 < cap，映射到比例口径 < 1。
        error = 1
    return WaterLevel(
        node_id=node_id, balance=balance, cap=cap, rate=rate, scale=scale,
        ratio_scaled=ratio, quantization_error=error,
        truncation=TRUNC_TOWARD_ZERO,
    )


def global_quota_error_bound(scales: list[int]) -> int:
    """全局额度聚合误差上界：每节点至多 1 个比例单位，求和为节点数。

    额度本身按 scale 量化时单节点截断余数 < scale；跨节点聚合统一以
    “单位数”报告，上界 = 节点个数（每个节点贡献严格小于 1 个比例单位）。
    """
    if not scales:
        return 0
    return len(scales)


def comparison_error_bound(scale: int) -> int:
    """逐节点水位比较判定误差界：两次截断各贡献 < 1 单位，合界 < 2，取 2。"""
    validate_scale(scale)
    return 2


def demand_gap(balance: int, demand_scaled: int) -> int:
    """剩余额度与候选需求之差（整数，可为负）。"""
    return balance - demand_scaled


def choose_candidate(candidates: list[dict[str, Any]], balances: dict[str, int],
                     caps: dict[str, int], scale: int,
                     topo_index: dict[str, int]) -> dict[str, Any]:
    """在误差界内按显式决胜规则选一个候选分片放行。

    candidates 每项含 shard_id / node_id / demand_scaled / priority。
    决胜顺序：
      1. 需求差落在误差界外：需求差更大者优先（额度更宽裕）；
      2. 界内不可区分：稳定拓扑序（topo_index 小者优先）；
      3. 仍相同：分片标识字典序；
      4. 再相同：优先级数值小者优先。
    结论附带 ``error_influenced`` 标记。
    """
    if not candidates:
        raise ValueError("candidate set must be non-empty")
    bound = comparison_error_bound(scale)

    def sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        node_id = candidate["node_id"]
        gap = demand_gap(balances[node_id], candidate["demand_scaled"])
        # 第一主键：可负担（gap>=0）优先；同档内完全交给稳定序裁决。
        affordable_rank = 0 if gap >= 0 else 1
        return (
            affordable_rank,
            topo_index[candidate["shard_id"]],
            candidate["shard_id"],
            candidate["priority"],
            -gap,
        )

    ordered = sorted(candidates, key=sort_key)
    winner = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    error_influenced = False
    if runner_up is not None:
        winner_gap = demand_gap(balances[winner["node_id"]],
                                winner["demand_scaled"])
        runner_gap = demand_gap(balances[runner_up["node_id"]],
                                runner_up["demand_scaled"])
        if abs(winner_gap - runner_gap) <= bound:
            error_influenced = True
    if abs(demand_gap(balances[winner["node_id"]],
                      winner["demand_scaled"])) <= bound:
        error_influenced = True
    return {
        "winner_shard_id": winner["shard_id"],
        "winner_node_id": winner["node_id"],
        "error_influenced": error_influenced,
        "comparison_error_bound": bound,
        "truncation": TRUNC_TOWARD_ZERO,
        "ordered_shard_ids": [item["shard_id"] for item in ordered],
    }
