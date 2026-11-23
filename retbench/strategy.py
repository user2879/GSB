"""数据规模自适应策略选择器。

输入规模指标（确定性可审计）：
- n_samples          当前规模档位下参与计算的样本数；
- n_primitives       这些样本的图元总数；
- avg_assertion_len  文本断言平均长度（含补齐前的真实长度）；
- unit_cost          当前档位单样本估计代价。

预置策略表中的每条策略带四项容量阈值与单位代价上限，选择器对每条策略
按"利用率余量 + 单位代价余量"评分（数值完全确定），取可行策略中评分最高者，
同分按策略 id 决断。选择结果直接决定后续计算路径：
- feature_precision=reduced 丢弃边缘密度统计；
- tile_range=center_half 仅对中心半区参与对齐；
- channels 改变实际参与融合的召回通道；
- cap_discount 改变实际候选上限。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config, KNOWN_CHANNELS
from .errors import BenchError, E_CONFIG_INVALID

REASON_CAPACITY = "R_CAPACITY_MARGIN"
REASON_COST = "R_UNIT_COST_HEADROOM"
REASON_TIE = "R_TIE_BREAK_ID"
REASON_SOLE = "R_SOLE_FEASIBLE"


@dataclass
class StrategyDecision:
    strategy_id: str
    metrics: dict[str, float]
    scores: list[dict[str, Any]]
    reason_codes: list[str]
    feature_precision: str
    tile_range: str
    channels: list[str]
    candidate_cap: int
    effective_tier: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "effective_tier": self.effective_tier,
            "metrics": self.metrics,
            "scores": self.scores,
            "reason_codes": self.reason_codes,
            "feature_precision": self.feature_precision,
            "tile_range": self.tile_range,
            "channels": list(self.channels),
            "candidate_cap": self.candidate_cap,
        }


def compute_metrics(active_samples: list, effective_tier: int) -> dict[str, float]:
    n_samples = len(active_samples)
    n_primitives = sum(sample.primitive_count() for sample in active_samples)
    if n_samples:
        avg_assertion_len = sum(sample.assertion_len() for sample in active_samples) / n_samples
    else:
        avg_assertion_len = 0.0
    # 单样本估计代价：图元数 * 断言长度 * 100（确定性标量公式，写进决策记录）
    unit_cost = int(round((n_primitives / n_samples) * (avg_assertion_len + 1.0) * 100.0)) \
        if n_samples else 0
    return {
        "n_samples": float(n_samples),
        "n_primitives": float(n_primitives),
        "avg_assertion_len": round(avg_assertion_len, 6),
        "unit_cost": float(unit_cost),
        "effective_tier": float(effective_tier),
    }


def select_strategy(config: Config, active_samples: list,
                    effective_tier: int) -> StrategyDecision:
    metrics = compute_metrics(active_samples, effective_tier)
    scored: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []

    for entry in config.strategy_table:
        sample_margin = (entry.capacity_samples - metrics["n_samples"]) / entry.capacity_samples
        primitive_margin = (entry.capacity_primitives - metrics["n_primitives"]) / entry.capacity_primitives
        assertion_margin = (entry.capacity_assertion_len - metrics["avg_assertion_len"]) \
            / entry.capacity_assertion_len
        cost_margin = (entry.unit_cost_limit - metrics["unit_cost"]) / entry.unit_cost_limit
        feasible_flag = (
            metrics["n_samples"] <= entry.capacity_samples
            and metrics["n_primitives"] <= entry.capacity_primitives
            and metrics["avg_assertion_len"] <= entry.capacity_assertion_len
            and metrics["unit_cost"] <= entry.unit_cost_limit
        )
        # 可行策略中优先选择"最贴合当前规模"者：容量/代价余量越小得分越高；
        # 基础偏移 1000 保证所有可行策略排序时余量项主导，确定性可复现。
        capacity_margin = round(
            0.5 * sample_margin + 0.3 * primitive_margin + 0.2 * assertion_margin, 9
        )
        capacity_score = round(1000.0 - capacity_margin, 9)
        cost_score = round(100.0 - cost_margin, 9)
        total = round(capacity_score + cost_score, 9)
        row = {
            "strategy_id": entry.id,
            "feasible": feasible_flag,
            "capacity_score": capacity_score,
            "cost_score": cost_score,
            "total_score": total,
        }
        scored.append(row)
        if feasible_flag:
            feasible.append(row)

    if not feasible:
        raise BenchError(E_CONFIG_INVALID, "预置策略表中无可行策略",
                         {"metrics": metrics})

    best_total = max(row["total_score"] for row in feasible)
    tied = sorted((row for row in feasible if row["total_score"] == best_total),
                  key=lambda row: row["strategy_id"])
    chosen_id = tied[0]["strategy_id"]
    chosen = next(entry for entry in config.strategy_table if entry.id == chosen_id)

    reason_codes = [REASON_CAPACITY, REASON_COST]
    if len(feasible) == 1:
        reason_codes = [REASON_SOLE]
    elif len(tied) > 1:
        reason_codes.append(REASON_TIE)

    enabled = [channel for channel in KNOWN_CHANNELS
               if config.channels_enabled[channel] and channel in chosen.channels]
    if not enabled:
        enabled = [chosen.channels[0]]
    candidate_cap = max(1, int(round(config.candidate_cap * chosen.cap_discount)))

    return StrategyDecision(
        strategy_id=chosen.id,
        metrics=metrics,
        scores=scored,
        reason_codes=reason_codes,
        feature_precision=chosen.feature_precision,
        tile_range=chosen.tile_range,
        channels=enabled,
        candidate_cap=min(candidate_cap, len(active_samples)) if active_samples else candidate_cap,
        effective_tier=effective_tier,
    )
