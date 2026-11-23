"""修复顺序判定的不可区分裁决（需求 5）。

紧迫度合成（整数定点）：
  urgency = fp.mul(quantize_ratio(mismatch_permille, 1000), quantize_units(weight))
  即失配程度（千分比定点化）× 节点优先级权重（定点化），全部整数运算。

合成误差界：
  失配千分比量化截断误差 < 1 量子，权重为整数量化（无误差），
  故单个候选紧迫度误差界 err(u) = weight_quanta（量子）；
  两个候选的合成误差界 = err(u1) + err(u2)。
  字节消耗比较同理：err(c) = 该方案各批次量化截断误差界之和。

三类结论：
  EXACT_SEPARABLE        确切可分：定点差 > 合成误差界；
  ERROR_INDISTINGUISHABLE 误差不可区分：定点差 <= 合成误差界，按稳定决胜规则
                         （块标识序、再节点标识序）选定，并标注受误差影响；
  INSUFFICIENT_EVIDENCE  证据不足：任一候选失配程度缺失。

排序与节点处理顺序无关：全序键 = (-urgency_quanta, block_id, node_id)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .fixedpoint import FixedPoint
from .spec import BlockSpec, NodeSpec

EXACT_SEPARABLE = "EXACT_SEPARABLE"
ERROR_INDISTINGUISHABLE = "ERROR_INDISTINGUISHABLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class Candidate:
    block: BlockSpec
    node: NodeSpec


@dataclass(frozen=True)
class RankedCandidate:
    block_id: str
    node_id: str
    urgency_quanta: int | None       # None = 证据不足
    urgency_err_quanta: int          # 单侧误差界
    cost_quanta: int                 # 定点字节消耗（块字节数量化）
    cost_err_quanta: int             # 消耗误差界


class UrgencyRanker:
    def __init__(self, fp: FixedPoint):
        self._fp = fp

    def evaluate(self, cand: Candidate) -> RankedCandidate:
        fp = self._fp
        weight_q = fp.quantize_units(cand.node.priority_weight)
        cost_q = fp.quantize_bytes(cand.block.byte_count)
        # 消耗误差界：块字节量化截断 < 1 量子，取保守界 1。
        cost_err = 1
        if cand.block.mismatch_permille is None:
            return RankedCandidate(
                cand.block.block_id, cand.node.node_id, None, 0, cost_q, cost_err
            )
        mismatch_q = fp.quantize_ratio(cand.block.mismatch_permille, 1000)
        urgency = fp.mul(mismatch_q, weight_q)
        # err(u) <= err(mismatch_q) * weight_quanta / scale = 1 量子 * weight（定点乘法截断再 +1）
        urgency_err = cand.node.priority_weight + 1
        return RankedCandidate(
            cand.block.block_id, cand.node.node_id, urgency, urgency_err, cost_q, cost_err
        )

    def rank(self, candidates: list) -> list:
        """稳定排序：证据不足者排尾；其余按 (-urgency, block_id, node_id) 全序。"""
        ranked = [self.evaluate(c) for c in candidates]

        def key(item: RankedCandidate):
            if item.urgency_quanta is None:
                return (1, 0, item.block_id, item.node_id)
            return (0, -item.urgency_quanta, item.block_id, item.node_id)

        return sorted(ranked, key=key)

    def compare(self, a: RankedCandidate, b: RankedCandidate) -> dict:
        """两候选紧迫度裁决。"""
        if a.urgency_quanta is None or b.urgency_quanta is None:
            return {
                "verdict": INSUFFICIENT_EVIDENCE,
                "reason": "失配程度缺失，证据不足",
                "a": a.block_id,
                "b": b.block_id,
            }
        diff = abs(a.urgency_quanta - b.urgency_quanta)
        bound = a.urgency_err_quanta + b.urgency_err_quanta
        if diff > bound:
            winner = a if a.urgency_quanta > b.urgency_quanta else b
            return {
                "verdict": EXACT_SEPARABLE,
                "diff_quanta": diff,
                "bound_quanta": bound,
                "winner": winner.block_id,
                "error_affected": False,
            }
        # 误差不可区分：稳定决胜（块标识序、再节点标识序）。
        if a.block_id != b.block_id:
            winner = a if a.block_id < b.block_id else b
        else:
            winner = a if a.node_id <= b.node_id else b
        return {
            "verdict": ERROR_INDISTINGUISHABLE,
            "diff_quanta": diff,
            "bound_quanta": bound,
            "winner": winner.block_id,
            "tiebreak": "block_id,node_id",
            "error_affected": True,
        }

    def compare_cost(self, a: RankedCandidate, b: RankedCandidate) -> dict:
        """两候选修复方案定点字节消耗裁决。"""
        diff = abs(a.cost_quanta - b.cost_quanta)
        bound = a.cost_err_quanta + b.cost_err_quanta
        if diff > bound:
            cheaper = a if a.cost_quanta < b.cost_quanta else b
            return {
                "verdict": EXACT_SEPARABLE,
                "diff_quanta": diff,
                "bound_quanta": bound,
                "cheaper": cheaper.block_id,
                "error_affected": False,
            }
        if a.block_id != b.block_id:
            cheaper = a if a.block_id < b.block_id else b
        else:
            cheaper = a if a.node_id <= b.node_id else b
        return {
            "verdict": ERROR_INDISTINGUISHABLE,
            "diff_quanta": diff,
            "bound_quanta": bound,
            "cheaper": cheaper.block_id,
            "tiebreak": "block_id,node_id",
            "error_affected": True,
        }

    def adjudicate_all(self, candidates: list) -> dict:
        """整体裁决：排序 + 相邻对比较，输出三类结论清单（顺序无关）。"""
        ranked = self.rank(candidates)
        pair_verdicts = []
        for i in range(len(ranked) - 1):
            a, b = ranked[i], ranked[i + 1]
            if a.urgency_quanta is None or b.urgency_quanta is None:
                continue
            pair_verdicts.append(self.compare(a, b))
        order = [
            {
                "block_id": r.block_id,
                "node_id": r.node_id,
                "urgency_quanta": r.urgency_quanta,
                "cost_quanta": r.cost_quanta,
            }
            for r in ranked
        ]
        classes = {
            EXACT_SEPARABLE: sum(1 for v in pair_verdicts if v["verdict"] == EXACT_SEPARABLE),
            ERROR_INDISTINGUISHABLE: sum(
                1 for v in pair_verdicts if v["verdict"] == ERROR_INDISTINGUISHABLE
            ),
            INSUFFICIENT_EVIDENCE: sum(
                1 for r in ranked if r.urgency_quanta is None
            ),
        }
        return {"order": order, "pairs": pair_verdicts, "classes": classes}
