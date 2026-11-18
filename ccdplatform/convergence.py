"""回执收敛与派生状态一致性判定。"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from .dispatcher import DispatchOutcome
from .errors import ConvergenceError
from .fixedpoint import trunc_zero
from .graph import DependencyGraph
from .model import CompiledItem, DispatchPlan


def expected_digest(
    node_id: str,
    version: int,
    keys: List[str],
    compiled: Dict[str, CompiledItem],
    graph: DependencyGraph,
    scale: int,
) -> str:
    """按版本与派生关系重算节点期望摘要（与 node_simulator 同口径）。"""
    # 节点摘要覆盖其全部目标键（含在多批中重复出现的键），
    # 与 node_simulator 跨批累积的 qmap 口径一致。
    key_set = set(keys)
    order = [k for k in graph.topological_order() if k in key_set]
    qmap: Dict[str, int] = {}
    for key in order:
        item = compiled[key]
        if not item.is_derived:
            qmap[key] = item.quantized
            continue
        total = 0
        for ref_key, coef_q in item.coef_quantized:
            # 与节点口径一致：未下发到本节点的上游不参与该节点的摘要计算
            if ref_key not in key_set:
                continue
            total += coef_q * qmap[ref_key]
        # 同尺度乘加后除以 scale 回到同一格度，向零截断与节点侧完全一致
        qmap[key] = trunc_zero(total, scale)
    material = json.dumps(
        {"node": node_id, "version": version, "q": qmap},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(material).hexdigest()


def check_convergence(
    plan: DispatchPlan,
    compiled: Dict[str, CompiledItem],
    graph: DependencyGraph,
    outcome: DispatchOutcome,
    scale: int,
) -> Dict[str, Dict[str, Any]]:
    """逐节点比对期望摘要与最终回执摘要。

    不一致或未回执统一抛 E_CONVERGE，稳定排序列出偏差节点、期望/实际摘要、
    判定所用回执序号。
    """
    report: Dict[str, Dict[str, Any]] = {}
    deviations: List[Dict[str, Any]] = []
    for node_id in sorted(plan.node_batches):
        keys = [k for batch in plan.node_batches[node_id] for k in batch]
        expect = expected_digest(
            node_id, plan.version, keys, compiled, graph, scale
        )
        receipt = outcome.receipts.get(node_id)
        actual = receipt.digest if receipt is not None else None
        seq = receipt.seq if receipt is not None else None
        attempt = receipt.attempt if receipt is not None else None
        entry = {
            "node": node_id,
            "expected_digest": expect,
            "actual_digest": actual,
            "receipt_seq": seq,
            "receipt_attempt": attempt,
            "matched": actual == expect,
        }
        report[node_id] = entry
        if not entry["matched"]:
            deviations.append(entry)
    if deviations:
        raise ConvergenceError(
            f"{len(deviations)} 个节点回执未收敛",
            {
                "version": plan.version,
                "deviations": deviations,
                "node_failures": outcome.node_failures,
            },
        )
    return report
