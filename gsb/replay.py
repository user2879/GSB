"""恢复后的一致性重放校验。

按账本重放分流、隔离与归并，逐项与账本比较；
首个不一致以 E_REPLAY 定位到批次、请求与字段。
"""

from __future__ import annotations

import json

from .errors import e_replay
from .merge import MergeCandidate, merge_receipts
from .receipts import NormalizedReceipt
from .routing import bucket_for
from .spec import Spec


def _check(condition: bool, message: str, *, batch_id: str, request_id: str, field: str):
    if not condition:
        raise e_replay(message, batch_id=batch_id, request_id=request_id, field=field)


def verify_replay(ledger, spec: Spec) -> int:
    """逐项重放已提交批次，返回校验通过的批次数。"""
    rules = spec.rules
    verified = 0
    for batch_id in ledger.committed_batches():
        # 1) 重放分流：逐请求重算分桶与去向
        for hit in ledger.hits_for(batch_id):
            rid = hit["request_id"]
            rule = rules.get(hit["rule_id"])
            _check(
                rule is not None,
                "账本引用了未知规则",
                batch_id=batch_id,
                request_id=rid,
                field="rule_id",
            )
            expected_bucket = bucket_for(hit["rule_id"], hit["client_key"])
            _check(
                expected_bucket == hit["bucket"],
                f"分桶值不一致: 重放 {expected_bucket} != 账本 {hit['bucket']}",
                batch_id=batch_id,
                request_id=rid,
                field="bucket",
            )
            _check(
                rule.weight == hit["weight"],
                "权重与当前规则不一致",
                batch_id=batch_id,
                request_id=rid,
                field="weight",
            )
            expected_route = "gray" if expected_bucket < rule.weight else "stable"
            _check(
                expected_route == hit["route"],
                f"去向不一致: 重放 {expected_route} != 账本 {hit['route']}",
                batch_id=batch_id,
                request_id=rid,
                field="route",
            )
            expected_backend = (
                rule.gray_backend if expected_route == "gray" else rule.stable_backend
            )
            _check(
                expected_backend == hit["backend_id"],
                "后端绑定不一致",
                batch_id=batch_id,
                request_id=rid,
                field="backend_id",
            )
        # 2) 重放隔离：同一批次同一后端不得混装两个参数版本
        versions_per_backend: dict[str, set] = {}
        for hit in ledger.hits_for(batch_id):
            versions_per_backend.setdefault(hit["backend_id"], set()).add(
                hit["param_version"]
            )
        for backend_id in sorted(versions_per_backend):
            versions = versions_per_backend[backend_id]
            _check(
                len(versions) <= 1,
                f"批次在后端 {backend_id} 混装多个参数版本: {sorted(versions)}",
                batch_id=batch_id,
                request_id="-",
                field="param_version",
            )
        # 3) 重放归并：由账本内归一化回执重算结论并逐项比较
        rows = ledger.receipts_for(batch_id)
        candidates = [
            MergeCandidate(
                receipt=NormalizedReceipt.from_dict(json.loads(row["normalized_json"])),
                merge_seq=row["merge_seq"],
            )
            for row in rows
        ]
        recomputed = merge_receipts(candidates).to_dict()
        stored = ledger.conclusion_for(batch_id)
        _check(stored is not None, "缺少已提交结论", batch_id=batch_id, request_id="-", field="conclusion")
        for key in sorted(recomputed):
            _check(
                recomputed[key] == stored.get(key),
                f"结论字段不一致: 重放 {recomputed[key]!r} != 账本 {stored.get(key)!r}",
                batch_id=batch_id,
                request_id="-",
                field=f"conclusion.{key}",
            )
        verified += 1
    return verified
