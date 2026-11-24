"""批次资源视图：整数定点乘加量、利用率与参数版本失效传播。

所有指标只使用整数（含 numpy 整数向量），定点比例保存在
    :data:`~receipt_ledger.util.FIXED_SCALE`。
numpy 仅做确定性的整数向量化聚合，结果转回 Python int，无浮点参与。
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from .models import BatchResource, LedgerEntry, Receipt, ReuseRecord, Worker
from .util import FIXED_SCALE, fixed_div

QUANT_FACTOR_FIXED = {
    "int8": FIXED_SCALE,
    "int4": FIXED_SCALE // 2,
    "fp16": 2 * FIXED_SCALE,
}


def build_batch_resources(
    receipts: list[Receipt],
    workers: tuple[Worker, ...],
    unknown_versions: set[tuple[str, int]],
    request_tokens: dict[str, int],
) -> list[BatchResource]:
    worker_map = {worker.worker_id: worker for worker in workers}
    batches: dict[tuple[str, str], dict] = {}
    for receipt in receipts:
        if receipt.worker_id is None or receipt.duration_ms is None:
            continue
        key = (receipt.batch_id, receipt.worker_id)
        bucket = batches.setdefault(
            key,
            {
                "param_version": receipt.param_version,
                "durations": [],
                "tokens": [],
                "request_ids": [],
            },
        )
        if bucket["param_version"] is None and receipt.param_version is not None:
            bucket["param_version"] = receipt.param_version
        bucket["durations"].append(int(receipt.duration_ms))
        bucket["tokens"].append(int(request_tokens.get(receipt.request_id, 1)))
        bucket["request_ids"].append(receipt.request_id)

    results: list[BatchResource] = []
    for (batch_id, worker_id), bucket in sorted(batches.items()):
        worker = worker_map.get(worker_id)
        if worker is None:
            continue
        param_version = bucket["param_version"]
        durations = np.asarray(bucket["durations"], dtype=np.int64)
        tokens = np.asarray(bucket["tokens"], dtype=np.int64)
        request_count = int(durations.shape[0])
        total_duration = int(durations.sum())
        max_duration = int(durations.max())
        dims = np.int64(worker.model_dim) * np.int64(worker.hidden_dim)
        macs = int(
            tokens.sum()
            * dims
            * 2
            * QUANT_FACTOR_FIXED[worker.quant]
            // FIXED_SCALE
        )
        capacity_token_slots = int(tokens.max()) * request_count
        capacity_macs = (
            capacity_token_slots
            * worker.model_dim
            * worker.hidden_dim
            * 2
            * QUANT_FACTOR_FIXED[worker.quant]
            // FIXED_SCALE
        )
        utilization = fixed_div(macs, capacity_macs) if capacity_macs else 0
        macs_per_ms = fixed_div(macs, max(1, total_duration))

        invalid = False
        reason = None
        path: tuple[str, ...] = ()
        if (worker_id, param_version) in unknown_versions:
            invalid = True
            reason = f"未知参数版本 v{param_version}（工作器 {worker_id} 未声明）"
            path = (
                f"worker:{worker_id}",
                f"unknown_param_version:v{param_version}",
                f"batch:{batch_id}",
            )
        elif param_version is None:
            invalid = True
            reason = "回执缺失参数版本，无法判定依赖"
            path = (f"worker:{worker_id}", f"batch:{batch_id}")
        elif param_version != worker.param_version:
            invalid = True
            reason = (
                f"参数版本过期: 回执 v{param_version} -> 工作器当前 v{worker.param_version}"
            )
            path = (
                f"worker:{worker_id}",
                f"param_version:v{param_version}->v{worker.param_version}",
                f"batch:{batch_id}",
            )

        results.append(
            BatchResource(
                batch_id=batch_id,
                worker_id=worker_id,
                param_version=param_version if param_version is not None else -1,
                request_count=request_count,
                total_duration_ms=total_duration,
                max_duration_ms=max_duration,
                macs_fixed=macs,
                utilization_fixed=utilization,
                macs_per_ms_fixed=macs_per_ms,
                quant=worker.quant,
                invalid=invalid,
                invalid_reason=reason,
                propagation_path=path,
            )
        )
    return results


def propagate_invalidation(
    entries: list[LedgerEntry],
    resources: list[BatchResource],
) -> tuple[list[LedgerEntry], list[ReuseRecord]]:
    """沿 工作器参数版本 -> 批次 -> 请求归属 链传播失效。"""

    invalid_batches = {
        (resource.batch_id, resource.worker_id): resource for resource in resources if resource.invalid
    }
    updated: list[LedgerEntry] = []
    reuse: list[ReuseRecord] = []
    for entry in entries:
        key = (entry.batch_id, entry.worker_id)
        resource = invalid_batches.get(key)
        if resource is not None:
            path = resource.propagation_path + (
                f"request:{entry.request_id}",
                "attribution",
            )
            updated.append(
                replace(
                    entry,
                    invalid=True,
                    invalid_reason=resource.invalid_reason,
                    propagation_path=path,
                )
            )
            reuse.append(
                ReuseRecord(
                    stage="invalidation",
                    key=f"{entry.batch_id}/{entry.request_id}",
                    action="invalidate",
                    reason=resource.invalid_reason or "批次失效",
                    propagation_path=path,
                )
            )
        else:
            updated.append(entry)
            reuse.append(
                ReuseRecord(
                    stage="invalidation",
                    key=f"{entry.batch_id}/{entry.request_id}",
                    action="reuse",
                    reason="参数版本未变化，复用持久化归属",
                )
            )
    return updated, reuse
