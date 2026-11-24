"""界面快速视图与权威重算的分离判定。

* ``preview``：一次 SQLite 读取返回缓存结论，不触发任何推理或重算；
* ``recompute``：按账本逐批重放规划、分发、归并与 NumPy 推理，重算统计；
* 两者逐字节不一致时抛 E_PREVIEW，报告首个偏差批次；
* 单调时钟耗时按整数定点（均值 x1000）聚合。
"""

import json

import numpy as np

from . import codec
from .deterministic import build_parameters, forward, output_digest
from .digest import canonical_json
from .errors import PreviewError
from .merge import build_batch_plan, merge_receipts, synthesize_conclusion
from .scheduler import now_ms


def preview(ledger, spec) -> dict:
    """快速视图：单次读取账本缓存。"""
    started = now_ms()
    rows = ledger.preview_rows(spec.spec_version)
    view = {
        "spec_version": spec.spec_version,
        "batches": [
            {
                "batch_id": row["batch_id"],
                "worker_id": row["worker_id"],
                "request_seqs": json.loads(row["request_seqs"]),
                "output_digest": row["output_digest"],
                "destination_distribution": json.loads(
                    row["destination_distribution"]
                ),
                "conclusion": json.loads(row["conclusion_json"]),
            }
            for row in rows
        ],
    }
    elapsed = now_ms() - started
    ledger.record_latency("preview", elapsed)
    view["elapsed_ms"] = elapsed
    return view


def rederive_conclusions(spec, plan):
    """不依赖套接字，按账本声明确定性重推导全部批次结论（用于恢复重放）。"""
    plan = plan or build_batch_plan(spec)
    derived = {}
    param_cache = {}
    for task in plan:
        worker = spec.worker(task.worker_id)
        key = (worker.worker_id, task.param_version)
        if key not in param_cache:
            param_cache[key] = build_parameters(worker, task.param_version)
        weight, bias, param_digest = param_cache[key]
        per_request = [
            (req.seq, output_digest(forward(weight, bias, req.vector)))
            for req in task.requests
        ]
        batch_digest = output_digest(
            np.frombuffer(
                b"".join(d.encode("ascii") for _, d in per_request),
                dtype=np.uint8,
            )
        )
        normalized = []
        context = task.context(param_digest, worker.quant)
        for merge_seq, fmt in enumerate(task.receipt_formats):
            fields = {
                "batch_id": task.batch_id,
                "worker_id": task.worker_id,
                "destination": task.worker_id,
                "output_digest": batch_digest,
                "param_digest": param_digest,
                "quant": worker.quant,
                "elapsed_ms": 0,
                "request_seqs": list(task.request_seqs),
                "param_version": task.param_version,
            }
            raw = codec.encode_receipt(fmt, fields)
            decoded_fmt, obj, raw_len = codec.decode_receipt(raw)
            view = codec.normalize_receipt(
                decoded_fmt, obj, context, raw_len
            )
            view["merge_seq"] = merge_seq
            normalized.append(view)
        winner, _ = merge_receipts(normalized)
        derived[task.batch_id] = synthesize_conclusion(
            task, winner, per_request
        )
    return derived


def recompute(ledger, spec, pool=None) -> dict:
    """权威重算：逐批重放分发/归并/推理，并与缓存结论逐字节比对。"""
    started = now_ms()
    plan = build_batch_plan(spec)
    rederived = rederive_conclusions(spec, plan)
    cached = {
        row["batch_id"]: json.loads(row["conclusion_json"])
        for row in ledger.preview_rows(spec.spec_version)
    }
    for task in plan:
        batch_id = task.batch_id
        if batch_id not in cached:
            raise PreviewError(
                "权威重算发现账本缺少批次缓存", batch_id=batch_id,
                field="<batch>",
            )
        if canonical_json(cached[batch_id]) != canonical_json(rederived[batch_id]):
            field, seq = _first_diverging_field(
                cached[batch_id], rederived[batch_id]
            )
            raise PreviewError(
                "快速缓存与权威重算不逐字节一致",
                batch_id=batch_id, field=field, request_seq=seq,
            )
    elapsed = now_ms() - started
    ledger.record_latency("recompute", elapsed)
    stats = {
        "spec_version": spec.spec_version,
        "batches": len(plan),
        "matched": len(plan),
        "elapsed_ms": elapsed,
        "latency": ledger.latency_stats("recompute"),
    }
    return stats


def _first_diverging_field(cached, derived):
    for key in (
        "worker_id", "param_version", "source_format", "param_digest",
        "output_digest", "content_digest",
    ):
        if cached.get(key) != derived.get(key):
            return key, None
    cached_dest = dict(cached["request_destinations"])
    for seq, destination in derived["request_destinations"]:
        if cached_dest.get(seq) != destination:
            return "request_destinations", seq
    cached_out = dict(cached["per_request_outputs"])
    for seq, digest in derived["per_request_outputs"]:
        if cached_out.get(seq) != digest:
            return "per_request_outputs", seq
    return "<conclusion>", None
