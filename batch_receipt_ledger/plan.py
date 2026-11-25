"""确定性批切分计划（需求 3）。

规则（完全显式、与字典遍历顺序无关）：
  1. 归并后的有效请求按 ``arrival_seq`` 升序排列；
  2. 按声明的 ``batch_size`` 顺序切片；
  3. 工作进程按 ``worker_id`` 排序后以批序号取模轮转；
  4. 批标识 = 稳定 JSON 序列化（sort_keys、固定分隔符）后的 BLAKE2b。
"""

import hashlib
import json
from dataclasses import dataclass


def canonical_json(obj):
    """逐字节稳定的 JSON 序列化。"""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def stable_hash(payload, digest_size=16):
    return hashlib.blake2b(canonical_json(payload), digest_size=digest_size).hexdigest()


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    worker_id: str
    batch_seq: int
    request_ids: tuple
    arrival_seqs: tuple

    def identity_payload(self):
        return {
            "worker_id": self.worker_id,
            "batch_seq": self.batch_seq,
            "arrival_seqs": list(self.arrival_seqs),
        }

    def to_dict(self):
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "batch_seq": self.batch_seq,
            "request_ids": list(self.request_ids),
            "arrival_seqs": list(self.arrival_seqs),
        }


def derive_batch_id(worker_id, batch_seq, arrival_seqs):
    payload = {
        "worker_id": worker_id,
        "batch_seq": int(batch_seq),
        "arrival_seqs": [int(seq) for seq in arrival_seqs],
    }
    return "b_" + stable_hash(payload, digest_size=12)


def build_plan(spec):
    """根据声明生成确定性批计划列表（批序号升序）。"""
    active = spec.effective_deliveries()
    worker_ids = sorted(spec.worker_ids_sorted)
    batch_size = spec.batch_size

    plans = []
    batch_seq = 0
    for offset in range(0, len(active), batch_size):
        chunk = active[offset : offset + batch_size]
        worker_id = worker_ids[batch_seq % len(worker_ids)]
        request_ids = tuple(item.request_id for item in chunk)
        arrival_seqs = tuple(item.arrival_seq for item in chunk)
        batch_id = derive_batch_id(worker_id, batch_seq, arrival_seqs)
        plans.append(
            BatchPlan(
                batch_id=batch_id,
                worker_id=worker_id,
                batch_seq=batch_seq,
                request_ids=request_ids,
                arrival_seqs=arrival_seqs,
            )
        )
        batch_seq += 1
    return plans


def plan_fingerprint(plans):
    """整个计划的逐字节指纹，用于判定两次划分是否完全一致。"""
    payload = [plan.to_dict() for plan in plans]
    return hashlib.blake2b(canonical_json(payload), digest_size=16).hexdigest()
