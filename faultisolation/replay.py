"""独立确定性重放与双路径逐字节一致性校验。

校验命令只读取原始声明、事件流与账本文件，自行用逻辑时钟重放三条路径，
完全不依赖主流程内存状态。比较字段覆盖：结局状态、租约版本、所属槽位、
到期时刻、等待时长等全部确定性列；首个不一致抛 ``E_DIVERGE``。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .clock import LogicalClock
from .engine import Engine, Outcome
from .errors import DivergenceError
from .ledger import Ledger
from .planner import PATH_BATCH, PATH_EVENT, PATH_MERGE
from .spec import SpecBundle, load_bundle

COMPARE_FIELDS = (
    "status", "lease_id", "version", "slots", "expiry_ms",
    "wait_start_ms", "wait_end_ms", "wait_ms", "clock_ms", "domain_id", "action",
)


def replay_path(bundle: SpecBundle, path: str) -> list[Outcome]:
    engine = Engine(bundle, LogicalClock(0))
    if path == PATH_EVENT:
        return engine.run_event_path()
    if path == PATH_BATCH:
        return engine.run_batch_path()
    if path == PATH_MERGE:
        return engine.run_merge_path()
    raise ValueError(f"未知仲裁路径 {path}")


def _compare_outcomes(a: Outcome, b: Outcome) -> str | None:
    for field_name in COMPARE_FIELDS:
        if getattr(a, field_name) != getattr(b, field_name):
            return field_name
    return None


def verify_paths(bundle: SpecBundle) -> dict:
    """重放三条路径并逐字节比较 canonical JSON，返回核对摘要。"""
    results = {path: replay_path(bundle, path) for path in (PATH_EVENT, PATH_BATCH, PATH_MERGE)}
    lengths = {path: len(items) for path, items in results.items()}
    if len(set(lengths.values())) != 1:
        first_path = PATH_EVENT
        first_len = lengths[first_path]
        bad_path = next(p for p, n in lengths.items() if n != first_len)
        raise DivergenceError(
            "路径输出事件数不一致", seq=-1, instance="", field="length",
            path_a=first_path, path_b=bad_path,
        )
    reference = results[PATH_EVENT]
    byte_hashes = {
        path: hashlib.sha256(b"".join(o.canonical_json() for o in items)).hexdigest()
        for path, items in results.items()
    }
    for path in (PATH_BATCH, PATH_MERGE):
        for idx, (ref, other) in enumerate(zip(reference, results[path])):
            if ref.canonical_json() != other.canonical_json():
                bad_field = _compare_outcomes(ref, other) or "canonical_json"
                raise DivergenceError(
                    f"{PATH_EVENT} 与 {path} 在事件 {ref.seq} 字段 {bad_field} 出现首个不一致",
                    seq=ref.seq, instance=ref.instance_id, field=bad_field,
                    path_a=PATH_EVENT, path_b=path,
                )
    return {
        "event_count": lengths[PATH_EVENT],
        "byte_hashes": byte_hashes,
        "byte_identical": len(set(byte_hashes.values())) == 1,
        "fields_compared": list(COMPARE_FIELDS),
    }


def verify_ledger(bundle: SpecBundle, ledger_path: str | Path) -> dict:
    """把账本已提交结局与独立重放结果逐事件逐字段比对。"""
    ledger = Ledger(ledger_path)
    replaced = ledger.replace_if_digest_bad()
    stored = ledger.load_outcomes()
    recovery = ledger.recover()
    replay = {o.seq: o for o in replay_path(bundle, PATH_EVENT)}
    first_diverge = None
    for item in stored:
        expected = replay.get(item["seq"])
        if expected is None:
            raise DivergenceError("账本包含重放中不存在的事件", seq=item["seq"],
                                  instance=item.get("instance_id", ""), field="seq")
        for field_name in COMPARE_FIELDS:
            if item.get(field_name) != getattr(expected, field_name):
                first_diverge = DivergenceError(
                    f"账本与重放在事件 {item['seq']} 字段 {field_name} 不一致",
                    seq=item["seq"], instance=item["instance_id"], field=field_name,
                )
                break
        if first_diverge:
            raise first_diverge
    missing = sorted(set(replay) - {item["seq"] for item in stored})
    ledger.close()
    return {
        "stored_events": len(stored),
        "replay_events": len(replay),
        "missing_seqs": missing,
        "digest_ok": recovery.digest_ok,
        "replaced": replaced,
        "active_leases": len(recovery.active_leases),
        "last_committed_seq": recovery.last_committed_seq,
    }


def verify_slot_uniqueness(bundle: SpecBundle) -> dict:
    """重放并逐时刻检查同域活跃租约槽位不重叠（E_ALIAS 防线的独立验证）。"""
    engine = Engine(bundle, LogicalClock(0))
    engine.run_event_path()
    violations = []
    for domain_id, arbiter in engine.arbiters.items():
        seen: dict[int, str] = {}
        for lease in arbiter.history.values():
            if lease.state in ("granted", "renewed"):
                for slot in lease.slots:
                    if slot in seen and seen[slot] != lease.lease_id:
                        violations.append({
                            "domain_id": domain_id, "slot": slot,
                            "lease_a": seen[slot], "lease_b": lease.lease_id,
                        })
                    seen[slot] = lease.lease_id
    return {
        "domains": len(engine.arbiters),
        "violations": violations,
        "ok": not violations,
    }
