"""恢复后一致性重放校验与异常注入判定。

重放依据账本记录的槽位参数版本、阈值与哈希分区数，在独立临时账本上
从干净状态重跑全部已完整提交批次（不写回原账本），每个批次结束的同一
时刻捕获连接水位与聚合中间量快照并同账本逐项比较，首个差异以
E_REPLAY 报明依据。
"""

import os
import tempfile

from .digest import content_digest
from .engine import Engine
from .errors import DigestError, ReplayError
from .scan import build_scan_plan


def _check(condition, message, location, detail):
    if not condition:
        raise ReplayError(message, location, detail)


def _expected_thresholds(spec):
    return {
        "hotspot_row_threshold": spec.hotspot.row_threshold,
        "subpartition_count": spec.hotspot.subpartition_count,
        "partition_count": spec.topology.partition_count,
        "hash_rule": "FNV1A32_V1",
    }


def verify_snapshot(ledger, spec):
    snap = ledger.read_snapshot()
    if snap is None:
        return None
    if snap["format_version"] != 1:
        raise ReplayError("快照格式版本不一致", "snapshot.format_version",
                          {"expected": 1, "actual": snap["format_version"]})
    if snap["slot_params_version"] != spec.slot_params_version:
        raise ReplayError("槽位参数版本不一致", "snapshot.slot_params_version",
                          {"expected": spec.slot_params_version,
                           "actual": snap["slot_params_version"]})
    if snap["thresholds"] != _expected_thresholds(spec):
        raise ReplayError("快照阈值与当前规格不一致", "snapshot.thresholds",
                          {"snapshot": snap["thresholds"],
                           "expected": _expected_thresholds(spec)})
    return snap


def verify_digests(ledger, inject_digest=False):
    mismatch = ledger.verify_batch_digests()
    if mismatch is not None:
        batch_id, actual, stored = mismatch
        raise DigestError(
            "分片摘要不一致" + ("（异常注入）" if inject_digest else ""),
            f"batches[{batch_id}].digest",
            {"expected": stored, "actual": actual},
        )
    snap = ledger.read_snapshot()
    if snap is not None and snap["ledger_digest"] != ledger.ledger_digest():
        raise DigestError(
            "快照账本摘要不一致" + ("（异常注入）" if inject_digest else ""),
            "snapshot.ledger_digest",
            {"expected": snap["ledger_digest"],
             "actual": ledger.ledger_digest()},
        )


def replay_and_compare(spec, out_dir, clock, inject_digest=False):
    from .ledger import Ledger
    source_ledger = Ledger(out_dir + "/ledger.sqlite")
    verify_snapshot(source_ledger, spec)
    verify_digests(source_ledger, inject_digest)
    batch_ids = source_ledger.all_batch_ids()
    _check(bool(batch_ids), "账本中没有已提交批次，无法重放", "batches", {})

    tmp_dir = tempfile.mkdtemp(prefix="hashlot_replay_", dir=out_dir)
    replay_engine = Engine(spec, tmp_dir, clock, workers=1)
    os.unlink(os.path.join(tmp_dir, "ledger.sqlite"))
    replay_engine.ledger = source_ledger
    plan = build_scan_plan(spec, redistribute=True)
    tasks = replay_engine._build_tasks(plan)
    _check(len(tasks) >= len(batch_ids), "重放任务数少于账本批次数", "batches",
           {"ledger_batches": len(batch_ids), "replay_tasks": len(tasks)})

    crash_events = source_ledger.crash_events()
    unknown = sorted({cid for event in crash_events for cid in event["connections"]})

    for index, batch_id in enumerate(batch_ids):
        stored = source_ledger.load_batch(batch_id)
        replay_engine._run_task(tasks[index])
        live_wm = replay_engine.slot_manager.watermark_snapshot()
        live_by_id = {c["conn_id"]: c for c in live_wm["connections"]}

        for stored_conn in stored["connections"]:
            cid = stored_conn["conn_id"]
            if stored_conn["state"] == "released" or cid in unknown:
                continue
            live = live_by_id.get(cid)
            _check(live is not None, "重放缺少账本记录的连接",
                   f"batches[{batch_id}].connections[{cid}]",
                   {"field": "existence"})
            for field in ("owner_partition", "home_partition", "is_borrowed",
                          "key_cap", "key_count", "peak_keys", "lender",
                          "subpartition"):
                _check(live.get(field) == stored_conn.get(field),
                       f"连接字段重放不一致: {field}",
                       f"batches[{batch_id}].connections[{cid}].{field}",
                       {"expected": stored_conn.get(field),
                        "actual": live.get(field)})

        live_partials = {p["conn_id"]: p
                         for p in replay_engine.capture_partial_snapshot()}
        for partial in sorted(stored["partials"], key=lambda p: p["conn_id"]):
            cid = partial["conn_id"]
            _check(cid in live_partials, "重放缺少账本记录的聚合连接",
                   f"batches[{batch_id}].partials[{cid}]", {"field": "existence"})
            _check(live_partials[cid] == partial,
                   "分区聚合中间量重放不一致",
                   f"batches[{batch_id}].partials[{cid}]",
                   {"expected_digest": content_digest(partial),
                    "actual_digest": content_digest(live_partials[cid])})

        if stored["borrows"]:
            live_borrows = [{
                "borrower": b.borrower, "lender": b.lender,
                "key_cap": b.borrowed_key_cap, "deadline_ms": b.deadline_ms,
            } for b in sorted(spec.topology.borrows, key=lambda x: x.borrower)]
            _check(live_borrows == stored["borrows"], "借用归属重放不一致",
                   f"batches[{batch_id}].borrows",
                   {"expected": stored["borrows"], "actual": live_borrows})

    final = replay_engine.finalize(plan)
    ledger_digest = source_ledger.ledger_digest()
    if crash_events:
        source_ledger.clear_crash_events()
    replay_engine.close()
    _cleanup_tmp(tmp_dir)
    return {
        "batches_replayed": len(batch_ids),
        "unknown_connections": unknown,
        "crash_events": crash_events,
        "result_digest": final["result_digest"],
        "ledger_digest": ledger_digest,
    }


def _cleanup_tmp(tmp_dir):
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
