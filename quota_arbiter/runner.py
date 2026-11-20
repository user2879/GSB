"""仲裁运行编排：批次落账、三路径一致性、崩溃恢复与重放。"""
from __future__ import annotations

import os

from . import ALL_PATHS
from .consistency import compare_results
from .engine import ArbiterEngine
from .ledger import Ledger
from .paths import build_batch_plan, canonical_requests, run_path
from .selector import select_default_path
from .snapshot import verify_replay, write_snapshot
from .util import digest_of


def run_all_paths(tenants, pools, requests):
    results = {name: run_path(name, tenants, pools, requests) for name in ALL_PATHS}
    report = compare_results(results)
    return results, report


def persist_run(tenants, pools, requests, path_name: str, ledger: Ledger,
                *, crash_after_batch=None, logger=None):
    """按执行路径的批次边界驱动引擎，并在每个批次同一事务落账。

    引擎始终按全局规范序推进；批次计划仅决定提交边界与统计口径，
    因此三条路径的恢复点不同，但结局序列完全一致。
    """
    ordered = canonical_requests(requests)
    by_rid = {req.rid: req for req in ordered}
    plan = build_batch_plan(path_name, tenants, ordered)
    engine = ArbiterEngine(tenants, pools)

    # 第一遍：完整驱动引擎，得到与路径无关的确定性最终结局与排队轨迹。
    for batch in plan:
        for rid in batch["rids"]:
            engine.process_request(by_rid[rid])
    engine.finalize()

    # 第二遍：按路径批次边界在同一事务中落账。结局按 rid UPSERT 到最终态，
    # 排队轨迹逐条追加，批次表记录逐批余量、摘要链与进度游标。
    committed = 0
    for batch_index, batch in enumerate(plan):
        batch_requests = [by_rid[rid] for rid in batch["rids"]]
        outcomes = [engine.outcomes[req.rid] for req in batch_requests]
        batch_events = [e for e in engine.queue_events if e.rid in set(batch["rids"])]
        tenant_state = {
            tid: (
                engine.quota_remaining[tid],
                engine.burst_remaining[tid],
                engine.inflight[tid],
            )
            for tid in sorted(engine.quota_remaining)
        }
        pool_state = {pid: engine.pool_remaining[pid] for pid in sorted(engine.pool_remaining)}
        crash_here = crash_after_batch is not None and batch_index == crash_after_batch
        ledger.append_batch(
            batch_index,
            path_name,
            {
                "request_count": batch["request_count"],
                "skipped_count": batch["skipped_count"],
                "rids": batch["rids"],
            },
            outcomes,
            batch_events,
            tenant_state,
            pool_state,
            crash_in_batch=crash_here,
        )
        committed += 1
        if logger:
            logger.info(
                "batch_committed",
                path=path_name,
                batch_index=batch_index,
                request_count=batch["request_count"],
                skipped_count=batch["skipped_count"],
            )
    return {
        "committed_batches": committed,
        "last_rid": ordered[-1].rid if ordered else None,
        "total_requests": len(ordered),
    }


def recover_and_rerun(tenants, pools, requests, ledger: Ledger, spec_digest: str,
                      path_name: str, *, logger=None):
    """强制终止后恢复：只承认完整提交批次，未提交请求标记未知并重跑。"""
    ledger.verify_format()
    ledger.verify_spec_digest(spec_digest)
    ledger.verify_digest_chain()
    ordered = canonical_requests(requests)
    all_rids = [r.rid for r in ordered]
    point = ledger.recovery_point(all_rids)
    if logger:
        logger.info(
            "recovery_point",
            last_committed_rid=point["last_committed_rid"],
            unknown_count=point["unknown_count"],
            unknown_rids=point["unknown_rids"],
        )

    rerun_ledger_path = ledger.path + ".rerun"
    if os.path.exists(rerun_ledger_path):
        os.remove(rerun_ledger_path)
    rerun = Ledger(rerun_ledger_path)
    rerun.initialize_session(tenants, pools, path_name, spec_digest)
    summary = persist_run(tenants, pools, requests, path_name, rerun, logger=logger)
    rerun.verify_digest_chain()
    rerun.close()
    point["rerun_ledger"] = rerun_ledger_path
    point["rerun_summary"] = summary
    return point


def snapshot_and_replay(tenants, pools, requests, selection, result, snapshot_path: str,
                        *, cursor_rid: str | None = None):
    ordered = canonical_requests(requests)
    cursor = cursor_rid or (ordered[-1].rid if ordered else "")
    write_snapshot(snapshot_path, tenants, pools, selection, result, cursor)
    from .snapshot import read_snapshot

    payload = read_snapshot(snapshot_path)
    replay_result = run_path(selection["selected"], tenants, pools, requests)
    verify_replay(payload, replay_result, cursor)
    return replay_result


def choose_and_describe(tenants, pools, requests):
    return select_default_path(len(tenants), len(requests), len(pools))


def spec_digest(tenants, pools, requests):
    return digest_of(
        {
            "tenants": [
                {"tid": t.tid, "quota": t.quota,
                 "concurrency_cap": t.concurrency_cap, "burst": t.burst}
                for t in sorted(tenants, key=lambda x: x.tid)
            ],
            "pools": [
                {"pid": p.pid, "capacity": p.capacity,
                 "priority_order": list(p.priority_order)}
                for p in sorted(pools, key=lambda x: x.pid)
            ],
            "requests": [
                [r.rid, r.tid, r.pid, r.demand, r.arrival, r.priority]
                for r in canonical_requests(requests)
            ],
        }
    )
