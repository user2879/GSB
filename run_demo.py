"""可复现实验入口：跑通全部业务要求并把结果写入 results/。

场景：
1. 加载初始记录 + 顺序执行轨迹，校验索引/计数一致性（E_INDEX 守门）。
2. 过期语义：按注入逻辑时钟验证可见/过期/缺失，并保留判定依据步骤。
3. 并发重建：屏障挂起重建线程，期间执行前台查询与写入，验证一致视图。
4. 快速查询 vs 权威重放（E_VISIBLE 守门）。
5. 快照保存/恢复 + 恢复后逐项重放（E_REPLAY 守门）。
6. 三类异常注入：E_TRACE / E_VERSION / E_SNAPSHOT，写入结构化日志。
7. 产出基准摘要 baseline_summary.json，重复运行逐字节可比。
"""

from __future__ import annotations

import json
import os
import shutil
import threading

from kvstore import inject as inject_mod
from kvstore import numpyutil
from kvstore.engine import Engine
from kvstore.errors import KvError
from kvstore.replay import replay_authoritative, verify_visible
from kvstore.session import Session
from kvstore.snapshot import open_snapshot
from kvstore.storage import Storage
from kvstore.structlog import StructuredLogger
from kvstore.codec import prefixes_of
from kvstore.trace import load_trace, operations_digest

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def reset_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def remove_db(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = path + suffix
        if os.path.exists(target):
            os.remove(target)


def load_initial() -> list[dict]:
    with open(os.path.join(DATA, "initial_records.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    reset_dir(RESULTS)
    log = StructuredLogger(os.path.join(RESULTS, "events.jsonl"))
    trace_path = os.path.join(DATA, "sample_trace.jsonl")
    operations = load_trace(trace_path)
    initial = load_initial()
    summary: dict = {"trace_digest": operations_digest(operations), "scenarios": {}}

    # ---------- 场景 1：主执行 + 一致性 ----------
    main_db = os.path.join(RESULTS, "main.db")
    session = Session(main_db, log, initial)
    conclusions = session.run(operations)
    consistency = session.engine.verify_index_consistency()
    log.log("scenario1_consistency", **consistency)

    # NumPy 向量比对：缓存结构计数 vs 键集合重算
    recomputed: dict[str, int] = {}
    for row in session.engine.storage.all_records():
        for prefix in prefixes_of(row["key"]):
            recomputed[prefix] = recomputed.get(prefix, 0) + 1
    numpy_check = numpyutil.compare_count_vectors(
        session.engine.storage.all_counts(), recomputed
    )
    # NumPy 可见计数矩阵：在多个逻辑时钟上的批量校验
    prefixes = sorted(session.engine.storage.all_counts().keys())
    clocks = [250, 600, 750, 1000, 1300]
    visible_matrix = numpyutil.visible_count_matrix(
        [dict(r) for r in session.engine.storage.all_records()], prefixes, clocks
    )
    summary["scenarios"]["1_primary_run"] = {
        "ops_executed": len(conclusions),
        "keyspace_version": session.engine.keyspace_version,
        "consistency": consistency,
        "numpy_vector_check": numpy_check,
        "visible_count_matrix": {
            "prefixes": prefixes,
            "clocks": clocks,
            "matrix": visible_matrix.tolist(),
        },
    }

    # ---------- 场景 2：过期语义（纯逻辑时钟，独立会话停在 PURGE 之前） ----------
    expire_db = os.path.join(RESULTS, "expire.db")
    remove_db(expire_db)
    expire_session = Session(expire_db, log, initial)
    pre_purge = [op for op in operations if op["op_seq"] <= 4]  # 含已过期的 user/1/role
    expire_session.run(pre_purge)
    expire_cases = []
    for key, clock in [
        ("user/1/name", 1300),   # expire_ms=0 永不过期 -> OK
        ("user/1/role", 600),    # 600 >= 500 -> E_EXPIRED（记录仍在，未 PURGE）
        ("user/3/name", 100),    # 100 < 1500 -> OK
        ("user/9/name", 650),    # 不存在 -> MISSING
    ]:
        case = expire_session.engine.get(key, clock)
        expire_cases.append(case)
        log.log("expiration_case", key=key, clock=clock, status=case["status"],
                basis=case["basis"])
    # 过期键不计入前缀计数：clock=600 时 user/1 的可见计数应排除 role
    expired_prefix = expire_session.engine.query_prefix("user/1", 600)
    summary["scenarios"]["2_expiration"] = {
        "cases": expire_cases,
        "prefix_user_1_at_600": expired_prefix,
        "note": "判定仅使用注入逻辑时钟，system_clock_used=false；expire_ms=0 永不过期",
    }
    expire_session.close()
    remove_db(expire_db)

    # ---------- 场景 3：并发重建一致视图 ----------
    barrier = threading.Event()
    rebuild_db = os.path.join(RESULTS, "rebuild.db")
    remove_db(rebuild_db)
    rebuild_session = Session(rebuild_db, log, initial)
    # 先跑到 REBUILD_INDEX（op_seq=8），随后手动控制
    pre_ops = [op for op in operations if op["op_seq"] < 8]
    rebuild_session.run(pre_ops)
    rebuild_op = next(op for op in operations if op["op"] == "REBUILD_INDEX")
    rebuild_session.rebuild_worker.start(rebuild_op["clock"], barrier=barrier)
    while rebuild_session.rebuild_worker.phase != "BUILDING":
        threading.Event().wait(0.005)
    during_queries = []
    for prefix in ("user", "user/1", ""):
        q = rebuild_session.engine.query_prefix(prefix, 750)
        during_queries.append(q)
    # 重建期间前台写入（delta 同步路径）
    concurrent_put = rebuild_session.engine.put(
        "user/5/name", "erin", 3000, 99, 760
    )
    barrier.set()
    rebuild_result = rebuild_session.finish_rebuild()
    post_consistency = rebuild_session.engine.verify_index_consistency(clock=760)
    # 权威对照：把“预操作 + 并发写入”作为已提交状态重放
    auth_db = os.path.join(RESULTS, "rebuild_auth.db")
    remove_db(auth_db)
    auth_ops = pre_ops + [
        {"op_seq": 99, "op": "PUT", "clock": 760, "key": "user/5/name",
         "value": "erin", "expire_ms": 3000}
    ]
    authority = replay_authoritative(auth_ops, initial, auth_db)
    auth_engine = authority["engine"]
    concurrent_checks = []
    for prefix in ("user", "user/1", "", "user/5"):
        quick = rebuild_session.engine.query_prefix(prefix, 760)
        expected_keys = auth_engine.keyset(prefix, 760)
        concurrent_checks.append({
            "prefix": prefix,
            "quick_keys": quick["keys"],
            "authoritative_keys": expected_keys,
            "match": quick["keys"] == expected_keys,
        })
    authority["storage"].close()
    remove_db(auth_db)
    summary["scenarios"]["3_concurrent_rebuild"] = {
        "rebuild_result": rebuild_result,
        "during_rebuild_queries": during_queries,
        "concurrent_put": concurrent_put,
        "post_rebuild_consistency": post_consistency,
        "authoritative_comparison": concurrent_checks,
        "all_match": all(item["match"] for item in concurrent_checks),
    }
    rebuild_session.close()

    # ---------- 场景 4：快速 vs 权威重放 ----------
    verify_db = os.path.join(RESULTS, "verify_replay.db")
    remove_db(verify_db)
    visible = verify_visible(
        session.engine, operations, initial, verify_db,
        clock=1300, prefixes=prefixes + [""],
    )
    remove_db(verify_db)
    summary["scenarios"]["4_visible_vs_authority"] = visible

    # ---------- 场景 5：快照保存/恢复 + 重放比对 ----------
    snap_path = os.path.join(RESULTS, "snapshot.json")
    snap_info = session.snapshot(snap_path)
    # 在全新 DB 上恢复，验证整体替换与逐项重放
    restored_db = os.path.join(RESULTS, "restored.db")
    remove_db(restored_db)
    restore_storage = Storage(restored_db)
    restore_engine = Engine(restore_storage)
    opened = open_snapshot(
        snap_path, restore_engine,
        readonly_copy_dir=os.path.join(RESULTS, "readonly"),
    )
    from kvstore.replay import verify_replay_after_restore

    restore_replay_db = os.path.join(RESULTS, "restore_replay.db")
    remove_db(restore_replay_db)
    replay_check = verify_replay_after_restore(
        restore_engine, operations, initial,
        opened["conclusions"], restore_replay_db,
    )
    remove_db(restore_replay_db)
    summary["scenarios"]["5_snapshot_restore"] = {
        "snapshot": snap_info,
        "replaced_on_open": opened["replaced"],
        "diff_summary": opened["diff_summary"],
        "replay_check": replay_check,
    }
    restore_storage.close()

    # ---------- 场景 6：异常注入 ----------
    injected_dir = os.path.join(RESULTS, "injected")
    reset_dir(injected_dir)
    injection_results = {}

    # E_TRACE：轨迹序号跳号 / 坏 JSON
    for kind in ("trace_corrupt", "trace_bad_json"):
        path = inject_mod.INJECTORS[kind](trace_path, injected_dir)
        try:
            load_trace(path)
            injection_results[kind] = {"unexpected": "no error"}
        except KvError as exc:
            injection_results[kind] = exc.to_dict()
            log.log("injection", kind=kind, **exc.to_dict())

    # E_VERSION：索引条目版本回退
    rb_path = inject_mod.inject_index_rollback(snap_path, injected_dir)
    version_db = os.path.join(RESULTS, "version_check.db")
    remove_db(version_db)
    version_storage = Storage(version_db)
    try:
        open_snapshot(rb_path, Engine(version_storage))
        injection_results["index_rollback"] = {"unexpected": "no error"}
    except KvError as exc:
        injection_results["index_rollback"] = exc.to_dict()
        log.log("injection", kind="index_rollback", **exc.to_dict())
    version_storage.close()
    remove_db(version_db)

    # E_SNAPSHOT：摘要不匹配
    bad_digest_path = inject_mod.inject_snapshot_digest(snap_path, injected_dir)
    snap_db = os.path.join(RESULTS, "snapshot_check.db")
    remove_db(snap_db)
    snap_storage = Storage(snap_db)
    try:
        open_snapshot(bad_digest_path, Engine(snap_storage))
        injection_results["snapshot_digest"] = {"unexpected": "no error"}
    except KvError as exc:
        injection_results["snapshot_digest"] = exc.to_dict()
        log.log("injection", kind="snapshot_digest", **exc.to_dict())
    snap_storage.close()
    remove_db(snap_db)

    summary["scenarios"]["6_injection"] = injection_results
    session.close()

    # ---------- 基准摘要（确定性，剔除墙上时间） ----------
    baseline = _strip_wall_time(summary)
    baseline_path = os.path.join(RESULTS, "baseline_summary.json")
    with open(baseline_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(baseline, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    full_path = os.path.join(RESULTS, "run_summary.json")
    with open(full_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"ok": True,
                      "baseline": baseline_path,
                      "full_summary": full_path,
                      "log": os.path.join(RESULTS, "events.jsonl")},
                     ensure_ascii=False, indent=2))


def _strip_wall_time(value):
    if isinstance(value, dict):
        return {k: _strip_wall_time(v) for k, v in value.items()
                if k not in ("start_wall_ms", "finish_wall_ms", "ts_wall_ms")}
    if isinstance(value, list):
        return [_strip_wall_time(v) for v in value]
    return value


if __name__ == "__main__":
    main()
