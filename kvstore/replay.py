"""权威重放、快速查询一致性校验与快照恢复重放比对。

* :func:`replay_authoritative` 从初始记录出发，按轨迹逐操作重放到指定游标，
  返回权威键集合/计数/逐操作结论。全程仅使用轨迹注入的逻辑时钟。
* :func:`verify_visible` 对增量缓存的快速结论与权威重放逐项比对，首错抛 E_VISIBLE。
* :func:`verify_replay_after_restore` 在快照恢复后重新执行至快照游标并与快照内
  保存的历史结论逐项比对，首错抛 E_REPLAY，定位操作序号/键/前缀。
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from .engine import Engine
from .errors import ReplayError, VisibleError
from .storage import Storage
from . import codec


def replay_authoritative(
    operations: list[dict[str, Any]],
    initial_records: list[dict[str, Any]],
    db_path: str,
    upto: Optional[int] = None,
) -> dict[str, Any]:
    """在独立 DB 上做权威重放，不触碰前台会话。"""
    storage = Storage(db_path)
    try:
        engine = Engine(storage)
        if initial_records:
            engine.load_initial_records(copy.deepcopy(initial_records))
        conclusions: list[dict[str, Any]] = []
        target = len(operations) if upto is None else upto
        for operation in operations[:target]:
            conclusion = _apply_one(engine, operation)
            conclusions.append(conclusion)
        state = engine.export_state()
        return {
            "upto_op_seq": target,
            "conclusions": conclusions,
            "state": state,
            "engine": engine,
            "storage": storage,
        }
    except Exception:
        storage.close()
        raise


def _apply_one(engine: Engine, operation: dict[str, Any]) -> dict[str, Any]:
    seq = operation["op_seq"]
    name = operation["op"]
    clock = operation["clock"]
    if name == "PUT":
        result = engine.put(
            operation["key"], operation["value"],
            operation["expire_ms"], seq, clock,
        )
    elif name == "GET":
        result = engine.get(operation["key"], clock)
    elif name == "DEL":
        result = engine.delete(operation["key"], clock)
    elif name == "PURGE":
        result = engine.purge(clock)
    elif name == "REBUILD_INDEX":
        # 权威重放中重建是确定性的等价变换：重建后索引/计数必须不变
        before = engine.export_state()
        result = {"op": "REBUILD_INDEX", "op_seq": seq, "clock": clock,
                  "keyspace_version": engine.keyspace_version}
        after = engine.export_state()
        if (before["index_entries"], before["prefix_counts"]) != (
            after["index_entries"],
            after["prefix_counts"],
        ):
            raise AssertionError("权威重放中重建改变了索引内容")
    else:  # pragma: no cover - 轨迹加载阶段已拦截
        raise ValueError(name)
    result = dict(result)
    result["op_seq"] = seq
    return result


def verify_visible(
    live_engine: Engine,
    operations: list[dict[str, Any]],
    initial_records: list[dict[str, Any]],
    replay_db_path: str,
    clock: int,
    prefixes: list[str],
    upto: Optional[int] = None,
) -> dict[str, Any]:
    authority = replay_authoritative(
        operations, initial_records, replay_db_path, upto=upto
    )
    auth_engine: Engine = authority["engine"]
    divergence_seq = authority["upto_op_seq"]
    try:
        for prefix in prefixes:
            quick = live_engine.query_prefix(prefix, clock)
            auth_keys = auth_engine.keyset(prefix, clock)
            auth_count = auth_engine.visible_count(prefix, clock)
            if quick["keys"] != auth_keys or quick["visible_count"] != auth_count:
                raise VisibleError(
                    "快速查询结论与权威重放不一致",
                    details={
                        "prefix": prefix,
                        "divergence_op_seq": divergence_seq,
                        "quick": {"keys": quick["keys"],
                                  "visible_count": quick["visible_count"]},
                        "authoritative": {"keys": auth_keys,
                                          "visible_count": auth_count},
                    },
                )
        return {"ok": True, "prefixes_checked": len(prefixes),
                "divergence_op_seq": divergence_seq}
    finally:
        authority["storage"].close()


def verify_replay_after_restore(
    engine: Engine,
    operations: list[dict[str, Any]],
    initial_records: list[dict[str, Any]],
    saved_conclusions: list[dict[str, Any]],
    replay_db_path: str,
) -> dict[str, Any]:
    cursor = engine.trace_cursor
    authority = replay_authoritative(
        operations, initial_records, replay_db_path, upto=cursor
    )
    try:
        fresh_engine: Engine = authority["engine"]
        fresh_conclusions = authority["conclusions"]
        if len(fresh_conclusions) != len(saved_conclusions):
            raise ReplayError(
                "恢复后重放结论数量与快照不符",
                details={"cursor": cursor,
                         "saved": len(saved_conclusions),
                         "replayed": len(fresh_conclusions)},
            )
        for index, (saved, fresh) in enumerate(
            zip(saved_conclusions, fresh_conclusions)
        ):
            seq = fresh["op_seq"]
            if saved.get("op") != fresh.get("op") or saved.get("op_seq") != seq:
                raise ReplayError(
                    "重放操作类型/序号分歧",
                    details={"op_seq": seq, "saved": saved, "replayed": fresh},
                )
            _compare_conclusion(seq, saved, fresh)

        # 状态层逐项比对：键集合、索引条目、计数
        live_state = engine.export_state()
        auth_state = fresh_engine.export_state()
        _compare_states(live_state, auth_state, cursor)
        return {"ok": True, "replayed_ops": cursor,
                "keyspace_version": live_state["keyspace_version"]}
    finally:
        authority["storage"].close()


def _compare_conclusion(seq: int, saved: dict[str, Any], fresh: dict[str, Any]) -> None:
    name = fresh["op"]
    if name == "GET":
        if saved.get("status") != fresh.get("status"):
            raise ReplayError(
                "GET 过期/可见结论分歧",
                details={"op_seq": seq, "key": fresh.get("key"),
                         "prefix": fresh.get("key"),
                         "saved_status": saved.get("status"),
                         "replayed_status": fresh.get("status")},
            )
        if fresh.get("status") == "OK" and saved.get("value") != fresh.get("value"):
            raise ReplayError(
                "GET 返回值分歧",
                details={"op_seq": seq, "key": fresh.get("key"),
                         "saved_value": saved.get("value"),
                         "replayed_value": fresh.get("value")},
            )
    elif name in ("PUT", "DEL", "PURGE"):
        for field in ("keyspace_version",):
            if saved.get(field) != fresh.get(field):
                raise ReplayError(
                    f"{name} 结论字段分歧: {field}",
                    details={"op_seq": seq, "key": fresh.get("key"),
                             "field": field,
                             "saved": saved.get(field),
                             "replayed": fresh.get(field)},
                )
        if name == "PURGE" and sorted(saved.get("purged_keys", [])) != sorted(
            fresh.get("purged_keys", [])
        ):
            raise ReplayError(
                "PURGE 移除键集合分歧",
                details={"op_seq": seq,
                         "saved": saved.get("purged_keys"),
                         "replayed": fresh.get("purged_keys")},
            )


def _compare_states(live: dict[str, Any], auth: dict[str, Any], cursor: int) -> None:
    live_kv = {r["key"]: r for r in live["kv_records"]}
    auth_kv = {r["key"]: r for r in auth["kv_records"]}
    if set(live_kv) != set(auth_kv):
        sample = sorted(set(live_kv) ^ set(auth_kv))[0]
        raise ReplayError(
            "恢复后键集合与权威重放分歧",
            details={"op_seq": cursor, "key": sample, "prefix": sample,
                     "reason": "keyset_divergence"},
        )
    for key in sorted(auth_kv):
        if live_kv[key] != auth_kv[key]:
            raise ReplayError(
                "恢复后键值记录内容分歧",
                details={"op_seq": cursor, "key": key, "prefix": key,
                         "saved": live_kv[key], "replayed": auth_kv[key]},
            )
    live_index = sorted(
        (r["prefix"], r["ref_key"], r["index_value"], r["entry_version"])
        for r in live["index_entries"]
    )
    auth_index = sorted(
        (r["prefix"], r["ref_key"], r["index_value"], r["entry_version"])
        for r in auth["index_entries"]
    )
    if live_index != auth_index:
        first = next(
            (a, b) for a, b in zip(live_index, auth_index) if a != b
        )
        raise ReplayError(
            "恢复后索引条目/版本分歧",
            details={"op_seq": cursor, "prefix": first[0][0],
                     "key": first[0][1], "saved": first[0],
                     "replayed": first[1]},
        )
    if live["prefix_counts"] != auth["prefix_counts"]:
        saved_counts = {c["prefix"]: c["cnt"] for c in live["prefix_counts"]}
        auth_counts = {c["prefix"]: c["cnt"] for c in auth["prefix_counts"]}
        prefix = sorted(set(saved_counts) | set(auth_counts))[0]
        raise ReplayError(
            "恢复后前缀计数分歧",
            details={"op_seq": cursor, "prefix": prefix,
                     "saved": saved_counts.get(prefix, 0),
                     "replayed": auth_counts.get(prefix, 0)},
        )
