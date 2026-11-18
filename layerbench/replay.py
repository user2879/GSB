"""恢复后的重放校验与异常注入。

重放（无副作用、确定性）
------------------------
依据快照记录的清单版本与预算参数重建调度 DAG，按确定性就绪全序
（阶段拓扑序 + seq）遍历，重新推导：
- 每条指令的 precursor 与 layer_digest；
- 缓存命中结论（以快照缓存索引的 precursor 映射为准，不执行任何命令）；
- 业务码与槽位占用（从账本 slot 事件独立重建）。

随后与快照账本中的已完成指令逐项比较，差异定位到首个不一致的
阶段 / seq / 字段并以 E_REPLAY 报明依据。墙钟与单调时钟测量字段
（等待时长、执行耗时）不参与逐字节比较，这是显式排除而非遗漏。

异常注入
--------
仅通过修改输入文件或传入参数实现，绝不修改源码：
- corrupt-manifest：向清单副本写入非法 JSON / 非法字段；
- cache-mismatch：篡改缓存索引内容使其摘要与快照不符；
- zero-slots：以参数令 global_slots=0，触发 E_MANIFEST/E_BUDGET。
"""
from __future__ import annotations

import json
import os

from . import digest, layer as layer_mod
from .cache import CacheIndex, INDEX_FILENAME
from .errors import (
    E_BUDGET, E_MANIFEST, E_REPLAY, E_SNAPSHOT, LayerBenchError,
    ManifestError, OK, STATUS_SKIPPED,
)
from .manifest import load_manifest
from .scheduler import build_dag
from .snapshot import load_snapshot_meta, manifest_fingerprint
from .report import _rebuild_occupancy

# 重放参与逐项比较的确定性字段（显式白名单）
COMPARE_FIELDS = (
    "layer_digest", "precursor", "result_code", "status", "cache_hit",
    "exit_code", "stdout_digest", "result_digest", "skipped_trigger",
)


def replay_snapshot(snap_dir: str, workdir: str) -> dict:
    """重放快照并与快照账本逐项比对；任何差异抛 E_REPLAY。"""
    meta = load_snapshot_meta(snap_dir)
    manifest_path = os.path.join(snap_dir, "manifest.json")
    manifest = load_manifest(manifest_path, workspace=workdir)

    # 预算参数须与快照记录一致
    if manifest.global_slots != meta["budget"]["global_slots"]:
        raise LayerBenchError(
            E_REPLAY, "重放预算 global_slots 与快照记录不一致", snap_dir,
            {"manifest": manifest.global_slots,
             "snapshot": meta["budget"]["global_slots"]},
        )

    cache = CacheIndex.load(os.path.join(snap_dir, "cache"),
                            manifest.cache_capacity,
                            lambda: 0, lambda: 0)
    file_fp = CacheIndex.fingerprint_file(os.path.join(snap_dir, "cache"))
    if cache.fingerprint() != file_fp:
        raise LayerBenchError(
            E_SNAPSHOT, "快照内缓存索引内存/文件摘要不一致", snap_dir)

    snap_rows = _load_snapshot_rows(snap_dir, meta["build_id"])
    nodes = build_dag(manifest)
    order = sorted(nodes.values(),
                   key=lambda n: (n.stage_topo, n.seq))
    records: dict[str, dict] = {}

    for node in order:
        pred_layers = {
            uid: records[uid]["layer_digest"]
            for uid in node.predecessors
            if records[uid].get("layer_digest")
        }
        precursor, _inputs = layer_mod.precursor_digest(
            node.instruction, manifest, pred_layers
        )
        replayed = _replay_node(node, precursor, pred_layers, records, cache,
                                snap_rows)
        records[node.uid] = replayed

    comparison = _compare_with_ledger(snap_dir, manifest, records, meta)
    return {
        "ok": True,
        "snapshot_id": meta["snapshot_id"],
        "nodes_compared": comparison["count"],
        "fields": list(COMPARE_FIELDS),
        "occupancy": comparison["occupancy"],
    }


def _replay_node(node, precursor: str, pred_layers: dict,
                 records: dict, cache: CacheIndex,
                 snap_rows: dict) -> dict:
    base = {
        "node_uid": node.uid, "stage": node.stage, "seq": node.seq,
        "precursor": precursor, "cache_hit": False, "layer_digest": None,
        "result_code": OK, "status": "committed", "exit_code": None,
        "stdout_digest": None, "result_digest": None,
        "skipped_trigger": None,
    }
    # 跳过传播：上游 skipped 或失败
    trigger = None
    for pred_uid in sorted(node.predecessors):
        up = records[pred_uid]
        if up["status"] == STATUS_SKIPPED or up["result_code"] != OK:
            trigger = pred_uid
            break
    if trigger is not None:
        base.update({
            "status": STATUS_SKIPPED,
            "result_code": records[trigger]["result_code"],
            "skipped_trigger": trigger,
        })
        return base

    # cache_hit 以快照账本为权威事实（续建场景下节点可能在崩溃前已提交）。
    # 可用前驱集合仅用于解释命中来源；命中即从缓存复用，未命中则用账本
    # 中已提交的命令结果重算层摘要。判定不依赖 dict/set 遍历顺序。
    snap = snap_rows.get(node.uid)
    is_hit = bool(snap["cache_hit"]) if snap is not None else False
    entry = cache.lookup(precursor) if is_hit else None
    if entry is not None:
        base.update({
            "cache_hit": True,
            "layer_digest": entry.layer_digest,
            "result_digest": entry.result_digest,
            "exit_code": entry.exit_code,
            "stdout_digest": entry.stdout_digest,
            "result_code": OK,
        })
        return base

    # 未命中：非执行指令由 precursor 确定性推出层摘要；
    # run 指令不重新执行，而是复用快照账本中已提交的命令结果（退出码 +
    # 标准化 stdout 摘要）重算，保证与首次构建字节级一致。
    if node.instruction.type != "run":
        base["layer_digest"] = layer_mod.layer_digest_non_run(precursor)
        base["result_digest"] = layer_mod.NO_COMMAND_RESULT
    else:
        if snap is None:
            raise LayerBenchError(
                E_REPLAY, "未命中 run 指令缺少快照账本结果",
                node.uid, {"stage": node.stage, "seq": node.seq},
            )
        base["exit_code"] = snap["exit_code"]
        base["stdout_digest"] = snap["stdout_digest"]
        base["result_code"] = snap["result_code"]
        base["result_digest"] = layer_mod.command_result_digest(
            snap["exit_code"], snap["stdout_digest"]
        )
        base["layer_digest"] = layer_mod.layer_digest_run(
            precursor, base["result_digest"]
        )
    return base



def _load_snapshot_rows(snap_dir: str, build_id: str) -> dict:
    import sqlite3
    db_path = os.path.join(snap_dir, "ledger", "ledger.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM instructions WHERE build_id=?", (build_id,)
    ).fetchall()
    conn.close()
    return {row["node_uid"]: row for row in rows}


def _compare_with_ledger(snap_dir: str, manifest, replayed: dict,
                         meta: dict) -> dict:
    import sqlite3
    db_path = os.path.join(snap_dir, "ledger", "ledger.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    build_id = meta["build_id"]
    rows = conn.execute(
        "SELECT * FROM instructions WHERE build_id=? ORDER BY rowid",
        (build_id,),
    ).fetchall()
    snap_by_uid = {row["node_uid"]: row for row in rows}

    count = 0
    # 与构建相同的确定顺序遍历，保证首个不一致定位稳定
    for stage_name in manifest.topo_order:
        for ins in manifest.stage_by_name[stage_name].instructions:
            uid = f"{stage_name}#{ins.seq}:{ins.id}"
            count += 1
            if uid not in snap_by_uid:
                raise LayerBenchError(
                    E_REPLAY, "快照账本缺少指令记录", snap_dir,
                    {"stage": stage_name, "seq": ins.seq, "field": "row",
                     "basis": "账本中不存在该 node_uid"},
                )
            row = snap_by_uid[uid]
            got = replayed[uid]
            for field in COMPARE_FIELDS:
                expected = row[field]
                actual = got.get(field)
                # SQLite 用 0/1 表示 bool，重放用 True/False
                if field == "cache_hit":
                    expected = bool(expected)
                if expected != actual:
                    raise LayerBenchError(
                        E_REPLAY, "重放字段与快照记录不一致", snap_dir,
                        {"stage": stage_name, "seq": ins.seq, "field": field,
                         "snapshot_value": expected, "replay_value": actual,
                         "basis": f"按确定性规则重推 {field} 与账本不符"},
                    )

    # 槽位占用从快照事件时间线独立重建并复核预算上限
    peak_global, peak_stage, violation = _rebuild_occupancy_from_db(
        conn, build_id, manifest
    )
    conn.close()
    if violation is not None:
        raise LayerBenchError(E_REPLAY, "重放发现槽位占用越限", snap_dir,
                              violation)
    return {
        "count": count,
        "occupancy": {"peak_global": peak_global, "peak_stage": peak_stage},
    }


def _rebuild_occupancy_from_db(conn, build_id: str, manifest):
    events = conn.execute(
        "SELECT * FROM events WHERE build_id=? AND kind IN "
        "('slot_acquired','slot_released') ORDER BY seq", (build_id,)
    ).fetchall()
    cur_g = peak_g = 0
    cur_s = {s.name: 0 for s in manifest.stages}
    peak_s = dict(cur_s)
    violation = None
    for event in events:
        stage = event["node_uid"].split("#", 1)[0]
        if event["kind"] == "slot_acquired":
            cur_g += 1
            cur_s[stage] += 1
            peak_g = max(peak_g, cur_g)
            peak_s[stage] = max(peak_s[stage], cur_s[stage])
            if cur_g > manifest.global_slots:
                violation = {"basis": "全局占用越限", "node_uid": event["node_uid"]}
            if cur_s[stage] > manifest.stage_by_name[stage].quota:
                violation = {"basis": f"阶段 {stage} 配额越限",
                             "node_uid": event["node_uid"]}
        else:
            cur_g = max(0, cur_g - 1)
            cur_s[stage] = max(0, cur_s[stage] - 1)
    return peak_g, peak_s, violation



def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def inject_corrupt_manifest(target_manifest_path: str) -> str:
    """注入清单损坏：写入语法损坏的 JSON，加载须报 E_MANIFEST。"""
    backup = target_manifest_path + ".pristine"
    _ensure_writable(target_manifest_path)
    if os.path.exists(backup):
        _ensure_writable(backup)
    shutil_copy(target_manifest_path, backup)
    with open(target_manifest_path, "w", encoding="utf-8") as handle:
        handle.write('{"version": "broken", "stages": [ ,,, ]')
    return target_manifest_path


def _ensure_writable(path: str) -> None:
    if os.path.exists(path):
        os.chmod(path, 0o644)


def inject_cache_mismatch(cache_index_path: str) -> str:
    """注入缓存索引摘要不匹配：篡改条目内容，触发 E_SNAPSHOT/重放差异。"""
    if not os.path.isfile(cache_index_path):
        raise LayerBenchError(E_SNAPSHOT, "缓存索引文件不存在，无法注入",
                              cache_index_path)
    _ensure_writable(cache_index_path)
    raw = _read_json(cache_index_path)
    entries = raw.get("entries", [])
    if entries:
        entries[0]["refcount"] = entries[0].get("refcount", 0) + 999999
    else:
        raw["entries"] = [{"layer_digest": "injected-ghost",
                           "precursor": "injected-ghost"}]
    with open(cache_index_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(digest.canonical_json(raw) + "\n")
    return cache_index_path


def inject_zero_slots(manifest_path: str) -> str:
    """注入并行槽位上限为零：仅改参数，强校验须拒绝（E_MANIFEST/E_BUDGET）。"""
    _ensure_writable(manifest_path)
    raw = _read_json(manifest_path)
    raw["global_slots"] = 0
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(digest.canonical_json(raw) + "\n")
    return manifest_path


def shutil_copy(src: str, dst: str) -> None:
    import shutil
    shutil.copy2(src, dst)


def assert_injection_rejected(kind: str, manifest_path: str,
                              workspace: str) -> dict:
    """执行注入并验证对应业务码，返回观察结果（供 CLI/测试调用）。"""
    if kind == "corrupt-manifest":
        inject_corrupt_manifest(manifest_path)
        try:
            load_manifest(manifest_path, workspace)
        except ManifestError as exc:
            return {"injection": kind, "rejected": True, "code": exc.code}
        raise LayerBenchError(E_REPLAY, "损坏清单未被拒绝")
    if kind == "zero-slots":
        inject_zero_slots(manifest_path)
        try:
            load_manifest(manifest_path, workspace)
        except ManifestError as exc:
            return {"injection": kind, "rejected": True, "code": exc.code}
        raise LayerBenchError(E_REPLAY, "零槽位清单未被拒绝")
    raise LayerBenchError(E_BUDGET, f"未知注入类型 {kind}")
