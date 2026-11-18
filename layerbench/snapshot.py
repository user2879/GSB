"""构建会话快照：版本识别、整体替换恢复与强制终止恢复。

快照内容（自包含、只读归档）
----------------------------
- manifest.json：触发构建时的清单完整副本；
- cache/cache_index.json：构建结束时刻的缓存索引副本；
- ledger/ledger.db：账本副本（含完整提交记录与 running 残留）；
- meta.json：清单版本、缓存索引摘要、账本进度游标、已完成指令集合、
  预算参数与整体内容摘要 content_digest。

恢复规则（显式）
----------------
1. 重开快照先校验 meta.content_digest，损坏 -> E_SNAPSHOT。
2. 比较快照清单版本/指纹与当前磁盘清单、快照缓存索引摘要与当前缓存：
   任一不一致 -> 整体替换当前 manifest/cache/ledger（先把当前状态归档到
   ``replaced_at_<时间戳>`` 只读目录），绝不增量合并。
3. 强制终止恢复：仅承认 instructions 中完整提交的记录；running 表里残留
   （执行中未提交）的指令标记 unknown 并要求重跑，不承诺生成退出报告。
"""
from __future__ import annotations

import json
import os
import shutil
import stat

from . import digest
from .cache import INDEX_FILENAME
from .errors import E_SNAPSHOT, LayerBenchError, STATUS_UNKNOWN
from .ledger import LEDGER_FILENAME

META_FILENAME = "meta.json"


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_fingerprint(manifest) -> str:
    """清单指纹：版本 + 规范化后的完整结构（与文件字节解耦，稳定序）。"""
    payload = {
        "version": manifest.version,
        "global_slots": manifest.global_slots,
        "default_timeout_ms": manifest.default_timeout_ms,
        "cache_capacity": manifest.cache_capacity,
        "stages": [
            {
                "name": s.name,
                "depends_on": list(s.depends_on),
                "quota": s.quota,
                "instructions": [
                    {
                        "seq": i.seq, "id": i.id, "type": i.type,
                        "params": i.params, "after": list(i.after),
                    }
                    for i in s.instructions
                ],
            }
            for s in [manifest.stage_by_name[n] for n in manifest.topo_order]
        ],
    }
    return digest.stable_digest_of_mapping(payload)


def create_snapshot(snapshot_root: str, snapshot_id: str, manifest, ledger,
                    cache, build_id: str, completed_uids: set[str],
                    progress_cursor: int,
                    baseline_precursors: set[str] | None = None) -> str:
    """在 snapshot_root/snapshot_id 下生成自包含只读快照。"""
    snap_dir = os.path.abspath(os.path.join(snapshot_root, snapshot_id))
    os.makedirs(snap_dir, exist_ok=True)
    cache_dir = os.path.join(snap_dir, "cache")
    ledger_dir = os.path.join(snap_dir, "ledger")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ledger_dir, exist_ok=True)

    # 先落盘缓存与账本副本，再写 manifest，最后写带摘要的 meta
    cache.save()
    ledger.checkpoint()  # WAL 合并回主库，确保单文件副本完整
    shutil.copy2(cache.path, os.path.join(cache_dir, INDEX_FILENAME))
    shutil.copy2(ledger.path, os.path.join(ledger_dir, LEDGER_FILENAME))
    for suffix in ("-wal", "-shm"):
        stale = os.path.join(ledger_dir, LEDGER_FILENAME + suffix)
        if os.path.exists(stale):
            os.remove(stale)
    shutil.copy2(manifest.source_path, os.path.join(snap_dir, "manifest.json"))

    from .cache import CacheIndex
    cache_fp = CacheIndex.fingerprint_file(cache_dir)
    man_fp = manifest_fingerprint(manifest)
    meta = {
        "snapshot_version": 1,
        "snapshot_id": snapshot_id,
        "build_id": build_id,
        "manifest_version": manifest.version,
        "manifest_fingerprint": man_fp,
        "cache_index_fingerprint": cache_fp,
        "budget": {
            "global_slots": manifest.global_slots,
            "cache_capacity": manifest.cache_capacity,
            "stage_quota": {s.name: s.quota for s in manifest.stages},
        },
        "progress_cursor": progress_cursor,
        "completed_node_uids": sorted(completed_uids),
        "baseline_precursors": sorted(baseline_precursors or set()),
    }
    meta["content_digest"] = digest.stable_digest_of_mapping(
        {k: v for k, v in meta.items() if k != "content_digest"}
    )
    meta_path = os.path.join(snap_dir, META_FILENAME)
    with open(meta_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(digest.canonical_json(meta) + "\n")
    _make_readonly(snap_dir)
    return snap_dir


def _make_readonly(directory: str) -> None:
    for root, _dirs, files in os.walk(directory):
        for name in files:
            os.chmod(os.path.join(root, name), stat.S_IREAD | stat.S_IRGRP)


def load_snapshot_meta(snap_dir: str) -> dict:
    meta_path = os.path.join(snap_dir, META_FILENAME)
    if not os.path.isfile(meta_path):
        raise LayerBenchError(E_SNAPSHOT, "快照缺少 meta.json", snap_dir)
    meta = _read_json(meta_path)
    stored = meta.get("content_digest")
    recomputed = digest.stable_digest_of_mapping(
        {k: v for k, v in meta.items() if k != "content_digest"}
    )
    if stored != recomputed:
        raise LayerBenchError(
            E_SNAPSHOT, "快照内容摘要不匹配，快照已损坏", snap_dir,
            {"stored": stored, "recomputed": recomputed},
        )
    return meta


def restore_snapshot(snap_dir: str, workdir: str, current_manifest_path: str,
                     cache_dir: str, ledger_path: str,
                     current_manifest_fp: str,
                     current_cache_fp: str) -> dict:
    """比较版本/摘要，不一致则整体替换当前清单、缓存与账本。

    返回恢复决策说明 dict；只读快照本身永不被修改。
    """
    meta = load_snapshot_meta(snap_dir)
    mismatch = []
    if meta["manifest_fingerprint"] != current_manifest_fp:
        mismatch.append("manifest")
    if meta["cache_index_fingerprint"] != current_cache_fp:
        mismatch.append("cache_index")

    replaced = False
    archive_dir = None
    if mismatch:
        # 整体替换前，把当前工作区状态归档为只读，杜绝增量合并
        archive_dir = os.path.join(
            workdir, "snapshots",
            "replaced_at_" + str(digest.sha256_hex(snap_dir)[:12]),
        )
        os.makedirs(archive_dir, exist_ok=True)
        if os.path.isfile(current_manifest_path):
            shutil.copy2(current_manifest_path,
                         os.path.join(archive_dir, "manifest.json"))
        if os.path.isfile(os.path.join(cache_dir, INDEX_FILENAME)):
            os.makedirs(os.path.join(archive_dir, "cache"), exist_ok=True)
            shutil.copy2(os.path.join(cache_dir, INDEX_FILENAME),
                         os.path.join(archive_dir, "cache", INDEX_FILENAME))
        if os.path.isfile(ledger_path):
            os.makedirs(os.path.join(archive_dir, "ledger"), exist_ok=True)
            shutil.copy2(ledger_path,
                         os.path.join(archive_dir, "ledger", LEDGER_FILENAME))
        _make_readonly(archive_dir)

        # 用快照副本整体覆盖当前活动状态
        shutil.copy2(os.path.join(snap_dir, "manifest.json"),
                     current_manifest_path)
        os.makedirs(cache_dir, exist_ok=True)
        shutil.copy2(os.path.join(snap_dir, "cache", INDEX_FILENAME),
                     os.path.join(cache_dir, INDEX_FILENAME))
        shutil.copy2(os.path.join(snap_dir, "ledger", LEDGER_FILENAME),
                     ledger_path)
        replaced = True

    return {
        "snapshot_dir": snap_dir,
        "mismatch": mismatch,
        "replaced": replaced,
        "archive_dir": archive_dir,
        "meta": meta,
    }


def detect_in_flight(ledger, build_id: str, dag_uids: set[str]) -> dict:
    """强制终止恢复：区分已完整提交与执行中未提交(unknown)指令。"""
    completed = ledger.completed_node_uids(build_id)
    running_rows = ledger.running_nodes(build_id)
    in_flight = sorted({row["node_uid"] for row in running_rows}
                       | (dag_uids - completed))
    # running 表残留是权威的「未提交」信号；dag 差集兜底
    authoritative = sorted({row["node_uid"] for row in running_rows})
    unknown = sorted(set(authoritative) | set(in_flight))
    return {
        "completed": sorted(completed),
        "unknown": unknown,
        "policy": (
            "仅依据完整提交记录恢复；unknown 指令标记 "
            + STATUS_UNKNOWN + " 并要求重跑，不生成退出报告"
        ),
    }
