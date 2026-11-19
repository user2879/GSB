"""会话快照的版本识别、整体替换恢复与条目版本回退检测。

快照为单个 JSON 文件，持久化键空间版本、轨迹游标、键值记录、索引条目与
条目版本、计数视图、重建进度、逐操作结论及内容摘要。打开快照时先重算内容
摘要并比对，同时比较版本；任一不一致即整体替换当前状态（不做增量合并），
并保留旧快照只读、输出按版本排列的差异摘要。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Optional

from . import codec
from .engine import Engine
from .errors import SnapshotError, VersionError

SNAPSHOT_FORMAT = 2
DIGEST_FIELDS = (
    "keyspace_version",
    "trace_cursor",
    "kv_records",
    "index_entries",
    "prefix_counts",
    "rebuild_progress",
    "conclusions",
)


WALL_FIELDS = ("start_wall_ms", "finish_wall_ms", "ts_wall_ms")


def _strip_wall(value: Any) -> Any:
    """墙上时间用于人工观测与日志，不参与内容摘要，保证摘要确定性。"""
    if isinstance(value, dict):
        return {k: _strip_wall(v) for k, v in value.items() if k not in WALL_FIELDS}
    if isinstance(value, list):
        return [_strip_wall(v) for v in value]
    return value


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    material = {field: _strip_wall(payload.get(field)) for field in DIGEST_FIELDS}
    return codec.canonical_json(material)


def compute_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def save_snapshot(
    path: str,
    engine: Engine,
    conclusions: list[dict[str, Any]],
    rebuild_progress: Optional[dict[str, Any]] = None,
    trace_digest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    state = engine.export_state()
    payload: dict[str, Any] = {
        "snapshot_format": SNAPSHOT_FORMAT,
        "keyspace_version": state["keyspace_version"],
        "trace_cursor": state["trace_cursor"],
        "kv_records": state["kv_records"],
        "index_entries": state["index_entries"],
        "prefix_counts": state["prefix_counts"],
        "rebuild_progress": rebuild_progress or {"phase": "IDLE"},
        "conclusions": conclusions,
        "trace_digest": trace_digest or {},
    }
    payload["content_digest"] = compute_digest(payload)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    os.replace(tmp_path, path)
    return {"path": path, "content_digest": payload["content_digest"],
            "keyspace_version": payload["keyspace_version"],
            "trace_cursor": payload["trace_cursor"]}


def read_snapshot(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_version_rollback(index_entries: list[dict[str, Any]]) -> None:
    """检测注入的条目版本回退：同一 (prefix, ref_key) 出现负版本或重复降版。"""
    seen: dict[tuple[str, str], int] = {}
    for entry in index_entries:
        key_pair = (entry["prefix"], entry["ref_key"])
        version = entry["entry_version"]
        if not isinstance(version, int) or version < 0:
            raise VersionError(
                "索引条目版本号非法/回退",
                details={"prefix": entry["prefix"], "ref_key": entry["ref_key"],
                         "entry_version": version, "reason": "bad_version"},
            )
        if key_pair in seen and version < seen[key_pair]:
            raise VersionError(
                "检测到索引条目版本回退",
                details={"prefix": entry["prefix"], "ref_key": entry["ref_key"],
                         "previous": seen[key_pair], "actual": version},
            )
        seen[key_pair] = version


def open_snapshot(
    path: str,
    engine: Engine,
    *,
    expected_digest: Optional[str] = None,
    mismatch_digest: Optional[str] = None,
    readonly_copy_dir: Optional[str] = None,
) -> dict[str, Any]:
    """打开快照并恢复。

    * 内容摘要不匹配（含注入的 ``mismatch_digest`` 参数）=> E_SNAPSHOT。
    * 摘要正确但条目版本回退 => E_VERSION（在整体替换前拦截）。
    * 版本与当前会话不一致 => 整体替换，不增量合并，旧状态只读留档并给差异摘要。
    """
    payload = read_snapshot(path)
    stored_digest = payload.get("content_digest")
    actual_digest = compute_digest(payload)
    if mismatch_digest is not None:
        stored_digest = mismatch_digest
        expected_digest = mismatch_digest
    if stored_digest != actual_digest or (
        expected_digest is not None and expected_digest != actual_digest
    ):
        raise SnapshotError(
            "快照内容摘要不匹配，拒绝恢复",
            details={
                "path": path,
                "stored_digest": stored_digest,
                "actual_digest": actual_digest,
                "reason": "digest_mismatch",
            },
        )
    if payload.get("snapshot_format") != SNAPSHOT_FORMAT:
        raise SnapshotError(
            "快照格式版本不支持",
            details={"path": path, "actual": payload.get("snapshot_format"),
                     "expected": SNAPSHOT_FORMAT},
        )

    detect_version_rollback(payload["index_entries"])

    current = engine.export_state()
    version_differs = current["keyspace_version"] != payload["keyspace_version"]
    digest_differs = compute_digest(
        {field: payload.get(field) for field in DIGEST_FIELDS}
    ) != compute_digest(
        {
            "keyspace_version": current["keyspace_version"],
            "trace_cursor": current["trace_cursor"],
            "kv_records": current["kv_records"],
            "index_entries": current["index_entries"],
            "prefix_counts": current["prefix_counts"],
            "rebuild_progress": {"phase": "IDLE"},
            "conclusions": payload.get("conclusions", []),
        }
    )

    diff_summary = _diff_summary(current, payload)
    replaced = False
    readonly_copy = None
    if version_differs or digest_differs:
        # 旧快照只读留档（这里留档的是被替换前的当前会话 DB 状态摘要）
        if readonly_copy_dir:
            os.makedirs(readonly_copy_dir, exist_ok=True)
            import time

            readonly_copy = os.path.join(
                readonly_copy_dir,
                f"prev_session_v{current['keyspace_version']}_"
                f"{int(time.time() * 1000)}.json",
            )
            if not os.path.exists(readonly_copy):
                with open(readonly_copy, "w", encoding="utf-8") as handle:
                    json.dump(
                        {"keyspace_version": current["keyspace_version"],
                         "trace_cursor": current["trace_cursor"],
                         "state_digest": hashlib.sha256(
                             codec.canonical_json(current)
                         ).hexdigest(),
                         "diff_summary": diff_summary},
                        handle, ensure_ascii=False, sort_keys=True, indent=2,
                    )
                os.chmod(readonly_copy, 0o444)
        state = {
            "keyspace_version": payload["keyspace_version"],
            "trace_cursor": payload["trace_cursor"],
            "kv_records": payload["kv_records"],
            "index_entries": payload["index_entries"],
            "prefix_counts": payload["prefix_counts"],
        }
        engine.replace_all(state)
        replaced = True

    return {
        "path": path,
        "replaced": replaced,
        "content_digest": actual_digest,
        "keyspace_version": payload["keyspace_version"],
        "trace_cursor": payload["trace_cursor"],
        "diff_summary": diff_summary,
        "readonly_copy": readonly_copy,
        "rebuild_progress": payload.get("rebuild_progress"),
        "conclusions": payload.get("conclusions", []),
        "trace_digest": payload.get("trace_digest", {}),
    }


def _diff_summary(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    current_counts = current["all_counts"] if "all_counts" in current else {
        item["prefix"]: item["cnt"] for item in current.get("prefix_counts", [])
    }
    incoming_counts = {item["prefix"]: item["cnt"] for item in incoming["prefix_counts"]}
    all_prefixes = sorted(set(current_counts) | set(incoming_counts))
    return {
        "version": {
            "current": current["keyspace_version"],
            "snapshot": incoming["keyspace_version"],
        },
        "trace_cursor": {
            "current": current["trace_cursor"],
            "snapshot": incoming["trace_cursor"],
        },
        "kv_record_count": {
            "current": len(current["kv_records"]),
            "snapshot": len(incoming["kv_records"]),
        },
        "index_entry_count": {
            "current": len(current["index_entries"]),
            "snapshot": len(incoming["index_entries"]),
        },
        "prefix_count_deltas": [
            {"prefix": prefix,
             "current": current_counts.get(prefix, 0),
             "snapshot": incoming_counts.get(prefix, 0)}
            for prefix in all_prefixes
            if current_counts.get(prefix, 0) != incoming_counts.get(prefix, 0)
        ],
    }
