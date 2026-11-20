"""路由会话快照的持久化与整体替换恢复。

快照携带整数格式版本 :data:`SNAPSHOT_FORMAT_VERSION` 与内容摘要；重新打开时
比较快照内版本摘要与当前磁盘配置推导出的版本，任一不一致即整体替换配置、
缓存与依赖图状态（绝不增量合并），并以 E_VERSION 记录差异。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from .determinism import content_digest
from .errors import VersionError
from .versioning import VersionState

SNAPSHOT_FORMAT_VERSION = 1


@dataclass
class Snapshot:
    format_version: int
    versions: dict[str, Any]
    cache_digest: str
    history_digest: str
    ledger_cursor: int | None
    config_path: str
    units_path: str | None
    content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "versions": self.versions,
            "cache_digest": self.cache_digest,
            "history_digest": self.history_digest,
            "ledger_cursor": self.ledger_cursor,
            "config_path": self.config_path,
            "units_path": self.units_path,
            "content_digest": self.content_digest,
        }


def build_snapshot(
    *,
    versions: VersionState,
    cache_digest: str,
    history_digest: str,
    ledger_cursor: int | None,
    config_path: str,
    units_path: str | None,
) -> Snapshot:
    body = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "versions": versions.as_dict(),
        "cache_digest": cache_digest,
        "history_digest": history_digest,
        "ledger_cursor": ledger_cursor,
        "config_path": config_path,
        "units_path": units_path,
    }
    return Snapshot(
        format_version=SNAPSHOT_FORMAT_VERSION,
        versions=versions.as_dict(),
        cache_digest=cache_digest,
        history_digest=history_digest,
        ledger_cursor=ledger_cursor,
        config_path=config_path,
        units_path=units_path,
        content_digest=content_digest(body),
    )


def save_snapshot(path: str, snapshot: Snapshot) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = snapshot.to_dict()
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_snapshot(path: str) -> Snapshot:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    format_version = payload.get("format_version")
    if not isinstance(format_version, int) or format_version != SNAPSHOT_FORMAT_VERSION:
        raise VersionError(
            f"快照格式版本不支持: {format_version!r}，期望 {SNAPSHOT_FORMAT_VERSION}",
            paths=[path],
            details={"actual": format_version},
        )
    stored_digest = payload.get("content_digest")
    body = {key: value for key, value in payload.items() if key != "content_digest"}
    if stored_digest != content_digest(body):
        raise VersionError(
            "快照内容摘要校验失败，文件可能已损坏",
            paths=[path],
        )
    return Snapshot(
        format_version=format_version,
        versions=payload["versions"],
        cache_digest=payload["cache_digest"],
        history_digest=payload["history_digest"],
        ledger_cursor=payload["ledger_cursor"],
        config_path=payload["config_path"],
        units_path=payload["units_path"],
        content_digest=stored_digest,
    )


def compare_snapshot_versions(
    snapshot: Snapshot, current: VersionState
) -> dict[str, dict[str, Any]]:
    """逐类比较快照版本与当前磁盘版本，返回全部不一致项。"""
    mismatches: dict[str, dict[str, Any]] = {}
    current_dict = current.as_dict()
    for kind in ("config", "ring", "dependency", "budget"):
        snapshot_digest = snapshot.versions.get(f"{kind}_digest")
        current_digest = current_dict[f"{kind}_digest"]
        if snapshot_digest != current_digest:
            mismatches[kind] = {
                "snapshot_version": snapshot.versions.get(f"{kind}_version"),
                "current_version": current_dict[f"{kind}_version"],
                "snapshot_digest": snapshot_digest,
                "current_digest": current_digest,
            }
    return mismatches


def assert_snapshot_matches(
    snapshot: Snapshot, current: VersionState
) -> dict[str, dict[str, Any]]:
    mismatches = compare_snapshot_versions(snapshot, current)
    if mismatches:
        raise VersionError(
            "快照版本与当前磁盘配置不一致，须整体替换缓存与依赖图状态",
            paths=["snapshot.versions"],
            details={"mismatches": mismatches},
        )
    return mismatches
