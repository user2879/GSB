"""恢复会话快照的 SQLite 持久化、版本识别与整体替换(要求 7)。"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .buffer import DoubleWriteBuffer
from .errors import SnapshotMismatchError
from .paging import MainDataFile
from .util import StructuredLogger, canonical_json, digest_of_json
from .versions import VersionMap, PageVersion
from .views import ViewManager

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_seq     INTEGER NOT NULL,
    buffer_digest   TEXT NOT NULL,
    main_digest     TEXT NOT NULL,
    version_map     TEXT NOT NULL,
    view_versions   TEXT NOT NULL,
    repair_trace    TEXT NOT NULL,
    invalidation    TEXT NOT NULL,
    created_ts_ms   INTEGER NOT NULL
);
"""


@dataclass
class Snapshot:
    session_id: str
    created_seq: int
    buffer_digest: str
    main_digest: str
    version_map: dict[str, Any]
    view_versions: dict[str, int]
    repair_trace: list[dict[str, Any]]
    invalidation: list[dict[str, Any]]
    created_ts_ms: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Snapshot":
        return cls(
            session_id=row["session_id"],
            created_seq=int(row["created_seq"]),
            buffer_digest=row["buffer_digest"],
            main_digest=row["main_digest"],
            version_map=json.loads(row["version_map"]),
            view_versions=json.loads(row["view_versions"]),
            repair_trace=json.loads(row["repair_trace"]),
            invalidation=json.loads(row["invalidation"]),
            created_ts_ms=int(row["created_ts_ms"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_seq": self.created_seq,
            "buffer_digest": self.buffer_digest,
            "main_digest": self.main_digest,
            "version_map": self.version_map,
            "view_versions": self.view_versions,
            "repair_trace": self.repair_trace,
            "invalidation": self.invalidation,
            "created_ts_ms": self.created_ts_ms,
        }


class SnapshotStore:
    def __init__(self, db_path: Path, logger: StructuredLogger) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def save(
        self,
        session_id: str,
        created_seq: int,
        buffer_digest: str,
        main: MainDataFile,
        versions: VersionMap,
        view_manager: ViewManager,
        repair_trace: list[dict[str, Any]],
        invalidation: list[dict[str, Any]],
        created_ts_ms: int,
    ) -> Snapshot:
        main_digest = digest_of_json(main.iter_digests())
        snap = Snapshot(
            session_id=session_id,
            created_seq=created_seq,
            buffer_digest=buffer_digest,
            main_digest=main_digest,
            version_map=versions.as_dict(),
            view_versions={
                name: view_manager.states[name].version
                for name in view_manager.catalog.names_sorted()
            },
            repair_trace=repair_trace,
            invalidation=invalidation,
            created_ts_ms=created_ts_ms,
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, created_seq, buffer_digest, main_digest, version_map,
                    view_versions, repair_trace, invalidation, created_ts_ms)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    snap.session_id,
                    snap.created_seq,
                    snap.buffer_digest,
                    snap.main_digest,
                    canonical_json(snap.version_map),
                    canonical_json(snap.view_versions),
                    canonical_json(snap.repair_trace),
                    canonical_json(snap.invalidation),
                    snap.created_ts_ms,
                ),
            )
        self.logger.log("snapshot_saved", detail={"session_id": session_id})
        return snap

    def load(self, session_id: str) -> Snapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return Snapshot.from_row(row) if row is not None else None

    def latest(self) -> Snapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions ORDER BY created_seq DESC, session_id ASC LIMIT 1"
            ).fetchone()
        return Snapshot.from_row(row) if row is not None else None

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, created_seq, buffer_digest, main_digest, created_ts_ms "
                "FROM sessions ORDER BY created_seq DESC, session_id ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 版本识别与整体替换 ----------
    def reconcile(
        self,
        snapshot: Snapshot,
        dwb: DoubleWriteBuffer,
        main: MainDataFile,
        versions: VersionMap,
        view_manager: ViewManager,
    ) -> dict[str, Any]:
        """比较快照摘要与当前磁盘; 不一致整体替换内存状态, 旧快照备份为只读。"""
        current_buffer = dwb.file_digest_hex()
        current_main_digests = main.iter_digests()
        current_main = digest_of_json(current_main_digests)

        buffer_match = current_buffer == snapshot.buffer_digest
        main_match = current_main == snapshot.main_digest
        if buffer_match and main_match:
            self._apply_in_memory(snapshot, versions, view_manager)
            self.logger.log(
                "snapshot_match", detail={"session_id": snapshot.session_id}
            )
            return {
                "status": "match",
                "changed_pages": [],
                "changed_views": [],
                "backup": None,
            }

        # E_SNAPSHOT: 快照与磁盘不一致 -> 整体替换而非增量合并
        changed_pages = self._diff_pages(snapshot, current_main_digests)
        changed_views = self._diff_views(snapshot, view_manager)
        backup_path = self._backup_readonly(snapshot.session_id)
        self.logger.log(
            "snapshot_mismatch",
            code="E_SNAPSHOT",
            detail={
                "session_id": snapshot.session_id,
                "buffer_match": buffer_match,
                "main_match": main_match,
                "changed_pages": changed_pages,
                "changed_views": changed_views,
                "backup": str(backup_path),
            },
        )
        # 以当前磁盘为真值重建内存: 页缓存/版本映射/视图状态整体替换
        self._replace_from_disk(snapshot, main, versions, view_manager)
        raise SnapshotMismatchError(
            "snapshot digests differ from current disk; state fully replaced",
            detail={
                "session_id": snapshot.session_id,
                "changed_pages": changed_pages,
                "changed_views": changed_views,
                "backup": str(backup_path),
            },
        )

    def _diff_pages(
        self, snapshot: Snapshot, current_digests: dict[int, str]
    ) -> list[dict[str, Any]]:
        snap_pages = snapshot.version_map
        changes: list[dict[str, Any]] = []
        all_keys = sorted(
            {int(k) for k in snap_pages} | set(current_digests),
        )
        for page_no in all_keys:
            old = snap_pages.get(str(page_no), {}).get("digest")
            new = current_digests.get(page_no)
            if old != new:
                changes.append(
                    {"page_no": page_no, "snapshot_digest": old, "disk_digest": new}
                )
        return changes

    def _diff_views(
        self, snapshot: Snapshot, view_manager: ViewManager
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for name in view_manager.catalog.names_sorted():
            old = snapshot.view_versions.get(name, 0)
            new = view_manager.states[name].version
            if old != new:
                changes.append(
                    {"view": name, "snapshot_version": old, "disk_version": new}
                )
        return changes

    def _backup_readonly(self, session_id: str) -> Path:
        backup = self.db_path.with_name(f"{self.db_path.name}.{session_id}.bak")
        shutil.copy2(self.db_path, backup)
        # 应用层只读备份: 组件不会再向该文件写入; 不强行 chmod 0o444,
        # 否则 SQLite 重新打开时无法创建 journal/wal 而读取失败(Windows 尤甚)。
        self.logger.log(
            "snapshot_backup_readonly",
            detail={"session_id": session_id, "backup": str(backup)},
        )
        return backup

    def _apply_in_memory(
        self, snapshot: Snapshot, versions: VersionMap, view_manager: ViewManager
    ) -> None:
        pages = {
            int(k): PageVersion(
                int(k), int(v["seq"]), str(v["digest"])
            )
            for k, v in snapshot.version_map.items()
        }
        versions.replace_all(pages)
        view_manager.replace_states(snapshot.view_versions)

    def _replace_from_disk(
        self,
        snapshot: Snapshot,
        main: MainDataFile,
        versions: VersionMap,
        view_manager: ViewManager,
    ) -> None:
        # 整体丢弃快照中的页缓存/版本, 以主文件当前页摘要重建版本映射
        pages: dict[int, PageVersion] = {}
        for page_no, digest in main.iter_digests().items():
            prior = versions.get(page_no)
            seq = prior.seq if prior is not None else snapshot.created_seq
            pages[page_no] = PageVersion(page_no, seq, digest)
        versions.replace_all(pages)
        versions.save()
        # 视图状态整体替换并标记待重建
        view_manager.replace_states(
            {name: 0 for name in view_manager.catalog.names_sorted()}
        )
