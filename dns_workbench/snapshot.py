"""缓存账本快照：SQLite 持久化与恢复重放校验。

库内表：
- meta：快照级元数据（格式版本集合、区版本、代数、账本内容摘要）；
- entries：逐条来源序列号与内容摘要；
- hits：命中历史，每条携带命中时账本摘要。
恢复时比较格式版本集合与逐条摘要，不一致返回差异并交由账本整体替换。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SnapshotError
from .ledger import CacheLedger
from .records import canonical_digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    snapshot_id INTEGER PRIMARY KEY,
    created_at_ms INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    zone_version INTEGER NOT NULL,
    format_versions TEXT NOT NULL,
    ledger_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entries (
    snapshot_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    qtype INTEGER NOT NULL,
    source_version TEXT NOT NULL,
    seq INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, name, qtype)
);
CREATE TABLE IF NOT EXISTS hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    hit_at_ms INTEGER NOT NULL,
    name TEXT NOT NULL,
    qtype INTEGER NOT NULL,
    remaining_ttl_s INTEGER NOT NULL,
    ledger_digest TEXT NOT NULL
);
"""


@dataclass
class HitEvent:
    hit_at_ms: int
    name: str
    qtype: int
    remaining_ttl_s: int


@dataclass
class RestoreReport:
    snapshot_id: int
    matched: bool
    format_versions_changed: bool
    diff: dict = field(default_factory=dict)


class SnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save(self, ledger: CacheLedger, *, created_at_ms: int, hits: list[HitEvent] | None = None) -> int:
        generation = ledger.current()
        snapshot = generation.cache.export_snapshot()
        ledger_digest = self._digest_snapshot(snapshot.items, generation.format_versions, generation.zone_version)
        cur = self._conn.execute(
            "INSERT INTO meta(created_at_ms, generation, zone_version, format_versions, ledger_digest) VALUES (?,?,?,?,?)",
            (created_at_ms, generation.generation, generation.zone_version, json.dumps(generation.format_versions), ledger_digest),
        )
        snapshot_id = int(cur.lastrowid)
        rows = []
        for item in snapshot.items:
            name, qtype = item["key"]
            rows.append(
                (
                    snapshot_id,
                    name,
                    qtype,
                    item["source_version"],
                    item["entry"].get("seq", 0),
                    item["content_digest"],
                )
            )
        self._conn.executemany(
            "INSERT INTO entries(snapshot_id, name, qtype, source_version, seq, content_digest) VALUES (?,?,?,?,?,?)",
            rows,
        )
        for event in hits or []:
            self._conn.execute(
                "INSERT INTO hits(snapshot_id, hit_at_ms, name, qtype, remaining_ttl_s, ledger_digest) VALUES (?,?,?,?,?,?)",
                (snapshot_id, event.hit_at_ms, event.name, event.qtype, event.remaining_ttl_s, ledger_digest),
            )
        self._conn.commit()
        return snapshot_id

    def list_snapshots(self) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT snapshot_id, created_at_ms, generation, zone_version, format_versions, ledger_digest FROM meta ORDER BY snapshot_id"
        )
        return [
            {
                "snapshot_id": row[0],
                "created_at_ms": row[1],
                "generation": row[2],
                "zone_version": row[3],
                "format_versions": json.loads(row[4]),
                "ledger_digest": row[5],
            }
            for row in cursor.fetchall()
        ]

    def load(self, snapshot_id: int) -> dict:
        meta = self._conn.execute(
            "SELECT snapshot_id, created_at_ms, generation, zone_version, format_versions, ledger_digest FROM meta WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if meta is None:
            raise SnapshotError("快照不存在", context={"snapshot_id": snapshot_id})
        entries = self._conn.execute(
            "SELECT name, qtype, source_version, seq, content_digest FROM entries WHERE snapshot_id=? ORDER BY name, qtype",
            (snapshot_id,),
        ).fetchall()
        hits = self._conn.execute(
            "SELECT hit_at_ms, name, qtype, remaining_ttl_s, ledger_digest FROM hits WHERE snapshot_id=? ORDER BY id",
            (snapshot_id,),
        ).fetchall()
        return {
            "snapshot_id": meta[0],
            "created_at_ms": meta[1],
            "generation": meta[2],
            "zone_version": meta[3],
            "format_versions": tuple(json.loads(meta[4])),
            "ledger_digest": meta[5],
            "entries": [
                {"name": row[0], "qtype": row[1], "source_version": row[2], "seq": row[3], "content_digest": row[4]}
                for row in entries
            ],
            "hits": [
                {"hit_at_ms": row[0], "name": row[1], "qtype": row[2], "remaining_ttl_s": row[3], "ledger_digest": row[4]}
                for row in hits
            ],
        }

    def verify_against(self, snapshot_id: int, ledger: CacheLedger) -> RestoreReport:
        saved = self.load(snapshot_id)
        generation = ledger.current()
        live_snapshot = generation.cache.export_snapshot()
        live_versions = generation.format_versions
        format_changed = tuple(sorted(saved["format_versions"])) != tuple(sorted(live_versions))
        saved_map = {(item["name"], item["qtype"]): item["content_digest"] for item in saved["entries"]}
        live_map = {tuple(item["key"]): item["content_digest"] for item in live_snapshot.items}
        added = sorted(live_map.keys() - saved_map.keys())
        removed = sorted(saved_map.keys() - live_map.keys())
        digest_changed = [
            {"key": list(key), "saved_digest": saved_map[key], "live_digest": live_map[key]}
            for key in sorted(saved_map.keys() & live_map.keys())
            if saved_map[key] != live_map[key]
        ]
        matched = not format_changed and not added and not removed and not digest_changed
        return RestoreReport(
            snapshot_id=snapshot_id,
            matched=matched,
            format_versions_changed=format_changed,
            diff={"added": [list(k) for k in added], "removed": [list(k) for k in removed], "digest_changed": digest_changed},
        )

    @staticmethod
    def _digest_snapshot(items: list[dict], format_versions: tuple[str, ...], zone_version: int) -> str:
        payload = {
            "format_versions": list(format_versions),
            "zone_version": zone_version,
            "items": [
                {"key": item["key"], "source_version": item["source_version"], "content_digest": item["content_digest"]}
                for item in items
            ],
        }
        return canonical_digest(payload)
