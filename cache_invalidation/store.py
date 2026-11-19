"""SQLite 持久化：条目状态、注释边、继承链与逐轮失效账本。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import STATE_VERSION
from .model import EntryID, Graph, RoundRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS namespaces(
    name TEXT PRIMARY KEY,
    parents_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entries(
    namespace TEXT NOT NULL,
    entry_key INTEGER NOT NULL,
    expire_ms INTEGER NOT NULL,
    digest TEXT NOT NULL,
    digest_algorithm TEXT NOT NULL,
    annotation_version INTEGER NOT NULL,
    alive INTEGER NOT NULL,
    PRIMARY KEY(namespace, entry_key)
);
CREATE TABLE IF NOT EXISTS annotation_edges(
    source_ns TEXT NOT NULL,
    source_key INTEGER NOT NULL,
    target_ns TEXT NOT NULL,
    target_key INTEGER NOT NULL,
    PRIMARY KEY(source_ns, source_key, target_ns, target_key)
);
CREATE TABLE IF NOT EXISTS rounds(
    round_no INTEGER PRIMARY KEY,
    clock_ms INTEGER NOT NULL,
    trigger_reason TEXT NOT NULL,
    explicit_triggers_json TEXT NOT NULL,
    expired_triggers_json TEXT NOT NULL,
    refresh_triggers_json TEXT NOT NULL,
    affected_json TEXT NOT NULL,
    versions_json TEXT NOT NULL,
    format_version INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class EntryDiff:
    entry_id: str
    old_version: int | None
    new_version: int | None
    change: str

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "change": self.change,
        }


@dataclass(frozen=True)
class RestoreReport:
    replaced: bool
    snapshot_format_version: int
    snapshot_state_version: int
    current_state_version: int
    entry_diffs: list[EntryDiff]
    replaced_edges: int
    replaced_namespaces: int

    def to_dict(self) -> dict:
        return {
            "replaced": self.replaced,
            "snapshot_format_version": self.snapshot_format_version,
            "snapshot_state_version": self.snapshot_state_version,
            "current_state_version": self.current_state_version,
            "entry_diffs": [diff.to_dict() for diff in self.entry_diffs],
            "replaced_edges": self.replaced_edges,
            "replaced_namespaces": self.replaced_namespaces,
        }


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(
                str(self.path), check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
        else:
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._put_meta("state_version", str(STATE_VERSION))
        self.conn.commit()

    def _put_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()

    def bootstrap(self, graph: Graph) -> None:
        """新会话建立时写入完整初始条目、注释边与继承链。"""
        with self.conn:
            if self.conn.execute(
                    "SELECT COUNT(*) FROM namespaces").fetchone()[0]:
                return
            for name in sorted(graph.namespaces):
                self.conn.execute(
                    "INSERT INTO namespaces(name, parents_json) VALUES(?, ?)",
                    (name, json.dumps(list(graph.namespace_parents[name]))),
                )
            for entry_id in graph.entry_ids_sorted():
                entry = graph.entries[entry_id]
                self.conn.execute(
                    "INSERT INTO entries(namespace, entry_key, expire_ms, "
                    "digest, digest_algorithm, annotation_version, alive) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (entry_id.namespace, entry_id.key, entry.expire_ms,
                     entry.digest, entry.digest_algorithm,
                     entry.annotation_version, int(entry.alive)),
                )
            for source in graph.entry_ids_sorted():
                for target in sorted(graph.annotation_edges[source]):
                    self.conn.execute(
                        "INSERT INTO annotation_edges(source_ns, source_key, "
                        "target_ns, target_key) VALUES(?, ?, ?, ?)",
                        (source.namespace, source.key,
                         target.namespace, target.key),
                    )

    def replace_all(self, graph: Graph, rounds: list[RoundRecord],
                    format_version: int) -> RestoreReport:
        """整体替换：先删除全部业务表，再按快照重建，绝不做增量合并。"""
        # bootstrap 已把初始图写入 SQLite；先采集旧状态，再整表删除重建。
        old_versions = {
            row_id: version for row_id, version in
            self.load_annotation_versions().items()
        }
        old_edges = self.load_edge_count()
        old_ns = self.load_namespace_count()
        snapshot_state = int(self.get_meta("state_version") or STATE_VERSION)

        with self.conn:
            for table in ("rounds", "annotation_edges", "entries",
                          "namespaces"):
                self.conn.execute(f"DELETE FROM {table}")
            for name in sorted(graph.namespaces):
                self.conn.execute(
                    "INSERT INTO namespaces(name, parents_json) VALUES(?, ?)",
                    (name, json.dumps(list(graph.namespace_parents[name]))),
                )
            for entry_id in graph.entry_ids_sorted():
                entry = graph.entries[entry_id]
                self.conn.execute(
                    "INSERT INTO entries(namespace, entry_key, expire_ms, "
                    "digest, digest_algorithm, annotation_version, alive) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (entry_id.namespace, entry_id.key, entry.expire_ms,
                     entry.digest, entry.digest_algorithm,
                     entry.annotation_version, int(entry.alive)),
                )
            for source in graph.entry_ids_sorted():
                for target in sorted(graph.annotation_edges[source]):
                    self.conn.execute(
                        "INSERT INTO annotation_edges(source_ns, source_key, "
                        "target_ns, target_key) VALUES(?, ?, ?, ?)",
                        (source.namespace, source.key,
                         target.namespace, target.key),
                    )
            for record in rounds:
                self._insert_round(record)
            self._put_meta("snapshot_format_version", str(format_version))
            self._put_meta("state_version", str(STATE_VERSION))

        new_versions = {
            entry.id.to_str(): entry.annotation_version
            for entry in graph.entries.values()
        }
        return RestoreReport(
            replaced=True,
            snapshot_format_version=format_version,
            snapshot_state_version=snapshot_state,
            current_state_version=STATE_VERSION,
            entry_diffs=diff_versions(old_versions, new_versions),
            replaced_edges=sum(
                len(targets) for targets in graph.annotation_edges.values()),
            replaced_namespaces=len(graph.namespaces),
        )

    def _insert_round(self, record: RoundRecord) -> None:
        self.conn.execute(
            "INSERT INTO rounds(round_no, clock_ms, trigger_reason, "
            "explicit_triggers_json, expired_triggers_json, "
            "refresh_triggers_json, affected_json, versions_json, "
            "format_version) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.round_no,
                record.clock_ms,
                record.trigger_reason,
                json.dumps(list(record.explicit_triggers)),
                json.dumps(list(record.expired_triggers)),
                json.dumps(list(record.refresh_triggers)),
                json.dumps(list(record.affected)),
                json.dumps(record.annotation_versions, sort_keys=True),
                record.format_version,
            ),
        )

    def append_round(self, record: RoundRecord) -> None:
        with self.conn:
            self._insert_round(record)

    def load_rounds(self) -> list[RoundRecord]:
        rows = self.conn.execute(
            "SELECT round_no, clock_ms, trigger_reason, "
            "explicit_triggers_json, expired_triggers_json, "
            "refresh_triggers_json, affected_json, versions_json, "
            "format_version FROM rounds ORDER BY round_no"
        ).fetchall()
        records = []
        for row in rows:
            records.append(RoundRecord(
                round_no=row[0],
                clock_ms=row[1],
                trigger_reason=row[2],
                explicit_triggers=tuple(json.loads(row[3])),
                expired_triggers=tuple(json.loads(row[4])),
                refresh_triggers=tuple(json.loads(row[5])),
                affected=tuple(json.loads(row[6])),
                annotation_versions=json.loads(row[7]),
                format_version=row[8],
            ))
        return records

    def load_annotation_versions(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT namespace, entry_key, annotation_version FROM entries"
        ).fetchall()
        return {
            EntryID(row[0], row[1]).to_str(): row[2] for row in rows
        }

    def load_edge_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM annotation_edges").fetchone()[0]

    def load_namespace_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM namespaces").fetchone()[0]

    def update_entry_state(self, entry_id: EntryID,
                           annotation_version: int, alive: bool) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE entries SET annotation_version=?, alive=? "
                "WHERE namespace=? AND entry_key=?",
                (annotation_version, int(alive),
                 entry_id.namespace, entry_id.key),
            )


def diff_versions(old: dict[str, int],
                  new: dict[str, int]) -> list[EntryDiff]:
    diffs: list[EntryDiff] = []
    for entry_id in sorted(set(old) | set(new)):
        old_value = old.get(entry_id)
        new_value = new.get(entry_id)
        if old_value == new_value:
            continue
        if old_value is None:
            change = "added"
        elif new_value is None:
            change = "removed"
        else:
            change = "version_changed"
        diffs.append(EntryDiff(entry_id, old_value, new_value, change))
    return diffs
