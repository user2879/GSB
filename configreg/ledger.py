"""SQLite 回归账本与会话快照（需求 7、8）。

每个采集批次在同一个数据库事务内提交：差异结论、归因结论、定点误差界、
观测明细、调度/重试记录与进度游标一起落盘，因此强制终止后只会看到
“完整批次”或“完全没有该批次”两种状态，不存在半批次。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canon import canonical_bytes, canonical_digest

LEDGER_FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS replacement_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT NOT NULL,
    old_identity TEXT,
    new_identity TEXT,
    affected_items TEXT NOT NULL,
    affected_nodes TEXT NOT NULL,
    created_at_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY,
    baseline_version INTEGER NOT NULL,
    candidate_version INTEGER NOT NULL,
    precision INTEGER NOT NULL,
    sample_seed INTEGER NOT NULL,
    task_keys TEXT NOT NULL,
    diff_digest TEXT NOT NULL,
    regression_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    committed_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS diff_records (
    batch_id INTEGER NOT NULL,
    item_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (batch_id, item_key)
);
CREATE TABLE IF NOT EXISTS item_verdicts (
    batch_id INTEGER NOT NULL,
    item_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (batch_id, item_key)
);
CREATE TABLE IF NOT EXISTS observations (
    batch_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    item_key TEXT NOT NULL,
    node_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (batch_id, version, item_key, node_id)
);
CREATE TABLE IF NOT EXISTS scheduler_records (
    batch_id INTEGER NOT NULL,
    task_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (batch_id, task_key)
);
CREATE TABLE IF NOT EXISTS snapshot (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class SessionIdentity:
    baseline_version: int
    candidate_version: int
    baseline_digest: str
    candidate_digest: str
    precision: int
    precision_param_version: int
    sample_seed: int
    ledger_format_version: int = LEDGER_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "baseline_digest": self.baseline_digest,
            "candidate_digest": self.candidate_digest,
            "precision": self.precision,
            "precision_param_version": self.precision_param_version,
            "sample_seed": self.sample_seed,
            "ledger_format_version": self.ledger_format_version,
        }


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _put_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
        )

    def get_meta(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row["value"])

    def initialize_identity(self, identity: SessionIdentity) -> None:
        self._put_meta("identity", identity.to_dict())
        self.conn.commit()

    def current_identity(self) -> dict[str, Any] | None:
        return self.get_meta("identity")

    def identity_matches(self, identity: SessionIdentity) -> tuple[bool, list[str]]:
        stored = self.current_identity()
        if stored is None:
            return True, []
        expected = identity.to_dict()
        mismatched_fields = sorted(key for key in expected if stored.get(key) != expected[key])
        return not mismatched_fields, mismatched_fields

    def replace_all(
        self,
        identity: SessionIdentity,
        reason: str,
        affected_items: list[str],
        affected_nodes: list[str],
        now_monotonic_ms: int,
    ) -> dict[str, Any]:
        """版本或摘要不一致时的整体替换：单事务清空全部业务表。"""
        old_identity = self.current_identity()
        with self.conn:
            for table in (
                "batches",
                "diff_records",
                "item_verdicts",
                "observations",
                "scheduler_records",
                "snapshot",
            ):
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute(
                "INSERT INTO replacement_events(reason, old_identity, new_identity, affected_items, "
                "affected_nodes, created_at_monotonic_ms) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    reason,
                    json.dumps(old_identity, ensure_ascii=False, sort_keys=True),
                    json.dumps(identity.to_dict(), ensure_ascii=False, sort_keys=True),
                    json.dumps(affected_items),
                    json.dumps(affected_nodes),
                    now_monotonic_ms,
                ),
            )
            self._put_meta("identity", identity.to_dict())
        return {
            "replaced": True,
            "reason": reason,
            "old_identity": old_identity,
            "new_identity": identity.to_dict(),
            "affected_items": affected_items,
            "affected_nodes": affected_nodes,
        }

    def list_replacements(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM replacement_events ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def append_batch(
        self,
        batch_id: int,
        identity: SessionIdentity,
        task_keys: list[str],
        diff_payload: dict[str, Any],
        regression_payload: dict[str, Any],
        observations: dict[tuple[int, str, str], dict[str, Any]],
        scheduler_records: list[dict[str, Any]],
        committed_at_ms: int,
    ) -> str:
        diff_digest = canonical_digest(diff_payload)
        regression_digest = canonical_digest(regression_payload)
        content_digest = canonical_digest(
            {
                "batch_id": batch_id,
                "task_keys": task_keys,
                "diff": diff_payload,
                "regression": regression_payload,
                "observations": [observations[key] for key in sorted(observations)],
                "scheduler": scheduler_records,
            }
        )
        with self.conn:
            self.conn.execute(
                "INSERT INTO batches(batch_id, baseline_version, candidate_version, precision, sample_seed, "
                "task_keys, diff_digest, regression_digest, content_digest, committed_at_ms) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    identity.baseline_version,
                    identity.candidate_version,
                    identity.precision,
                    identity.sample_seed,
                    json.dumps(task_keys),
                    diff_digest,
                    regression_digest,
                    content_digest,
                    committed_at_ms,
                ),
            )
            self.conn.executemany(
                "INSERT INTO diff_records(batch_id, item_key, payload) VALUES(?, ?, ?)",
                [
                    (batch_id, item_key, json.dumps(payload, ensure_ascii=False, sort_keys=True))
                    for item_key, payload in sorted(diff_payload["item_diffs"].items())
                ],
            )
            self.conn.executemany(
                "INSERT INTO item_verdicts(batch_id, item_key, payload) VALUES(?, ?, ?)",
                [
                    (batch_id, item_key, json.dumps(payload, ensure_ascii=False, sort_keys=True))
                    for item_key, payload in sorted(regression_payload["item_verdicts"].items())
                ],
            )
            self.conn.executemany(
                "INSERT INTO observations(batch_id, version, item_key, node_id, payload) "
                "VALUES(?, ?, ?, ?, ?)",
                [
                    (
                        batch_id,
                        version,
                        item_key,
                        node_id,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    )
                    for (version, item_key, node_id), payload in sorted(observations.items())
                ],
            )
            self.conn.executemany(
                "INSERT INTO scheduler_records(batch_id, task_key, payload) VALUES(?, ?, ?)",
                [
                    (batch_id, record["task_key"], json.dumps(record, ensure_ascii=False, sort_keys=True))
                    for record in scheduler_records
                ],
            )
            self._put_meta(
                "progress_cursor",
                {"last_committed_batch": batch_id, "committed_task_keys": self.committed_task_keys() + task_keys},
            )
        return content_digest

    def committed_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM batches ORDER BY batch_id"))

    def next_batch_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(batch_id), 0) AS max_id FROM batches").fetchone()
        return int(row["max_id"]) + 1

    def committed_task_keys(self) -> list[str]:
        keys: list[str] = []
        for row in self.conn.execute("SELECT task_keys FROM batches ORDER BY batch_id"):
            keys.extend(json.loads(row["task_keys"]))
        return keys

    def save_snapshot(self, snapshot: dict[str, Any]) -> str:
        digest = canonical_digest(snapshot)
        stored = dict(snapshot)
        stored["snapshot_digest"] = digest
        with self.conn:
            self.conn.execute(
                "INSERT INTO snapshot(key, payload) VALUES('session', ?) "
                "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                (json.dumps(stored, ensure_ascii=False, sort_keys=True),),
            )
            self._put_meta("snapshot_digest", digest)
        return digest

    def load_snapshot(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT payload FROM snapshot WHERE key='session'").fetchone()
        return None if row is None else json.loads(row["payload"])

    def load_batch_payload(self, batch_id: int) -> dict[str, Any]:
        batch = self.conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        if batch is None:
            raise KeyError(batch_id)
        diffs = {
            row["item_key"]: json.loads(row["payload"])
            for row in self.conn.execute("SELECT * FROM diff_records WHERE batch_id=? ORDER BY item_key", (batch_id,))
        }
        verdicts = {
            row["item_key"]: json.loads(row["payload"])
            for row in self.conn.execute("SELECT * FROM item_verdicts WHERE batch_id=? ORDER BY item_key", (batch_id,))
        }
        observations = {
            tuple(json.loads(row["payload"])["identity_key"]): json.loads(row["payload"])["observation"]
            for row in self.conn.execute("SELECT * FROM observations WHERE batch_id=?", (batch_id,))
        }
        return {
            "batch": dict(batch),
            "item_diffs": diffs,
            "item_verdicts": verdicts,
            "observations": observations,
        }

    def verify_digests(self) -> list[str]:
        problems = []
        for row in self.committed_batches():
            payload = self.load_batch_payload(row["batch_id"])
            diff_digest = canonical_digest({"item_diffs": payload["item_diffs"]})
            if diff_digest != row["diff_digest"]:
                problems.append(f"batch {row['batch_id']} diff digest mismatch")
        return problems

    def canonical_identity_bytes(self, identity: SessionIdentity) -> bytes:
        return canonical_bytes(identity.to_dict())
