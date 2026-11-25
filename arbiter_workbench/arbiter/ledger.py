"""SQLite arbitration ledger with atomic transactions and snapshot replacement.

Schema contract (integer FORMAT_VERSION stored in ``meta``):

* meta(format_version, session_scale, content_digest)
* view_commit(view_id, version, status, content_digest, committed_at_ms)
* key_record(view_id, version, key, policy, authority_q, adopted_envs,
             rejected_envs, error_quant, error_boundary, error_norm,
             error_total, attribution_json, evidence_json, replay_status)

Each view arbitration is written in one transaction. Snapshot recovery is
whole-state replacement: a mismatched version/digest never triggers an
incremental merge. WAL mode gives waiting readers a consistent view.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import ErrorContext, ReplayError, StateError

FORMAT_VERSION = 1
UNKNOWN = "UNKNOWN"
COMMITTED = "COMMITTED"


@dataclass(frozen=True)
class KeyRecord:
    view_id: str
    version: int
    key: str
    policy: str
    authority_q: int
    adopted_envs: tuple[str, ...]
    rejected_envs: tuple[str, ...]
    error_quant: int
    error_boundary: int
    error_norm: int
    error_total: int
    attribution: dict
    evidence: dict
    replay_status: str = COMMITTED


def _digest_payload(payload: dict | list) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class Ledger:
    def __init__(self, db_path: str | Path, session_scale: int):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_scale = session_scale
        self.conn = sqlite3.connect(str(self.db_path), timeout=15, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                format_version INTEGER NOT NULL,
                session_scale INTEGER NOT NULL,
                content_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS view_commit (
                view_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                committed_at_ms INTEGER NOT NULL,
                PRIMARY KEY (view_id, version)
            );
            CREATE TABLE IF NOT EXISTS key_record (
                view_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                key TEXT NOT NULL,
                policy TEXT NOT NULL,
                authority_q INTEGER NOT NULL,
                adopted_envs TEXT NOT NULL,
                rejected_envs TEXT NOT NULL,
                error_quant INTEGER NOT NULL,
                error_boundary INTEGER NOT NULL,
                error_norm INTEGER NOT NULL,
                error_total INTEGER NOT NULL,
                attribution_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                replay_status TEXT NOT NULL,
                PRIMARY KEY (view_id, version, key)
            );
            """
        )
        row = self.conn.execute("SELECT format_version, session_scale FROM meta WHERE id=1").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(id, format_version, session_scale, content_digest) "
                "VALUES (1, ?, ?, ?)",
                (FORMAT_VERSION, self.session_scale, _digest_payload({"v": FORMAT_VERSION})),
            )
        elif row["format_version"] != FORMAT_VERSION or row["session_scale"] != self.session_scale:
            raise StateError(
                "ledger format/scale mismatch requires snapshot replacement",
                ErrorContext(str(self.db_path), "meta",
                             f"disk=({row['format_version']},{row['session_scale']}) "
                             f"want=({FORMAT_VERSION},{self.session_scale})"),
            )

    # ---- write path ------------------------------------------------------
    def commit_view(
        self,
        view_id: str,
        version: int,
        records: list[KeyRecord],
    ) -> str:
        if not records:
            raise StateError("refusing to commit empty view arbitration",
                             ErrorContext(view_id))
        payload = [asdict(r) for r in sorted(records, key=lambda r: r.key)]
        digest = _digest_payload(payload)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO view_commit VALUES (?, ?, ?, ?, ?)",
                (view_id, version, COMMITTED, digest, int(time.time() * 1000)),
            )
            self.conn.execute(
                "DELETE FROM key_record WHERE view_id=? AND version=?",
                (view_id, version),
            )
            for r in payload:
                self.conn.execute(
                    "INSERT INTO key_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        r["view_id"], r["version"], r["key"], r["policy"],
                        r["authority_q"],
                        json.dumps(r["adopted_envs"], separators=(",", ":")),
                        json.dumps(r["rejected_envs"], separators=(",", ":")),
                        r["error_quant"], r["error_boundary"], r["error_norm"],
                        r["error_total"],
                        json.dumps(r["attribution"], sort_keys=True, separators=(",", ":")),
                        json.dumps(r["evidence"], sort_keys=True, separators=(",", ":")),
                        r["replay_status"],
                    ),
                )
            self._refresh_meta_digest_locked()
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return digest

    def begin_unknown(self, view_id: str, version: int) -> None:
        """Marker row written outside the view's commit transaction.

        On crash recovery, any view whose latest state is UNKNOWN must be
        recomputed; no exit report is fabricated.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO view_commit VALUES (?, ?, ?, ?, ?)",
            (view_id, version, UNKNOWN, "", int(time.time() * 1000)),
        )

    def _refresh_meta_digest_locked(self) -> None:
        rows = self.conn.execute(
            "SELECT view_id, version, status, content_digest FROM view_commit "
            "ORDER BY view_id, version"
        ).fetchall()
        digest = _digest_payload([dict(r) for r in rows])
        self.conn.execute("UPDATE meta SET content_digest=? WHERE id=1", (digest,))

    # ---- read path -------------------------------------------------------
    def meta_digest(self) -> str:
        return self.conn.execute(
            "SELECT content_digest FROM meta WHERE id=1"
        ).fetchone()["content_digest"]

    def latest_versions(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT vc.* FROM view_commit vc JOIN ("
            " SELECT view_id, MAX(version) AS mv FROM view_commit GROUP BY view_id"
            ") m ON m.view_id=vc.view_id AND m.mv=vc.version"
        ).fetchall()
        return {r["view_id"]: dict(r) for r in rows}

    def load_records(self, view_id: str, version: int) -> list[KeyRecord]:
        rows = self.conn.execute(
            "SELECT * FROM key_record WHERE view_id=? AND version=? ORDER BY key",
            (view_id, version),
        ).fetchall()
        return [
            KeyRecord(
                view_id=r["view_id"], version=r["version"], key=r["key"],
                policy=r["policy"], authority_q=r["authority_q"],
                adopted_envs=tuple(json.loads(r["adopted_envs"])),
                rejected_envs=tuple(json.loads(r["rejected_envs"])),
                error_quant=r["error_quant"], error_boundary=r["error_boundary"],
                error_norm=r["error_norm"], error_total=r["error_total"],
                attribution=json.loads(r["attribution_json"]),
                evidence=json.loads(r["evidence_json"]),
                replay_status=r["replay_status"],
            )
            for r in rows
        ]

    def open_snapshot(self, snapshot_path: str | Path) -> dict:
        data = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        if data.get("format_version") != FORMAT_VERSION:
            raise StateError(
                "snapshot format version mismatch; whole snapshot replacement required",
                ErrorContext(str(snapshot_path), "format_version"),
            )
        return data

    def replace_with_snapshot(self, snapshot_path: str | Path) -> str:
        """Whole-state replacement; never an incremental merge."""
        data = self.open_snapshot(snapshot_path)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("DELETE FROM key_record")
            self.conn.execute("DELETE FROM view_commit")
            for commit in sorted(data["commits"], key=lambda c: (c["view_id"], c["version"])):
                self.conn.execute(
                    "INSERT INTO view_commit VALUES (?,?,?,?,?)",
                    (commit["view_id"], commit["version"], commit["status"],
                     commit["content_digest"], commit["committed_at_ms"]),
                )
            for r in data["records"]:
                self.conn.execute(
                    "INSERT INTO key_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        r["view_id"], r["version"], r["key"], r["policy"],
                        r["authority_q"],
                        json.dumps(r["adopted_envs"], separators=(",", ":")),
                        json.dumps(r["rejected_envs"], separators=(",", ":")),
                        r["error_quant"], r["error_boundary"], r["error_norm"],
                        r["error_total"],
                        json.dumps(r["attribution"], sort_keys=True, separators=(",", ":")),
                        json.dumps(r["evidence"], sort_keys=True, separators=(",", ":")),
                        r["replay_status"],
                    ),
                )
            self._refresh_meta_digest_locked()
            want_digest = self.conn.execute(
                "SELECT content_digest FROM meta WHERE id=1"
            ).fetchone()[0]
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        if want_digest != data.get("content_digest"):
            raise StateError(
                "snapshot content digest mismatch after replacement",
                ErrorContext(str(snapshot_path), "content_digest",
                             f"rebuilt={want_digest} snapshot={data.get('content_digest')}"),
            )
        return want_digest

    def export_snapshot(self, snapshot_path: str | Path) -> str:
        commits = [
            dict(r) for r in self.conn.execute(
                "SELECT * FROM view_commit ORDER BY view_id, version"
            ).fetchall()
        ]
        records_rows = self.conn.execute(
            "SELECT * FROM key_record ORDER BY view_id, version, key"
        ).fetchall()
        records = []
        for r in records_rows:
            d = dict(r)
            d["adopted_envs"] = json.loads(d["adopted_envs"])
            d["rejected_envs"] = json.loads(d["rejected_envs"])
            d["attribution"] = json.loads(d.pop("attribution_json"))
            d["evidence"] = json.loads(d.pop("evidence_json"))
            records.append(d)
        digest = self.meta_digest()
        payload = {
            "format_version": FORMAT_VERSION,
            "session_scale": self.session_scale,
            "content_digest": digest,
            "commits": commits,
            "records": records,
        }
        Path(snapshot_path).write_text(
            json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
        )
        return digest

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def verify_replay(view_id: str, stored: list[KeyRecord], replayed: list[KeyRecord]) -> None:
    stored_by_key = {r.key: r for r in stored}
    replayed_by_key = {r.key: r for r in replayed}
    for key in sorted(set(stored_by_key) | set(replayed_by_key)):
        a, b = stored_by_key.get(key), replayed_by_key.get(key)
        if a is None or b is None:
            raise ReplayError(
                "replay key set mismatch",
                ErrorContext(view_id, key, "key missing in one side"),
            )
        for field_name in (
            "authority_q", "adopted_envs", "rejected_envs",
            "error_quant", "error_boundary", "error_norm", "error_total",
            "attribution", "policy",
        ):
            if getattr(a, field_name) != getattr(b, field_name):
                raise ReplayError(
                    "replay field mismatch",
                    ErrorContext(view_id, field_name, key),
                )
