"""SQLite lease ledger - the single persistence backend.

Schema versioning is explicit (``FORMAT_VERSION``).  Each dispatched batch is
committed in *one* transaction containing the batch row, all receipts, quota
consumption and the progress cursor, so a hard kill can never leave a
half-committed batch.  On reopen, batches lacking a COMMITTED row are reported
as UNKNOWN and must be rerun; the ledger never fabricates an exit report.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .content import canonical_bytes, fnv1a_32
from .logging_setup import JsonlLogger
from .models import PlanItem, Receipt

FORMAT_VERSION = 1


def content_digest(payload: Any) -> str:
    return f"sha-fnv64:{fnv1a_32(canonical_bytes(payload)):08x}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_mono_ms INTEGER NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    version INTEGER PRIMARY KEY,
    manifest_digest TEXT NOT NULL,
    quant_version TEXT NOT NULL,
    scale INTEGER NOT NULL,
    content_seed INTEGER NOT NULL,
    plan_digest TEXT NOT NULL,
    items_json TEXT NOT NULL,
    error_rows_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    state TEXT NOT NULL,
    committed INTEGER NOT NULL DEFAULT 0,
    total_items INTEGER NOT NULL,
    ok_items INTEGER NOT NULL,
    quota_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    committed_ms INTEGER
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    receipt_json TEXT NOT NULL,
    UNIQUE(batch_id, item_id, attempt)
);
CREATE TABLE IF NOT EXISTS progress (
    batch_id TEXT PRIMARY KEY,
    cursor_item TEXT NOT NULL,
    done_items_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_mono_ms INTEGER NOT NULL,
    code TEXT NOT NULL,
    location TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: Path, logger: Optional[JsonlLogger] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._init_meta()

    def _init_meta(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(FORMAT_VERSION),),
            )
        elif int(row["value"]) != FORMAT_VERSION:
            from .errors import SnapshotError

            raise SnapshotError(
                f"unsupported snapshot format version {row['value']}",
                "meta.format_version",
                {"expected": FORMAT_VERSION, "actual": int(row["value"])},
            )

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, sort_keys=True)),
        )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def record_state(
        self, ts_ms: int, from_state: str, to_state: str, reason: str, detail: Any
    ) -> None:
        self.conn.execute(
            "INSERT INTO state_events(ts_mono_ms, from_state, to_state, reason, detail)"
            " VALUES(?, ?, ?, ?, ?)",
            (ts_ms, from_state, to_state, reason, json.dumps(detail, sort_keys=True)),
        )

    def state_history(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM state_events ORDER BY seq ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def save_plan(
        self,
        version: int,
        manifest_digest: str,
        quant_version: str,
        scale: int,
        content_seed: int,
        items: List[PlanItem],
        error_rows: List[Dict[str, int]],
    ) -> str:
        items_payload = [item.to_dict() for item in items]
        plan_digest = content_digest(items_payload)
        self.conn.execute(
            "INSERT OR REPLACE INTO plans(version, manifest_digest, quant_version,"
            " scale, content_seed, plan_digest, items_json, error_rows_json)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version,
                manifest_digest,
                quant_version,
                scale,
                content_seed,
                plan_digest,
                json.dumps(items_payload, sort_keys=True),
                json.dumps(error_rows, sort_keys=True),
            ),
        )
        return plan_digest

    def load_plan(self, version: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM plans WHERE version=?", (version,)
        ).fetchone()
        if row is None:
            return None
        doc = dict(row)
        doc["items"] = json.loads(doc.pop("items_json"))
        doc["error_rows"] = json.loads(doc.pop("error_rows_json"))
        return doc

    def begin_batch(
        self, batch_id: str, version: int, total_items: int, created_ms: int
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO batches(batch_id, version, state, committed,"
            " total_items, ok_items, quota_json, created_ms, committed_ms)"
            " VALUES(?, ?, 'UNKNOWN', 0, ?, 0, '{}', ?, NULL)",
            (batch_id, version, total_items, created_ms),
        )

    def commit_batch(
        self,
        batch_id: str,
        receipts: List[Receipt],
        quota: Dict[str, int],
        cursor_item: str,
        done_items: List[str],
        ok_items: int,
        committed_ms: int,
    ) -> None:
        """Atomic batch commit: rows + receipts + quota + cursor in one tx."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE batches SET state='COMMITTED', committed=1, ok_items=?,"
                " quota_json=?, committed_ms=? WHERE batch_id=?",
                (ok_items, json.dumps(quota, sort_keys=True), committed_ms, batch_id),
            )
            conn.execute("DELETE FROM receipts WHERE batch_id=?", (batch_id,))
            for receipt in receipts:
                conn.execute(
                    "INSERT INTO receipts(batch_id, item_id, attempt, receipt_json)"
                    " VALUES(?, ?, ?, ?)",
                    (
                        batch_id,
                        receipt.item_id,
                        receipt.attempt,
                        json.dumps(receipt.to_dict(), sort_keys=True),
                    ),
                )
            conn.execute(
                "INSERT OR REPLACE INTO progress(batch_id, cursor_item, done_items_json)"
                " VALUES(?, ?, ?)",
                (batch_id, cursor_item, json.dumps(done_items, sort_keys=True)),
            )

    def unknown_batches(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM batches WHERE committed=0 ORDER BY created_ms ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def committed_batches(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM batches WHERE committed=1 ORDER BY committed_ms ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def receipts_for(self, batch_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT receipt_json FROM receipts WHERE batch_id=? ORDER BY id ASC",
            (batch_id,),
        ).fetchall()
        return [json.loads(row["receipt_json"]) for row in rows]

    def all_committed_receipts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT r.receipt_json FROM receipts r JOIN batches b"
            " ON r.batch_id=b.batch_id WHERE b.committed=1 ORDER BY r.id ASC"
        ).fetchall()
        return [json.loads(row["receipt_json"]) for row in rows]

    def mark_batch_rerun(self, batch_id: str, note: str) -> None:
        self.conn.execute(
            "UPDATE batches SET state='RERUN_REQUIRED' WHERE batch_id=?",
            (batch_id,),
        )
        if self.logger is not None:
            self.logger.warning("batch_rerun_required", batch_id=batch_id, note=note)

    def record_error(self, ts_ms: int, code: str, location: str, detail: Any) -> None:
        self.conn.execute(
            "INSERT INTO errors(ts_mono_ms, code, location, detail_json)"
            " VALUES(?, ?, ?, ?)",
            (ts_ms, code, location, json.dumps(detail, sort_keys=True)),
        )

    def replace_all_runtime_state(self) -> None:
        """Whole-state replacement used when snapshot identity mismatches."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM receipts")
            conn.execute("DELETE FROM progress")
            conn.execute("DELETE FROM batches")
            conn.execute("DELETE FROM state_events")
            conn.execute("DELETE FROM plans")

    def snapshot_identity(self) -> Dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "session": self.get_meta("session_identity"),
            "plan_versions": [
                row["version"]
                for row in self.conn.execute(
                    "SELECT version FROM plans ORDER BY version ASC"
                ).fetchall()
            ],
        }
