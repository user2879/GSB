from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .canonical import canonical_json, sha256_value
from .errors import LabError
from .receipts import decode_receipt


LEDGER_FORMAT_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    batch_index INTEGER NOT NULL UNIQUE,
    rule_id TEXT NOT NULL,
    destination TEXT NOT NULL,
    raw_duration_ms INTEGER NOT NULL,
    raw_sample_count INTEGER NOT NULL,
    scale_exp INTEGER NOT NULL,
    duration_q INTEGER NOT NULL,
    samples_q INTEGER NOT NULL,
    output_digest TEXT NOT NULL,
    backend_param_digest TEXT,
    backend_param_version INTEGER,
    receipt_count INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    batch_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    format TEXT NOT NULL,
    merge_seq INTEGER NOT NULL,
    destination TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    client_bucket INTEGER,
    weight_bp INTEGER,
    duration_ms INTEGER,
    sample_ids_json TEXT NOT NULL,
    backend_param_digest TEXT,
    backend_param_version INTEGER,
    extras_json TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_receipts_batch ON receipts(batch_id, merge_seq);
CREATE TABLE IF NOT EXISTS cursor (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    completed_batches INTEGER NOT NULL,
    next_batch_index INTEGER NOT NULL,
    cursor_json TEXT NOT NULL
);
"""


def _ledger_error(message: str, **details: Any) -> LabError:
    return LabError("E_LEDGER", message, details=details)


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(
                str(self.path),
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self.conn.row_factory = sqlite3.Row
            self.thread_lock = threading.RLock()
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO cursor(singleton, completed_batches, next_batch_index, cursor_json) "
                "VALUES(1, 0, 0, ?)",
                (canonical_json({"completed_batches": 0, "next_batch_index": 0}),),
            )
            self._verify_format_version()
            with self.transaction():
                self._refresh_metadata()
        except sqlite3.Error as exc:
            raise _ledger_error(f"cannot open ledger: {exc}", path=str(self.path)) from exc

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def _verify_format_version(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='ledger_format_version'"
        ).fetchone()
        if row is None:
            raise _ledger_error("ledger_format_version is missing")
        try:
            version = int(row["value"])
        except ValueError as exc:
            raise _ledger_error("ledger_format_version is not an integer") from exc
        if version != LEDGER_FORMAT_VERSION:
            raise _ledger_error(
                "unsupported ledger format version",
                expected=LEDGER_FORMAT_VERSION,
                actual=version,
            )

    @staticmethod
    def _batch_digest(record: dict[str, Any]) -> str:
        clean = {key: value for key, value in record.items() if key != "batch_digest"}
        return sha256_value(
            {
                "ledger_format_version": LEDGER_FORMAT_VERSION,
                "record": clean,
                "receipts": sorted(
                    record.get("receipts", []),
                    key=lambda item: decode_receipt(item).identity_digest(),
                ),
            }
        )

    def _ledger_digest(self) -> str:
        rows = self.conn.execute(
            "SELECT batch_id, batch_digest FROM batches ORDER BY batch_index, batch_id"
        ).fetchall()
        return sha256_value(
            {
                "ledger_format_version": LEDGER_FORMAT_VERSION,
                "batches": [
                    {"batch_id": row["batch_id"], "batch_digest": row["batch_digest"]}
                    for row in rows
                ],
            }
        )

    def _refresh_metadata(self) -> None:
        rows = self.conn.execute(
            "SELECT batch_id, batch_index FROM batches ORDER BY batch_index, batch_id"
        ).fetchall()
        completed = len(rows)
        next_index = max((int(row["batch_index"]) for row in rows), default=-1) + 1
        cursor = {
            "completed_batches": completed,
            "next_batch_index": next_index,
            "batch_ids": [row["batch_id"] for row in rows],
        }
        self.conn.execute(
            "UPDATE cursor SET completed_batches=?, next_batch_index=?, cursor_json=? WHERE singleton=1",
            (completed, next_index, canonical_json(cursor)),
        )
        ledger_digest = self._ledger_digest()
        meta = {
            "ledger_format_version": str(LEDGER_FORMAT_VERSION),
            "content_digest": ledger_digest,
            "completed_batches": str(completed),
            "next_batch_index": str(next_index),
        }
        for key, value in meta.items():
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def record_batch(
        self,
        record: dict[str, Any],
        receipts: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        batch_id = record.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise _ledger_error("batch record lacks batch_id")
        required = {
            "rule_id",
            "destination",
            "raw_duration_ms",
            "raw_sample_count",
            "scale_exp",
            "duration_q",
            "samples_q",
            "output_digest",
            "record",
        }
        missing = sorted(required - set(record))
        if missing:
            raise _ledger_error("batch record is incomplete", field=missing[0], batch_id=batch_id)
        receipt_values = [dict(item) for item in receipts]
        decoded = [decode_receipt(item) for item in receipt_values]
        if not decoded:
            raise _ledger_error("batch record requires at least one receipt", batch_id=batch_id)
        if any(item.batch_id != batch_id for item in decoded):
            raise _ledger_error("receipt batch_id does not match record", batch_id=batch_id)

        stored_record = dict(record)
        stored_record["receipts"] = receipt_values
        stored_record["ledger_format_version"] = LEDGER_FORMAT_VERSION

        try:
            with self.transaction():
                scale_row = self.conn.execute(
                    "SELECT value FROM meta WHERE key='scale_exp'"
                ).fetchone()
                batch_scale = int(record["scale_exp"])
                if scale_row is None:
                    self.conn.execute(
                        "INSERT INTO meta(key, value) VALUES('scale_exp', ?)", (str(batch_scale),)
                    )
                elif int(scale_row["value"]) != batch_scale:
                    raise _ledger_error(
                        "batch scale_exp differs from the ledger session scale",
                        batch_id=batch_id,
                        details={"ledger_scale_exp": int(scale_row["value"]), "batch_scale_exp": batch_scale},
                    )
                existing = self.conn.execute(
                    "SELECT batch_index FROM batches WHERE batch_id=?", (batch_id,)
                ).fetchone()
                if existing is None:
                    cursor_row = self.conn.execute(
                        "SELECT next_batch_index FROM cursor WHERE singleton=1"
                    ).fetchone()
                    batch_index = int(cursor_row["next_batch_index"])
                else:
                    batch_index = int(existing["batch_index"])
                stored_record["batch_index"] = batch_index
                batch_digest = self._batch_digest(stored_record)
                stored_record["batch_digest"] = batch_digest

                self.conn.execute(
                    "INSERT INTO batches("
                    "batch_id,batch_index,rule_id,destination,raw_duration_ms,raw_sample_count,"
                    "scale_exp,duration_q,samples_q,output_digest,backend_param_digest,"
                    "backend_param_version,receipt_count,record_json,batch_digest"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(batch_id) DO UPDATE SET "
                    "rule_id=excluded.rule_id,destination=excluded.destination,"
                    "raw_duration_ms=excluded.raw_duration_ms,raw_sample_count=excluded.raw_sample_count,"
                    "scale_exp=excluded.scale_exp,duration_q=excluded.duration_q,samples_q=excluded.samples_q,"
                    "output_digest=excluded.output_digest,backend_param_digest=excluded.backend_param_digest,"
                    "backend_param_version=excluded.backend_param_version,"
                    "receipt_count=excluded.receipt_count,record_json=excluded.record_json,"
                    "batch_digest=excluded.batch_digest",
                    (
                        batch_id,
                        batch_index,
                        record["rule_id"],
                        record["destination"],
                        record["raw_duration_ms"],
                        record["raw_sample_count"],
                        record["scale_exp"],
                        record["duration_q"],
                        record["samples_q"],
                        record["output_digest"],
                        record.get("backend_param_digest"),
                        record.get("backend_param_version"),
                        len(receipt_values),
                        canonical_json(stored_record),
                        batch_digest,
                    ),
                )

                self.conn.execute("DELETE FROM receipts WHERE batch_id=?", (batch_id,))
                for raw, receipt in zip(receipt_values, decoded):
                    self.conn.execute(
                        "INSERT INTO receipts("
                        "receipt_id,batch_id,format,merge_seq,destination,output_digest,"
                        "client_bucket,weight_bp,duration_ms,sample_ids_json,"
                        "backend_param_digest,backend_param_version,extras_json,raw_json"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(receipt_id) DO UPDATE SET "
                        "format=excluded.format,merge_seq=excluded.merge_seq,"
                        "destination=excluded.destination,output_digest=excluded.output_digest,"
                        "client_bucket=excluded.client_bucket,weight_bp=excluded.weight_bp,"
                        "duration_ms=excluded.duration_ms,sample_ids_json=excluded.sample_ids_json,"
                        "backend_param_digest=excluded.backend_param_digest,"
                        "backend_param_version=excluded.backend_param_version,"
                        "extras_json=excluded.extras_json,raw_json=excluded.raw_json",
                        (
                            receipt.identity_digest(),
                            batch_id,
                            receipt.format,
                            receipt.effective_merge_seq,
                            receipt.destination,
                            receipt.output_digest,
                            receipt.client_bucket,
                            receipt.weight_bp,
                            receipt.duration_ms,
                            canonical_json(list(receipt.sample_ids)),
                            receipt.backend_param_digest,
                            receipt.backend_param_version,
                            canonical_json(receipt.extras),
                            canonical_json(raw),
                        ),
                    )

                self._refresh_metadata()
        except sqlite3.Error as exc:
            raise _ledger_error(f"batch transaction failed: {exc}", batch_id=batch_id) from exc
        return stored_record

    def batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT record_json FROM batches ORDER BY batch_index, batch_id"
        ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def batch(self, batch_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT record_json FROM batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        return json.loads(row["record_json"]) if row else None

    def receipt_rows(self, batch_id: str | None = None) -> list[dict[str, Any]]:
        if batch_id is None:
            rows = self.conn.execute(
                "SELECT raw_json FROM receipts ORDER BY batch_id, merge_seq, receipt_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT raw_json FROM receipts WHERE batch_id=? ORDER BY merge_seq, receipt_id",
                (batch_id,),
            ).fetchall()
        return [json.loads(row["raw_json"]) for row in rows]

    def cursor(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM cursor WHERE singleton=1").fetchone()
        value = json.loads(row["cursor_json"])
        value["completed_batches"] = int(row["completed_batches"])
        value["next_batch_index"] = int(row["next_batch_index"])
        return value

    def meta(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def content_digest(self) -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key='content_digest'").fetchone()
        if row is None:
            raise _ledger_error("content_digest is missing")
        return row["value"]

    def verify_integrity(self) -> dict[str, Any]:
        self._verify_format_version()
        stored = self.content_digest()
        computed = self._ledger_digest()
        if stored != computed:
            raise _ledger_error(
                "ledger content digest mismatch",
                details={"stored": stored, "computed": computed},
            )
        for row in self.conn.execute(
            "SELECT batch_id, record_json, batch_digest FROM batches ORDER BY batch_index"
        ):
            record = json.loads(row["record_json"])
            computed_batch = self._batch_digest(record)
            if computed_batch != row["batch_digest"] or record.get("batch_digest") != row["batch_digest"]:
                raise _ledger_error(
                    "batch digest mismatch",
                    batch_id=row["batch_id"],
                    details={"stored": row["batch_digest"], "computed": computed_batch},
                )
            table_rows = self.conn.execute(
                "SELECT raw_json FROM receipts WHERE batch_id=? ORDER BY receipt_id",
                (row["batch_id"],),
            ).fetchall()
            table_receipts = sorted(
                (json.loads(item["raw_json"]) for item in table_rows),
                key=lambda item: decode_receipt(item).identity_digest(),
            )
            record_receipts = sorted(
                record.get("receipts", []),
                key=lambda item: decode_receipt(item).identity_digest(),
            )
            if table_receipts != record_receipts:
                raise _ledger_error(
                    "receipts table differs from batch record",
                    batch_id=row["batch_id"],
                    details={"table_count": len(table_receipts), "record_count": len(record_receipts)},
                )
        return {
            "ok": True,
            "format_version": LEDGER_FORMAT_VERSION,
            "content_digest": stored,
            "completed_batches": self.cursor()["completed_batches"],
        }
