from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_text, content_digest


LEDGER_FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    seq INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    request_ids_json TEXT NOT NULL,
                    planned_at_ms INTEGER NOT NULL,
                    committed_at_ms INTEGER,
                    batch_digest TEXT
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    request_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    canonical_json TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    PRIMARY KEY(request_id, node_id)
                );
                CREATE TABLE IF NOT EXISTS conclusions (
                    request_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    canonical_json TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attributions (
                    request_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    canonical_json TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scheduler_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cursor_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_batch_id TEXT,
                    last_batch_seq INTEGER NOT NULL DEFAULT 0,
                    last_request_id TEXT,
                    updated_at_ms INTEGER NOT NULL
                );
                """
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO cursor_state(id, updated_at_ms) VALUES(1, ?)",
                (int(time.time() * 1000),),
            )
        stored_version = int(self.get_meta("ledger_format_version", "0"))
        if stored_version != LEDGER_FORMAT_VERSION:
            raise ValueError(f"unsupported ledger format version {stored_version}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

    def configure_versions(self, *, hop_set_version: int,
                           scale_caliber_version: int) -> None:
        self.set_meta("hop_set_version", hop_set_version)
        self.set_meta("scale_caliber_version", scale_caliber_version)

    def plan_batch(self, batch_id: str, seq: int, request_ids: list[str]) -> None:
        now_ms = int(time.time() * 1000)
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO batches"
                "(batch_id, seq, status, request_ids_json, planned_at_ms) "
                "VALUES(?, ?, 'planned', ?, ?)",
                (batch_id, seq, canonical_text(request_ids), now_ms),
            )

    def mark_unknown_batches(self) -> list[str]:
        unknown: list[str] = []
        rows = self.conn.execute(
            "SELECT batch_id FROM batches WHERE status != 'committed' ORDER BY seq, batch_id"
        ).fetchall()
        with self.conn:
            for row in rows:
                self.conn.execute(
                    "UPDATE batches SET status='unknown' WHERE batch_id=? AND status != 'committed'",
                    (row["batch_id"],),
                )
                unknown.append(row["batch_id"])
        return unknown

    def unknown_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM batches WHERE status != 'committed' ORDER BY seq, batch_id"
        ).fetchall()
        return [
            {
                "batch_id": row["batch_id"],
                "seq": row["seq"],
                "status": row["status"],
                "request_ids": json.loads(row["request_ids_json"]),
            }
            for row in rows
        ]

    def committed_batch_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT batch_id FROM batches WHERE status='committed'").fetchall()
        return {row["batch_id"] for row in rows}

    def commit_batch(
        self,
        *,
        batch_id: str,
        seq: int,
        request_ids: list[str],
        receipts_by_request: dict[str, list[dict[str, Any]]],
        conclusions: dict[str, dict[str, Any]],
        attributions: dict[str, dict[str, Any]],
        events_by_request: dict[str, list[dict[str, Any]]],
        hop_set_version: int,
        scale_caliber_version: int,
        session_scale: int | None = None,
        last_request_id: str | None = None,
    ) -> str:
        batch_payload = {
            "batch_id": batch_id,
            "seq": seq,
            "request_ids": request_ids,
            "conclusion_digests": {
                request_id: content_digest(conclusions[request_id])
                for request_id in sorted(conclusions)
            },
        }
        batch_digest = content_digest(batch_payload)
        now_ms = int(time.time() * 1000)

        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('hop_set_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(hop_set_version),),
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('scale_caliber_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(scale_caliber_version),),
            )
            if session_scale is None and conclusions:
                first_conclusion = next(iter(conclusions.values()))
                session_scale = first_conclusion.get("session_scale")
            if session_scale is not None:
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('session_scale',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(session_scale),),
                )
            for request_id in request_ids:
                for seq_index, receipt in enumerate(receipts_by_request.get(request_id, [])):
                    canonical = canonical_text(receipt)
                    digest = content_digest(receipt)
                    self.conn.execute(
                        "INSERT INTO receipts(request_id,node_id,batch_id,seq,"
                        "canonical_json,digest) VALUES(?,?,?,?,?,?) "
                        "ON CONFLICT(request_id,node_id) DO UPDATE SET "
                        "batch_id=excluded.batch_id,seq=excluded.seq,"
                        "canonical_json=excluded.canonical_json,digest=excluded.digest",
                        (request_id, receipt["node_id"], batch_id, seq_index,
                         canonical, digest),
                    )

                conclusion = conclusions[request_id]
                conclusion_canonical = canonical_text(conclusion)
                self.conn.execute(
                    "INSERT INTO conclusions(request_id,batch_id,canonical_json,digest) "
                    "VALUES(?,?,?,?) ON CONFLICT(request_id) DO UPDATE SET "
                    "batch_id=excluded.batch_id,canonical_json=excluded.canonical_json,"
                    "digest=excluded.digest",
                    (request_id, batch_id, conclusion_canonical,
                     content_digest(conclusion)),
                )

                attribution = attributions[request_id]
                attribution_canonical = canonical_text(attribution)
                self.conn.execute(
                    "INSERT INTO attributions(request_id,batch_id,canonical_json,digest) "
                    "VALUES(?,?,?,?) ON CONFLICT(request_id) DO UPDATE SET "
                    "batch_id=excluded.batch_id,canonical_json=excluded.canonical_json,"
                    "digest=excluded.digest",
                    (request_id, batch_id, attribution_canonical,
                     content_digest(attribution)),
                )

            self.conn.execute("DELETE FROM scheduler_events WHERE batch_id=?", (batch_id,))
            for request_id in request_ids:
                for event in events_by_request.get(request_id, []):
                    self.conn.execute(
                        "INSERT INTO scheduler_events(batch_id,request_id,event_json) "
                        "VALUES(?,?,?)",
                        (batch_id, request_id, canonical_text(event)),
                    )

            self.conn.execute(
                "INSERT INTO batches(batch_id,seq,status,request_ids_json,"
                "planned_at_ms,committed_at_ms,batch_digest) "
                "VALUES(?,?, 'committed', ?, ?, ?, ?) "
                "ON CONFLICT(batch_id) DO UPDATE SET status='committed',"
                "request_ids_json=excluded.request_ids_json,"
                "committed_at_ms=excluded.committed_at_ms,"
                "batch_digest=excluded.batch_digest",
                (batch_id, seq, canonical_text(request_ids), now_ms, now_ms, batch_digest),
            )
            self.conn.execute(
                "UPDATE cursor_state SET last_batch_id=?, last_batch_seq=?, "
                "last_request_id=?, updated_at_ms=? WHERE id=1",
                (batch_id, seq, last_request_id or (request_ids[-1] if request_ids else None),
                 now_ms),
            )
            digest = self._compute_content_digest_locked()
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('content_digest',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (digest,),
            )
        return batch_digest

    def _digest_rows(self) -> dict[str, Any]:
        receipt_rows = self.conn.execute(
            "SELECT request_id,node_id,digest FROM receipts "
            "ORDER BY request_id,node_id"
        ).fetchall()
        conclusion_rows = self.conn.execute(
            "SELECT request_id,digest FROM conclusions ORDER BY request_id"
        ).fetchall()
        attribution_rows = self.conn.execute(
            "SELECT request_id,digest FROM attributions ORDER BY request_id"
        ).fetchall()
        cursor = self.conn.execute("SELECT * FROM cursor_state WHERE id=1").fetchone()
        return {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "hop_set_version": self.get_meta("hop_set_version"),
            "scale_caliber_version": self.get_meta("scale_caliber_version"),
            "session_scale": self.get_meta("session_scale"),
            "receipts": [(r["request_id"], r["node_id"], r["digest"]) for r in receipt_rows],
            "conclusions": [(r["request_id"], r["digest"]) for r in conclusion_rows],
            "attributions": [(r["request_id"], r["digest"]) for r in attribution_rows],
            "cursor": dict(cursor) if cursor else None,
        }

    def _compute_content_digest_locked(self) -> str:
        return content_digest(self._digest_rows())

    def compute_content_digest(self) -> str:
        return content_digest(self._digest_rows())

    def verify_content_digest(self) -> bool:
        expected = self.get_meta("content_digest")
        return expected is not None and expected == self.compute_content_digest()

    def cursor(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM cursor_state WHERE id=1").fetchone()
        return dict(row) if row else {}

    def digest_index(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in self.conn.execute("SELECT request_id,node_id,digest FROM receipts"):
            result[f"receipt:{row['request_id']}:{row['node_id']}"] = row["digest"]
        for row in self.conn.execute("SELECT request_id,digest FROM conclusions"):
            result[f"conclusion:{row['request_id']}"] = row["digest"]
        for row in self.conn.execute("SELECT request_id,digest FROM attributions"):
            result[f"attribution:{row['request_id']}"] = row["digest"]
        return result

    def receipt_version_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.conn.execute(
            "SELECT request_id,node_id,canonical_json,digest FROM receipts"
        ):
            receipt = json.loads(row["canonical_json"])
            result[(row["request_id"], row["node_id"])] = {
                "config_version": receipt.get("config_version"),
                "scale_factor": receipt.get("scale_factor"),
                "digest": row["digest"],
            }
        return result

    def get_receipts(self, request_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT canonical_json FROM receipts WHERE request_id=? ORDER BY seq",
            (request_id,),
        ).fetchall()
        return [json.loads(row["canonical_json"]) for row in rows]

    def get_conclusion(self, request_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT canonical_json FROM conclusions WHERE request_id=?",
            (request_id,),
        ).fetchone()
        return json.loads(row["canonical_json"]) if row else None

    def get_attribution(self, request_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT canonical_json FROM attributions WHERE request_id=?",
            (request_id,),
        ).fetchone()
        return json.loads(row["canonical_json"]) if row else None

    def committed_request_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT request_id FROM conclusions ORDER BY request_id").fetchall()
        return [row["request_id"] for row in rows]

    def scheduler_events(self, batch_id: str | None = None) -> list[dict[str, Any]]:
        if batch_id:
            rows = self.conn.execute(
                "SELECT event_json FROM scheduler_events WHERE batch_id=? ORDER BY id",
                (batch_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT event_json FROM scheduler_events ORDER BY id"
            ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def all_results(self) -> dict[str, dict[str, Any]]:
        return {
            request_id: {
                "receipts": self.get_receipts(request_id),
                "conclusion": self.get_conclusion(request_id),
                "attribution": self.get_attribution(request_id),
            }
            for request_id in self.committed_request_ids()
        }

    def iter_batch_requests(self) -> Iterable[tuple[str, int, list[str], str]]:
        rows = self.conn.execute("SELECT * FROM batches ORDER BY seq,batch_id").fetchall()
        for row in rows:
            yield row["batch_id"], row["seq"], json.loads(row["request_ids_json"]), row["status"]
