from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import canonical_text, digest_hex, first_difference
from .errors import DomainError
from .models import Spec


LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path: str | Path, create: bool = True) -> None:
        self.path = Path(path)
        if not create and not self.path.exists():
            raise DomainError("E_REPLAY", f"ledger does not exist: {self.path}", field_path="ledger.path")
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(str(self.path), timeout=15.0, isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA synchronous = FULL")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
            self._validate_format()
        except sqlite3.DatabaseError as exc:
            raise DomainError(
                "E_REPLAY",
                f"cannot open ledger: {exc}",
                field_path="ledger",
            ) from exc

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_info (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_index INTEGER PRIMARY KEY,
                through_seq INTEGER NOT NULL UNIQUE,
                progress_cursor INTEGER NOT NULL,
                spec_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY,
                batch_index INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                record_digest TEXT NOT NULL,
                FOREIGN KEY(batch_index) REFERENCES batches(batch_index)
            );
            CREATE TABLE IF NOT EXISTS session_snapshot (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                format_version INTEGER NOT NULL,
                disk_digest TEXT NOT NULL,
                body_json TEXT NOT NULL
            );
            """
        )
        with self.conn:
            row = self.conn.execute("SELECT id FROM ledger_info WHERE id = 1").fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO ledger_info(id, format_version, content_digest) VALUES(1, ?, ?)",
                    (LEDGER_FORMAT_VERSION, digest_hex([])),
                )

    def _validate_format(self) -> None:
        row = self.conn.execute(
            "SELECT format_version, content_digest FROM ledger_info WHERE id = 1"
        ).fetchone()
        if row is None:
            raise DomainError("E_REPLAY", "ledger_info row is missing", field_path="ledger.format_version")
        version = row["format_version"]
        if not isinstance(version, int):
            raise DomainError("E_REPLAY", "ledger format_version is not an integer", field_path="ledger.format_version")
        if version != LEDGER_FORMAT_VERSION:
            raise DomainError(
                "E_REPLAY",
                f"unsupported ledger format_version: {version}",
                field_path="ledger.format_version",
            )
        actual = self._compute_content_digest()
        if actual != row["content_digest"]:
            raise DomainError(
                "E_REPLAY",
                "ledger content digest does not match persisted batches",
                field_path="ledger.content_digest",
                details={"stored": row["content_digest"], "actual": actual},
            )

    def _batch_rows_for_digest(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT batch_index, through_seq, progress_cursor, spec_digest,
                   payload_json, payload_digest
            FROM batches ORDER BY batch_index
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def _compute_content_digest(self) -> str:
        return digest_hex(self._batch_rows_for_digest())

    def _update_content_digest(self) -> str:
        value = self._compute_content_digest()
        self.conn.execute(
            "UPDATE ledger_info SET content_digest = ? WHERE id = 1",
            (value,),
        )
        return value

    def progress_cursor(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(progress_cursor), 0) AS cursor FROM batches").fetchone()
        return int(row["cursor"])

    def next_batch_index(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(batch_index), 0) AS value FROM batches").fetchone()
        return int(row["value"]) + 1

    def commit_batch(
        self,
        batch_index: int,
        records: list[dict[str, Any]],
        spec_digest: str,
        progress_cursor: int,
        invalidation_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("records must not be empty")
        seqs = [record["seq"] for record in records]
        if seqs != sorted(seqs):
            raise ValueError("records must be ordered by sequence")
        through_seq = max(seqs)
        payload = {
            "batch_index": batch_index,
            "through_seq": through_seq,
            "progress_cursor": progress_cursor,
            "spec_digest": spec_digest,
            "messages": records,
            "invalidation_trace": invalidation_trace,
        }
        payload_json = canonical_text(payload)
        payload_digest = digest_hex(payload)
        message_rows = [
            (
                record["seq"],
                batch_index,
                canonical_text(record),
                digest_hex(record),
            )
            for record in records
        ]
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.conn.execute("SELECT COALESCE(MAX(batch_index), 0) AS value FROM batches").fetchone()
            if batch_index != int(current["value"]) + 1:
                raise DomainError(
                    "E_REPLAY",
                    f"batch index must be continuous; expected {int(current['value']) + 1}",
                    field_path="ledger.batch_index",
                )
            cursor = self.conn.execute("SELECT COALESCE(MAX(progress_cursor), 0) AS value FROM batches").fetchone()
            if progress_cursor < int(cursor["value"]):
                raise DomainError("E_REPLAY", "progress cursor moved backwards", field_path="ledger.progress_cursor")
            self.conn.execute(
                """
                INSERT INTO batches(batch_index, through_seq, progress_cursor, spec_digest, payload_json, payload_digest)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (batch_index, through_seq, progress_cursor, spec_digest, payload_json, payload_digest),
            )
            self.conn.executemany(
                "INSERT INTO messages(seq, batch_index, record_json, record_digest) VALUES(?, ?, ?, ?)",
                message_rows,
            )
            content_digest = self._update_content_digest()
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return {
            "batch_index": batch_index,
            "through_seq": through_seq,
            "progress_cursor": progress_cursor,
            "payload_digest": payload_digest,
            "content_digest": content_digest,
        }

    def list_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT batch_index, through_seq, progress_cursor, spec_digest, payload_digest
            FROM batches ORDER BY batch_index
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def read_batch_payloads(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT batch_index, payload_json, payload_digest FROM batches ORDER BY batch_index"
        ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except Exception as exc:
                raise DomainError(
                    "E_REPLAY",
                    f"cannot parse batch payload {row['batch_index']}: {exc}",
                    field_path="ledger.payload_json",
                ) from exc
            actual = digest_hex(payload)
            if actual != row["payload_digest"]:
                seq = payload.get("through_seq") if isinstance(payload, dict) else None
                raise DomainError(
                    "E_REPLAY",
                    "batch payload digest mismatch",
                    message_seq=seq,
                    field_path="ledger.payload_digest",
                    details={"batch_index": row["batch_index"], "stored": row["payload_digest"], "actual": actual},
                )
            payloads.append(payload)
        return payloads

    def committed_records(self) -> dict[int, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT seq, record_json, record_digest FROM messages ORDER BY seq"
        ).fetchall()
        records: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                record = json.loads(row["record_json"])
            except Exception as exc:
                raise DomainError(
                    "E_REPLAY",
                    f"cannot parse committed message record: {exc}",
                    message_seq=row["seq"],
                    field_path="ledger.record_json",
                ) from exc
            actual = digest_hex(record)
            if actual != row["record_digest"]:
                raise DomainError(
                    "E_REPLAY",
                    "committed record digest mismatch",
                    message_seq=row["seq"],
                    field_path="ledger.record_digest",
                )
            records[int(row["seq"])] = record
        return records

    def unknown_sequences(self, spec: Spec) -> list[int]:
        cursor = self.progress_cursor()
        return [message.seq for message in spec.messages if message.seq > cursor]

    def _snapshot_body(self, spec: Spec) -> dict[str, Any]:
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "certificates": [cert.as_dict() for cert in spec.certificates],
            "trust_anchors": list(spec.trust_anchors),
            "fixed": spec.fixed.as_dict(),
        }

    def sync_snapshot(self, spec: Spec) -> dict[str, Any]:
        body = self._snapshot_body(spec)
        body_json = canonical_text(body)
        disk_digest = digest_hex(body)
        row = self.conn.execute(
            "SELECT format_version, disk_digest, body_json FROM session_snapshot WHERE id = 1"
        ).fetchone()
        replaced = False
        reason = "unchanged"
        diff = {"added_cert_ids": [], "removed_cert_ids": [], "changed_cert_ids": []}
        if row is None:
            replaced = True
            reason = "missing"
            diff["added_cert_ids"] = sorted(cert.cert_id for cert in spec.certificates)
        elif row["format_version"] != SNAPSHOT_FORMAT_VERSION:
            replaced = True
            reason = "format_version_mismatch"
        elif row["disk_digest"] != disk_digest:
            replaced = True
            reason = "content_digest_mismatch"
        if replaced:
            old_body: dict[str, Any] = {}
            if row is not None:
                try:
                    old_body = json.loads(row["body_json"])
                except Exception:
                    old_body = {}
            old_certs = {
                cert.get("id"): cert
                for cert in old_body.get("certificates", [])
                if isinstance(cert, dict) and isinstance(cert.get("id"), str)
            }
            new_certs = {cert.cert_id: cert.as_dict() for cert in spec.certificates}
            diff = {
                "added_cert_ids": sorted(set(new_certs) - set(old_certs)),
                "removed_cert_ids": sorted(set(old_certs) - set(new_certs)),
                "changed_cert_ids": sorted(
                    cert_id
                    for cert_id in set(old_certs) & set(new_certs)
                    if canonical_text(old_certs[cert_id]) != canonical_text(new_certs[cert_id])
                ),
            }
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute("DELETE FROM session_snapshot WHERE id = 1")
                self.conn.execute(
                    """
                    INSERT INTO session_snapshot(id, format_version, disk_digest, body_json)
                    VALUES(1, ?, ?, ?)
                    """,
                    (SNAPSHOT_FORMAT_VERSION, disk_digest, body_json),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return {
            "replaced": replaced,
            "reason": reason,
            "disk_digest": disk_digest,
            "replacement_mode": "whole_snapshot_replace" if replaced else "not_needed",
            **diff,
        }

    def snapshot(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT format_version, disk_digest, body_json FROM session_snapshot WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "format_version": row["format_version"],
            "disk_digest": row["disk_digest"],
            "body": json.loads(row["body_json"]),
        }

    def content_digest(self) -> str:
        row = self.conn.execute("SELECT content_digest FROM ledger_info WHERE id = 1").fetchone()
        return str(row["content_digest"])

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def first_replay_difference(
    expected: dict[int, dict[str, Any]],
    actual: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    for seq in sorted(set(expected) | set(actual)):
        if seq not in expected or seq not in actual:
            return {
                "seq": seq,
                "cert_id": None,
                "field": "message.presence",
                "expected": seq in expected,
                "actual": seq in actual,
            }
        difference = first_difference(expected[seq], actual[seq], f"$.messages[{seq}]")
        if difference is not None:
            cert_id = None
            path = difference["path"]
            if ".certificates[" in path:
                try:
                    index_text = path.split(".certificates[", 1)[1].split("]", 1)[0]
                    index = int(index_text)
                    certs = actual[seq].get("selected_chain", {}).get("certificates", [])
                    if index < len(certs):
                        cert_id = certs[index].get("cert_id")
                except (ValueError, IndexError, AttributeError):
                    cert_id = None
            if cert_id is None:
                certs = actual[seq].get("selected_chain", {}).get("certificates", [])
                if certs:
                    cert_id = certs[0].get("cert_id")
            return {
                "seq": seq,
                "cert_id": cert_id,
                "field": path,
                "expected": difference["left"],
                "actual": difference["right"],
            }
    return None
