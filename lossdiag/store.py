"""SQLite persistence for snapshot replacement and the diagnostic ledger."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import snapshot_error
from .pipeline import DiagnosisSession
from .util import canonical_json, content_digest, pretty_json


SCHEMA_VERSION = "workbench-db-v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_uid TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    event_versions_json TEXT NOT NULL,
    basis_version TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    version_signature_json TEXT NOT NULL,
    report_digest TEXT NOT NULL,
    report_json TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    ledger_uid TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    version_combo_json TEXT NOT NULL,
    basis_version TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    report_digest TEXT NOT NULL,
    report_json TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class SnapshotRecord:
    snapshot_uid: str
    created_at: str
    model_version: str
    event_versions: list[str]
    basis_version: str
    input_digest: str
    content_digest: str
    version_signature: dict[str, Any]
    report_digest: str
    report: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class LedgerRecord:
    ledger_uid: str
    created_at: str
    model_version: str
    version_combo: dict[str, Any]
    basis_version: str
    input_digest: str
    report_digest: str
    report: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class RestoredSessionView:
    """Authoritative read-only session state restored from a snapshot."""

    snapshot_uid: str
    payload: dict[str, Any]
    report: dict[str, Any]
    version_signature: dict[str, Any]

    @property
    def report_digest(self) -> str:
        return str(self.report["report_digest"])

    @property
    def input_digest(self) -> str:
        return str(self.payload["inputs"]["input_digest"])

    @property
    def version_combo(self) -> dict[str, Any]:
        return self.payload["version_combo"]

    @property
    def event_versions(self) -> list[str]:
        return list(self.version_combo["event_list_versions"])

    @property
    def alignment(self) -> Any:
        return _RestoredAlignment(
            basis_version=str(self.version_combo["alignment_basis_version"]),
            unaligned_flow_ids=list(self.report["alignment"]["unaligned_flow_ids"]),
        )

    @property
    def rate_report(self) -> dict[str, Any]:
        return self.payload["rates"]

    def model_payload(self) -> dict[str, Any]:
        return self.payload

    def content_digest(self) -> str:
        return content_digest(self.payload)


@dataclass
class _RestoredAlignment:
    basis_version: str
    unaligned_flow_ids: list[str]


class WorkbenchStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(SCHEMA)
        self.connection.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (SCHEMA_VERSION,),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "WorkbenchStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def save_snapshot(self, session: Any) -> str:
        payload = session.model_payload()
        digest = content_digest(payload)
        signature = session.version_signature()
        uid = f"snap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        self.connection.execute(
            "INSERT INTO snapshots("
            "snapshot_uid, created_at, model_version, event_versions_json, basis_version, "
            "input_digest, content_digest, version_signature_json, report_digest, report_json, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uid,
                utc_now(),
                session.version_combo["model_version"],
                canonical_json(session.event_versions),
                session.alignment.basis_version,
                session.input_digest,
                digest,
                canonical_json(signature),
                session.report_digest,
                canonical_json(session.report),
                canonical_json(payload),
            ),
        )
        self.connection.commit()
        return uid

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT snapshot_uid, created_at, model_version, event_versions_json, basis_version, "
            "content_digest, report_digest FROM snapshots ORDER BY created_at ASC, snapshot_uid ASC"
        ).fetchall()
        return [
            {
                "snapshot_uid": row["snapshot_uid"],
                "created_at": row["created_at"],
                "model_version": row["model_version"],
                "event_versions": json.loads(row["event_versions_json"]),
                "basis_version": row["basis_version"],
                "content_digest": row["content_digest"],
                "report_digest": row["report_digest"],
            }
            for row in rows
        ]

    def load_snapshot(
        self,
        snapshot_uid: str,
        *,
        inject_digest_mismatch: bool = False,
    ) -> SnapshotRecord:
        row = self.connection.execute(
            "SELECT * FROM snapshots WHERE snapshot_uid = ?", (snapshot_uid,)
        ).fetchone()
        if row is None:
            raise snapshot_error(
                "snapshot was not found",
                snapshot_uid=snapshot_uid,
            )
        payload = json.loads(row["payload_json"])
        if inject_digest_mismatch:
            payload["window_ms"] = int(payload.get("window_ms", 0)) + 1
        actual_digest = content_digest(payload)
        if actual_digest != row["content_digest"]:
            raise snapshot_error(
                "snapshot content digest does not match its stored digest",
                snapshot_uid=snapshot_uid,
                stored_digest=row["content_digest"],
                recomputed_digest=actual_digest,
            )
        return SnapshotRecord(
            snapshot_uid=row["snapshot_uid"],
            created_at=row["created_at"],
            model_version=row["model_version"],
            event_versions=json.loads(row["event_versions_json"]),
            basis_version=row["basis_version"],
            input_digest=row["input_digest"],
            content_digest=row["content_digest"],
            version_signature=json.loads(row["version_signature_json"]),
            report_digest=row["report_digest"],
            report=json.loads(row["report_json"]),
            payload=payload,
        )

    def save_ledger(self, session: Any) -> str:
        payload = session.model_payload()
        uid = f"ledger-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        self.connection.execute(
            "INSERT INTO ledger("
            "ledger_uid, created_at, model_version, version_combo_json, basis_version, "
            "input_digest, report_digest, report_json, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uid,
                utc_now(),
                session.version_combo["model_version"],
                canonical_json(session.version_combo),
                session.alignment.basis_version,
                session.input_digest,
                session.report_digest,
                canonical_json(session.report),
                canonical_json(payload),
            ),
        )
        self.connection.commit()
        return uid

    def load_ledger(self, ledger_uid: str) -> LedgerRecord:
        row = self.connection.execute(
            "SELECT * FROM ledger WHERE ledger_uid = ?", (ledger_uid,)
        ).fetchone()
        if row is None:
            raise snapshot_error("ledger record was not found", ledger_uid=ledger_uid)
        return LedgerRecord(
            ledger_uid=row["ledger_uid"],
            created_at=row["created_at"],
            model_version=row["model_version"],
            version_combo=json.loads(row["version_combo_json"]),
            basis_version=row["basis_version"],
            input_digest=row["input_digest"],
            report_digest=row["report_digest"],
            report=json.loads(row["report_json"]),
            payload=json.loads(row["payload_json"]),
        )

    def list_ledger(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT ledger_uid, created_at, model_version, version_combo_json, basis_version, "
            "input_digest, report_digest FROM ledger ORDER BY created_at ASC, ledger_uid ASC"
        ).fetchall()
        return [
            {
                "ledger_uid": row["ledger_uid"],
                "created_at": row["created_at"],
                "model_version": row["model_version"],
                "version_combo": json.loads(row["version_combo_json"]),
                "basis_version": row["basis_version"],
                "input_digest": row["input_digest"],
                "report_digest": row["report_digest"],
            }
            for row in rows
        ]


def restore_snapshot(
    store: WorkbenchStore,
    snapshot_uid: str,
    *,
    current_session: DiagnosisSession | None = None,
    inject_digest_mismatch: bool = False,
) -> dict[str, Any]:
    snapshot = store.load_snapshot(
        snapshot_uid,
        inject_digest_mismatch=inject_digest_mismatch,
    )
    snapshot_signature = snapshot.version_signature
    current_signature = current_session.version_signature() if current_session else None
    field_order = (
        "model_version",
        "event_versions",
        "basis_version",
        "window_ms",
        "conclusion_digest",
        "content_digest",
    )
    difference_sources = {
        "model_version": "internal_model_version",
        "event_versions": "event_format_version_set",
        "basis_version": "alignment_basis_version",
        "window_ms": "session_window_parameter",
        "conclusion_digest": "conclusion_content_digest",
        "content_digest": "snapshot_model_content_digest",
    }
    differences: list[dict[str, Any]] = []
    if current_signature is not None:
        for field_name in field_order:
            left = snapshot_signature.get(field_name)
            right = current_signature.get(field_name)
            if left != right:
                differences.append(
                    {
                        "field": field_name,
                        "difference_source": difference_sources[field_name],
                        "snapshot_value": left,
                        "current_session_value": right,
                        "replacement_reason": "version_or_content_signature_changed",
                    }
                )
    replaced = current_signature is None or bool(differences)
    if current_signature is None:
        reason = "no_current_session_snapshot_is_authoritative"
    elif differences:
        reason = "version_set_or_digest_mismatch_requires_wholesale_replacement"
    else:
        reason = "snapshot_matches_current_session"
    return {
        "ok": True,
        "snapshot_uid": snapshot.snapshot_uid,
        "snapshot_created_at": snapshot.created_at,
        "comparison": {
            "current_session_present": current_signature is not None,
            "differences": differences,
            "replacement_reason": reason,
            "replacement_mode": "whole_payload_replace",
            "incremental_merge": False,
            "replaced": replaced,
        },
        "restored_model": snapshot.payload,
        "restored_report": snapshot.report,
        "restored_version_signature": snapshot.version_signature,
        "restored_content_digest": snapshot.content_digest,
    }


def restored_session_view(result: dict[str, Any]) -> RestoredSessionView:
    return RestoredSessionView(
        snapshot_uid=str(result["snapshot_uid"]),
        payload=result["restored_model"],
        report=result["restored_report"],
        version_signature=result["restored_version_signature"],
    )


def pretty_snapshot_list(store: WorkbenchStore) -> str:
    return pretty_json(store.list_snapshots())
