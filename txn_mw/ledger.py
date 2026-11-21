"""Whole-ledger export, replacement, backup and post-replacement validation."""

from __future__ import annotations

import base64
import binascii
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from . import FORMAT_VERSION
from .codec import VALID_STATES, decode_record
from .errors import MWError, ProtocolError
from .store import Store, ledger_content_digest


def load_ledger(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MWError("E_REPLACE", f"cannot read ledger: {exc}", "$") from exc
    if not isinstance(value, dict):
        raise MWError("E_REPLACE", "ledger root must be an object", "$")
    if isinstance(value.get("format_version"), bool) or not isinstance(value.get("format_version"), int):
        raise MWError("E_REPLACE", "format_version must be an integer", "$.format_version")
    if not isinstance(value.get("content_digest"), str) or not value["content_digest"]:
        raise MWError("E_REPLACE", "content_digest is required", "$.content_digest")
    for key in ("transactions", "events", "idempotency"):
        if not isinstance(value.get(key), list):
            raise MWError("E_REPLACE", f"{key} must be an array", f"$.{key}")
    return value


def inspect_ledger(value: dict[str, Any]) -> dict[str, Any]:
    calculated = ledger_content_digest(value)
    return {
        "format_version": value["format_version"],
        "declared_content_digest": value["content_digest"],
        "calculated_content_digest": calculated,
        "digest_matches": calculated == value["content_digest"],
        "version_matches": value["format_version"] == FORMAT_VERSION,
        "transaction_count": len(value["transactions"]),
        "event_count": len(value["events"]),
        "idempotency_count": len(value["idempotency"]),
    }


def _event_record(row: dict[str, Any]) -> bytes:
    encoded = row.get("record_b64")
    if not isinstance(encoded, str):
        raise MWError("E_REPLACE", "event record_b64 is required", "$.events[].record_b64")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise MWError("E_REPLACE", "event record_b64 is not valid base64", "$.events[].record_b64") from exc


def validate_event_rows(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    previous_global = 0
    per_txn: dict[str, int] = {}
    for index, row in enumerate(events):
        try:
            payload = decode_record(_event_record(row))
            seq = int(payload["event_seq"])
            if seq <= previous_global:
                raise ProtocolError("global event_seq is not strictly increasing", "$.events[].event_seq")
            previous = per_txn.get(payload["txn_id"])
            if previous is not None and seq <= previous:
                raise ProtocolError("per-transaction event_seq is not strictly increasing", "$.events[].event_seq")
            for field in ("txn_id", "version", "state", "body_digest"):
                if row.get(field) != payload[field]:
                    raise ProtocolError(
                        f"event row {field} does not match framed record", f"$.events[{index}].{field}"
                    )
            if row.get("event_type") != payload["type"]:
                raise ProtocolError(
                    "event row type does not match framed record",
                    f"$.events[{index}].event_type",
                )
            if int(row.get("event_seq", -1)) != seq:
                raise ProtocolError("event row sequence does not match frame", f"$.events[{index}].event_seq")
            previous_global = seq
            per_txn[payload["txn_id"]] = seq
        except MWError as exc:
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": row.get("txn_id"),
                    "field": exc.path or "$.events",
                    "message": exc.message,
                }
            )
    return errors


def validate_replaced_ledger(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Validate each transaction and every idempotency link independently."""
    errors: list[dict[str, Any]] = []
    tx_rows = conn.execute("SELECT * FROM txns ORDER BY txn_id").fetchall()
    event_rows = conn.execute("SELECT * FROM events WHERE valid=1 ORDER BY txn_id, event_seq").fetchall()
    by_txn: dict[str, list[sqlite3.Row]] = {}
    for row in event_rows:
        by_txn.setdefault(row["txn_id"], []).append(row)
    idem_rows = {row["txn_id"]: row for row in conn.execute("SELECT * FROM idempotency").fetchall()}
    eviction_txns = {
        row["txn_id"] for row in conn.execute("SELECT txn_id FROM idempotency_evictions").fetchall()
    }
    allowed_dispositions = {"pending", "committable", "consumed", "rolled_back", "manual_review"}
    for tx in tx_rows:
        txn_id = tx["txn_id"]
        events = by_txn.get(txn_id, [])
        if tx["state"] not in VALID_STATES:
            errors.append({"code": "E_REPLACE", "txn_id": txn_id, "field": "state", "message": "unknown state"})
        if tx["disposition"] not in allowed_dispositions:
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": txn_id,
                    "field": "disposition",
                    "message": "unknown disposition",
                }
            )
        valid_dispositions = {
            "half": {"pending", "rolled_back", "manual_review"},
            "committable": {"committable"},
            "consumed": {"consumed"},
        }
        if tx["disposition"] not in valid_dispositions.get(tx["state"], set()):
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": txn_id,
                    "field": "disposition",
                    "message": f"disposition {tx['disposition']!r} is invalid for state {tx['state']!r}",
                }
            )
        if tx["body_digest"] is None or len(tx["body_digest"]) != 64:
            errors.append({"code": "E_REPLACE", "txn_id": txn_id, "field": "body_digest", "message": "invalid digest"})
        half_events = [e for e in events if e["event_type"] == "half"]
        if len(half_events) != 1:
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": txn_id,
                    "field": "events.half",
                    "message": f"expected exactly one half event, found {len(half_events)}",
                }
            )
        else:
            try:
                half_payload = decode_record(bytes(half_events[0]["record"]))
                if int(half_payload["version"]) != int(tx["version"]):
                    errors.append(
                        {
                            "code": "E_REPLACE",
                            "txn_id": txn_id,
                            "field": "version",
                            "message": "half event version does not match transaction",
                        }
                    )
                if half_payload["body_digest"] != tx["body_digest"]:
                    errors.append(
                        {
                            "code": "E_REPLACE",
                            "txn_id": txn_id,
                            "field": "body_digest",
                            "message": "half event digest does not match transaction",
                        }
                    )
            except MWError as exc:
                errors.append(
                    {
                        "code": "E_REPLACE",
                        "txn_id": txn_id,
                        "field": "events.half",
                        "message": exc.message,
                    }
                )
        query_starts = [e for e in events if e["event_type"] == "query_started"]
        if int(tx["query_count"]) != len(query_starts):
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": txn_id,
                    "field": "query_count",
                    "message": f"row={tx['query_count']} events={len(query_starts)}",
                }
            )
        if int(tx["query_count"]) > int(tx["query_max"]):
            errors.append(
                {"code": "E_REPLACE", "txn_id": txn_id, "field": "query_count", "message": "exceeds query_max"}
            )
        first_from_events: int | None = None
        if query_starts:
            try:
                decoded = decode_record(bytes(query_starts[0]["record"]))
                first_from_events = int(decoded["data"].get("first_query_mono_ns", 0))
            except MWError:
                first_from_events = None
        if tx["first_query_mono_ns"] != first_from_events:
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": txn_id,
                    "field": "first_query_mono_ns",
                    "message": f"row={tx['first_query_mono_ns']} events={first_from_events}",
                }
            )
        if tx["state"] == "consumed":
            if not tx["effect_id"]:
                errors.append(
                    {"code": "E_REPLACE", "txn_id": txn_id, "field": "effect_id", "message": "missing effect"}
                )
            if not any(e["event_type"] == "consume" for e in events):
                errors.append(
                    {"code": "E_REPLACE", "txn_id": txn_id, "field": "events.consume", "message": "missing consume event"}
                )
            idem = idem_rows.get(txn_id)
            if idem is None and txn_id not in eviction_txns:
                errors.append(
                    {
                        "code": "E_REPLACE",
                        "txn_id": txn_id,
                        "field": "idempotency",
                        "message": "consumed transaction has neither key nor eviction evidence",
                    }
                )
            if idem is not None and idem["effect_id"] != tx["effect_id"]:
                errors.append(
                    {
                        "code": "E_REPLACE",
                        "txn_id": txn_id,
                        "field": "idempotency.effect_id",
                        "message": "effect identifier does not match transaction",
                    }
                )
            if idem is not None and idem["idem_key"] != txn_id:
                errors.append(
                    {
                        "code": "E_REPLACE",
                        "txn_id": txn_id,
                        "field": "idempotency.idem_key",
                        "message": "idempotency key must equal the transaction identifier",
                    }
                )
            if idem is not None and int(idem["consume_count"]) != 1:
                errors.append(
                    {
                        "code": "E_REPLACE",
                        "txn_id": txn_id,
                        "field": "idempotency.consume_count",
                        "message": "idempotent consumption count must remain exactly one",
                    }
                )
        elif tx["effect_id"] is not None:
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": txn_id,
                    "field": "effect_id",
                    "message": "non-consumed transaction has an effect identifier",
                }
            )
    for txn_id, idem in idem_rows.items():
        tx = next((row for row in tx_rows if row["txn_id"] == txn_id), None)
        if tx is None:
            errors.append(
                {"code": "E_REPLACE", "txn_id": txn_id, "field": "idempotency.txn_id", "message": "orphan key"}
            )
        elif tx["state"] != "consumed":
            errors.append(
                {
                    "code": "E_REPLACE",
                    "txn_id": txn_id,
                    "field": "idempotency.state",
                    "message": "key exists for a non-consumed transaction",
                }
            )
    return errors


def _backup_path(db_path: str | os.PathLike[str]) -> Path:
    source = Path(db_path).resolve()
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return source.with_name(f"{source.stem}.pre-replace-{tag}-{time.time_ns()}.readonly.db")


def _insert_snapshot_row(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    """Insert only known columns; reject unknown snapshot fields explicitly."""
    allowed = {item["name"] for item in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    unknown = set(row) - allowed
    if unknown:
        raise ValueError(f"unknown columns for {table}: {sorted(unknown)}")
    columns = list(row)
    conn.execute(
        f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
        tuple(row[column] for column in columns),
    )


def replace_ledger(
    store: Store,
    ledger_path: str | os.PathLike[str],
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    value = load_ledger(ledger_path)
    inspection = inspect_ledger(value)
    event_errors = validate_event_rows(value["events"])
    backup = store.sqlite_backup_readonly(_backup_path(store.path))
    old_info = {
        "format_version": int(store.get_meta("format_version", str(FORMAT_VERSION)) or FORMAT_VERSION),
        "content_digest": str(store.get_meta("content_digest", "") or ""),
    }
    version_changed = old_info["format_version"] != value["format_version"]
    digest_changed = old_info["content_digest"] != value["content_digest"]
    insertion_errors: list[dict[str, Any]] = []
    with store.transaction() as conn:
        for table in (
            "txns",
            "events",
            "idempotency",
            "idempotency_evictions",
            "pulls",
            "latency",
            "conflicts",
            "quarantine",
        ):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM sqlite_sequence")
        for row in value["transactions"]:
            try:
                _insert_snapshot_row(conn, "txns", row)
            except (sqlite3.Error, KeyError) as exc:
                insertion_errors.append(
                    {
                        "code": "E_REPLACE",
                        "txn_id": row.get("txn_id") if isinstance(row, dict) else None,
                        "field": "transactions",
                        "message": str(exc),
                    }
                )
        for row in value["events"]:
            try:
                record = _event_record(row)
                conn.execute(
                    """INSERT INTO events(row_id,event_seq,txn_id,version,state,body_digest,event_type,record,created_mono_ns,valid)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        row.get("row_id"),
                        row["event_seq"],
                        row["txn_id"],
                        row["version"],
                        row["state"],
                        row["body_digest"],
                        row["event_type"],
                        sqlite3.Binary(record),
                        row.get("created_mono_ns", time.monotonic_ns()),
                        int(row.get("valid", 1)),
                    ),
                )
            except (sqlite3.Error, KeyError, MWError) as exc:
                insertion_errors.append(
                    {
                        "code": "E_REPLACE",
                        "txn_id": row.get("txn_id") if isinstance(row, dict) else None,
                        "field": "events",
                        "message": str(exc),
                    }
                )
        for row in value["idempotency"]:
            try:
                _insert_snapshot_row(conn, "idempotency", row)
            except (sqlite3.Error, KeyError) as exc:
                insertion_errors.append(
                    {"code": "E_REPLACE", "txn_id": row.get("txn_id"), "field": "idempotency", "message": str(exc)}
                )
        optional = {
            "idempotency_evictions": value.get("idempotency_evictions", []),
            "pulls": value.get("pulls", []),
            "latency": value.get("latency", []),
            "conflicts": value.get("conflicts", []),
        }
        for table, rows in optional.items():
            for row in rows:
                try:
                    _insert_snapshot_row(conn, table, row)
                except (sqlite3.Error, KeyError) as exc:
                    insertion_errors.append(
                        {"code": "E_REPLACE", "txn_id": row.get("txn_id"), "field": table, "message": str(exc)}
                    )
        embedded_meta = value.get("meta", {})
        next_seq = int(embedded_meta.get("next_event_seq", 1))
        if value["events"]:
            next_seq = max(next_seq, max(int(row["event_seq"]) for row in value["events"]) + 1)
        store.set_meta(conn, "format_version", str(value["format_version"]))
        store.set_meta(conn, "next_event_seq", str(next_seq))
        store.set_meta(conn, "created_mono_ns", str(embedded_meta.get("created_mono_ns", time.monotonic_ns())))
        validation_errors = validate_replaced_ledger(conn)
        all_errors = event_errors + insertion_errors + validation_errors
        for error in all_errors:
            if logger is not None:
                logger.error(
                    "E_REPLACE",
                    error["message"],
                    txn_id=error.get("txn_id"),
                    field=error.get("field"),
                )
        store.set_meta(conn, "content_digest", ledger_content_digest(store.ledger_snapshot(conn)))
        if not inspection["version_matches"] and logger is not None:
            logger.error(
                "E_REPLACE",
                "ledger format version differs; whole replacement was applied without incremental merge",
                incoming_format_version=value["format_version"],
                supported_format_version=FORMAT_VERSION,
            )
        if not inspection["digest_matches"] and logger is not None:
            logger.error(
                "E_REPLACE",
                "ledger declared digest mismatch; whole replacement was applied and revalidated",
                declared_content_digest=value["content_digest"],
                calculated_content_digest=inspection["calculated_content_digest"],
            )
    return {
        "ok": not all_errors,
        "op": "replace_ledger",
        "full_replacement": True,
        "incremental_merge": False,
        "backup_path": backup,
        "backup_readonly": True,
        "previous_ledger": old_info,
        "incoming_ledger": inspection,
        "version_changed": version_changed,
        "digest_changed": digest_changed,
        "validation_errors": all_errors,
        "content_digest": store.get_meta("content_digest"),
    }
