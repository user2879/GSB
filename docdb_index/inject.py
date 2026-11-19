"""Deterministic fault injection for the three required business checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import digest_bytes
from .errors import StateError
from .storage import log_event, open_session, transaction


KIND_ALIASES = {
    "segment": "segment",
    "segment_digest": "segment",
    "digest": "segment",
    "bound": "bound",
    "boundary": "bound",
    "segment_bound": "bound",
    "ledger": "ledger",
    "ledger_missing": "ledger",
}


def inject_anomaly(
    session_dir: str | Path,
    *,
    collection_name: str,
    index_path: str,
    kind: str,
    segment_id: str | None = None,
) -> dict[str, Any]:
    normalized = KIND_ALIASES.get(kind)
    if normalized is None:
        raise StateError("unknown injection type", allowed=sorted(KIND_ALIASES))
    conn = open_session(session_dir)
    try:
        with transaction(conn):
            row = None
            if segment_id is not None:
                row = conn.execute(
                    "SELECT segment_id, version, entry_count, digest "
                    "FROM segments WHERE collection_name=? AND index_path=? AND segment_id=?",
                    (collection_name, index_path, segment_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT segment_id, version, entry_count, digest "
                    "FROM segments WHERE collection_name=? AND index_path=? "
                    "ORDER BY segment_id LIMIT 1",
                    (collection_name, index_path),
                ).fetchone()
            if normalized in {"segment", "bound"} and row is None:
                raise StateError(
                    "no segment is available for injection",
                    collection=collection_name,
                    field_path=index_path,
                )
            if normalized == "segment":
                new_digest = digest_bytes(
                    f"injected-segment-corruption\0{row['segment_id']}".encode("utf-8")
                )
                conn.execute(
                    "UPDATE segments SET digest=? "
                    "WHERE collection_name=? AND index_path=? AND segment_id=?",
                    (new_digest, collection_name, index_path, row["segment_id"]),
                )
                result = {
                    "injection": "segment_digest",
                    "segment_id": str(row["segment_id"]),
                    "field": "digest",
                    "expected_code": "E_SEGMENT",
                }
            elif normalized == "bound":
                conn.execute(
                    "UPDATE segments SET entry_count=entry_count+1 "
                    "WHERE collection_name=? AND index_path=? AND segment_id=?",
                    (collection_name, index_path, row["segment_id"]),
                )
                result = {
                    "injection": "segment_bound",
                    "segment_id": str(row["segment_id"]),
                    "field": "entry_count",
                    "expected_code": "E_BOUND",
                }
            else:
                ledger_row = conn.execute(
                    "SELECT collection_name, index_path, pk_json, stale_value_json, reason "
                    "FROM stale_ledger WHERE collection_name=? AND index_path=? "
                    "ORDER BY pk_json, stale_value_json LIMIT 1",
                    (collection_name, index_path),
                ).fetchone()
                if ledger_row is None:
                    raise StateError(
                        "no stale-ledger entry is available for injection",
                        collection=collection_name,
                        field_path=index_path,
                    )
                conn.execute(
                    "DELETE FROM stale_ledger WHERE collection_name=? AND index_path=? "
                    "AND pk_json=? AND stale_value_json=?",
                    (
                        ledger_row["collection_name"],
                        ledger_row["index_path"],
                        ledger_row["pk_json"],
                        ledger_row["stale_value_json"],
                    ),
                )
                result = {
                    "injection": "ledger_missing",
                    "primary_key": json.loads(ledger_row["pk_json"]),
                    "stale_key": json.loads(ledger_row["stale_value_json"]),
                    "field": "stale_ledger",
                    "expected_code": "E_LEDGER",
                }
        log_event(
            session_dir,
            level="warning",
            code="INJECT",
            message="anomaly injected",
            context=result,
        )
        return {"ok": True, **result}
    finally:
        conn.close()
