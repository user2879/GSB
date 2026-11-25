from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_text, diff_flat, sha256_hex
from .errors import E_VERIFY, VerifyError
from .ownership import adjudicate_events
from .spec import digest_declaration, load_declaration


def _first_difference(differences: list[dict[str, Any]], index: Optional[int] = None) -> VerifyError:
    first = differences[0] if differences else {"path": "$", "kind": "unknown"}
    return VerifyError(
        "ledger replay verification failed at first difference",
        field=str(first.get("path")),
        details={"index": index, "first_difference": first, "difference_count": len(differences)},
    )


def verify_ledger(ledger_path: str | Path, declaration_path: str | Path) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise VerifyError(f"ledger does not exist: {ledger_path}", field="ledger_path")
    spec = load_declaration(declaration_path)
    expected_digest = digest_declaration(spec)

    uri = f"file:{ledger_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if user_version != LEDGER_FORMAT_VERSION:
            raise VerifyError(
                "integer SQLite user_version format mismatch",
                field="PRAGMA.user_version",
                details={"expected": LEDGER_FORMAT_VERSION, "actual": user_version},
            )

        format_row = conn.execute("SELECT * FROM ledger_format WHERE id=1").fetchone()
        if format_row is None or int(format_row["format_version"]) != LEDGER_FORMAT_VERSION:
            raise VerifyError("missing integer ledger_format row", field="ledger_format.format_version")
        if format_row["declaration_digest"] != expected_digest:
            raise VerifyError(
                "ledger declaration digest does not match original declaration",
                field="declaration_digest",
                details={"expected": expected_digest, "actual": format_row["declaration_digest"]},
            )

        meta_rows = {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM meta")}
        if meta_rows.get("declaration_digest") != expected_digest:
            raise VerifyError("meta declaration_digest mismatch", field="meta.declaration_digest")

        commit_rows = conn.execute("SELECT * FROM commits ORDER BY seq").fetchall()
        previous_digest: Optional[str] = None
        events: list[dict[str, Any]] = []
        for row in commit_rows:
            seq = int(row["seq"])
            if row["prev_digest"] != previous_digest:
                raise VerifyError(
                    "commit prev_digest mismatch",
                    field="commits.prev_digest",
                    details={"index": seq, "expected": previous_digest, "actual": row["prev_digest"]},
                )
            payload = json.loads(row["payload_json"])
            envelope = {
                "seq": seq,
                "kind": row["kind"],
                "request_id": row["request_id"],
                "event_seq": row["event_seq"],
                "payload": payload,
                "prev_digest": previous_digest,
            }
            expected_hash = sha256_hex(envelope)
            if expected_hash != row["digest"]:
                raise VerifyError(
                    "commit content digest mismatch",
                    field="commits.digest",
                    details={"index": seq, "expected": expected_hash, "actual": row["digest"]},
                )
            previous_digest = row["digest"]
            event = payload.get("event") if isinstance(payload, dict) else None
            if event is not None:
                events.append(event)

        if meta_rows.get("head_digest") != previous_digest:
            raise VerifyError(
                "meta head_digest does not match final commit",
                field="meta.head_digest",
                details={"expected": previous_digest, "actual": meta_rows.get("head_digest")},
            )

        expected = adjudicate_events(events, spec)
        stored_projection_raw = meta_rows.get("projection_json")
        if stored_projection_raw is None:
            raise VerifyError("missing projection_json", field="meta.projection_json")
        stored_projection = json.loads(stored_projection_raw)
        differences = diff_flat(expected, stored_projection)
        if differences:
            raise _first_difference(differences, len(commit_rows))

        lease_rows = conn.execute("SELECT * FROM leases ORDER BY lease_id").fetchall()
        expected_leases = sorted(expected["leases"], key=lambda item: item["lease_id"])
        if len(lease_rows) != len(expected_leases):
            raise VerifyError(
                "lease row count mismatch",
                field="leases",
                details={"expected": len(expected_leases), "actual": len(lease_rows)},
            )
        for index, (row, expected_lease) in enumerate(zip(lease_rows, expected_leases)):
            actual_lease = {
                "lease_id": row["lease_id"],
                "node_id": row["node_id"],
                "slot_id": row["slot_id"],
                "state": row["state"],
                "generation": int(row["generation"]),
                "quota_version": row["quota_version"],
                "last_event_seq": row["last_event_seq"],
                "history": json.loads(row["history_json"]),
            }
            comparable_expected = {
                "lease_id": expected_lease["lease_id"],
                "node_id": expected_lease["node_id"],
                "slot_id": expected_lease["slot_id"],
                "state": expected_lease["state"],
                "generation": int(expected_lease["generation"]),
                "quota_version": expected_lease.get("quota_version"),
                "last_event_seq": expected_lease.get("last_event_seq"),
                "history": expected_lease.get("history", []),
            }
            differences = diff_flat(comparable_expected, actual_lease)
            if differences:
                raise _first_difference(differences, index)

        owner_rows = conn.execute("SELECT * FROM ownership ORDER BY slot_id").fetchall()
        expected_owners = sorted(expected["owners"], key=lambda item: item["slot_id"])
        actual_owners = [
            {
                "slot_id": row["slot_id"],
                "lease_id": row["lease_id"],
                "node_id": row["node_id"],
                "generation": int(row["generation"]),
                "event_seq": int(row["event_seq"]),
                "basis_generation": row["basis_generation"],
            }
            for row in owner_rows
        ]
        differences = diff_flat(expected_owners, actual_owners)
        if differences:
            raise _first_difference(differences, len(expected_leases))

        decision_rows = conn.execute("SELECT payload_json FROM decisions ORDER BY event_seq").fetchall()
        actual_decisions = [json.loads(row["payload_json"]) for row in decision_rows]
        differences = diff_flat(expected["decisions"], actual_decisions)
        if differences:
            raise _first_difference(differences, len(commit_rows))

        transition_rows = conn.execute("SELECT record_json FROM transitions ORDER BY id").fetchall()
        actual_transitions = [json.loads(row["record_json"]) for row in transition_rows]
        expected_transitions = [
            record
            for lease in sorted(expected["leases"], key=lambda item: item["lease_id"])
            for record in lease.get("history", [])
        ]
        differences = diff_flat(expected_transitions, actual_transitions)
        if differences:
            raise _first_difference(differences, len(commit_rows))

        return {
            "ok": True,
            "code": "OK",
            "ledger": str(ledger_path),
            "checked_commits": len(commit_rows),
            "replayed_events": len(events),
            "leases": len(expected_leases),
            "owners": len(expected_owners),
            "decisions": len(expected["decisions"]),
            "head_digest": previous_digest,
            "projection_digest": sha256_hex(canonical_text(expected).encode("utf-8")),
            "first_difference": None,
        }
    finally:
        conn.close()
