from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import spec_identity
from .engine import replay_spec
from .errors import VerifyError
from .invalidation import deletion_events_to_ledger_keys
from .sampling import is_selected


def _read_identity(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT value FROM meta WHERE key='session_identity'").fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def expected_block_projection(spec: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    replay = replay_spec(spec, verify_declared_digests=True)
    invalidated: set[tuple[str, str, int]] = set()
    affected, _ = deletion_events_to_ledger_keys(replay)
    for item in affected:
        invalidated.add((item["rollback_id"], item["volume_id"], item["block_index"]))

    expected: dict[tuple[str, int, str], dict[str, Any]] = {}
    permille = int(spec["budget"]["verification_sample_permille"])
    for plan in replay.rollback_plans:
        for block in plan["blocks"]:
            volume_id = plan["volume_id"]
            index = int(block["block_index"])
            selected = is_selected(plan["rollback_id"], volume_id, index, permille)
            key = (volume_id, index, plan["rollback_id"])
            is_invalidated = (plan["rollback_id"], volume_id, index) in invalidated
            if is_invalidated:
                status = "invalidated"
            elif selected:
                status = "pass"
            else:
                status = "unchecked"
            expected[key] = {
                "rollback_id": plan["rollback_id"],
                "volume_id": volume_id,
                "block_index": index,
                "block_version": int(block["replay_block_version"]),
                "baseline_version": int(block["baseline_version"]),
                "expected_digest": block["expected_digest"],
                "restored_digest": block["expected_digest"],
                "sample_selected": 1 if selected else 0,
                "verification_status": status,
            }
    return expected


def verify_ledger(
    spec: dict[str, Any],
    ledger_path: str | Path,
    *,
    check_identity: bool = True,
) -> dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        raise VerifyError(f"ledger does not exist: {path}", field="ledger")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if check_identity:
            ledger_identity = _read_identity(conn)
            identity = spec_identity(spec)
            if ledger_identity is None:
                raise VerifyError("ledger is missing session identity", field="session_identity")
            for field in ("format_version", "volume_set_version", "content_digest"):
                if ledger_identity.get(field) != identity[field]:
                    raise VerifyError(
                        "session identity mismatch",
                        field=field,
                        expected=identity[field],
                        actual=ledger_identity.get(field),
                    )

        rows = conn.execute("SELECT * FROM block_state_latest").fetchall()
        ledger_map: dict[tuple[str, int, str], dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            key = (item["volume_id"], int(item["block_index"]), item["rollback_id"])
            ledger_map[key] = item

        expected_map = expected_block_projection(spec)
        all_keys = sorted(set(expected_map) | set(ledger_map), key=lambda key: (key[0], key[1], key[2]))
        fields = (
            "rollback_id",
            "block_version",
            "baseline_version",
            "expected_digest",
            "restored_digest",
            "sample_selected",
            "verification_status",
        )
        checked = 0
        for key in all_keys:
            volume_id, block_index, rollback_id = key
            expected = expected_map.get(key)
            actual = ledger_map.get(key)
            if expected is None:
                raise VerifyError(
                    "ledger contains block absent from independent replay",
                    volume_id=volume_id,
                    block_index=block_index,
                    field="rollback_id",
                    expected=None,
                    actual=rollback_id,
                )
            if actual is None:
                raise VerifyError(
                    "ledger is missing replayed block",
                    volume_id=volume_id,
                    block_index=block_index,
                    field="block_index",
                    expected=rollback_id,
                    actual=None,
                )
            for field in fields:
                expected_value = expected[field]
                actual_value = actual.get(field)
                if field == "sample_selected":
                    actual_value = int(bool(actual_value))
                if expected_value != actual_value:
                    raise VerifyError(
                        "ledger field differs from independent replay",
                        volume_id=volume_id,
                        block_index=block_index,
                        field=field,
                        expected=expected_value,
                        actual=actual_value,
                    )
            checked += 1

        status_counts: dict[str, int] = {}
        for item in expected_map.values():
            status_counts[item["verification_status"]] = status_counts.get(item["verification_status"], 0) + 1
        return {
            "ok": True,
            "checked_blocks": checked,
            "status_counts": status_counts,
            "ledger_content_digest": conn.execute(
                "SELECT value FROM meta WHERE key='ledger_content_digest'"
            ).fetchone()[0]
            if conn.execute("SELECT 1 FROM meta WHERE key='ledger_content_digest'").fetchone()
            else None,
        }
    finally:
        conn.close()
