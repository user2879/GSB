"""Replay verification: recompute windows from the ledger's recorded rule versions."""

from __future__ import annotations

from typing import Any

from .engine import WindowResult
from .errors import BusinessError
from .ledger import Ledger


def verify_committed_windows(
    ledger: Ledger,
    computed: list[WindowResult],
    *,
    rule_signatures: dict[str, str],
) -> dict[str, Any]:
    """Compare freshly computed windows against every committed ledger row."""
    rows = ledger.conn.execute(
        "SELECT * FROM window_results WHERE status='committed' ORDER BY rule_id, window_start_ms"
    ).fetchall()
    by_key = {(result.rule_id, result.window_start_ms): result for result in computed}
    missing: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    verified = 0
    for row in rows:
        key = (row["rule_id"], row["window_start_ms"])
        current = by_key.get(key)
        if current is None:
            missing.append(
                {
                    "window_id": row["window_id"],
                    "rule_id": row["rule_id"],
                    "window_start_ms": row["window_start_ms"],
                    "basis": "window absent from the recomputed stream",
                }
            )
            continue
        ledger_signature = rule_signatures.get(row["rule_id"])
        recorded_signature = ledger.conn.execute(
            "SELECT signature FROM rule_versions WHERE rule_id=? AND version=?",
            (row["rule_id"], row["rule_version"]),
        ).fetchone()["signature"]
        if ledger_signature is not None and recorded_signature != ledger_signature:
            drift.append(
                {
                    "window_id": row["window_id"],
                    "rule_id": row["rule_id"],
                    "rule_version": row["rule_version"],
                    "ledger_signature": recorded_signature,
                    "current_signature": ledger_signature,
                    "basis": "rule signature mismatch under the same rule version",
                }
            )
            continue
        if row["content_digest"] != current.content_digest:
            drift.append(
                {
                    "window_id": row["window_id"],
                    "rule_id": row["rule_id"],
                    "window_start_ms": row["window_start_ms"],
                    "ledger_digest": row["content_digest"],
                    "current_digest": current.content_digest,
                    "basis": "recomputed aggregate content differs from the ledger",
                }
            )
            continue
        verified += 1

    if drift:
        first = drift[0]
        raise BusinessError(
            "E_REPLAY",
            "rule parameters are not reproducible for the recorded version",
            f"rule_versions[{first['rule_id']}:{first.get('rule_version', '?')}]",
            {"rule_id": first["rule_id"], "windows": drift},
        )
    return {
        "status": "ok",
        "code": "OK",
        "verified_windows": verified,
        "missing_windows": missing,
        "requires_rerun": bool(missing),
    }
