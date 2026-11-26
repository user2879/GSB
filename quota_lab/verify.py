from __future__ import annotations

from pathlib import Path
from typing import Any

from .accounting import build_shard_accounting
from .canonical import canonical_dumps, sha256_hex
from .leases import new_lease, ownership_view
from .ledger import Ledger
from .spec import SessionSpec


def expected_business_fields(
    spec: SessionSpec,
    samples: dict[str, list[list[int]]],
    shard_id: str,
) -> dict[str, Any]:
    accounting = build_shard_accounting(spec, shard_id, samples[shard_id])
    lease = new_lease(spec, shard_id)
    for _ in range(spec.shard(shard_id).planned_renewals):
        lease["lease_version"] += 1
        lease["renew_count"] += 1
    return {
        "aggregate_summary": accounting["summary"],
        "watermark": {key: value for key, value in accounting.items() if key != "summary"},
        "lease_ownership": ownership_view(lease),
    }


def rebuild_consistency_check(
    spec: SessionSpec,
    samples: dict[str, list[list[int]]],
    ledger: Ledger,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    all_ok = True
    for shard in spec.shards:
        row = ledger.conn.execute(
            "SELECT * FROM shard_records WHERE shard_id=?",
            (shard.shard_id,),
        ).fetchone()
        field_checks: dict[str, dict[str, Any]] = {}
        shard_ok = row is not None and row["state"] == "completed"
        if row is None:
            results.append({"shard_id": shard.shard_id, "ok": False, "reason": "missing_record"})
            all_ok = False
            continue

        expected = expected_business_fields(spec, samples, shard.shard_id)
        comparisons = (
            ("summary_json", "aggregate_summary", expected["aggregate_summary"]),
            ("watermark_json", "watermark", expected["watermark"]),
            ("lease_ownership_json", "lease_ownership", expected["lease_ownership"]),
        )
        for column, field_name, expected_value in comparisons:
            stored_raw = row[column]
            try:
                stored_value = __import__("json").loads(stored_raw)
            except Exception:
                stored_value = None
            equal = canonical_dumps(stored_value) == canonical_dumps(expected_value)
            field_checks[field_name] = {
                "equal": equal,
                "stored_digest": sha256_hex((stored_raw or "").encode("utf-8")),
                "rebuilt_digest": sha256_hex(expected_value),
            }
            shard_ok = shard_ok and equal
        results.append(
            {
                "shard_id": shard.shard_id,
                "ok": shard_ok,
                "fields": field_checks,
            }
        )
        all_ok = all_ok and shard_ok
    digest_check = ledger.verify_digest()
    return {
        "ok": all_ok and digest_check["ok"],
        "method": "recompute_numpy_fixed_point_and_deterministic_ownership_from_blank_declaration_events",
        "shard_checks": results,
        "ledger_digest": digest_check,
        "timing_fields_note": "monotonic grant/expiry/elapsed fields are runtime observations and are checked for ordering, not cross-process equality",
    }


def byte_preservation_check(
    old_ledger_path: str | Path,
    new_ledger: Ledger,
    unchanged_shard_ids: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    import sqlite3

    old_conn = sqlite3.connect(f"file:{old_ledger_path}?mode=ro", uri=True)
    old_conn.row_factory = sqlite3.Row
    old = Ledger(old_ledger_path, old_conn)
    try:
        checks = []
        all_ok = True
        columns = ("summary_json", "lease_ownership_json", "watermark_json")
        for shard_id in unchanged_shard_ids:
            old_row = old.conn.execute(
                "SELECT * FROM shard_records WHERE shard_id=?",
                (shard_id,),
            ).fetchone()
            new_row = new_ledger.conn.execute(
                "SELECT * FROM shard_records WHERE shard_id=?",
                (shard_id,),
            ).fetchone()
            field_results = {}
            shard_ok = old_row is not None and new_row is not None
            for column in columns:
                old_bytes = old_row[column].encode("utf-8") if old_row is not None and old_row[column] is not None else b""
                new_bytes = new_row[column].encode("utf-8") if new_row is not None and new_row[column] is not None else b""
                equal = old_bytes == new_bytes
                field_results[column] = {
                    "byte_identical": equal,
                    "sha256": sha256_hex(old_bytes),
                    "bytes": len(old_bytes),
                }
                shard_ok = shard_ok and equal
            checks.append({"shard_id": shard_id, "ok": shard_ok, "fields": field_results})
            all_ok = all_ok and shard_ok
        return {"ok": all_ok, "checks": checks}
    finally:
        old.close()
