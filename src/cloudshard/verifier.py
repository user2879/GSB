from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .adjudicate import Adjudicator
from .canonical import canonical_text, content_digest
from .errors import VerifyError
from .invalidate import build_invalidation_plan
from .ledger import Ledger, digest_record
from .aggregate import aggregate_samples
from .samples import generate_samples
from .spec import Spec, load_spec, topological_shards


def _verify(condition: bool, message: str, *, shard_id: str | None = None, field: str, basis: Any = None) -> None:
    if not condition:
        raise VerifyError(message, shard_id=shard_id, field=field, basis=basis)


def _parse_json(value: str, *, shard_id: str | None, field_name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise VerifyError(
            f"stored {field_name} is not valid JSON: {exc}",
            shard_id=shard_id,
            field=field_name,
        ) from exc


def verify_spec_file(spec_path: str | Path, ledger_path: str | Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    return verify_ledger(spec, ledger_path)


def verify_ledger(spec: Spec, ledger_path: str | Path) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    _verify(ledger_path.exists(), "ledger file does not exist", field="ledger", basis={"path": str(ledger_path)})
    ledger = Ledger(ledger_path, read_only=True)
    try:
        report = _verify_open_ledger(spec, ledger)
    finally:
        ledger.close()
    report["ok"] = True
    report["code"] = "E_VERIFY_OK"
    return report


def _verify_open_ledger(spec: Spec, ledger: Ledger) -> dict[str, Any]:
    conn = ledger.conn
    meta_row = conn.execute("SELECT value FROM meta WHERE key='ledger_format_version'").fetchone()
    _verify(meta_row is not None, "ledger is missing format version meta", field="format_version")
    _verify(int(meta_row["value"]) == 1, "unsupported ledger format version", field="format_version", basis={"value": meta_row["value"]})

    spec_meta = conn.execute("SELECT value FROM meta WHERE key='spec'").fetchone()
    _verify(spec_meta is not None, "ledger is missing spec digest meta", field="spec_digest")
    spec_meta_obj = _parse_json(spec_meta["value"], shard_id=None, field_name="meta.spec")
    _verify(
        spec_meta_obj.get("spec_digest") == content_digest(spec.raw),
        "original declaration digest does not match ledger",
        field="spec_digest",
        basis={"ledger": spec_meta_obj.get("spec_digest"), "spec": content_digest(spec.raw)},
    )

    _audit_content_digests(conn)
    declared_ownership = _verify_declared_events(spec, conn)
    runtime_ownership = _verify_runtime_events(spec, conn)
    _verify_no_unknown_dispatch(conn)
    outcome_report = _verify_outcomes(spec, conn)
    invalidation_report = _verify_invalidation(spec, conn)
    progress_report = _verify_progress(conn)
    recovery_report = _verify_recovery(conn)

    active_duplicates = conn.execute(
        """
        SELECT node_id, shard_id, COUNT(*) AS c FROM leases
        WHERE status IN ('active','HELD')
        GROUP BY node_id, shard_id HAVING c > 1
        """
    ).fetchall()
    _verify(not active_duplicates, "two active lease rows share node and shard", field="lease_id", basis=[dict(r) for r in active_duplicates])

    return {
        "ledger": str(ledger.path),
        "spec_digest": content_digest(spec.raw),
        "declared_ownership_records": len(declared_ownership),
        "runtime_ownership_records": len(runtime_ownership),
        "outcomes_verified": outcome_report["verified"],
        "invalidation": invalidation_report,
        "progress": progress_report,
        "recovery": recovery_report,
        "ownership_canonical_digest": content_digest(declared_ownership + runtime_ownership),
    }


def _audit_content_digests(conn: sqlite3.Connection) -> None:
    checks = [
        ("event_log", "payload_json", "content_digest", "stream", "event_seq"),
        ("ownership", "record_json", "content_digest", "stream", "sequence_no"),
        ("decisions", "record_json", "content_digest", "stream", "id"),
        ("leases", "record_json", "content_digest", "lease_id", "lease_id"),
        ("attempts", "record_json", "content_digest", "shard_id", "shard_id"),
        ("outcomes", "record_json", "content_digest", "shard_id", "shard_id"),
        ("recovery_marks", "detail_json", "content_digest", "shard_id", "shard_id"),
    ]
    for table, json_col, digest_col, loc_col, _ in checks:
        rows = conn.execute(f"SELECT *, {json_col} AS audit_json, {digest_col} AS audit_digest FROM {table} ORDER BY rowid").fetchall()
        for row in rows:
            try:
                record = json.loads(row["audit_json"])
            except json.JSONDecodeError as exc:
                raise VerifyError(f"{table} row contains invalid JSON", shard_id=row["shard_id"] if "shard_id" in row.keys() else None, field=json_col) from exc
            if table == "event_log":
                expected = digest_record({"stream": row["stream"], "event": record})
            elif table == "recovery_marks":
                expected = digest_record(record)
            else:
                expected = digest_record(record)
            _verify(
                expected == row["audit_digest"],
                f"{table} content digest mismatch",
                shard_id=row["shard_id"] if "shard_id" in row.keys() else None,
                field="content_digest",
                basis={"table": table, "locator": row[loc_col], "expected": expected, "actual": row["audit_digest"]},
            )
            _verify(
                int(row["format_version"]) == 1,
                f"{table} has unsupported format_version",
                shard_id=row["shard_id"] if "shard_id" in row.keys() else None,
                field="format_version",
                basis={"table": table, "value": row["format_version"]},
            )


def _verify_declared_events(spec: Spec, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    adj = Adjudicator.from_spec(spec).replay(list(spec.events))
    rows = conn.execute(
        "SELECT record_json FROM ownership WHERE stream='declared' ORDER BY sequence_no, id"
    ).fetchall()
    stored = [json.loads(row["record_json"]) for row in rows]
    expected = adj.ownership_sequence
    _verify(
        len(stored) == len(expected),
        "declared ownership sequence length mismatch",
        field="ownership_sequence",
        basis={"expected": len(expected), "ledger": len(stored)},
    )
    for index, (actual_record, expected_record) in enumerate(zip(stored, expected)):
        if canonical_text(actual_record) != canonical_text(expected_record):
            shard_id = actual_record.get("shard_id") or expected_record.get("shard_id")
            for field_name in ("change", "reason", "lease_id", "node_id", "shard_id", "event_seq", "quota_version", "holder_key"):
                if actual_record.get(field_name) != expected_record.get(field_name):
                    raise VerifyError(
                        "declared ownership field mismatch",
                        shard_id=shard_id,
                        field=field_name,
                        basis={"index": index, "expected": expected_record.get(field_name), "actual": actual_record.get(field_name)},
                    )
            raise VerifyError("declared ownership record mismatch", shard_id=shard_id, field="basis", basis={"index": index})
    return stored


KNOWN_REPLAY_EVENTS = {
    "lease_apply",
    "lease_renew",
    "lease_release",
    "lease_expire",
    "expire_due",
    "quota_change",
    "cancel_shard",
}


def _verify_runtime_events(spec: Spec, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT payload_json FROM event_log WHERE stream='runtime' ORDER BY event_seq").fetchall()
    events = [json.loads(row["payload_json"]) for row in rows]
    adj = Adjudicator.from_spec(spec)
    for event in events:
        if event.get("type") in KNOWN_REPLAY_EVENTS:
            adj.handle_event(event)
    stored_rows = conn.execute(
        "SELECT record_json FROM ownership WHERE stream='runtime' ORDER BY sequence_no, id"
    ).fetchall()
    stored = [json.loads(row["record_json"]) for row in stored_rows]
    expected = adj.ownership_sequence
    _verify(
        len(stored) == len(expected),
        "runtime ownership sequence length mismatch",
        field="ownership_sequence",
        basis={"expected": len(expected), "ledger": len(stored), "events": len(events)},
    )
    for index, (actual_record, expected_record) in enumerate(zip(stored, expected)):
        for field_name in ("change", "reason", "lease_id", "node_id", "shard_id", "event_seq", "holder_key"):
            if actual_record.get(field_name) != expected_record.get(field_name):
                raise VerifyError(
                    "runtime ownership field mismatch",
                    shard_id=actual_record.get("shard_id"),
                    field=field_name,
                    basis={"index": index, "expected": expected_record.get(field_name), "actual": actual_record.get(field_name)},
                )
    return stored


def _empty_limits() -> Any:
    from .adjudicate import CapacityLimits

    return CapacityLimits(
        global_budget_units=10**12,
        node_lease_limits={},
        node_concurrency_quotas={},
        node_renewal_limits={},
    )


def _verify_no_unknown_dispatch(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT COUNT(*) AS c FROM attempts WHERE status='DISPATCHED'").fetchone()
    _verify(
        int(row["c"]) == 0,
        "ledger contains a dispatch without committed outcome; run recover and rerun",
        field="attempts.status",
        basis={"unknown_attempts": int(row["c"])},
    )


def _verify_outcomes(spec: Spec, conn: sqlite3.Connection) -> dict[str, Any]:
    order = [s.shard_id for s in topological_shards(spec)]
    order_index = {sid: i for i, sid in enumerate(order)}
    rows = conn.execute("SELECT * FROM outcomes WHERE state='COMMITTED'").fetchall()
    rows = sorted(rows, key=lambda r: (order_index.get(r["shard_id"], 10**9), int(r["revision"]), int(r["attempt"])))
    verified = 0
    for row in rows:
        shard_id = row["shard_id"]
        shard = spec.shard_by_id.get(shard_id)
        _verify(shard is not None, "ledger outcome for unknown shard", shard_id=shard_id, field="shard_id")
        assert shard is not None
        node = spec.node_by_id.get(row["node_id"])
        _verify(node is not None, "outcome uses unknown node", shard_id=shard_id, field="node_id", basis={"node_id": row["node_id"]})
        assert node is not None
        summary = _parse_json(row["summary_json"], shard_id=shard_id, field_name="summary")
        _verify(
            content_digest(summary) == row["summary_digest"],
            "summary digest mismatch",
            shard_id=shard_id,
            field="summary_digest",
        )
        revision = next((r for r in spec.revisions if r.revision == int(row["revision"])), None)
        _verify(revision is not None, "outcome uses unknown revision", shard_id=shard_id, field="revision")
        assert revision is not None
        samples = generate_samples(spec, shard, node)
        expected_summary, expected_digest = aggregate_samples(samples, node, revision.config)
        _verify(
            expected_digest == row["summary_digest"],
            "independently recomputed aggregate differs",
            shard_id=shard_id,
            field="summary_digest",
            basis={"expected": expected_digest, "ledger": row["summary_digest"]},
        )
        _verify(
            int(row["resource_units"]) == shard.resource_units,
            "resource_units mismatch",
            shard_id=shard_id,
            field="resource_units",
            basis={"expected": shard.resource_units, "actual": row["resource_units"]},
        )
        verified += 1

    view_rows = conn.execute("SELECT * FROM current_view").fetchall()
    for row in sorted(view_rows, key=lambda r: order_index.get(r["shard_id"], 10**9)):
        shard_id = row["shard_id"]
        outcome = conn.execute(
            "SELECT * FROM outcomes WHERE shard_id=? AND revision=? AND attempt=?",
            (shard_id, row["revision"], row["attempt"]),
        ).fetchone()
        _verify(outcome is not None, "current_view points to missing outcome", shard_id=shard_id, field="outcome_key")
        assert outcome is not None
        expected_canonical = canonical_text(
            {
                "shard_id": shard_id,
                "revision": int(row["revision"]),
                "summary": json.loads(outcome["summary_json"]),
                "summary_digest": outcome["summary_digest"],
            }
        )
        _verify(
            row["outcome_canonical"] == expected_canonical,
            "current view canonical outcome bytes differ",
            shard_id=shard_id,
            field="outcome_canonical",
        )
    return {"verified": verified, "current_view": len(view_rows)}


def _verify_invalidation(spec: Spec, conn: sqlite3.Connection) -> dict[str, Any]:
    plans: dict[int, list[str]] = {}
    previous_outcomes: dict[str, int] = {}
    first_revision = spec.revisions[0].revision
    for shard in spec.shards:
        row = conn.execute(
            "SELECT revision FROM outcomes WHERE shard_id=? AND state='COMMITTED' ORDER BY revision DESC, attempt DESC LIMIT 1",
            (shard.shard_id,),
        ).fetchone()
        if row is not None:
            previous_outcomes[shard.shard_id] = int(row["revision"])

    for revision in spec.revisions[1:]:
        plan = build_invalidation_plan(spec, revision)
        invalid = set(plan.invalid_shard_ids)
        plans[revision.revision] = list(plan.invalid_shard_ids)
        for shard_id in plan.invalid_shard_ids:
            row = conn.execute(
                "SELECT 1 FROM outcomes WHERE shard_id=? AND revision=? AND state='COMMITTED' LIMIT 1",
                (shard_id, revision.revision),
            ).fetchone()
            _verify(
                row is not None,
                "invalidated shard was not recomputed at new revision",
                shard_id=shard_id,
                field="revision",
                basis={"revision": revision.revision, "plan": plan.to_dict()},
            )
        for shard in spec.shards:
            if shard.shard_id in invalid:
                continue
            row = conn.execute(
                "SELECT 1 FROM outcomes WHERE shard_id=? AND revision=? AND state='COMMITTED' LIMIT 1",
                (shard.shard_id, revision.revision),
            ).fetchone()
            _verify(
                row is None,
                "unaffected shard was unnecessarily recomputed",
                shard_id=shard.shard_id,
                field="revision",
                basis={"revision": revision.revision},
            )
            old = conn.execute(
                "SELECT outcome_canonical FROM current_view WHERE shard_id=?",
                (shard.shard_id,),
            ).fetchone()
            initial = conn.execute(
                "SELECT summary_json, summary_digest FROM outcomes WHERE shard_id=? AND revision=? ORDER BY attempt DESC LIMIT 1",
                (shard.shard_id, first_revision),
            ).fetchone()
            if old is not None and initial is not None:
                expected = canonical_text(
                    {
                        "shard_id": shard.shard_id,
                        "revision": first_revision,
                        "summary": json.loads(initial["summary_json"]),
                        "summary_digest": initial["summary_digest"],
                    }
                )
                _verify(
                    old["outcome_canonical"] == expected,
                    "unaffected shard outcome bytes changed",
                    shard_id=shard.shard_id,
                    field="outcome_canonical",
                    basis={"revision": revision.revision},
                )
    return {"plans": plans}


def _verify_progress(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM progress WHERE id=1").fetchone()
    _verify(row is not None, "missing progress cursor", field="progress")
    assert row is not None
    cursor = _parse_json(row["cursor_json"], shard_id=None, field_name="cursor_json")
    _verify(digest_record(cursor) == row["content_digest"], "progress cursor digest mismatch", field="content_digest")
    count = conn.execute("SELECT COUNT(*) AS c FROM current_view").fetchone()["c"]
    _verify(
        int(cursor["completed_count"]) == int(count) == int(row["completed_count"]),
        "progress completed_count does not match current_view",
        field="completed_count",
        basis={"cursor": cursor["completed_count"], "view": count, "row": row["completed_count"]},
    )
    return {"completed_count": int(count), "topology_cursor": row["topology_cursor"]}


def _verify_recovery(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute("SELECT * FROM recovery_marks ORDER BY id").fetchall()
    marks = [dict(row) for row in rows]
    for row in rows:
        later = conn.execute(
            """
            SELECT o.shard_id FROM outcomes o
            JOIN attempts a ON a.shard_id=o.shard_id AND a.attempt=o.attempt
            WHERE o.shard_id=? AND o.attempt > ? AND o.state='COMMITTED'
            LIMIT 1
            """,
            (row["shard_id"], row["attempt"]),
        ).fetchone()
        _verify(
            later is not None,
            "recovered unknown shard has no later committed rerun",
            shard_id=row["shard_id"],
            field="recovery_marks",
            basis={"unknown_attempt": row["attempt"]},
        )
    return {"marks": len(marks)}
