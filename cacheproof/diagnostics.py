"""Read-only diagnostic projections and human-readable rendering."""

from __future__ import annotations

from typing import Any

from .canonical import canonical_text, utf8_key
from .content import digest_file
from .db import get_ledger_state, loads_json, verify_ledger_digest
from .errors import DiagnosticError
from .proof import build_segment_proof, verify_proof_digest


def dashboard_snapshot(store: Any) -> dict[str, Any]:
    """Build a stable, read-only dashboard projection.

    This function never rebuilds views, recovers unknowns, saves snapshots or
    changes ledger rows. Repeated refreshes therefore leave ledger batch and
    content digest unchanged.
    """

    ledger_error = None
    try:
        ledger = verify_ledger_digest(store.connection)
    except DiagnosticError as exc:
        ledger_error = exc.to_dict()
        state = get_ledger_state(store.connection)
        ledger = {
            "format_version": None if state is None else state["format_version"],
            "batch_id": None if state is None else state["batch_id"],
            "segment_set_version": None
            if state is None
            else state["segment_set_version"],
            "completed_cursor": None if state is None else state["completed_cursor"],
            "content_digest": None if state is None else state["content_digest"],
        }

    state = get_ledger_state(store.connection)
    segment_rows = store.connection.execute("SELECT * FROM segments").fetchall()
    segment_rows = sorted(
        segment_rows, key=lambda row: utf8_key(row["segment_id"])
    )
    segments = []
    for row in segment_rows:
        entries = store._entries_for_segment(store.connection, row["segment_id"])
        proof = build_segment_proof(
            segment_id=row["segment_id"],
            capacity_bytes=row["capacity_bytes"],
            segment_set_version=0 if state is None else state["segment_set_version"],
            segment_version=row["segment_version"],
            entries=entries,
        )
        proof_ok = True
        proof_error = None
        try:
            stored = loads_json(
                row["proof_json"], field=f"segments[{row['segment_id']}].proof_json"
            )
            verify_proof_digest(stored)
            if canonical_text(stored) != canonical_text(proof):
                proof_ok = False
                proof_error = "stored proof differs from rebuilt proof"
        except (DiagnosticError, KeyError, TypeError) as exc:
            proof_ok = False
            proof_error = str(exc)

        entry_reports = []
        for entry in entries:
            report = {
                "key": entry["entry_key"],
                "size_bytes": entry["size_bytes"],
                "digest": entry["digest"],
                "actual_digest": None,
                "digest_ok": False,
                "error": None,
            }
            try:
                actual = digest_file(
                    store.paths.absolute_entry_path(entry["file_relpath"]),
                    entry["entry_key"],
                    entry["size_bytes"],
                )
                report["actual_digest"] = actual
                report["digest_ok"] = actual == entry["digest"]
                if not report["digest_ok"]:
                    report["error"] = "digest mismatch"
            except DiagnosticError as exc:
                report["error"] = exc.message
            entry_reports.append(report)
        entry_reports.sort(key=lambda item: utf8_key(item["key"]))
        segments.append(
            {
                "segment_id": row["segment_id"],
                "capacity_bytes": proof["capacity_bytes"],
                "used_bytes": proof["used_bytes"],
                "remaining_bytes": proof["remaining_bytes"],
                "entry_count": proof["entry_count"],
                "segment_version": proof["segment_version"],
                "segment_set_version": proof["segment_set_version"],
                "proof_content_digest": proof["content_digest"],
                "proof_ok": proof_ok,
                "proof_error": proof_error,
                "entries": entry_reports,
            }
        )

    views = []
    view_rows = store.connection.execute("SELECT * FROM read_views").fetchall()
    view_rows = sorted(view_rows, key=lambda row: utf8_key(row["view_id"]))
    for row in view_rows:
        try:
            preview = store.preview_view(row["view_id"])
            views.append(
                {
                    "view_id": row["view_id"],
                    "view_version": row["view_version"],
                    "stale": preview["stale"],
                    "snapshot_digest": row["snapshot_digest"],
                    "last_rebuild_differences": preview["last_rebuild_differences"],
                    "pending_differences": preview["pending_differences"],
                    "error": None,
                }
            )
        except DiagnosticError as exc:
            views.append(
                {
                    "view_id": row["view_id"],
                    "view_version": row["view_version"],
                    "stale": None,
                    "snapshot_digest": row["snapshot_digest"],
                    "last_rebuild_differences": {},
                    "pending_differences": {},
                    "error": exc.to_dict(),
                }
            )

    integrity_error = any(
        not segment["proof_ok"]
        or any(not entry["digest_ok"] for entry in segment["entries"])
        for segment in segments
    )
    snapshot_error = any(view.get("error") for view in views)
    if ledger_error is not None:
        code = ledger_error["code"]
    elif integrity_error:
        code = "E_INTEGRITY"
    elif snapshot_error:
        code = "E_SNAPSHOT"
    else:
        code = "OK"
    return {
        "ok": code == "OK",
        "code": code,
        "workspace": str(store.paths.root),
        "ledger": ledger,
        "ledger_error": ledger_error,
        "segments": segments,
        "views": views,
        "unknown_entries": store.list_unknown_entries(),
        "recovery": {
            "exit_report_available": False,
            "basis": "fully committed SQLite rows plus filesystem observations",
        },
    }


def render_text_dashboard(dashboard: dict[str, Any]) -> str:
    lines: list[str] = []
    ledger = dashboard["ledger"]
    lines.append("Cacheproof diagnostic dashboard")
    lines.append(f"workspace: {dashboard['workspace']}")
    lines.append(
        "ledger: format={} batch={} segment_set_version={} cursor={}".format(
            ledger.get("format_version"),
            ledger.get("batch_id"),
            ledger.get("segment_set_version"),
            ledger.get("completed_cursor"),
        )
    )
    lines.append(f"ledger content digest: {ledger.get('content_digest')}")
    if dashboard.get("ledger_error"):
        lines.append(
            f"LEDGER ERROR {dashboard['ledger_error']['code']}: {dashboard['ledger_error']['message']}"
        )
    lines.append("")
    lines.append("Segments")
    for segment in dashboard["segments"]:
        lines.append(
            "- {segment_id}: capacity={capacity_bytes} used={used_bytes} remaining={remaining_bytes} entries={entry_count} version={segment_version} proof={digest} ok={proof_ok}".format(
                digest=segment["proof_content_digest"], **segment
            )
        )
        if segment.get("proof_error"):
            lines.append(f"  proof error: {segment['proof_error']}")
        for entry in segment["entries"]:
            lines.append(
                "  entry {key}: size={size_bytes} digest={digest} actual={actual_digest} ok={digest_ok}{error_suffix}".format(
                    error_suffix=""
                    if not entry.get("error")
                    else f" error={entry['error']}",
                    **entry,
                )
            )
    lines.append("")
    lines.append("Read views")
    for view in dashboard["views"]:
        lines.append(
            f"- {view['view_id']}: version={view['view_version']} stale={view['stale']} snapshot={view['snapshot_digest']}"
        )
        if view.get("error"):
            lines.append(f"  error: {view['error']}")
        if view.get("last_rebuild_differences"):
            lines.append(
                "  last rebuild: "
                + canonical_text(view["last_rebuild_differences"])
            )
        if view.get("pending_differences"):
            lines.append(
                "  pending rebuild: "
                + canonical_text(view["pending_differences"])
            )
    lines.append("")
    lines.append("Unknown entries")
    if not dashboard["unknown_entries"]:
        lines.append("- none")
    for unknown in dashboard["unknown_entries"]:
        lines.append(
            "- {segment_id}/{entry_key}: reason={reason} observed={observed_digest} size={observed_size_bytes} operation={operation_id}".format(
                **unknown
            )
        )
    lines.append("")
    lines.append(
        "crash exit report available: false (unknowns are inferred from committed prepare rows and observed files)"
    )
    return "\n".join(lines)


def inject_anomaly(
    store: Any,
    kind: str,
    *,
    segment_id: str | None = None,
    entry_key: str | None = None,
) -> None:
    """Inject one supported anomaly and then invoke the normal verifier.

    - ``truncate`` changes an entry content file.
    - ``capacity`` changes only the verifier's capacity parameter.
    - ``ledger-digest`` changes the persisted ledger digest field in SQLite.
    """

    if kind == "truncate":
        entry = store.connection.execute(
            "SELECT * FROM entries WHERE size_bytes > 0 ORDER BY segment_id, entry_key"
        ).fetchall()
        chosen = None
        for row in entry:
            if segment_id is not None and row["segment_id"] != segment_id:
                continue
            if entry_key is not None and row["entry_key"] != entry_key:
                continue
            chosen = row
            break
        if chosen is None:
            raise DiagnosticError(
                "E_NOT_FOUND",
                "no non-empty committed entry is available for truncation injection",
                segment_id=segment_id,
                entry_key=entry_key,
                field="entry",
            )
        path = store.paths.absolute_entry_path(chosen["file_relpath"])
        with path.open("r+b") as handle:
            handle.truncate(max(0, chosen["size_bytes"] - 1))
            handle.flush()
        store.logger.emit(
            "anomaly_injected",
            business_code="E_INTEGRITY",
            segment_id=chosen["segment_id"],
            entry_key=chosen["entry_key"],
            field="content_file",
            details={"injection": "truncate", "truncated_to_size": chosen["size_bytes"] - 1},
        )
        store.verify_all()
        raise DiagnosticError(
            "E_IO",
            "truncation injection did not trigger an integrity failure",
            segment_id=chosen["segment_id"],
            entry_key=chosen["entry_key"],
            field="content_file",
        )

    if kind == "capacity":
        rows = store.connection.execute(
            "SELECT * FROM segments ORDER BY segment_id"
        ).fetchall()
        chosen = None
        for row in rows:
            if segment_id is not None and row["segment_id"] != segment_id:
                continue
            if row["used_bytes"] > 0:
                chosen = row
                break
        if chosen is None:
            raise DiagnosticError(
                "E_CAPACITY",
                "no segment with used bytes is available for capacity injection",
                segment_id=segment_id,
                field="capacity_bytes",
                details={"required_bytes": 0, "available_bytes": 0},
            )
        override = max(0, chosen["used_bytes"] - 1)
        store.logger.emit(
            "anomaly_injected",
            business_code="E_CAPACITY",
            segment_id=chosen["segment_id"],
            field="capacity_bytes",
            details={
                "injection": "capacity",
                "capacity_override": override,
                "used_bytes": chosen["used_bytes"],
            },
        )
        store.verify_all(capacity_override={chosen["segment_id"]: override})
        raise DiagnosticError(
            "E_IO",
            "capacity injection did not trigger E_CAPACITY",
            segment_id=chosen["segment_id"],
            field="capacity_bytes",
        )

    if kind == "ledger-digest":
        wrong = "sha256:" + "f" * 64
        store.connection.execute(
            "UPDATE ledger_state SET content_digest = ? WHERE singleton = 1", (wrong,)
        )
        store.logger.emit(
            "anomaly_injected",
            business_code="E_SNAPSHOT",
            field="ledger.content_digest",
            details={"injection": "ledger-digest", "stored_digest": wrong},
        )
        store.verify_all()
        raise DiagnosticError(
            "E_IO",
            "ledger digest injection did not trigger E_SNAPSHOT",
            field="ledger.content_digest",
        )

    raise DiagnosticError(
        "E_SPEC",
        "unsupported anomaly injection kind",
        field="inject",
        details={"kind": kind, "supported": ["truncate", "capacity", "ledger-digest"]},
    )
