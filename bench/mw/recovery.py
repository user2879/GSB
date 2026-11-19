"""Post-snapshot continuation, replay verification and fault injection."""
from __future__ import annotations

from typing import Any

from .config import load_events
from .errors import OrderError, ReplayError, StructuredLogger
from .ledger import Ledger
from .session import Session

COMPARE_FIELDS = ("watermark_ms", "max_ts_seen_ms", "finalized_count",
                  "late_count", "sum_raw", "sum_fixed", "cursor_seq")


def verify_against_snapshot(session: Session,
                            snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    """Compare current committed state against every snapshotted partition."""
    current = {v["partition"]: v for v in
               session.ledger.cache_views(session.param_version, session.spec)}
    for expected in sorted(snapshot_payload["partition_views"],
                           key=lambda v: v["partition"]):
        key = expected["partition"]
        actual = current.get(key)
        if actual is None:
            raise ReplayError(
                "partition missing after replay", partition=key,
                seq=None, field="<partition>")
        for field in COMPARE_FIELDS:
            if actual[field] != expected[field]:
                raise ReplayError(
                    "replay diverges from snapshot",
                    partition=key,
                    seq=expected.get("cursor_seq"),
                    field=field,
                    snapshot=expected[field], replay=actual[field],
                    basis="snapshot content digest "
                          f"{snapshot_payload.get('content_digest')}")
    snap_queries = snapshot_payload.get("query_ledger", [])
    live_queries = session.ledger.fetch_queries(session.param_version)
    if len(snap_queries) > len(live_queries):
        raise ReplayError(
            "query ledger shorter than snapshot after replay",
            partition=None, seq=None, field="query_ledger",
            snapshot=len(snap_queries), replay=len(live_queries))
    for index, expected_q in enumerate(snap_queries):
        actual_q = dict(live_queries[index])
        for field in ("query_seq", "partition", "latency_ms", "budget_ms",
                      "over_budget"):
            if actual_q[field] != expected_q[field]:
                raise ReplayError(
                    "query ledger diverges from snapshot",
                    partition=expected_q.get("partition"),
                    seq=expected_q["query_seq"], field=field,
                    snapshot=expected_q[field], replay=actual_q[field])
    return {"status": "identical",
            "partitions": len(snapshot_payload["partition_views"]),
            "queries_verified": len(snap_queries)}


def continue_from_snapshot(session: Session, remaining_events_path: str,
                           logger: StructuredLogger,
                           *, allow_regression_ms: int | None = None
                           ) -> dict[str, Any]:
    """Resume ingestion beyond the snapshot cursor with replay checks.

    Events at or below a partition's committed cursor are treated as already
    durable and skipped. Any event beyond the cursor whose timestamp regresses
    by more than the spec out-of-order allowance raises E_ORDER; ordinary
    late/within-allowance events follow the normal E_LATE path instead.
    """
    events = load_events(remaining_events_path, session.spec)
    _, cursors = session.ledger.hydrate_engine(
        session.param_version, session.spec)
    # Cursor authority is the durable committed boundary, independent of the
    # in-memory engine's replayed state.
    rows = session.ledger.conn.execute(
        "SELECT partition, MAX(seq) AS m FROM raw_events "
        "WHERE param_version=? GROUP BY partition",
        (session.param_version,)).fetchall()
    cursors = {r["partition"]: r["m"] for r in rows}
    max_seen = _max_committed_ts(session.ledger, session.param_version)
    allowance = (session.spec.out_of_order_ms
                 if allow_regression_ms is None else allow_regression_ms)
    for event in events:
        if event.seq <= cursors.get(event.partition, -1):
            continue
        prior = max_seen.get(event.partition, 0)
        if prior - event.ts_ms > allowance:
            record = logger.emit(
                "E_ORDER",
                "timestamp regression beyond allowed watermark allowance",
                partition=event.partition, seq=event.seq,
                ts_ms=event.ts_ms, previous_max_ts_ms=prior,
                allowance_ms=allowance, source_line=event.source_line)
            raise OrderError(
                "timestamp regression beyond allowed out-of-order allowance",
                partition=event.partition, seq=event.seq,
                field="ts_ms", ts_ms=event.ts_ms,
                previous_max_ts_ms=prior, allowance_ms=allowance,
                log_record=record)
    ingest = session.ingest_events([
        e for e in events
        if e.seq > cursors.get(e.partition, -1)])
    session.drain()
    return ingest


def _max_committed_ts(ledger: Ledger,
                      param_version: int) -> dict[str, int]:
    rows = ledger.conn.execute(
        "SELECT partition, MAX(ts_ms) AS m FROM raw_events "
        "WHERE param_version=? GROUP BY partition",
        (param_version,)).fetchall()
    return {r["partition"]: r["m"] for r in rows}


def unknown_after_crash(session: Session) -> list[dict[str, Any]]:
    """List read-but-undeterminable state that must be reloaded.

    A forced kill can only lose uncommitted work; the committed cursor per
    partition is the durable boundary. Everything past it is reported as
    unknown rather than guessed, and no exit report is fabricated.
    """
    _, cursors = session.ledger.hydrate_engine(
        session.param_version, session.spec)
    return [{"partition": key, "committed_cursor_seq": cursor,
             "status": "unknown_reload_required"}
            for key, cursor in sorted(cursors.items())]
