from __future__ import annotations

import json
from typing import Any

from .content import content_digest_from_versions, topology_from_versions
from .errors import PipelineError
from .fixed import FixedGrid
from .ledger import Ledger
from .spec import parse_spec
from .stats import VALID_SOURCES, compute_batch_metrics, error_summary_from_metrics


def first_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        if not (isinstance(expected, (int, float)) and isinstance(actual, (int, float))):
            return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        expected_keys = sorted(expected)
        actual_keys = sorted(actual)
        if expected_keys != actual_keys:
            return {
                "path": path,
                "expected_keys": expected_keys,
                "actual_keys": actual_keys,
            }
        for key in expected_keys:
            found = first_difference(expected[key], actual[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {"path": path, "expected_length": len(expected), "actual_length": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            found = first_difference(left, right, f"{path}[{index}]")
            if found:
                return found
        return None
    if expected != actual:
        return {"path": path, "expected": expected, "actual": actual}
    return None


def _verify_mismatch(
    *,
    session_id: str,
    batch_id: int,
    node_id: str,
    field: str,
    expected: Any,
    actual: Any,
    basis: str,
) -> PipelineError:
    return PipelineError(
        "E_REPLAY",
        "ledger self-consistency check failed",
        path=f"session={session_id}/batch={batch_id}/node={node_id}",
        field=field,
        details={
            "first_inconsistency": {
                "batch_id": batch_id,
                "node_id": node_id,
                "field": field,
                "expected": expected,
                "actual": actual,
            },
            "basis": basis,
        },
    )


def _check_metric_shape(metric: dict[str, Any], expected_key: str) -> str | None:
    if metric.get("metric") != expected_key:
        return "metric"
    if not isinstance(metric.get("value_units"), int) or metric["value_units"] < 0:
        return "value_units"
    if not isinstance(metric.get("error_bound_units"), int) or metric["error_bound_units"] < 0:
        return "error_bound_units"
    if not isinstance(metric.get("dominant_error_source"), str) or not metric["dominant_error_source"]:
        return "dominant_error_source"
    if not isinstance(metric.get("segments"), list):
        return "segments"
    for index, segment in enumerate(metric["segments"]):
        if not isinstance(segment.get("share_units"), int):
            return f"segments[{index}].share_units"
        if segment["share_units"] < 0:
            return f"segments[{index}].share_units"
    return None


def verify_ledger(ledger_path: str, *, session_id: str | None = None) -> dict[str, Any]:
    ledger = Ledger(ledger_path, create=False)
    try:
        sessions = ledger.list_sessions()
        if session_id is not None:
            sessions = [row for row in sessions if row["session_id"] == session_id]
            if not sessions:
                raise PipelineError(
                    "E_STATE", "session not found for verification", path=session_id, field="session_id"
                )
        checked_batches = 0
        checked_snapshots = 0
        session_summaries: list[dict[str, Any]] = []
        for session_row in sessions:
            sid = str(session_row["session_id"])
            session = ledger.get_session(sid)
            grid = FixedGrid(int(session["grid_power"]))
            spec = parse_spec(session["spec"], grid_power=grid.power, source_name=f"ledger:{sid}")
            if spec.session.session_id != sid:
                raise _verify_mismatch(
                    session_id=sid,
                    batch_id=0,
                    node_id="session",
                    field="session_id",
                    expected=sid,
                    actual=spec.session.session_id,
                    basis="session row id must match stored spec",
                )
            snapshots = ledger.list_snapshots(sid)
            snapshot_map: dict[int, dict[str, Any]] = {}
            for snapshot in snapshots:
                full = ledger.get_snapshot(str(snapshot["snapshot_id"]))
                ledger.verify_snapshot_integrity(full)
                checked_snapshots += 1
                snapshot_map[int(full["batch_id"])] = full
                if full["grid_power"] != grid.power or full["grid_scale"] != grid.scale:
                    raise PipelineError(
                        "E_SNAPSHOT",
                        "snapshot grid differs from its session grid",
                        path=str(full["snapshot_id"]),
                        field="grid",
                    )
            if 0 not in snapshot_map:
                raise PipelineError(
                    "E_SNAPSHOT", "session has no genesis snapshot", path=sid, field="genesis_snapshot"
                )
            batch_rows = ledger.get_batch_rows(sid)
            expected_batch_ids = sorted({request.batch for request in spec.requests})
            observed_batch_ids = [int(row["batch_id"]) for row in batch_rows]
            if observed_batch_ids != expected_batch_ids:
                raise _verify_mismatch(
                    session_id=sid,
                    batch_id=0,
                    node_id="session",
                    field="batch_ids",
                    expected=expected_batch_ids,
                    actual=observed_batch_ids,
                    basis="ledger batches must exactly cover the recorded request batches",
                )
            cumulative_hits = 0
            cumulative_requests = 0
            cumulative_consistent = 0
            for row in batch_rows:
                batch_id = int(row["batch_id"])
                batch = ledger.get_batch(sid, batch_id)
                records = ledger.get_batch_requests(sid, batch_id)
                if not records:
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="session",
                        field="requests",
                        expected="non-empty",
                        actual=[],
                        basis="a committed batch must contain request records",
                    )
                if records[0].seq != int(batch["start_seq"]) or records[-1].seq != int(batch["end_seq"]):
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="session",
                        field="seq_range",
                        expected=[batch["start_seq"], batch["end_seq"]],
                        actual=[records[0].seq, records[-1].seq],
                        basis="batch seq range must match request rows",
                    )
                if int(batch["request_count"]) != len(records):
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="session",
                        field="request_count",
                        expected=len(records),
                        actual=int(batch["request_count"]),
                        basis="batch request_count must equal the number of committed request rows",
                    )
                stored_request_rows = {
                    int(row["seq"]): dict(row)
                    for row in ledger.conn.execute(
                        "SELECT * FROM batch_requests WHERE session_id=? AND batch_id=? ORDER BY seq",
                        (sid, batch_id),
                    ).fetchall()
                }
                for record in records:
                    stored = stored_request_rows.get(record.seq)
                    if stored is None:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}]",
                            expected="present",
                            actual="missing",
                            basis="normalized request columns must exist for every record_json row",
                        )
                    expected_columns = {
                        "request_id": record.request_id,
                        "node_id": record.node_id,
                        "path": record.path,
                        "query": record.query,
                        "at_ms": record.at_ms,
                        "hit_source": record.hit_source,
                        "returned_etag": record.returned_etag,
                        "authoritative_etag": record.authoritative_etag,
                        "version_consistent": 1 if record.version_consistent else 0,
                        "remaining_ttl_ms": record.remaining_ttl_ms,
                        "entry_version": record.entry_version,
                        "stale": 1 if record.stale else 0,
                        "error_code": record.error_code,
                    }
                    actual_columns = {key: stored[key] for key in expected_columns}
                    diff = first_difference(expected_columns, actual_columns)
                    if diff:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].{diff.get('path', '$')}",
                            expected=diff.get("expected"),
                            actual=diff.get("actual"),
                            basis="normalized request columns must match record_json",
                        )
                previous_at = -1
                for record in records:
                    if record.hit_source not in VALID_SOURCES:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].hit_source",
                            expected=sorted(VALID_SOURCES),
                            actual=record.hit_source,
                            basis="hit source must belong to the supported classification",
                        )
                    if record.version_consistent != (record.returned_etag == record.authoritative_etag):
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].version_consistent",
                            expected=record.returned_etag == record.authoritative_etag,
                            actual=record.version_consistent,
                            basis="version-consistency flag must equal returned_etag == authoritative_etag",
                        )
                    if record.node_id not in spec.nodes:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].node_id",
                            expected=sorted(spec.nodes),
                            actual=record.node_id,
                            basis="request node must be declared by the session spec",
                        )
                    if record.path not in spec.content:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].path",
                            expected=sorted(spec.content),
                            actual=record.path,
                            basis="request path must be declared by the session spec",
                        )
                    if record.remaining_ttl_ms < 0:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].remaining_ttl_ms",
                            expected=">=0",
                            actual=record.remaining_ttl_ms,
                            basis="remaining TTL cannot be negative",
                        )
                    if record.stale and (record.error_code != "E_STALE" or record.remaining_ttl_ms != 0):
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].stale",
                            expected={"error_code": "E_STALE", "remaining_ttl_ms": 0},
                            actual={"error_code": record.error_code, "remaining_ttl_ms": record.remaining_ttl_ms},
                            basis="a stale hit must carry E_STALE and zero remaining TTL",
                        )
                    if record.error_code == "E_STALE" and not record.stale:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].error_code",
                            expected="stale=true",
                            actual=record.stale,
                            basis="E_STALE must only annotate stale hits",
                        )
                    if record.at_ms < previous_at:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].at_ms",
                            expected=f">={previous_at}",
                            actual=record.at_ms,
                            basis="request timestamps must be non-decreasing in a deterministic replay",
                        )
                    previous_at = record.at_ms
                    if record.entry_version <= 0:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=record.node_id,
                            field=f"requests[{record.seq}].entry_version",
                            expected=">0",
                            actual=record.entry_version,
                            basis="every cache entry version starts at one and only increases",
                        )
                batch_snapshot = snapshot_map.get(batch_id)
                topology = (
                    topology_from_versions(batch_snapshot["content_versions"])
                    if batch_snapshot is not None
                    else None
                )
                recomputed = compute_batch_metrics(spec, grid, batch_id, records, topology)
                diff = first_difference(recomputed, batch["metrics"])
                if diff:
                    field = str(diff.get("path", "metrics"))
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="multiple",
                        field=field,
                        expected=diff.get("expected", diff.get("expected_length", diff.get("expected_keys"))),
                        actual=diff.get("actual", diff.get("actual_length", diff.get("actual_keys"))),
                        basis="batch metrics must be reproducible from per-request hit records",
                    )
                recomputed_summary = error_summary_from_metrics(recomputed)
                diff = first_difference(recomputed_summary, batch["error_summary"])
                if diff:
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="multiple",
                        field=str(diff.get("path", "error_summary")),
                        expected=diff.get("expected"),
                        actual=diff.get("actual"),
                        basis="stored error bounds and dominant sources must match recomputed metrics",
                    )
                metric_rows = {
                    (str(metric_row["node_id"]), str(metric_row["metric_name"])): dict(metric_row)
                    for metric_row in ledger.conn.execute(
                        "SELECT * FROM node_metrics WHERE session_id=? AND batch_id=?",
                        (sid, batch_id),
                    ).fetchall()
                }
                expected_metric_rows: dict[tuple[str, str], dict[str, Any]] = {}
                for node_id in sorted(recomputed["nodes"]):
                    for metric_name in sorted(recomputed["nodes"][node_id]["metrics"]):
                        expected_metric_rows[(node_id, metric_name)] = recomputed["nodes"][node_id]["metrics"][metric_name]
                for metric_name in sorted(recomputed["aggregate"]["metrics"]):
                    expected_metric_rows[("__aggregate__", metric_name)] = recomputed["aggregate"]["metrics"][metric_name]
                if sorted(metric_rows) != sorted(expected_metric_rows):
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="multiple",
                        field="node_metrics.keys",
                        expected=sorted(expected_metric_rows),
                        actual=sorted(metric_rows),
                        basis="normalized metric rows must cover every node and aggregate metric exactly once",
                    )
                for key in sorted(expected_metric_rows):
                    metric = expected_metric_rows[key]
                    shape_error = _check_metric_shape(metric, key[1])
                    if shape_error:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=key[0],
                            field=shape_error,
                            expected="valid fixed-point metric",
                            actual=metric,
                            basis="metric shape validation",
                        )
                    row = metric_rows[key]
                    actual_row = {
                        "value_units": int(row["value_units"]),
                        "error_bound_units": int(row["error_bound_units"]),
                        "dominant_error_source": str(row["dominant_error_source"]),
                        "segments": json.loads(row["segments_json"]),
                    }
                    expected_row = {
                        "value_units": metric["value_units"],
                        "error_bound_units": metric["error_bound_units"],
                        "dominant_error_source": metric["dominant_error_source"],
                        "segments": metric["segments"],
                    }
                    diff = first_difference(expected_row, actual_row)
                    if diff:
                        raise _verify_mismatch(
                            session_id=sid,
                            batch_id=batch_id,
                            node_id=key[0],
                            field=f"{key[1]}.{diff.get('path', '$')}",
                            expected=diff.get("expected"),
                            actual=diff.get("actual"),
                            basis="normalized node_metrics row must match batch metrics JSON",
                        )
                snapshot = snapshot_map.get(batch_id)
                if snapshot is None:
                    raise PipelineError(
                        "E_SNAPSHOT",
                        "batch has no matching snapshot",
                        path=f"session={sid}/batch={batch_id}",
                        field="snapshot_id",
                    )
                if snapshot["snapshot_id"] != batch["snapshot_id"]:
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="session",
                        field="snapshot_id",
                        expected=batch["snapshot_id"],
                        actual=snapshot["snapshot_id"],
                        basis="batch row must reference the immutable snapshot committed in the same transaction",
                    )
                if snapshot["content_digest"] != batch["content_digest"]:
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="session",
                        field="content_digest",
                        expected=batch["content_digest"],
                        actual=snapshot["content_digest"],
                        basis="batch and snapshot must record the same content version-set digest",
                    )
                recomputed_digest = content_digest_from_versions(snapshot["content_versions"])
                if recomputed_digest != snapshot["content_digest"]:
                    raise PipelineError(
                        "E_SNAPSHOT",
                        "snapshot content digest is not reproducible",
                        path=snapshot["snapshot_id"],
                        field="content_digest",
                        details={"stored": snapshot["content_digest"], "recomputed": recomputed_digest},
                    )
                expected_cursor = {
                    "next_request_index": records[-1].seq + 1,
                    "next_batch_id": (
                        expected_batch_ids[expected_batch_ids.index(batch_id) + 1]
                        if batch_id != expected_batch_ids[-1]
                        else None
                    ),
                    "request_count": len(spec.requests),
                }
                if batch["cursor"] != expected_cursor:
                    raise _verify_mismatch(
                        session_id=sid,
                        batch_id=batch_id,
                        node_id="session",
                        field="cursor",
                        expected=expected_cursor,
                        actual=batch["cursor"],
                        basis="cursor must point to the next unprocessed request batch",
                    )
                batch_nodes = recomputed["nodes"]
                cumulative_hits += sum(int(batch_nodes[node_id]["hit_count"]) for node_id in batch_nodes)
                cumulative_requests += len(records)
                cumulative_consistent += sum(
                    int(batch_nodes[node_id]["version_consistent_count"]) for node_id in batch_nodes
                )
                checked_batches += 1
            active = ledger.active_state()
            if active is not None:
                active_snapshot = ledger.get_snapshot(active["snapshot_id"])
                ledger.verify_snapshot_integrity(active_snapshot)
            session_summaries.append(
                {
                    "session_id": sid,
                    "batches": len(batch_rows),
                    "requests": cumulative_requests,
                    "hit_rate_units": grid.q_ratio(cumulative_hits, cumulative_requests) if cumulative_requests else 0,
                    "version_consistency_rate_units": (
                        grid.q_ratio(cumulative_consistent, cumulative_requests) if cumulative_requests else 0
                    ),
                    "grid": grid.as_dict(),
                }
            )
        active = ledger.active_state()
        active_summary = None
        if active is not None:
            active_summary = {
                "snapshot_id": active["snapshot_id"],
                "session_id": active["session_id"],
                "batch_id": active["batch_id"],
                "content_digest": active["content_digest"],
                "grid_power": active["grid_power"],
                "grid_scale": active["grid_scale"],
                "active_cache_entries": ledger.active_cache_count(),
            }
        return {
            "ok": True,
            "ledger": ledger_path,
            "checks": {
                "batch_metrics_vs_request_records": "ok",
                "error_bounds_and_dominant_sources": "ok",
                "normalized_metric_rows": "ok",
                "snapshot_content_digests": "ok",
                "immutable_snapshot_references": "ok",
                "per_request_consistency": "ok",
            },
            "sessions": session_summaries,
            "batch_count": checked_batches,
            "snapshot_count": checked_snapshots,
            "active_state": active_summary,
        }
    finally:
        ledger.close()
