"""Authoritative replay: rebuild receipts from the ledger and compare bytes.

The ledger's per-request rows are the sole truth source.  Replay folds the
four outcomes into per-window counters in a *sorted, order-independent* way and
recomputes each receipt with the same fixed-point rules used online.  Cached
rows that differ byte-for-byte are reported with the first offending field.
"""

from .budget import WindowCounters, compute_receipt, window_index_for
from .canonical import canonical_dumps
from .errors import PrevailError, ReplayError


def _fold_windows(request_rows, route):
    windows = {}
    for row in sorted(request_rows, key=lambda item: item["sequence"]):
        window_index = window_index_for(row["started_clock_ms"], route)
        counters = windows.get(window_index, WindowCounters())
        windows[window_index] = counters.add(row["final_outcome"])
    return windows


def replay_receipts(config, request_rows_by_route):
    """Recompute authoritative receipts keyed by window_id."""
    receipts = {}
    for route_id, rows in request_rows_by_route.items():
        route = config.routes[route_id]
        folded = _fold_windows(rows, route)
        for window_index, counters in sorted(folded.items()):
            receipt = compute_receipt(
                route, config.version, window_index, counters
            )
            receipts[receipt.window_id] = receipt
    return receipts


def _first_field_diff(expected_payload, actual_payload):
    keys = sorted(set(expected_payload) | set(actual_payload))
    for key in keys:
        if expected_payload.get(key) != actual_payload.get(key):
            return key
    return "<canonical-bytes>"


def compare_receipts(expected_receipts, stored_rows, error_cls):
    """Byte-level comparison; raise ``error_cls`` on the first mismatch."""
    import json as _json

    stored = {row["window_id"]: row for row in stored_rows}
    expected_ids = sorted(expected_receipts)
    stored_ids = sorted(stored)
    if expected_ids != stored_ids:
        missing = sorted(set(expected_ids) - set(stored_ids))
        extra = sorted(set(stored_ids) - set(expected_ids))
        window_id = (missing or extra)[0]
        raise error_cls(
            "receipt window set mismatch",
            window_id=window_id, field="<window-set>", missing=missing, extra=extra,
        )
    for window_id in expected_ids:
        expected = expected_receipts[window_id]
        row = stored[window_id]
        if expected.canonical_bytes != row["canonical_bytes"]:
            try:
                actual_payload = _json.loads(row["canonical_bytes"].decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                actual_payload = {}
            field = _first_field_diff(expected.payload, actual_payload)
            raise error_cls(
                "authoritative receipt differs from stored bytes",
                window_id=window_id, field=field,
                expected_digest=expected.digest, stored_digest=row["digest"],
            )
    return True


def prevail_check(config, ledger):
    """E_PREVAIL: cached view vs authoritative recomputation."""
    rows_by_route = {}
    for route_id in config.routes:
        rows_by_route[route_id] = ledger.requests_for_route(route_id, config.version)
    expected = replay_receipts(config, rows_by_route)
    stored = [row for row in ledger.current_receipts()
              if row["rule_version"] == config.version]
    compare_receipts(expected, stored, PrevailError)
    return expected


def replay_after_recovery(config, ledger):
    """E_REPLAY: full recovery verification over every stored receipt row."""
    rows_by_route = {}
    for route_id in config.routes:
        rows_by_route[route_id] = ledger.requests_for_route(route_id, config.version)
    expected = replay_receipts(config, rows_by_route)
    stored = [row for row in ledger.stored_receipts_all()
              if row["rule_version"] == config.version and row["stale"] == 0]
    compare_receipts(expected, stored, ReplayError)
    return expected


def canonical_view_digest(receipts):
    """Stable digest over a mapping of window_id -> Receipt for snapshots."""
    payload = {
        window_id: receipt.digest
        for window_id, receipt in sorted(receipts.items())
    }
    return canonical_dumps(payload)
