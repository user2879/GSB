from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .aggregate import FixedPointAggregator
from .cache import PageCache
from .canonical import canonical_bytes, canonical_json
from .errors import EngineError, ErrorCode
from .session import checkpoint_record
from .snapshot import load_snapshot, restore_snapshot


def replay_from_snapshot(
    snapshot_path: str | Path,
    *,
    cache: PageCache,
    aggregator: FixedPointAggregator,
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path)
    restore_report = restore_snapshot(snapshot, cache=cache, aggregator=aggregator, exact_disk=True)
    scan_order = [str(value) for value in snapshot["scan_order"]]
    position = int(snapshot["position"])
    if position < 0 or position > len(scan_order):
        raise EngineError(ErrorCode.E_REPLAY, "snapshot position is invalid", path="$snapshot.position", details={"position": position})
    reference_trace = snapshot["reference_trace"]
    if not isinstance(reference_trace, list) or len(reference_trace) != len(scan_order):
        raise EngineError(ErrorCode.E_REPLAY, "reference trace length is invalid", path="$snapshot.reference_trace")

    compared = 0
    for index in range(position, len(scan_order)):
        expected = reference_trace[index]
        page_id = scan_order[index]
        try:
            page = cache.get_page(page_id)
        except EngineError as exc:
            basis = exc.details.get("basis") if isinstance(exc.details, dict) else None
            raise EngineError(
                ErrorCode.E_REPLAY,
                "page access failed during replay",
                page_id=page_id,
                path=exc.path,
                details={
                    "access_index": index,
                    "field": "overflow_segment" if basis == "missing_overflow_segment" else "cache_access",
                    "basis": basis or exc.code.value,
                    "cause": exc.to_dict(),
                },
            ) from exc
        aggregator.consume_page(page)
        actual = checkpoint_record(access_index=index, page_id=page_id, cache=cache, aggregator=aggregator)
        _compare_trace(expected, actual, index, page_id)
        compared += 1

    final_output = aggregator.output()
    expected_output_sha = str(snapshot["reference_final_output_sha256"])
    actual_output_sha = hashlib.sha256(canonical_bytes(final_output)).hexdigest()
    if actual_output_sha != expected_output_sha or canonical_json(final_output) != canonical_json(snapshot["reference_final_output"]):
        last_page = scan_order[-1] if scan_order else None
        raise EngineError(
            ErrorCode.E_REPLAY,
            "final aggregation output differs from the snapshot reference",
            page_id=last_page,
            path="$snapshot.reference_final_output",
            details={
                "access_index": len(scan_order),
                "field": "final_output",
                "expected_sha256": expected_output_sha,
                "actual_sha256": actual_output_sha,
                "basis": "reference_trace_mismatch",
            },
        )
    return {
        "status": "reproducible",
        "position": position,
        "compared_records": compared,
        "total_records": len(scan_order),
        "final_output_sha256": actual_output_sha,
        "disk_diff": restore_report["disk_diff"],
        "restore": restore_report,
    }


def _compare_trace(expected: dict[str, Any], actual: dict[str, Any], index: int, page_id: str) -> None:
    expected_page = str(expected.get("page_id"))
    if expected_page != page_id:
        raise EngineError(
            ErrorCode.E_REPLAY,
            "replay scan record differs at the first page",
            page_id=page_id,
            path=f"$snapshot.reference_trace[{index}].page_id",
            details={
                "access_index": index,
                "field": "page_id",
                "expected": expected_page,
                "actual": page_id,
                "basis": "reference_trace_mismatch",
            },
        )
    for field in ("overflow_sequence", "swap_in_sequence", "swap_in_count", "resident_page_ids", "cache_state_sha256", "aggregation_state"):
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if canonical_json(expected_value) != canonical_json(actual_value):
            raise EngineError(
                ErrorCode.E_REPLAY,
                f"replay differs at field {field}",
                page_id=page_id,
                path=f"$snapshot.reference_trace[{index}].{field}",
                details={
                    "access_index": index,
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                    "basis": "reference_trace_mismatch",
                },
            )
