from __future__ import annotations

from typing import Any

from .errors import MeshError
from .ledger import Ledger


def _attempt_map(ledger: Ledger) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for request in ledger.request_sequence():
        request_id = str(request["request_id"])
        result[request_id] = ledger.attempts_for_request(request_id)
    return result


def _outcome_map(ledger: Ledger) -> dict[str, dict[str, Any]]:
    return {str(row["request_id"]): row for row in ledger.latest_outcomes()}


def _mismatch(
    request_id: str,
    service: str | None,
    field: str,
    expected: Any,
    actual: Any,
    **details: Any,
) -> MeshError:
    merged = {"expected": expected, "actual": actual}
    merged.update(details)
    return MeshError(
        "E_REPLAY",
        f"replay mismatch for request {request_id!r} in field {field}: expected {expected!r}, got {actual!r}",
        field=field,
        location=f"request[{request_id}].{field}",
        request_id=request_id,
        service=service,
        details=merged,
    )


def compare_ledgers(
    expected: Ledger,
    actual: Ledger,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Compare semantic attempt records and locate the first mismatch."""

    expected_requests = expected.request_sequence()
    actual_requests = {
        str(row["request_id"]): row for row in actual.request_sequence()
    }
    expected_outcomes = _outcome_map(expected)
    actual_outcomes = _outcome_map(actual)
    expected_attempts = _attempt_map(expected)
    actual_attempts = _attempt_map(actual)
    compared = 0
    skipped = 0
    for request in expected_requests:
        request_id = str(request["request_id"])
        service = str(request["service"])
        if request_id not in actual_requests:
            raise _mismatch(request_id, service, "request", "present", "missing")
        actual_request = actual_requests[request_id]
        if str(actual_request["service"]) != service:
            raise _mismatch(
                request_id,
                service,
                "service",
                service,
                str(actual_request["service"]),
            )
        expected_outcome = expected_outcomes.get(request_id)
        if expected_outcome is None:
            if require_complete:
                skipped += 1
                continue
            raise _mismatch(
                request_id,
                service,
                "final_outcome",
                "complete",
                "missing",
            )
        actual_outcome = actual_outcomes.get(request_id)
        if actual_outcome is None:
            raise _mismatch(
                request_id,
                service,
                "final_outcome",
                expected_outcome.get("final_outcome"),
                "missing",
            )
        for field in (
            "final_outcome",
            "error_code",
            "attempt_count",
            "request_seq",
            "service",
        ):
            if expected_outcome.get(field) != actual_outcome.get(field):
                raise _mismatch(
                    request_id,
                    service,
                    f"outcomes.{field}",
                    expected_outcome.get(field),
                    actual_outcome.get(field),
                )
        expected_rows = expected_attempts.get(request_id, [])
        actual_rows = actual_attempts.get(request_id, [])
        if len(expected_rows) != len(actual_rows):
            raise _mismatch(
                request_id,
                service,
                "attempts.length",
                len(expected_rows),
                len(actual_rows),
            )
        for index, expected_attempt in enumerate(expected_rows):
            actual_attempt = actual_rows[index]
            attempt_no = int(expected_attempt["attempt_no"])
            for field in (
                "attempt_no",
                "instance_id",
                "select_seq",
                "outcome",
                "error_code",
                "budget_used",
                "route_revision",
                "upstream_delay_ms",
                "final_attempt",
            ):
                if expected_attempt.get(field) != actual_attempt.get(field):
                    raise _mismatch(
                        request_id,
                        service,
                        f"attempts[{index}].{field}",
                        expected_attempt.get(field),
                        actual_attempt.get(field),
                        attempt_no=attempt_no,
                    )
        compared += 1
    extra = sorted(set(actual_requests) - {str(row["request_id"]) for row in expected_requests})
    if extra:
        request_id = extra[0]
        raise _mismatch(
            request_id,
            actual_requests[request_id].get("service"),
            "request",
            "absent",
            "present",
        )
    return {
        "status": "consistent",
        "code": "OK",
        "compared_requests": compared,
        "skipped_incomplete": skipped,
        "expected_ledger": str(expected.path),
        "actual_ledger": str(actual.path),
        "expected_attempts": expected.count_attempts(),
        "actual_attempts": actual.count_attempts(),
    }
