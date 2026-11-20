from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .errors import E_VERIFY


def _violation(
    check: str,
    detail: str,
    *,
    request_id: str | None = None,
    request_index: int | None = None,
    attempt_no: int | None = None,
    expected: Any = None,
    actual: Any = None,
) -> dict[str, Any]:
    return {
        "check": check,
        "request_id": request_id,
        "request_index": request_index,
        "attempt_no": attempt_no,
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def verify_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    requests = list(snapshot.get("requests", []))
    attempts = list(snapshot.get("attempts", []))
    rule_version = snapshot.get("rule_version")
    limits = snapshot.get("limits", {})
    global_limit = int(limits.get("global_concurrency", 0))
    service_limits = {
        str(key): int(value)
        for key, value in dict(limits.get("services", {})).items()
    }

    if snapshot.get("complete") is not True:
        violations.append(
            _violation(
                "snapshot.complete",
                "snapshot is not marked complete and cannot be treated as an exit report",
            )
        )
    if not rule_version:
        violations.append(_violation("snapshot.rule_version", "missing rule version"))

    for collection_name in ("requests", "attempts", "services"):
        for index, item in enumerate(snapshot.get(collection_name, [])):
            if item.get("rule_version") != rule_version:
                violations.append(
                    _violation(
                        f"binding.{collection_name}.rule_version",
                        "display item is not bound to the snapshot rule version",
                        request_id=item.get("request_id"),
                        request_index=item.get("request_index", index),
                        expected=rule_version,
                        actual=item.get("rule_version"),
                    )
                )

    attempts_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_request_row: dict[str, dict[str, Any]] = {}
    for request in requests:
        by_request_row[str(request.get("request_id"))] = request
    for attempt in attempts:
        attempts_by_request[str(attempt.get("request_id"))].append(attempt)

    for request_id, rows in attempts_by_request.items():
        rows.sort(key=lambda item: int(item.get("attempt_no", 0)))
    for request in requests:
        request_id = str(request.get("request_id"))
        rows = attempts_by_request.get(request_id, [])
        if not rows:
            violations.append(
                _violation(
                    "final.no_attempt_trace",
                    "request has no completed attempt trace",
                    request_id=request_id,
                    request_index=request.get("request_index"),
                )
            )
            continue
        expected_attempts = int(request.get("attempts_used", len(rows)))
        if len(rows) != expected_attempts:
            violations.append(
                _violation(
                    "final.attempts_used",
                    "request attempts_used does not match trace length",
                    request_id=request_id,
                    request_index=request.get("request_index"),
                    expected=expected_attempts,
                    actual=len(rows),
                )
            )
        numbers = [int(row.get("attempt_no", 0)) for row in rows]
        if numbers != list(range(1, len(rows) + 1)):
            violations.append(
                _violation(
                    "final.attempt_sequence",
                    "attempt sequence must be contiguous and start at 1",
                    request_id=request_id,
                    request_index=request.get("request_index"),
                    expected=list(range(1, len(rows) + 1)),
                    actual=numbers,
                )
            )
        max_retries = int(request.get("max_retries", 0))
        for row in rows:
            attempt_no = int(row.get("attempt_no", 0))
            if attempt_no > max_retries + 1:
                violations.append(
                    _violation(
                        "retry.budget",
                        "attempt exceeds the route retry budget",
                        request_id=request_id,
                        request_index=request.get("request_index"),
                        attempt_no=attempt_no,
                        expected=f"<={max_retries + 1}",
                        actual=attempt_no,
                    )
                )
        last = rows[-1]
        final_status = request.get("final_status")
        final_code = request.get("final_code")
        prior_successes = [
            row for row in rows[:-1] if row.get("result_status") == "success"
        ]
        if prior_successes:
            violations.append(
                _violation(
                    "final.trace_after_success",
                    "attempt trace continues after a successful attempt",
                    request_id=request_id,
                    request_index=request.get("request_index"),
                    attempt_no=prior_successes[0].get("attempt_no"),
                )
            )
        if final_status == "success":
            if last.get("result_status") != "success":
                violations.append(
                    _violation(
                        "final.success_trace",
                        "successful final result must end in a successful attempt",
                        request_id=request_id,
                        request_index=request.get("request_index"),
                        attempt_no=last.get("attempt_no"),
                        expected="success",
                        actual=last.get("result_status"),
                    )
                )
            if final_code is not None:
                violations.append(
                    _violation(
                        "final.success_code",
                        "successful final result must not carry an error code",
                        request_id=request_id,
                        request_index=request.get("request_index"),
                        expected=None,
                        actual=final_code,
                    )
                )
        elif final_status == "failure":
            if last.get("result_status") != "failure":
                violations.append(
                    _violation(
                        "final.failure_trace",
                        "failed final result must end in a failed attempt",
                        request_id=request_id,
                        request_index=request.get("request_index"),
                        attempt_no=last.get("attempt_no"),
                        expected="failure",
                        actual=last.get("result_status"),
                    )
                )
            if final_code != last.get("code"):
                violations.append(
                    _violation(
                        "final.failure_code",
                        "final code does not match the terminal attempt code",
                        request_id=request_id,
                        request_index=request.get("request_index"),
                        expected=last.get("code"),
                        actual=final_code,
                    )
                )
        else:
            violations.append(
                _violation(
                    "final.status",
                    "final status must be success or failure",
                    request_id=request_id,
                    request_index=request.get("request_index"),
                    actual=final_status,
                )
            )

    # Concurrency sweep: one permit is held from start_mono_ms through
    # end_mono_ms. End events sort before starts at the same timestamp.
    dispatch_rows = [
        row
        for row in attempts
        if bool(row.get("selected")) and row.get("start_mono_ms") is not None
    ]
    events: list[tuple[float, int, str]] = []
    for row in dispatch_rows:
        start = float(row.get("permit_start_mono_ms") or row["start_mono_ms"])
        end = float(
            row.get("permit_end_mono_ms")
            or row.get("end_mono_ms")
            or start
        )
        service = str(row.get("service"))
        events.append((start, 1, service))
        events.append((end, -1, service))
    active_global = 0
    active_services: dict[str, int] = defaultdict(int)
    for _, delta, service in sorted(events, key=lambda item: (item[0], item[1])):
        active_global += delta
        active_services[service] += delta
        if active_global > global_limit:
            violations.append(
                _violation(
                    "concurrency.global",
                    "derived global overlap exceeds the configured limit",
                    expected=f"<={global_limit}",
                    actual=active_global,
                )
            )
            break
        limit = service_limits.get(service)
        if limit is not None and active_services[service] > limit:
            violations.append(
                _violation(
                    "concurrency.service",
                    "derived service overlap exceeds the configured quota",
                    expected=f"{service}<={limit}",
                    actual=active_services[service],
                )
            )
            break
    for row in attempts:
        expected_service_limit = service_limits.get(str(row.get("service")))
        recorded_global = row.get("global_active_at_start")
        recorded_service = row.get("service_active_at_start")
        if recorded_global is not None and int(recorded_global) > global_limit:
            violations.append(
                _violation(
                    "concurrency.recorded_global",
                    "recorded global occupancy exceeds the limit",
                    request_id=row.get("request_id"),
                    request_index=row.get("request_index"),
                    attempt_no=row.get("attempt_no"),
                    expected=f"<={global_limit}",
                    actual=recorded_global,
                )
            )
        if (
            expected_service_limit is not None
            and recorded_service is not None
            and int(recorded_service) > expected_service_limit
        ):
            violations.append(
                _violation(
                    "concurrency.recorded_service",
                    "recorded service occupancy exceeds the quota",
                    request_id=row.get("request_id"),
                    request_index=row.get("request_index"),
                    attempt_no=row.get("attempt_no"),
                    expected=f"<={expected_service_limit}",
                    actual=recorded_service,
                )
            )

    # During an open interval no attempt may have actually been dispatched to
    # that instance. E_CIRCUIT fast-fail attempts remain visible as targeted,
    # unselected trace rows.
    open_intervals: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for event in snapshot.get("circuit_events", []):
        if event.get("event") not in {"opened", "reopened"}:
            continue
        opened = event.get("opened_at_mono_ms")
        until = event.get("cooldown_until_mono_ms")
        if opened is None or until is None:
            continue
        open_intervals[(str(event.get("service")), int(event.get("port")))].append(
            (float(opened), float(until))
        )
    for row in dispatch_rows:
        key = (str(row.get("service")), int(row.get("target_port")))
        start = float(
            row.get("circuit_admitted_mono_ms") or row["start_mono_ms"]
        )
        for opened, until in open_intervals.get(key, []):
            if opened <= start < until:
                violations.append(
                    _violation(
                        "circuit.cooldown_dispatch",
                        "instance was dispatched while its circuit cooldown was active",
                        request_id=row.get("request_id"),
                        request_index=row.get("request_index"),
                        attempt_no=row.get("attempt_no"),
                        expected=f"no dispatch in [{opened:.3f},{until:.3f})",
                        actual=start,
                    )
                )
                break
    for row in attempts:
        if row.get("code") == "E_CIRCUIT" and bool(row.get("selected")):
            violations.append(
                _violation(
                    "circuit.fast_fail_selected",
                    "an E_CIRCUIT fast-fail attempt must not hold a selection permit",
                    request_id=row.get("request_id"),
                    request_index=row.get("request_index"),
                    attempt_no=row.get("attempt_no"),
                )
            )

    service_weights: dict[tuple[str, int], int] = {}
    service_weight_sums: dict[str, int] = {}
    for service in snapshot.get("services", []):
        name = str(service.get("service"))
        total = int(service.get("total_weight", 0))
        service_weight_sums[name] = total
        if total <= 0:
            violations.append(
                _violation(
                    "selection.weight_sum",
                    "service weight sum must be greater than zero",
                    expected=">0",
                    actual=total,
                )
            )
        calculated = 0
        for instance in service.get("instances", []):
            if instance.get("rule_version") != rule_version:
                violations.append(
                    _violation(
                        "binding.services.instance.rule_version",
                        "instance display row is not bound to the snapshot rule version",
                        request_index=0,
                        expected=rule_version,
                        actual=instance.get("rule_version"),
                    )
                )
            weight = int(instance.get("weight", 0))
            calculated += weight
            service_weights[(name, int(instance.get("port")))] = weight
            if weight < 0:
                violations.append(
                    _violation(
                        "selection.negative_weight",
                        "instance weight must be non-negative",
                        expected=">=0",
                        actual=weight,
                    )
                )
        if calculated != total:
            violations.append(
                _violation(
                    "selection.weight_sum",
                    "declared service total_weight does not equal instance weights",
                    expected=calculated,
                    actual=total,
                )
            )
    for row in snapshot.get("budget", {}).get("services", []):
        if row.get("rule_version") != rule_version:
            violations.append(
                _violation(
                    "binding.budget.rule_version",
                    "budget display row is not bound to the snapshot rule version",
                    request_index=0,
                    expected=rule_version,
                    actual=row.get("rule_version"),
                )
            )
    for row in attempts:
        key = (str(row.get("service")), int(row.get("target_port")))
        weight = service_weights.get(key)
        if weight is None:
            violations.append(
                _violation(
                    "selection.unknown_target",
                    "selected target is not part of the service definition",
                    request_id=row.get("request_id"),
                    request_index=row.get("request_index"),
                    attempt_no=row.get("attempt_no"),
                )
            )
        elif weight == 0:
            violations.append(
                _violation(
                    "selection.zero_weight",
                    "a zero-weight instance was selected",
                    request_id=row.get("request_id"),
                    request_index=row.get("request_index"),
                    attempt_no=row.get("attempt_no"),
                )
            )

    violations.sort(
        key=lambda item: (
            item.get("request_index") if item.get("request_index") is not None else -1,
            item.get("request_id") or "",
            item.get("attempt_no") if item.get("attempt_no") is not None else -1,
            item.get("check") or "",
        )
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in violations:
        key = repr(sorted(item.items(), key=lambda pair: pair[0]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    first = unique[0] if unique else None
    return {
        "schema_version": 1,
        "status": "pass" if not unique else "fail",
        "code": None if not unique else E_VERIFY,
        "snapshot_id": snapshot.get("snapshot_id"),
        "round_id": snapshot.get("round_id"),
        "rule_version_expected": rule_version,
        "checks": {
            "final_trace": not any(str(item["check"]).startswith("final.") for item in unique),
            "concurrency_budget": not any(
                str(item["check"]).startswith("concurrency.") for item in unique
            ),
            "circuit_cooldown": not any(
                str(item["check"]).startswith("circuit.") for item in unique
            ),
            "retry_budget": not any(
                str(item["check"]).startswith("retry.") for item in unique
            ),
            "selection_evidence": not any(
                str(item["check"]).startswith("selection.") for item in unique
            ),
            "single_rule_version": not any(
                str(item["check"]).startswith("binding.") for item in unique
            ),
        },
        "first_deviation": (
            {
                "request_id": first.get("request_id"),
                "attempt_no": first.get("attempt_no"),
                "check": first.get("check"),
                "detail": first.get("detail"),
            }
            if first
            else None
        ),
        "violations": unique,
        "violation_count": len(unique),
    }
