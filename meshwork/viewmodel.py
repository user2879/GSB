from __future__ import annotations

import time
from typing import Any

from .ledger import Ledger
from .model import MeshSpec, RouteRevision
from .selector import WeightedSelector
from .snapshot import SnapshotStore
from .stats import build_stats


def _breaker_view(
    instance_id: str,
    runtime_state: dict[str, Any] | None,
    ledger: Ledger,
) -> dict[str, Any]:
    if runtime_state is not None:
        for item in runtime_state.get("breakers", []):
            if item.get("instance_id") == instance_id:
                return dict(item)
    raw = ledger.latest_breakers().get(instance_id)
    if not raw:
        return {
            "instance_id": instance_id,
            "state": "CLOSED",
            "consecutive_failures": 0,
            "cooldown_remaining_ms": 0,
            "half_open_dispatched": 0,
            "half_open_successes": 0,
        }
    remaining = 0
    if raw.get("state") == "OPEN":
        remaining = max(0, int(raw.get("open_until_wall_ms", 0)) - int(time.time() * 1000))
    return {
        "instance_id": instance_id,
        "state": str(raw.get("state", "CLOSED")),
        "consecutive_failures": int(raw.get("consecutive_failures", 0)),
        "cooldown_remaining_ms": remaining,
        "half_open_dispatched": int(raw.get("half_open_dispatched", 0)),
        "half_open_successes": int(raw.get("half_open_successes", 0)),
    }


def build_view_model(
    spec: MeshSpec,
    revision: RouteRevision,
    ledger: Ledger,
    snapshot_store: SnapshotStore,
    runtime_state: dict[str, Any] | None = None,
    *,
    last_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fully sorted, read-only diagnostics view.

    This function performs SELECTs and in-memory calculations only. The caller
    can use ``ledger_refresh.unchanged`` to prove a refresh did not mutate the
    ledger content digest.
    """

    digest_before = ledger.content_digest()
    selector = WeightedSelector(spec, revision)
    attempts = ledger.latest_attempts()
    outcomes = ledger.latest_outcomes()
    budget_global, budget_by_service = ledger.budget_totals()
    selection_counts = ledger.attempts_by_instance()
    stats = build_stats(outcomes, attempts, spec.session.scale)
    attempts_by_instance: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        attempts_by_instance.setdefault(str(attempt["instance_id"]), []).append(attempt)
    services: list[dict[str, Any]] = []
    breaker_rows: list[dict[str, Any]] = []
    for service in spec.ordered_services():
        effective = spec.effective_weights(service.id, revision)
        instances: list[dict[str, Any]] = []
        for instance in service.instances:
            breaker = _breaker_view(instance.id, runtime_state, ledger)
            breaker["service"] = service.id
            breaker["failure_threshold"] = spec.session.breaker.failure_threshold
            breaker["half_open_probes"] = spec.session.breaker.half_open_probes
            breaker_rows.append(breaker)
            instance_attempts = attempts_by_instance.get(instance.id, [])
            last_attempt = instance_attempts[-1] if instance_attempts else None
            instances.append(
                {
                    "id": instance.id,
                    "host": instance.host,
                    "port": instance.port,
                    "base_weight": instance.weight,
                    "effective_weight": effective.get(instance.id, 0),
                    "selected_count": int(selection_counts.get(instance.id, 0)),
                    "last_selection": None
                    if last_attempt is None
                    else {
                        "request_id": last_attempt["request_id"],
                        "attempt_no": int(last_attempt["attempt_no"]),
                        "select_seq": int(last_attempt["select_seq"]),
                        "outcome": last_attempt["outcome"],
                    },
                    "breaker": breaker,
                }
            )
        services.append(
            {
                "id": service.id,
                "quota": service.quota,
                "budget_used": int(budget_by_service.get(service.id, 0)),
                "instances": instances,
            }
        )
    selections: list[dict[str, Any]] = []
    for attempt in attempts:
        selected = selector.select(
            str(attempt["service"]), str(attempt["request_id"]), int(attempt["attempt_no"])
        )
        selections.append(
            {
                "request_id": attempt["request_id"],
                "request_seq": int(attempt["request_seq"]),
                "service": attempt["service"],
                "attempt_no": int(attempt["attempt_no"]),
                "select_seq": int(attempt["select_seq"]),
                "instance_id": attempt["instance_id"],
                "weight_position": selected.position,
                "total_weight": selected.total_weight,
                "weight_ranges": [
                    {
                        "instance_id": item.instance.id,
                        "start": item.start,
                        "end": item.end,
                    }
                    for item in selected.ranges
                ],
                "outcome": attempt["outcome"],
            }
        )
    selections.sort(key=lambda item: (item["request_seq"], item["attempt_no"], item["select_seq"]))
    timeline_attempts = sorted(
        attempts,
        key=lambda item: (
            int(item["wait_start_mono_ms"]),
            int(item["request_seq"]),
            int(item["attempt_no"]),
        ),
    )
    timeline: list[dict[str, Any]] = []
    for attempt in timeline_attempts:
        timeline.append(
            {
                "request_id": attempt["request_id"],
                "service": attempt["service"],
                "instance_id": attempt["instance_id"],
                "attempt_no": int(attempt["attempt_no"]),
                "wait_start_mono_ms": int(attempt["wait_start_mono_ms"]),
                "wait_end_mono_ms": int(attempt["wait_end_mono_ms"]),
                "started_mono_ms": int(attempt["started_mono_ms"]),
                "finished_mono_ms": int(attempt["finished_mono_ms"]),
                "budget_wait_ms": int(attempt["budget_wait_ms"]),
                "duration_ms": int(attempt["duration_ms"]),
                "global_inflight": int(attempt["global_inflight"] or 0),
                "service_inflight": int(attempt["service_inflight"] or 0),
                "global_limit": int(attempt["global_limit"] or spec.session.global_concurrency),
                "service_quota": int(attempt["service_quota"] or spec.services[attempt["service"]].quota),
                "outcome": attempt["outcome"],
                "error_code": attempt["error_code"],
                "budget_used": bool(attempt["budget_used"]),
            }
        )
    digest_after = ledger.content_digest()
    snapshots = snapshot_store.list()
    return {
        "generated_wall_ms": int(time.time() * 1000),
        "session": {
            "route_revision": revision.revision,
            "description": revision.description,
            "scale": spec.session.scale,
            "global_concurrency": spec.session.global_concurrency,
            "attempt_timeout_ms": spec.session.attempt_timeout_ms,
            "retry_backoff_ms": list(spec.session.retry_backoff_ms),
            "breaker": {
                "failure_threshold": spec.session.breaker.failure_threshold,
                "cooldown_ms": spec.session.breaker.cooldown_ms,
                "half_open_probes": spec.session.breaker.half_open_probes,
            },
            "instance_version": spec.instance_version(),
            "config_digest": spec.config_digest(),
        },
        "services": services,
        "breakers": sorted(breaker_rows, key=lambda item: (item["service"], item["instance_id"])),
        "budget": {
            "global_limit": spec.session.global_concurrency,
            "global_used": int(budget_global),
            "by_service": {key: int(budget_by_service[key]) for key in sorted(budget_by_service)},
            "runtime": runtime_state.get("budget") if runtime_state else None,
        },
        "selections": selections,
        "recent_selections": selections[-40:],
        "timeline": timeline,
        "stats": stats,
        "versions": {
            "schema_version": spec.schema_version,
            "route_revision": revision.revision,
            "instance_version": spec.instance_version(),
            "config_digest": spec.config_digest(),
            "ledger_format_version": int(ledger.meta_dict().get("format_version", "0")),
            "ledger_content_digest": digest_after,
            "request_sequence_digest": ledger.meta_dict().get("request_sequence_digest", ""),
            "snapshot_repair": None if runtime_state is None else runtime_state.get("snapshot_repair"),
            "recovery": None if runtime_state is None else runtime_state.get("recovery"),
        },
        "snapshots": snapshots,
        "unknown_attempts": ledger.unknown_attempts(),
        "last_report": last_report,
        "ledger_refresh": {
            "digest_before": digest_before,
            "digest_after": digest_after,
            "unchanged": digest_before == digest_after,
        },
    }
