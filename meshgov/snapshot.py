from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from .errors import SnapshotError
from .ledger import Ledger
from .rules import RuleSet, diff_rules
from .util import (
    atomic_write_json,
    canonical_json,
    json_pretty_bytes,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)


def build_snapshot(
    round_id: str,
    ledger: Ledger,
    rules: RuleSet,
    *,
    rules_changed_during_round: bool,
    observed_rule_version: str,
) -> dict[str, Any]:
    data = ledger.round_data(round_id)
    round_row = data["round"]
    if round_row is None:
        raise SnapshotError("round not found", "$.round_id", {"round_id": round_id})
    requests = data["requests"]
    attempts = data["attempts"]
    circuit_events = data["circuit_events"]
    budget = ledger.budget_state(round_id) or {}

    attempts_by_instance: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for attempt in attempts:
        attempts_by_instance.setdefault(
            (attempt["service"], attempt["target_port"]), []
        ).append(attempt)

    circuit_by_port = {
        int(row["port"]): row for row in ledger.current_circuit_states().values()
    }
    services_payload: list[dict[str, Any]] = []
    for service in rules.services:
        instances_payload: list[dict[str, Any]] = []
        for instance in service.instances:
            rows = attempts_by_instance.get((service.name, instance.port), [])
            targeted_count = sum(bool(row["targeted"]) for row in rows)
            selected_count = sum(bool(row["selected"]) for row in rows)
            failure_count = sum(
                row["result_status"] == "failure" and bool(row["selected"]) for row in rows
            )
            success_count = sum(
                row["result_status"] == "success" and bool(row["selected"]) for row in rows
            )
            fast_fail_count = sum(row["code"] == "E_CIRCUIT" for row in rows)
            circuit = circuit_by_port.get(instance.port, {})
            instances_payload.append(
                {
                    "rule_version": rules.rules_version,
                    "instance": instance.name,
                    "host": instance.host,
                    "port": instance.port,
                    "weight": instance.weight,
                    "weight_sum": service.total_weight,
                    "failure_threshold": instance.failure_threshold,
                    "cooldown_ms": instance.cooldown_ms,
                    "targeted_count": targeted_count,
                    "selected_count": selected_count,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "fast_fail_count": fast_fail_count,
                    "circuit_state": circuit.get("state", "closed"),
                    "consecutive_failures": circuit.get("consecutive_failures", 0),
                    "opened_at_mono_ms": circuit.get("opened_at_mono_ms"),
                    "cooldown_until_mono_ms": circuit.get("cooldown_until_mono_ms"),
                }
            )
        services_payload.append(
            {
                "rule_version": rules.rules_version,
                "service": service.name,
                "concurrency_quota": service.concurrency_quota,
                "total_weight": service.total_weight,
                "targeted_count": sum(item["targeted_count"] for item in instances_payload),
                "selected_count": sum(item["selected_count"] for item in instances_payload),
                "failure_count": sum(item["failure_count"] for item in instances_payload),
                "fast_fail_count": sum(item["fast_fail_count"] for item in instances_payload),
                "instances": instances_payload,
            }
        )

    for request in requests:
        request["rule_version"] = rules.rules_version
    for attempt in attempts:
        attempt["rule_version"] = rules.rules_version
    for event in circuit_events:
        event["rule_version"] = rules.rules_version

    succeeded = sum(request["final_status"] == "success" for request in requests)
    failed = len(requests) - succeeded
    code_counts = Counter(
        request["final_code"] for request in requests if request["final_code"]
    )
    total_retries = sum(max(0, int(request["attempts_used"]) - 1) for request in requests)
    fast_fails = sum(attempt["code"] == "E_CIRCUIT" for attempt in attempts)
    selected_attempts = sum(bool(attempt["selected"]) for attempt in attempts)
    global_info = budget.get(
        "global",
        {
            "limit": rules.global_concurrency,
            "active": 0,
            "peak": 0,
        },
    )
    budget_services = budget.get("services", {})
    service_budget: list[dict[str, Any]] = []
    for service in rules.services:
        saved = budget_services.get(service.name, {})
        service_budget.append(
            {
                "rule_version": rules.rules_version,
                "service": service.name,
                "limit": service.concurrency_quota,
                "active": saved.get("active", 0),
                "peak": saved.get("peak", 0),
            }
        )
    conclusion = _conclusion(
        total=len(requests), succeeded=succeeded, failed=failed, fast_fails=fast_fails
    )
    summary = {
        "rule_version": rules.rules_version,
        "status": "success" if failed == 0 else "partial",
        "conclusion": conclusion,
        "requests_total": len(requests),
        "requests_succeeded": succeeded,
        "requests_failed": failed,
        "attempts_total": len(attempts),
        "selected_attempts": selected_attempts,
        "retries_used": total_retries,
        "circuit_fast_failures": fast_fails,
        "final_code_counts": dict(sorted(code_counts.items())),
        "global_concurrency": {
            "limit": global_info.get("limit", rules.global_concurrency),
            "peak": global_info.get("peak", 0),
            "active": global_info.get("active", 0),
        },
        "service_concurrency": service_budget,
        "circuits_open": sum(
            instance["circuit_state"] in {"open", "half_open"}
            for service in services_payload
            for instance in service["instances"]
        ),
        "rules_changed_during_round": rules_changed_during_round,
        "observed_rule_version": observed_rule_version,
    }
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "snapshot_id": None,
        "round_id": round_id,
        "rule_version": rules.rules_version,
        "source_rule_version": rules.rules_version,
        "observed_rule_version": observed_rule_version,
        "rules_changed_during_round": rules_changed_during_round,
        "created_at_utc": utc_now_iso(),
        "complete": True,
        "rules": rules.raw_document,
        "limits": {
            "global_concurrency": rules.global_concurrency,
            "services": {
                service.name: service.concurrency_quota for service in rules.services
            },
        },
        "summary": summary,
        "services": services_payload,
        "requests": requests,
        "attempts": attempts,
        "circuit_events": circuit_events,
        "budget": {
            "global": global_info,
            "services": service_budget,
        },
        "selection_algorithm": "smooth-weighted-round-robin-v1",
    }
    digest = sha256_bytes(canonical_json(snapshot).encode("utf-8"))
    snapshot["snapshot_id"] = f"snap-{round_id}-{digest[:12]}"
    return snapshot


def _conclusion(*, total: int, succeeded: int, failed: int, fast_fails: int) -> str:
    if failed == 0:
        return (
            f"PASS: {succeeded}/{total} requests succeeded; all concurrency and retry "
            "budgets were observed."
        )
    return (
        f"PARTIAL: {succeeded}/{total} requests succeeded, {failed} failed; "
        f"{fast_fails} circuit fast-fails were recorded. Inspect instance/code evidence."
    )


class SnapshotStore:
    def __init__(self, directory: str | Path, *, injected_failure: bool = False) -> None:
        self.directory = Path(directory)
        self.injected_failure = injected_failure

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.json"

    def publish(self, snapshot: dict[str, Any], ledger: Ledger) -> Path:
        if self.injected_failure:
            raise SnapshotError(
                "injected snapshot write failure",
                "$.snapshot",
                {"injection": "E_SNAPSHOT", "latest_path": str(self.latest_path)},
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        snapshot_id = str(snapshot["snapshot_id"])
        target = self.directory / f"{snapshot_id}.json"
        payload_sha = sha256_bytes(json_pretty_bytes(snapshot))
        try:
            if not target.exists():
                atomic_write_json(target, snapshot, read_only=True)
            else:
                if sha256_file(target) != payload_sha:
                    raise SnapshotError(
                        "immutable snapshot id collision",
                        "$.snapshot_id",
                        {"snapshot_id": snapshot_id},
                    )
            # The active view is replaced as one complete document. It never
            # merges fields from the previous active snapshot.
            atomic_write_json(self.latest_path, snapshot, read_only=False)
            existing = ledger.snapshot_for_round(str(snapshot["round_id"]))
            if existing is None:
                ledger.register_snapshot(
                    snapshot_id,
                    str(snapshot["round_id"]),
                    str(snapshot["rule_version"]),
                    str(target),
                    payload_sha,
                )
            elif existing["snapshot_id"] != snapshot_id:
                raise SnapshotError(
                    "round already has a different immutable snapshot",
                    "$.snapshot_id",
                    {
                        "round_id": snapshot["round_id"],
                        "existing_snapshot_id": existing["snapshot_id"],
                        "new_snapshot_id": snapshot_id,
                    },
                )
        except SnapshotError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise SnapshotError(
                "failed to persist snapshot atomically",
                "$.snapshot",
                {"snapshot_id": snapshot_id, "reason": str(exc)},
            ) from exc
        return target


def load_snapshot(path: str | Path) -> dict[str, Any]:
    snapshot_path = Path(path)
    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(
            "unable to read snapshot",
            "$",
            {"source": str(snapshot_path), "reason": str(exc)},
        ) from exc


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_rules = old.get("rules", {})
    new_rules = new.get("rules", {})
    return {
        "old_snapshot_id": old.get("snapshot_id"),
        "new_snapshot_id": new.get("snapshot_id"),
        "old_rule_version": old.get("rule_version"),
        "new_rule_version": new.get("rule_version"),
        "version_changed": old.get("rule_version") != new.get("rule_version"),
        "route_diff": diff_rules(old_rules, new_rules),
        "summary_diff": {
            "requests_total": [
                old.get("summary", {}).get("requests_total"),
                new.get("summary", {}).get("requests_total"),
            ],
            "requests_succeeded": [
                old.get("summary", {}).get("requests_succeeded"),
                new.get("summary", {}).get("requests_succeeded"),
            ],
            "circuit_fast_failures": [
                old.get("summary", {}).get("circuit_fast_failures"),
                new.get("summary", {}).get("circuit_fast_failures"),
            ],
        },
        "snapshot_replaced_wholesale": True,
        "old_snapshot_readonly": old.get("complete") is True,
    }
