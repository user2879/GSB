from __future__ import annotations

import json
from typing import Any

from .algorithms import run_full_path
from .allocation import allocate_path
from .canonical import canonical_bytes, sha256_hex
from .errors import ReplayError
from .leases import replay_quota_events
from .ledger import Ledger
from .scheduler import chunks_for_allocation
from .spec import Spec, load_spec_dict


def _expected_record_map(spec: Spec) -> dict[tuple[str, int, str], dict[str, Any]]:
    path_result = run_full_path(spec)
    allocations = allocate_path(spec, path_result)
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    for service_id in spec.service_ids:
        for item in path_result.minutes[service_id]:
            records[(service_id, item.minute, "demand")] = item.comparable_dict()
    for minute, rows in allocations.items():
        for row in rows:
            records[(row.service_id, minute, "allocation")] = row.comparable_dict()
    return records


def _first_value_difference(stored: Any, expected: Any) -> str | None:
    if stored == expected:
        return None
    if isinstance(stored, dict) and isinstance(expected, dict):
        for field in sorted(set(stored) | set(expected)):
            if stored.get(field) != expected.get(field):
                return field
    return "$value"


def verify_ledger_replay(ledger: Ledger, fallback_spec: Spec) -> dict[str, Any]:
    rounds = ledger.committed_rounds()
    previous_digest = sha256_hex(b"scale-review-ledger-genesis")
    replayed: list[dict[str, Any]] = []

    for round_row in rounds:
        round_seq = round_row["round_seq"]
        round_id = round_row["round_id"]
        payload_bytes = round_row["payload_json"].encode("utf-8")
        actual_payload_digest = sha256_hex(payload_bytes)
        if actual_payload_digest != round_row["payload_digest"]:
            raise ReplayError(
                "committed round payload digest mismatch",
                field="payload_digest",
                details={
                    "round_seq": round_seq,
                    "round_id": round_id,
                    "expected_digest": actual_payload_digest,
                    "stored_digest": round_row["payload_digest"],
                    "basis": "stored_payload_json_sha256",
                },
            )
        expected_ledger_digest = sha256_hex(
            (previous_digest + round_row["payload_digest"]).encode("utf-8")
        )
        if expected_ledger_digest != round_row["ledger_digest_after"]:
            raise ReplayError(
                "ledger digest chain mismatch",
                field="ledger_digest_after",
                details={
                    "round_seq": round_seq,
                    "round_id": round_id,
                    "expected": expected_ledger_digest,
                    "stored": round_row["ledger_digest_after"],
                },
            )
        previous_digest = round_row["ledger_digest_after"]

        spec = fallback_spec
        if round_row.get("spec_json"):
            try:
                spec = load_spec_dict(json.loads(round_row["spec_json"]))
            except Exception as exc:
                raise ReplayError(
                    "stored round spec cannot be loaded",
                    field="spec_json",
                    details={"round_seq": round_seq, "round_id": round_id},
                ) from exc

        expected_records = _expected_record_map(spec)
        stored_records = ledger.round_rows(round_seq, "minute_results")
        for stored_row in sorted(
            stored_records,
            key=lambda row: (row["minute"], row["service_id"], row["record_type"]),
        ):
            key = (
                stored_row["service_id"],
                stored_row["minute"],
                stored_row["record_type"],
            )
            expected_value = expected_records.get(key)
            if expected_value is None:
                raise ReplayError(
                    "ledger contains an unexpected replay record",
                    field=f"minute_results.{stored_row['record_type']}",
                    details={
                        "round_seq": round_seq,
                        "round_id": round_id,
                        "service_id": stored_row["service_id"],
                        "minute": stored_row["minute"],
                    },
                )
            stored_value = json.loads(stored_row["canonical_json"])
            field = _first_value_difference(stored_value, expected_value)
            if field is not None:
                raise ReplayError(
                    "replayed round differs from deterministic recomputation",
                    field=f"services.{stored_row['service_id']}.minutes[{stored_row['minute']}].{field}",
                    details={
                        "round_seq": round_seq,
                        "round_id": round_id,
                        "service_id": stored_row["service_id"],
                        "minute": stored_row["minute"],
                        "record_type": stored_row["record_type"],
                        "field": field,
                        "stored_value": stored_value.get(field)
                        if isinstance(stored_value, dict)
                        else stored_value,
                        "expected_value": expected_value.get(field)
                        if isinstance(expected_value, dict)
                        else expected_value,
                        "basis": "canonical_full_path_recomputation_using_committed_spec",
                    },
                )
            if stored_row["field_digest"] != sha256_hex(
                stored_row["canonical_json"].encode("utf-8")
            ):
                raise ReplayError(
                    "stored record digest does not match canonical_json",
                    field=f"minute_results[{stored_row['id']}].field_digest",
                    details={"round_seq": round_seq, "round_id": round_id},
                )

        event_rows = ledger.round_rows(round_seq, "quota_events")
        events = [json.loads(row["event_json"]) for row in event_rows]
        service_replica_quotas = {
            service_id: (
                spec.budget.service_concurrency_quotas[service_id]
                // spec.service(service_id).concurrency_per_replica
            )
            for service_id in spec.service_ids
        }
        quota_proof = replay_quota_events(
            spec.budget.global_replica_budget,
            service_replica_quotas,
            events,
        )
        if quota_proof["conflicts"]:
            raise ReplayError(
                "quota ownership replay contains a conflict",
                field="quota_events.owner_set",
                details={
                    "round_seq": round_seq,
                    "round_id": round_id,
                    "conflicts": quota_proof["conflicts"],
                },
            )

        for span in ledger.round_rows(round_seq, "wait_spans"):
            if span["waited_ms"] < 0:
                raise ReplayError(
                    "negative monotonic wait duration in ledger",
                    field="wait_spans.waited_ms",
                    details={"round_seq": round_seq, "task_id": span["task_id"]},
                )

        replayed.append(
            {
                "round_seq": round_seq,
                "round_id": round_id,
                "minute": round_row["minute"],
                "record_count": len(stored_records),
                "event_count": len(events),
                "quota_proof": quota_proof,
            }
        )

    expected_ledger_digest = ledger.meta("content_digest")
    if expected_ledger_digest != previous_digest:
        raise ReplayError(
            "ledger content digest does not match replayed chain",
            field="content_digest",
            details={
                "stored": expected_ledger_digest,
                "replayed": previous_digest,
            },
        )
    return {"rounds": replayed, "digest": previous_digest}


def recover_and_replay(
    ledger: Ledger,
    spec: Spec,
    *,
    recompute_unfinished: bool = True,
) -> dict[str, Any]:
    attempts = ledger.recover_unknown_attempts()
    replay = verify_ledger_replay(ledger, spec)
    cursor = ledger.get_cursor()
    next_minute = 0 if cursor is None else int(cursor["last_minute"]) + 1
    unfinished_round_id: str | None = None
    if attempts:
        next_minute = min(int(item["minute"]) for item in attempts)
        unfinished_round_id = attempts[0]["round_id"]

    unfinished: dict[str, Any] | None = None
    if recompute_unfinished and next_minute < spec.minute_count:
        path_result = run_full_path(spec)
        allocations = allocate_path(spec, path_result)
        rows = allocations[next_minute]
        tasks = []
        for row in rows:
            tasks.extend(
                task.__dict__
                for task in chunks_for_allocation(
                    round_id=unfinished_round_id or f"replay-round-{next_minute}",
                    batch_id=f"replay-batch-{next_minute}",
                    minute=next_minute,
                    service_id=row.service_id,
                    allocated_replicas=row.allocated_replicas,
                    max_task_replicas=spec.lease_policy.max_task_replicas,
                )
            )
        unfinished = {
            "round_id": unfinished_round_id or f"replay-round-{next_minute}",
            "minute": next_minute,
            "allocation": [row.comparable_dict() for row in rows],
            "tasks": tasks,
            "basis": "recomputed_from_last_committed_progress_cursor",
        }

    return {
        "recovered_attempts": attempts,
        "replay": replay,
        "cursor": cursor,
        "next_minute": next_minute,
        "unfinished_round": unfinished,
        "ok": True,
    }
