"""Independent per-transaction replay and ledger verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .codec import decode_record
from .errors import MWError
from .store import Store


class ReplayViolation(Exception):
    def __init__(self, txn_id: str, field: str, message: str, event_seq: int | None = None):
        super().__init__(message)
        self.txn_id = txn_id
        self.field = field
        self.message = message
        self.event_seq = event_seq

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": "E_VERIFY",
            "txn_id": self.txn_id,
            "field": self.field,
            "message": self.message,
            "event_seq": self.event_seq,
        }


def _replay_transaction(txn_id: str, events: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    state: str | None = None
    disposition: str | None = None
    query_count = 0
    query_max = int(expected.get("query_max", 0) or 0)
    query_inflight = False
    ack_sent = False
    effect_id: str | None = None
    terminal_reason: str | None = None
    pending_commands: list[str] = []
    trajectory: list[dict[str, Any]] = []
    half_seen = False
    last_seq = 0

    def violate(field: str, message: str, seq: int | None = None) -> None:
        raise ReplayViolation(txn_id, field, message, seq)

    for event in events:
        seq = int(event["event_seq"])
        if seq <= last_seq:
            violate("event_seq", "event sequence is not strictly increasing", seq)
        last_seq = seq
        event_type = event["type"]
        data = event.get("data", {})
        trajectory.append({"event_seq": seq, "type": event_type, "state_before": state})
        if event_type == "half":
            if half_seen:
                violate("half", "half message was registered twice", seq)
            if state is not None:
                violate("half", "half message must be the first transaction event", seq)
            half_seen = True
            state = "half"
            disposition = "pending"
            query_max = int(data.get("query_max", query_max))
            continue
        if not half_seen:
            violate("half", "event appeared before the half message", seq)
        if event_type == "half_ack":
            if ack_sent:
                violate("half_ack", "duplicate acknowledgment marker", seq)
            ack_sent = True
        elif event_type == "command":
            command = data.get("command")
            if command not in {"commit", "rollback"}:
                violate("command.command", "invalid command", seq)
            pending_commands.append(command)
            if state in {"committable", "consumed"} or disposition in {"rolled_back", "manual_review"}:
                # A conflicting post-terminal command is retained as evidence;
                # it must not rewrite the state.
                pass
        elif event_type == "query_started":
            if state != "half" or disposition != "pending":
                violate("query_started.state", "query can only start from pending half", seq)
            if query_inflight:
                violate("query_started.inflight", "overlapping query attempts for one transaction", seq)
            if not ack_sent:
                violate("query_started.half_ack", "query started before half-message acknowledgment marker", seq)
            query_count += 1
            attempt = int(data.get("attempt", query_count))
            if attempt != query_count:
                violate("query_started.attempt", f"attempt={attempt} replay_count={query_count}", seq)
            if query_max and query_count > query_max:
                violate("query_started.attempt", "query count exceeds query_max", seq)
            query_inflight = True
        elif event_type == "query_result":
            if not query_inflight:
                violate("query_result.inflight", "query result without a claimed query", seq)
            query_inflight = False
            if data.get("result") not in {"committed", "rolled_back", "unknown"}:
                violate("query_result.result", "invalid query result", seq)
        elif event_type == "late_query_result":
            if not (state in {"committable", "consumed"} or disposition in {"rolled_back", "manual_review"}):
                violate("late_query_result.state", "late query result is only valid after a terminal state", seq)
            query_inflight = False
            if data.get("result") not in {"committed", "rolled_back", "unknown"}:
                violate("late_query_result.result", "invalid query result", seq)
        elif event_type == "recovery_query_unknown":
            query_inflight = False
        elif event_type == "decision":
            if state in {"committable", "consumed"} or disposition in {"rolled_back", "manual_review"}:
                violate("decision.terminal", "decision attempted to rewrite a terminal result", seq)
            outcome = data.get("outcome")
            if outcome == "commit":
                state = "committable"
                disposition = "committable"
            elif outcome == "rollback":
                state = "half"
                disposition = "rolled_back"
            else:
                violate("decision.outcome", "invalid decision outcome", seq)
            target = data.get("target_state")
            if target != state:
                violate("decision.target_state", f"row target={target} replay target={state}", seq)
            declared_disposition = data.get("disposition")
            if declared_disposition != disposition:
                violate("decision.disposition", "decision disposition does not match outcome", seq)
            terminal_reason = data.get("reason", terminal_reason)
            query_inflight = False
            pending_commands.clear()
        elif event_type == "manual_review":
            if state != "half" or disposition == "manual_review":
                violate("manual_review.state", "manual review must follow a pending half", seq)
            disposition = "manual_review"
            terminal_reason = data.get("reason", "manual_review")
            query_inflight = False
        elif event_type == "query_exhausted":
            if query_count != int(data.get("query_count", query_count)):
                violate("query_exhausted.query_count", "query count does not match replay", seq)
        elif event_type == "consume":
            if state == "consumed":
                violate("consume.duplicate", "a second business effect was produced", seq)
            if state != "committable" or disposition != "committable":
                violate("consume.state", "consume requires a committable message", seq)
            effect_id = data.get("effect_id")
            declared = data.get("result", {}).get("effect_id")
            if not effect_id or (declared and declared != effect_id):
                violate("consume.effect_id", "effect identifier is missing or inconsistent", seq)
            state = "consumed"
            disposition = "consumed"
            terminal_reason = "consumed"
        elif event_type == "pull_duplicate":
            if state != "consumed":
                violate("pull_duplicate.state", "duplicate pull requires consumed state", seq)
            duplicate_effect = data.get("effect_id")
            if duplicate_effect is not None and duplicate_effect != effect_id:
                violate("pull_duplicate.effect_id", "duplicate pull returned a different effect", seq)
        elif event_type == "idempotency_evict":
            if state != "consumed":
                violate("idempotency_evict.state", "only consumed keys can be evicted", seq)
        elif event_type in {"pull_started", "pull_failed", "conflict", "recovery_pull_required"}:
            pass
        else:
            violate("type", f"unknown event type {event_type!r}", seq)
    if not half_seen:
        raise ReplayViolation(txn_id, "half", "transaction has no half event")

    expected_state = expected.get("state")
    expected_disposition = expected.get("disposition")
    expected_query_count = int(expected.get("query_count", 0))
    expected_effect = expected.get("effect_id")
    if state != expected_state:
        raise ReplayViolation(txn_id, "state", f"row={expected_state} replay={state}")
    if disposition != expected_disposition:
        raise ReplayViolation(txn_id, "disposition", f"row={expected_disposition} replay={disposition}")
    if query_count != expected_query_count:
        raise ReplayViolation(txn_id, "query_count", f"row={expected_query_count} replay={query_count}")
    if effect_id != expected_effect:
        raise ReplayViolation(txn_id, "effect_id", f"row={expected_effect} replay={effect_id}")
    return {
        "txn_id": txn_id,
        "state": state,
        "disposition": disposition,
        "query_count": query_count,
        "event_count": len(events),
        "effect_id": effect_id,
        "terminal_reason": terminal_reason,
        "trajectory": trajectory,
    }


def verify_database(store: Store, logger: Any | None = None) -> dict[str, Any]:
    with store.read() as conn:
        tx_rows = [dict(row) for row in conn.execute("SELECT * FROM txns").fetchall()]
        event_rows = [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY event_seq").fetchall()]
        idem_rows = [dict(row) for row in conn.execute("SELECT * FROM idempotency").fetchall()]
        evicted_txns = {
            row["txn_id"] for row in conn.execute("SELECT txn_id FROM idempotency_evictions").fetchall()
        }
    grouped: dict[str, list[dict[str, Any]]] = {}
    invalid_events: list[dict[str, Any]] = []
    for row in event_rows:
        if not bool(row.get("valid", 1)):
            invalid_events.append(
                {
                    "event_seq": row["event_seq"],
                    "txn_id": row["txn_id"],
                    "code": "E_LOG",
                    "message": "event row is quarantined and excluded from replay",
                }
            )
            continue
        try:
            payload = decode_record(bytes(row["record"]))
            grouped.setdefault(payload["txn_id"], []).append(payload)
        except MWError as exc:
            invalid_events.append({"event_seq": row["event_seq"], "txn_id": row["txn_id"], **exc.as_dict()})
    trajectory_order = sorted(
        tx_rows,
        key=lambda tx: min(
            (payload["event_seq"] for payload in grouped.get(tx["txn_id"], [])),
            default=10**30,
        ),
    )
    checked = []
    first_violation: dict[str, Any] | None = None
    for tx in trajectory_order:
        try:
            replay = _replay_transaction(tx["txn_id"], grouped.get(tx["txn_id"], []), tx)
            checked.append({k: v for k, v in replay.items() if k != "trajectory"})
        except ReplayViolation as exc:
            first_violation = exc.as_dict()
            break
    if first_violation is None:
        tx_by_id = {tx["txn_id"]: tx for tx in tx_rows}
        idem_txns = {idem["txn_id"] for idem in idem_rows}
        for tx in tx_rows:
            if tx["state"] == "consumed" and tx["txn_id"] not in idem_txns and tx["txn_id"] not in evicted_txns:
                first_violation = {
                    "code": "E_VERIFY",
                    "txn_id": tx["txn_id"],
                    "field": "idempotency",
                    "message": "consumed transaction has neither an idempotency key nor eviction evidence",
                }
                break
    if first_violation is None:
        for idem in idem_rows:
            tx = tx_by_id.get(idem["txn_id"])
            if tx is None or tx["state"] != "consumed" or tx["effect_id"] != idem["effect_id"]:
                first_violation = {
                    "code": "E_VERIFY",
                    "txn_id": idem["txn_id"],
                    "field": "idempotency.effect_id",
                    "message": "idempotency key is not linked to the consumed transaction effect",
                }
                break
    result = {
        "ok": first_violation is None,
        "op": "verify",
        "code": "E_VERIFY" if first_violation else None,
        "checked_transactions": len(checked) if first_violation is None else len(checked),
        "qualified_transactions": checked,
        "first_violation": first_violation,
        "invalid_event_records": invalid_events,
    }
    if first_violation and logger is not None:
        logger.error(
            "E_VERIFY",
            first_violation["message"],
            txn_id=first_violation.get("txn_id"),
            field=first_violation.get("field"),
            event_seq=first_violation.get("event_seq"),
        )
    return result


def verify_ledger_file(path: str | Path, logger: Any | None = None) -> dict[str, Any]:
    from .ledger import inspect_ledger, load_ledger, validate_event_rows

    value = load_ledger(path)
    inspection = inspect_ledger(value)
    event_errors = validate_event_rows(value["events"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in value["events"]:
        import base64

        try:
            payload = decode_record(base64.b64decode(row["record_b64"]))
            if int(row.get("valid", 1)) == 1:
                grouped.setdefault(payload["txn_id"], []).append(payload)
        except (MWError, ValueError, KeyError):
            continue
    first_violation: dict[str, Any] | None = None
    checked = []
    for tx in sorted(value["transactions"], key=lambda row: row["txn_id"]):
        try:
            replay = _replay_transaction(tx["txn_id"], grouped.get(tx["txn_id"], []), tx)
            checked.append({k: v for k, v in replay.items() if k != "trajectory"})
        except ReplayViolation as exc:
            first_violation = exc.as_dict()
            break
    if first_violation is None:
        replay_by_txn = {item["txn_id"]: item for item in checked}
        idem_txns = {row["txn_id"] for row in value["idempotency"]}
        evicted_txns = {row["txn_id"] for row in value.get("idempotency_evictions", [])}
        for tx in value["transactions"]:
            replay = replay_by_txn.get(tx["txn_id"])
            if replay is None:
                continue
            if replay["state"] == "consumed" and tx["txn_id"] not in idem_txns and tx["txn_id"] not in evicted_txns:
                first_violation = {
                    "code": "E_VERIFY",
                    "txn_id": tx["txn_id"],
                    "field": "idempotency",
                    "message": "consumed transaction lacks both key and eviction evidence",
                }
                break
        if first_violation is None:
            for idem in value["idempotency"]:
                replay = replay_by_txn.get(idem["txn_id"])
                if (
                    replay is None
                    or replay["state"] != "consumed"
                    or replay["effect_id"] != idem["effect_id"]
                ):
                    first_violation = {
                        "code": "E_VERIFY",
                        "txn_id": idem["txn_id"],
                        "field": "idempotency.effect_id",
                        "message": "idempotency key does not match the replayed consumption effect",
                    }
                    break
    result = {
        "ok": first_violation is None and not event_errors and inspection["digest_matches"],
        "op": "verify_ledger",
        "code": "E_VERIFY" if (first_violation or event_errors or not inspection["digest_matches"]) else None,
        "inspection": inspection,
        "event_errors": event_errors,
        "checked_transactions": len(checked) if first_violation is None else len(checked),
        "qualified_transactions": checked,
        "first_violation": first_violation,
    }
    if not result["ok"] and logger is not None:
        detail = first_violation or (event_errors[0] if event_errors else {"message": "ledger digest mismatch"})
        fields = {key: value for key, value in detail.items() if key not in {"code", "message"}}
        logger.error(
            "E_VERIFY",
            detail.get("message", "ledger verification failed"),
            **fields,
        )
    return result
