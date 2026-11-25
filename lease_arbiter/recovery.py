from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json, digest
from .engine import LeaseArbiterEngine
from .errors import ReplayError
from .ledger import Ledger
from .spec import Spec


OUTCOME_FIELDS = (
    "status",
    "label",
    "global_slot",
    "node_slot",
    "expires_at_ms",
    "state",
    "op",
    "lease_id",
    "node_id",
    "lease_type",
)


def replay_against_ledger(spec: Spec, ledger_path: str | Path) -> dict[str, Any]:
    initial_engine = LeaseArbiterEngine(spec)
    ledger = Ledger(ledger_path, spec, initial_engine.snapshot(), fresh=False)
    try:
        unknown = ledger.recover_pending()
        commits = ledger.read_commits()
        stored_snapshot = ledger.load_snapshot()
        engine = LeaseArbiterEngine(spec)
        expected_outcomes: dict[int, dict[str, Any]] = {}
        expected_reclaims: dict[int, list[dict[str, Any]]] = {}
        expected_transitions: dict[int, list[dict[str, Any]]] = {}
        expected_ownership: dict[int, list[dict[str, Any]]] = {}
        max_seq = max((item["event_seq"] for item in commits), default=0)
        for event in spec.events:
            if event.seq > max_seq:
                break
            expiry = engine.expire_due(event.at_ms, event.seq)
            applied = engine.apply_event(event)
            expected_outcomes[event.seq] = applied.outcome  # type: ignore[assignment]
            expected_reclaims[event.seq] = expiry.reclaims + applied.reclaims
            expected_transitions[event.seq] = expiry.transitions + applied.transitions
            expected_ownership[event.seq] = expiry.ownership_changes + applied.ownership_changes
        seen_sequences: list[int] = []
        for commit in commits:
            event_seq = int(commit["event_seq"])
            seen_sequences.append(event_seq)
            if event_seq not in expected_outcomes:
                raise ReplayError(
                    event_seq,
                    "event_seq",
                    "declared event",
                    "ledger event absent from declaration stream",
                    "ledger contains an event outside the replayed declaration prefix",
                )
            payload = commit["payload"]
            actual_outcome = payload.get("outcome", {})
            expected_outcome = expected_outcomes[event_seq]
            for field in OUTCOME_FIELDS:
                if expected_outcome.get(field) != actual_outcome.get(field):
                    raise ReplayError(
                        event_seq,
                        field,
                        expected_outcome.get(field),
                        actual_outcome.get(field),
                        "recalculated event outcome differs from the committed ledger",
                    )
            if canonical_json(expected_reclaims[event_seq]) != canonical_json(payload.get("reclaims", [])):
                raise ReplayError(
                    event_seq,
                    "reclaims",
                    expected_reclaims[event_seq],
                    payload.get("reclaims", []),
                    "recalculated reclaim decisions differ from the committed ledger",
                )
            if canonical_json(expected_transitions[event_seq]) != canonical_json(payload.get("transitions", [])):
                raise ReplayError(
                    event_seq,
                    "transitions",
                    expected_transitions[event_seq],
                    payload.get("transitions", []),
                    "recalculated transitions differ from the committed ledger",
                )
            if canonical_json(expected_ownership[event_seq]) != canonical_json(payload.get("ownership_changes", [])):
                raise ReplayError(
                    event_seq,
                    "ownership_changes",
                    expected_ownership[event_seq],
                    payload.get("ownership_changes", []),
                    "recalculated ownership adjudication differs from the committed ledger",
                )
        expected_snapshot = engine.snapshot()
        if canonical_json(expected_snapshot) != canonical_json(stored_snapshot):
            snapshot_seq = seen_sequences[-1] if seen_sequences else 0
            raise ReplayError(
                snapshot_seq,
                "snapshot",
                digest(expected_snapshot),
                digest(stored_snapshot),
                "recalculated snapshot differs from the committed snapshot",
            )
        session = ledger.session_info()
        return {
            "ok": True,
            "format_version": 1,
            "status": "replay_ok",
            "declaration_version": spec.declaration_version,
            "spec_digest": spec.digest,
            "snapshot_replaced": ledger.snapshot_replaced,
            "replacement_reason": ledger.replacement_reason,
            "run_id": session["run_id"],
            "snapshot_version": session["snapshot_version"],
            "snapshot_sha256": session["content_digest"],
            "committed_events": len(commits),
            "first_committed_event_seq": seen_sequences[0] if seen_sequences else None,
            "last_committed_event_seq": seen_sequences[-1] if seen_sequences else None,
            "unknown_requests": unknown,
            "unknown_request_count": len(unknown),
            "replayed_event_count": len(expected_outcomes),
            "replayed_outcome_sha256": digest(
                [expected_outcomes[seq] for seq in sorted(expected_outcomes)]
            ),
            "first_divergence": None,
        }
    finally:
        ledger.close()


def read_ledger_commit_payloads(ledger_path: str | Path) -> list[dict[str, Any]]:
    connection_payloads: list[dict[str, Any]] = []
    import sqlite3

    conn = sqlite3.connect(str(ledger_path))
    try:
        rows = conn.execute("SELECT event_seq, payload_json FROM commits ORDER BY event_seq").fetchall()
        for seq, payload_json in rows:
            connection_payloads.append({"event_seq": int(seq), "payload": json.loads(payload_json)})
    finally:
        conn.close()
    return connection_payloads
