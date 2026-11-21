"""Deterministic snapshot replay and first-difference localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest
from .engine import CascadeEngine
from .error_propagation import propagate_fixed_errors
from .errors import ReplayError, ReplayMismatch, SnapshotError
from .generator import DeterministicUdpGenerator, parse_generator_config
from .models import DecisionEvent, ErrorPropagationResult, RunSummary
from .policy import cascade_digest, load_policy, shaper_digest
from .snapshot import SnapshotDocument
from .transport import InMemoryTransport


EVENT_FIELDS = (
    "packet_id",
    "stage_id",
    "kind",
    "decision",
    "time_tick",
    "tokens_before",
    "tokens_after",
    "queue_len_before",
    "queue_len_after",
    "delay_ticks",
    "release_tick",
    "forwarded_to",
    "reason",
)


@dataclass(frozen=True)
class ReplayVerificationResult:
    ok: bool
    classification: str
    events_compared: int
    snapshot_cursor: int
    replay_cursor: int
    summary: RunSummary
    errors: ErrorPropagationResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "classification": self.classification,
            "events_compared": self.events_compared,
            "snapshot_cursor": self.snapshot_cursor,
            "replay_cursor": self.replay_cursor,
            "summary": self.summary.as_dict(),
            "errors": self.errors.as_dict(),
        }


def _unreproducible(reason: str, **details: Any) -> ReplayError:
    return ReplayError(
        reason,
        basis=(
            "snapshot cannot be reproduced from its recorded policy/version/sequence; "
            "this is classified as UNREPRODUCIBLE, not a ledger defect"
        ),
        details={"classification": "UNREPRODUCIBLE", **details},
    )


def _event_key(value: Mapping[str, Any] | DecisionEvent) -> tuple[Any, ...]:
    if isinstance(value, DecisionEvent):
        return (value.packet_seq, value.stage_id, value.kind, value.event_seq)
    return (
        int(value["packet_seq"]),
        str(value["stage_id"]),
        str(value["kind"]),
        int(value.get("event_seq", 0)),
    )


def _event_value(value: Mapping[str, Any] | DecisionEvent, field: str) -> Any:
    if isinstance(value, DecisionEvent):
        result = getattr(value, field)
        if field == "forwarded_to":
            return list(result)
        return result
    result = value.get(field)
    if field == "forwarded_to":
        return list(result or [])
    return result


def _first_event_mismatch(
    expected_values: list[Mapping[str, Any]],
    actual_values: tuple[DecisionEvent, ...],
) -> ReplayMismatch | None:
    expected = sorted(expected_values, key=_event_key)
    actual = sorted(actual_values, key=_event_key)
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected):
            event = actual[index]
            return ReplayMismatch(
                packet_id=event.packet_id,
                shaper_id=event.stage_id,
                field="event_presence",
                expected="no replay-only event",
                actual=event.as_dict(),
            )
        if index >= len(actual):
            event = expected[index]
            return ReplayMismatch(
                packet_id=str(event["packet_id"]),
                shaper_id=str(event["stage_id"]),
                field="event_presence",
                expected=dict(event),
                actual="no replayed event",
            )
        left = expected[index]
        right = actual[index]
        for field in EVENT_FIELDS:
            expected_value = _event_value(left, field)
            actual_value = _event_value(right, field)
            if expected_value != actual_value:
                return ReplayMismatch(
                    packet_id=str(left.get("packet_id", right.packet_id)),
                    shaper_id=str(left.get("stage_id", right.stage_id)),
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                )
    return None


def _stage_summary(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["stage_id"]): item for item in summary.get("stage_stats", [])
    }


def _first_summary_mismatch(
    expected: Mapping[str, Any] | None, actual: RunSummary
) -> ReplayMismatch | None:
    if expected is None:
        return ReplayMismatch(
            packet_id="<summary>",
            shaper_id="<cascade>",
            field="summary_presence",
            expected="recorded summary",
            actual="missing",
        )
    actual_dict = actual.as_dict()
    for field in (
        "final_released_bytes",
        "final_released_packets",
        "average_delay_ns",
        "unmatched_packets",
    ):
        if expected.get(field) != actual_dict.get(field):
            return ReplayMismatch(
                packet_id="<summary>",
                shaper_id="<cascade>",
                field=field,
                expected=expected.get(field),
                actual=actual_dict.get(field),
            )
    expected_stages = _stage_summary(expected)
    actual_stages = _stage_summary(actual_dict)
    for stage_id in sorted(set(expected_stages) | set(actual_stages)):
        if stage_id not in expected_stages or stage_id not in actual_stages:
            return ReplayMismatch(
                packet_id="<summary>",
                shaper_id=stage_id,
                field="stage_presence",
                expected=stage_id in expected_stages,
                actual=stage_id in actual_stages,
            )
        for field in (
            "passed_packets",
            "queued_packets",
            "dropped_packets",
            "released_packets",
            "forwarded_bytes",
            "max_queue_len",
            "total_delay_ticks",
        ):
            if expected_stages[stage_id].get(field) != actual_stages[stage_id].get(field):
                return ReplayMismatch(
                    packet_id="<summary>",
                    shaper_id=stage_id,
                    field=f"stage.{field}",
                    expected=expected_stages[stage_id].get(field),
                    actual=actual_stages[stage_id].get(field),
                )
    return None


def _first_error_mismatch(
    expected: Mapping[str, Any] | None,
    actual: ErrorPropagationResult,
) -> ReplayMismatch | None:
    if expected is None:
        return ReplayMismatch(
            packet_id="<error-bound>",
            shaper_id="<cascade>",
            field="error_conclusions_presence",
            expected="recorded error conclusions",
            actual="missing",
        )
    actual_dict = actual.as_dict()
    for field in (
        "scale",
        "tick_ns",
        "final_released_bytes",
        "bytes_error_bound",
        "average_delay_ns",
        "average_delay_error_bound_ns",
        "dominant_stage",
    ):
        if expected.get(field) != actual_dict.get(field):
            return ReplayMismatch(
                packet_id="<error-bound>",
                shaper_id="<cascade>",
                field=field,
                expected=expected.get(field),
                actual=actual_dict.get(field),
            )
    expected_terms = {
        str(term["stage_id"]): term for term in expected.get("terms", [])
    }
    actual_terms = {term.stage_id: term.as_dict() for term in actual.terms}
    for stage_id in sorted(set(expected_terms) | set(actual_terms)):
        if stage_id not in expected_terms or stage_id not in actual_terms:
            return ReplayMismatch(
                packet_id="<error-bound>",
                shaper_id=stage_id,
                field="term_presence",
                expected=stage_id in expected_terms,
                actual=stage_id in actual_terms,
            )
        for field in (
            "rate_quantization_error_bytes_per_s",
            "refill_cumulative_error_units",
            "refill_cumulative_error_bytes_ceiling",
            "delay_quantization_error_ns",
            "bytes_contribution",
            "delay_contribution_ns",
        ):
            if expected_terms[stage_id].get(field) != actual_terms[stage_id].get(field):
                return ReplayMismatch(
                    packet_id="<error-bound>",
                    shaper_id=stage_id,
                    field=f"error.{field}",
                    expected=expected_terms[stage_id].get(field),
                    actual=actual_terms[stage_id].get(field),
                )
    return None


def verify_snapshot_replay(
    snapshot: SnapshotDocument | str | Path | Mapping[str, Any],
    *,
    current_policy_version: str | None = None,
    current_scale: int | None = None,
    current_generator_digest: str | None = None,
) -> ReplayVerificationResult:
    if isinstance(snapshot, SnapshotDocument):
        document = snapshot
        document.verify()
    elif isinstance(snapshot, (str, Path)):
        document = SnapshotDocument.load(snapshot)
    else:
        payload = dict(snapshot)
        document = SnapshotDocument(path=None, payload=payload)
        document.verify()

    if (
        current_policy_version is not None
        and current_policy_version != document.policy_version
    ):
        raise _unreproducible(
            "policy_version_mismatch",
            expected=document.policy_version,
            current=current_policy_version,
        )
    if current_scale is not None and current_scale != document.scale:
        raise _unreproducible(
            "scale_mismatch", expected=document.scale, current=current_scale
        )
    if (
        current_generator_digest is not None
        and current_generator_digest != document.payload["packet_sequence_digest"]
    ):
        raise _unreproducible(
            "packet_sequence_digest_mismatch",
            expected=document.payload["packet_sequence_digest"],
            current=current_generator_digest,
        )
    if document.unknown_packet_ids or document.ledger.get("unknown_unresolved"):
        raise _unreproducible(
            "unknown_packets_require_rerun",
            unknown_packet_ids=list(document.unknown_packet_ids),
            unknown_unresolved=document.ledger.get("unknown_unresolved", []),
        )

    policy = document.policy
    if shaper_digest(policy) != document.payload.get("shaper_digest"):
        raise _unreproducible("snapshot_shaper_digest_mismatch")
    if cascade_digest(policy) != document.payload.get("cascade_digest"):
        raise _unreproducible("snapshot_cascade_digest_mismatch")

    full_generator = DeterministicUdpGenerator(document.generator_config)
    regenerated_manifest = list(full_generator.manifest)
    if digest(regenerated_manifest) != document.payload["packet_sequence_digest"]:
        raise _unreproducible(
            "packet_sequence_regeneration_digest_mismatch",
            expected=document.payload["packet_sequence_digest"],
            actual=digest(regenerated_manifest),
        )
    if regenerated_manifest != document.packet_sequence:
        first_index = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(regenerated_manifest, document.packet_sequence)
                )
                if left != right
            ),
            min(len(regenerated_manifest), len(document.packet_sequence)),
        )
        raise _unreproducible(
            "packet_sequence_regeneration_mismatch",
            packet_index=first_index,
            expected=(
                document.packet_sequence[first_index]
                if first_index < len(document.packet_sequence)
                else None
            ),
            actual=(
                regenerated_manifest[first_index]
                if first_index < len(regenerated_manifest)
                else None
            ),
        )

    cursor = int(document.ledger.get("cursor_seq", 0))
    if cursor < 0 or cursor > len(regenerated_manifest):
        raise _unreproducible("snapshot_cursor_out_of_range", cursor=cursor)
    complete = cursor == len(regenerated_manifest) and document.ledger.get("summary") is not None
    replay_config = dict(full_generator.config.as_dict())
    if not complete:
        if cursor == 0:
            replay_events: tuple[DecisionEvent, ...] = ()
            replay_summary = RunSummary(
                stage_stats=(),
                final_released_bytes=0,
                final_released_packets=0,
                average_delay_ns=0,
                total_generated_packets=0,
                unmatched_packets=0,
                transport_datagrams_sent=0,
                transport_datagrams_received=0,
            )
            replay_errors = propagate_fixed_errors(
                policy,
                replay_summary,
                replay_events,
                scale=document.scale,
                generator_digest=document.payload["packet_sequence_digest"],
            )
        else:
            replay_config["count"] = cursor
            replay_generator = DeterministicUdpGenerator(replay_config)
            transport = InMemoryTransport(
                [shaper.shaper_id for shaper in policy.shapers]
            )
            with CascadeEngine(
                policy,
                replay_generator,
                scale=document.scale,
                transport=transport,
                drain_final=False,
            ) as engine:
                replay = engine.run()
            replay_events = replay.events
            replay_summary = replay.summary
            replay_errors = propagate_fixed_errors(
                policy,
                replay_summary,
                replay_events,
                scale=document.scale,
                generator_digest=document.payload["packet_sequence_digest"],
            )
    else:
        transport = InMemoryTransport([shaper.shaper_id for shaper in policy.shapers])
        with CascadeEngine(
            policy,
            full_generator,
            scale=document.scale,
            transport=transport,
            drain_final=True,
        ) as engine:
            replay = engine.run()
        replay_events = replay.events
        replay_summary = replay.summary
        replay_errors = propagate_fixed_errors(
            policy,
            replay_summary,
            replay_events,
            scale=document.scale,
            generator_digest=document.payload["packet_sequence_digest"],
        )

    expected_events = document.ledger.get("decisions", [])
    event_mismatch = _first_event_mismatch(expected_events, replay_events)
    if event_mismatch is not None:
        raise ReplayError(
            "first_deterministic_mismatch",
            basis=(
                "replay used the recorded policy version, scale and regenerated packet "
                "sequence; stable comparison order is packet_seq, shaper id, event kind"
            ),
            mismatch=event_mismatch,
            details={"classification": "LEDGER_DEFECT"},
        )
    summary_mismatch = _first_summary_mismatch(document.ledger.get("summary"), replay_summary)
    if (
        summary_mismatch is not None
        and not complete
        and document.ledger.get("summary") is None
    ):
        raise _unreproducible(
            "snapshot_incomplete_requires_rerun",
            cursor=cursor,
            total_packets=len(regenerated_manifest),
            committed_events=len(replay_events),
        )
    if summary_mismatch is not None:
        raise ReplayError(
            "summary_mismatch",
            basis="per-event replay matched but aggregate summary fields differ",
            mismatch=summary_mismatch,
            details={"classification": "LEDGER_DEFECT"},
        )
    expected_errors = document.payload.get("error_conclusions")
    if expected_errors is None and not complete:
        raise _unreproducible(
            "snapshot_incomplete_requires_rerun",
            cursor=cursor,
            total_packets=len(regenerated_manifest),
            reason="no committed summary or error conclusions for a partial snapshot",
        )
    error_mismatch = _first_error_mismatch(expected_errors, replay_errors)
    if error_mismatch is not None:
        raise ReplayError(
            "error_bound_mismatch",
            basis="replay decisions matched but fixed-point error conclusions differ",
            mismatch=error_mismatch,
            details={"classification": "LEDGER_DEFECT"},
        )
    return ReplayVerificationResult(
        ok=True,
        classification="REPRODUCIBLE",
        events_compared=len(replay_events),
        snapshot_cursor=cursor,
        replay_cursor=cursor,
        summary=replay_summary,
        errors=replay_errors,
    )
