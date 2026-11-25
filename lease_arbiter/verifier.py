from __future__ import annotations

from typing import Any

from .canonical import canonical_bytes, canonical_json, digest
from .engine import run_arbitration
from .errors import DivergenceError
from .spec import Spec


EVENT_FIELDS = (
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
RECLAIM_FIELDS = (
    "event_seq",
    "lease_id",
    "reason",
    "label_after",
    "global_slot",
    "node_slot",
    "owner_before",
    "expires_at_ms",
)
FINAL_FIELDS = (
    "lease_id",
    "state",
    "label",
    "owner",
    "global_slot",
    "node_slot",
    "expires_at_ms",
)


def _first_difference(sequential: Any, batch: Any, *, event_seq: int | None) -> DivergenceError | None:
    if type(sequential) is not type(batch):
        return DivergenceError(event_seq, "type", sequential, batch, "projection value types differ")
    if isinstance(sequential, dict):
        for key in sorted(set(sequential) | set(batch)):
            if key not in sequential:
                return DivergenceError(event_seq, key, None, batch[key], "field is missing from sequential projection")
            if key not in batch:
                return DivergenceError(event_seq, key, sequential[key], None, "field is missing from batch projection")
            difference = _first_difference(sequential[key], batch[key], event_seq=event_seq)
            if difference is not None:
                return difference
        return None
    if isinstance(sequential, list):
        if len(sequential) != len(batch):
            return DivergenceError(event_seq, "length", len(sequential), len(batch), "projection list lengths differ")
        for index, (left, right) in enumerate(zip(sequential, batch)):
            difference = _first_difference(left, right, event_seq=event_seq)
            if difference is not None:
                return difference
        return None
    if sequential != batch:
        return DivergenceError(event_seq, "value", sequential, batch, "projection values differ")
    return None


def compare_arbitration_results(sequential: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    left = sequential["comparison_projection"]
    right = batch["comparison_projection"]
    left_bytes = canonical_bytes(left)
    right_bytes = canonical_bytes(right)
    if left_bytes != right_bytes:
        left_events = {item["seq"]: item for item in left.get("events", [])}
        right_events = {item["seq"]: item for item in right.get("events", [])}
        all_sequences = sorted(set(left_events) | set(right_events))
        for seq in all_sequences:
            if left_events.get(seq) != right_events.get(seq):
                left_event = left_events.get(seq, {})
                right_event = right_events.get(seq, {})
                for field in EVENT_FIELDS:
                    if left_event.get(field) != right_event.get(field):
                        raise DivergenceError(
                            seq,
                            field,
                            left_event.get(field),
                            right_event.get(field),
                            "arbitration paths diverged",
                        )
                raise DivergenceError(seq, "event", left_event, right_event, "arbitration paths diverged")
        left_reclaims = left.get("reclaims", [])
        right_reclaims = right.get("reclaims", [])
        if len(left_reclaims) != len(right_reclaims):
            raise DivergenceError(
                None,
                "reclaims.length",
                len(left_reclaims),
                len(right_reclaims),
                "reclaim counts differ",
            )
        for index, (left_reclaim, right_reclaim) in enumerate(zip(left_reclaims, right_reclaims)):
            if left_reclaim != right_reclaim:
                event_seq = int(left_reclaim.get("event_seq") or right_reclaim.get("event_seq") or 0)
                for field in RECLAIM_FIELDS:
                    if left_reclaim.get(field) != right_reclaim.get(field):
                        raise DivergenceError(
                            event_seq,
                            f"reclaims[{index}].{field}",
                            left_reclaim.get(field),
                            right_reclaim.get(field),
                            "reclaim adjudication diverged",
                        )
                raise DivergenceError(event_seq, f"reclaims[{index}]", left_reclaim, right_reclaim, "reclaim adjudication diverged")
        difference = _first_difference(left, right, event_seq=None)
        if difference is not None:
            raise difference
        raise DivergenceError(None, "projection", None, None, "canonical bytes differ")
    return {
        "ok": True,
        "byte_identical": True,
        "sequential_bytes": len(left_bytes),
        "batch_bytes": len(right_bytes),
        "projection_sha256": digest(left),
        "sequential_projection": left,
        "batch_projection": right,
    }


def verify_spec(spec: Spec) -> dict[str, Any]:
    sequential = run_arbitration(spec, "sequential")
    batch = run_arbitration(spec, "batch")
    comparison = compare_arbitration_results(sequential, batch)
    auto = run_arbitration(spec, "auto")
    return {
        "ok": True,
        "format_version": 1,
        "declaration_version": spec.declaration_version,
        "spec_digest": spec.digest,
        "auto_selection": auto["selection"],
        "sequential": {
            "path": sequential["path"],
            "selection": sequential["selection"],
            "summary": sequential["summary"],
            "comparison_sha256": sequential["comparison_sha256"],
        },
        "batch": {
            "path": batch["path"],
            "selection": batch["selection"],
            "summary": batch["summary"],
            "comparison_sha256": batch["comparison_sha256"],
            "batch_stats": batch.get("batch_stats"),
        },
        "comparison": comparison,
        "canonical_projection_json": canonical_json(comparison["sequential_projection"]),
    }
