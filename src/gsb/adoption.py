"""Stable multi-version receipt adjudication and backpressure attribution."""

from __future__ import annotations

from .receipts import FORMAT_ORDER, canonical_normalized_text

CALIBER_RANK = {"v1": 1}


def adjudicate(candidates, error_margin_ms: int) -> dict:
    """Pick the adopted receipt by explicit stable total ordering.

    Ordering keys, highest first:
      1. caliber version rank,
      2. format version (x3 > x2 > x1),
      3. merge sequence,
      4. source receipt text (final deterministic tiebreak).
    When the fixed-point duration delta of two equal-caliber candidates falls
    inside the declared error margin, the higher format still wins but the
    decision is explicitly marked as error-margin influenced.
    """
    if not candidates:
        raise ValueError("adjudication requires at least one candidate")

    def rank_key(candidate):
        normalized = candidate["normalized"]
        return (
            CALIBER_RANK.get(normalized["caliber_version"], 0),
            FORMAT_ORDER[normalized["format_version"]],
            candidate["merge_seq"],
        )

    ordered = sorted(
        candidates,
        key=lambda c: (rank_key(c), canonical_normalized_text(c["normalized"])),
        reverse=True,
    )
    winner = ordered[0]
    winner_norm = winner["normalized"]

    durations = sorted(
        {int(candidate["fixed_point_duration_ms"]) for candidate in candidates}
    )
    within_margin = bool(
        len(durations) >= 2 and (max(durations) - min(durations)) <= error_margin_ms
    )

    rejected = []
    for candidate in ordered[1:]:
        normalized = candidate["normalized"]
        reason = "lower format version under equal caliber"
        if normalized["caliber_version"] != winner_norm["caliber_version"]:
            reason = "lower caliber version"
        elif candidate["merge_seq"] != winner["merge_seq"]:
            reason = "lower merge sequence under equal caliber and format"
        rejected.append(
            {
                "format_version": normalized["format_version"],
                "reason": reason,
                "within_error_margin": within_margin,
            }
        )

    decision = {
        "batch_id": winner_norm["batch_id"],
        "winning_format": winner_norm["format_version"],
        "caliber_version": winner_norm["caliber_version"],
        "winning_event_seqs": winner_norm["event_seqs"],
        "covered_refs": winner_norm["covered_refs"],
        "winning_summary_digest": winner["summary_digest"],
        "merge_seq": winner["merge_seq"],
        "rejected_formats": rejected,
        "error_margin_ms": error_margin_ms,
        "error_margin_applied": within_margin,
        "attempt": winner.get("attempt", 1),
        "adoption_core": canonical_normalized_text(winner_norm),
    }
    return decision


def attribute_segments(queue_ms, backfill_ms, processing_ms) -> dict:
    """Split end-to-end latency into queueing, format backfill and processing."""
    queue_ms = max(0, int(queue_ms))
    backfill_ms = max(0, int(backfill_ms))
    processing_ms = max(0, int(processing_ms))
    total = queue_ms + backfill_ms + processing_ms
    if total == 0:
        ratio = {"queue": 0.0, "backfill": 0.0, "processing": 0.0}
        dominant = "processing"
    else:
        ratio = {
            "queue": round(queue_ms / total, 6),
            "backfill": round(backfill_ms / total, 6),
            "processing": round(processing_ms / total, 6),
        }
        dominant = max(
            (
                ("queue", queue_ms),
                ("backfill", backfill_ms),
                ("processing", processing_ms),
            ),
            key=lambda item: item[1],
        )[0]
    return {
        "segments_ms": {
            "queue_ms": queue_ms,
            "backfill_ms": backfill_ms,
            "processing_ms": processing_ms,
        },
        "contribution_ratio": ratio,
        "total_ms": total,
        "dominant_segment": dominant,
    }
