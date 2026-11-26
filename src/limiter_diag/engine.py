"""Deterministic per-event and millisecond-window token-bucket engines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Sequence

from .canon import canonical_bytes, sha256_value
from .fixed import (
    advance_many,
    decimal_text,
    fraction_text,
    lsb_fraction_to_tokens,
    saturating_add,
    saturating_mul,
)
from .spec import Bucket, DecisionPolicy, Event, EventStream, Spec


ALLOW = "ALLOW"
SHAPE = "SHAPE"
DROP = "DROP"
SKIP = "SKIP"
CHECKPOINT_FORMAT = "gateway-limit-checkpoint/v1"
EVENT_UNITS_PER_LEVEL = 2
WINDOW_UNITS_PER_LEVEL = 4
WRITEBACK_UNITS_PER_LEVEL = 1


@dataclass
class LevelRuntime:
    bucket: Bucket
    balance_fp: int
    queue_fp: deque[int]
    last_ms: int | None = None
    refill_count: int = 0
    saturated_refills: int = 0

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "level": self.bucket.level,
            "bucket_id": self.bucket.bucket_id,
            "balance_fp": self.balance_fp,
            "queue_fp": list(self.queue_fp),
            "last_ms": self.last_ms,
            "refill_count": self.refill_count,
            "saturated_refills": self.saturated_refills,
        }


@dataclass(frozen=True)
class PathChoice:
    path: str
    reason_code: str
    info: dict[str, Any]

    def canonical_dict(self) -> dict[str, Any]:
        return {"path": self.path, "reason_code": self.reason_code, **self.info}


@dataclass
class RunOutcome:
    path: str
    path_choice: PathChoice
    results: list[dict[str, Any]]
    checkpoint: dict[str, Any]
    error_report: dict[str, Any]
    analysis_digest: str

    def canonical_analysis(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "path_choice": self.path_choice.canonical_dict(),
            "results": self.results,
            "checkpoint": self.checkpoint,
            "error_report": self.error_report,
        }


def _initial_runtimes(spec: Spec) -> list[LevelRuntime]:
    runtimes: list[LevelRuntime] = []
    for bucket in spec.buckets:
        capacity_fp = saturating_mul(bucket.burst_capacity, spec.scale_factor)
        runtimes.append(LevelRuntime(bucket=bucket, balance_fp=capacity_fp, queue_fp=deque()))
    return runtimes


def _runtime_from_checkpoint(spec: Spec, checkpoint: dict[str, Any]) -> list[LevelRuntime]:
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if checkpoint.get("spec_digest") != spec.digest():
        raise ValueError("checkpoint spec digest mismatch")
    levels = checkpoint.get("levels")
    if not isinstance(levels, list) or len(levels) != len(spec.buckets):
        raise ValueError("checkpoint levels are invalid")
    runtimes: list[LevelRuntime] = []
    for bucket, level_value in zip(spec.buckets, levels):
        if not isinstance(level_value, dict):
            raise ValueError("checkpoint level entry is invalid")
        if level_value.get("level") != bucket.level or level_value.get("bucket_id") != bucket.bucket_id:
            raise ValueError("checkpoint bucket identity mismatch")
        balance = level_value.get("balance_fp")
        queue = level_value.get("queue_fp")
        last_ms = level_value.get("last_ms")
        refill_count = level_value.get("refill_count")
        saturated = level_value.get("saturated_refills")
        if (
            isinstance(balance, bool)
            or not isinstance(balance, int)
            or not isinstance(queue, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in queue)
            or (last_ms is not None and (isinstance(last_ms, bool) or not isinstance(last_ms, int)))
            or isinstance(refill_count, bool)
            or not isinstance(refill_count, int)
            or isinstance(saturated, bool)
            or not isinstance(saturated, int)
        ):
            raise ValueError("checkpoint state field type mismatch")
        runtimes.append(
            LevelRuntime(
                bucket=bucket,
                balance_fp=balance,
                queue_fp=deque(int(item) for item in queue),
                last_ms=last_ms,
                refill_count=refill_count,
                saturated_refills=saturated,
            )
        )
    return runtimes


def _advance_all(runtimes: Sequence[LevelRuntime], now_ms: int, spec: Spec) -> list[dict[str, int]]:
    elapsed_values: list[int] = []
    balances: list[int] = []
    capacities: list[int] = []
    rates: list[int] = []
    for runtime in runtimes:
        if runtime.last_ms is None:
            elapsed = 0
            runtime.last_ms = now_ms
        else:
            elapsed = now_ms - runtime.last_ms
            if elapsed < 0:
                raise ValueError("monotonic clock moved backwards")
            runtime.last_ms = now_ms
        elapsed_values.append(max(0, elapsed))
        balances.append(runtime.balance_fp)
        capacities.append(saturating_mul(runtime.bucket.burst_capacity, spec.scale_factor))
        rates.append(runtime.bucket.rate_per_sec)

    new_balances, residues, saturated = advance_many(
        balances,
        capacities,
        rates,
        elapsed_values,
        spec.scale_exponent,
    )
    updates: list[dict[str, int]] = []
    for runtime, balance, residue, was_saturated, elapsed in zip(
        runtimes, new_balances, residues, saturated, elapsed_values
    ):
        if elapsed > 0:
            runtime.refill_count += 1
        if was_saturated:
            runtime.saturated_refills += 1
        runtime.balance_fp = balance
        updates.append(
            {
                "elapsed_ms": elapsed,
                "refill_residual_numerator": residue,
            }
        )
    return updates


def _drain_queues(runtimes: Sequence[LevelRuntime]) -> None:
    for runtime in runtimes:
        while runtime.queue_fp and runtime.balance_fp >= runtime.queue_fp[0]:
            runtime.balance_fp -= runtime.queue_fp.popleft()


def _uncertainty_segments(
    runtime: LevelRuntime,
    upstream: Sequence[LevelRuntime],
    is_window_boundary: bool,
    refill_elapsed_ms: int,
) -> tuple[int, int, int]:
    """Local comparison budget, in LSBs.

    One LSB is allowed for the current replenishment truncation, one LSB for
    each upstream state handoff, and one LSB when the event sits on a window
    boundary. Cumulative refill counts are persisted separately for audit.
    """
    refill_lsb = 1 if refill_elapsed_ms > 0 else 0
    carry_lsb = len(upstream)
    window_lsb = 1 if is_window_boundary else 0
    return refill_lsb, carry_lsb, window_lsb


def _choose_cause(refill_lsb: int, carry_lsb: int, window_lsb: int) -> str:
    candidates = (("refill", refill_lsb), ("carry", carry_lsb), ("window", window_lsb))
    largest = max((value for _, value in candidates), default=0)
    if largest <= 0:
        return "none"
    for name, value in candidates:
        if value == largest:
            return name
    return "none"


def _process_event(
    runtimes: Sequence[LevelRuntime],
    event: Event,
    window_index: int,
    spec: Spec,
    refill_updates: Sequence[dict[str, int]],
) -> dict[str, Any]:
    levels: list[dict[str, Any]] = []
    eligible = True
    verdict = ALLOW
    reached_levels: list[int] = []
    window_boundary = event.arrival_ms % spec.decision.window_ms == 0
    for index, (runtime, update) in enumerate(zip(runtimes, refill_updates)):
        bucket = runtime.bucket
        needed_fp = saturating_mul(bucket.packet_len, spec.scale_factor)
        before = runtime.balance_fp
        queue_before = len(runtime.queue_fp)
        refill_lsb, carry_lsb, window_lsb = _uncertainty_segments(
            runtime,
            runtimes[:index],
            window_boundary,
            int(update["elapsed_ms"]),
        )
        uncertainty = refill_lsb + carry_lsb + window_lsb
        ambiguous = False
        overflow = False
        tiebreak: str | None = None

        if not eligible:
            status = SKIP
        else:
            reached_levels.append(bucket.level)
            if before >= needed_fp:
                status = ALLOW
                runtime.balance_fp -= needed_fp
            elif len(runtime.queue_fp) < bucket.queue_limit:
                status = SHAPE
                runtime.queue_fp.append(needed_fp)
            else:
                status = DROP
                overflow = True
                verdict = DROP
            if status != ALLOW and verdict == ALLOW:
                verdict = status
            difference = abs(before - needed_fp)
            if uncertainty > 0 and difference <= uncertainty:
                ambiguous = True
                tiebreak = (
                    f"level={bucket.level};seq={event.seq};"
                    "rule=conservative_numeric_then_level_then_seq"
                )
            eligible = status == ALLOW

        after = runtime.balance_fp
        cause = _choose_cause(refill_lsb, carry_lsb, window_lsb) if ambiguous else "none"
        levels.append(
            {
                "level": bucket.level,
                "name": bucket.name,
                "bucket_id": bucket.bucket_id,
                "reached": status != SKIP,
                "status": status,
                "before_fp": before,
                "after_fp": after,
                "needed_fp": needed_fp,
                "overflow": overflow,
                "queue_used_before": queue_before,
                "queue_used_after": len(runtime.queue_fp),
                "refill_elapsed_ms": int(update["elapsed_ms"]),
                "refill_residual_numerator": int(update["refill_residual_numerator"]),
                "refill_count_after": runtime.refill_count,
                "refill_lsb": refill_lsb,
                "carry_lsb": carry_lsb,
                "window_lsb": window_lsb,
                "uncertainty_lsb": uncertainty,
                "error_affected": ambiguous,
                "cause": cause,
                "tiebreak": tiebreak,
            }
        )

    pair_flags: list[dict[str, Any]] = []
    for left_index, right_index in zip(reached_levels, reached_levels[1:]):
        left = levels[left_index - 1]
        right = levels[right_index - 1]
        combined = int(left["uncertainty_lsb"]) + int(right["uncertainty_lsb"])
        delta = abs(int(left["before_fp"]) - int(right["before_fp"]))
        if combined > 0 and delta <= combined:
            pair_flags.append(
                {
                    "left_level": left_index,
                    "right_level": right_index,
                    "delta_fp": delta,
                    "bound_lsb": combined,
                    "tiebreak": f"level={left_index};seq={event.seq};rule=level_order",
                }
            )

    level_affected = any(bool(level["error_affected"]) for level in levels)
    causes = sorted(
        {
            str(level["cause"])
            for level in levels
            if level["error_affected"] and level["cause"] != "none"
        }
    )
    return {
        "seq": event.seq,
        "arrival_ms": event.arrival_ms,
        "window_index": window_index,
        "verdict": verdict,
        "levels": levels,
        "indistinguishable_level_pairs": pair_flags,
        "error_affected": level_affected or bool(pair_flags),
        "causes": causes,
        "window_boundary_affected": False,
    }


def _run_events(
    spec: Spec,
    events: EventStream,
    *,
    path: str,
    initial_checkpoint: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[LevelRuntime], dict[str, Any]]:
    runtimes = _runtime_from_checkpoint(spec, initial_checkpoint) if initial_checkpoint else _initial_runtimes(spec)
    results: list[dict[str, Any]] = []
    start_seq = (initial_checkpoint.get("last_seq", 0) if initial_checkpoint else 0) + 1
    pending_events = [event for event in events.events if event.seq >= start_seq]

    def consume(event: Event) -> None:
        window_index = event.arrival_ms // spec.decision.window_ms
        updates = _advance_all(runtimes, event.arrival_ms, spec)
        _drain_queues(runtimes)
        results.append(_process_event(runtimes, event, window_index, spec, updates))

    if path == "eventwise":
        for event in pending_events:
            consume(event)
    elif path == "window":
        windows: dict[int, list[Event]] = {}
        for event in pending_events:
            windows.setdefault(event.arrival_ms // spec.decision.window_ms, []).append(event)
        for window_index in sorted(windows):
            window_events = windows[window_index]
            subgroup: list[Event] = []
            current_ms: int | None = None
            for event in window_events:
                if current_ms is not None and event.arrival_ms != current_ms:
                    for grouped_event in subgroup:
                        consume(grouped_event)
                    subgroup = []
                subgroup.append(event)
                current_ms = event.arrival_ms
                if event.arrival_ms == current_ms and len(subgroup) == 1:
                    continue
            for event in subgroup:
                consume(event)
    else:
        raise ValueError(f"unknown path {path!r}")

    last_seq = results[-1]["seq"] if results else (initial_checkpoint or {}).get("last_seq")
    last_arrival = results[-1]["arrival_ms"] if results else (initial_checkpoint or {}).get("last_arrival_ms")
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "spec_digest": spec.digest(),
        "events_digest": events.digest(),
        "last_seq": last_seq,
        "last_arrival_ms": last_arrival,
        "levels": [runtime.checkpoint_dict() for runtime in runtimes],
    }
    return results, runtimes, checkpoint


def choose_path(spec: Spec, events: EventStream) -> PathChoice:
    event_count = events.event_count
    bucket_count = spec.bucket_count
    window_count = events.window_count(spec.decision.window_ms)
    mean_gap = events.mean_gap
    event_work = event_count * bucket_count * EVENT_UNITS_PER_LEVEL
    window_work = window_count * bucket_count * WINDOW_UNITS_PER_LEVEL + event_count * bucket_count * WRITEBACK_UNITS_PER_LEVEL
    tolerance_units = (max(event_work, window_work) * spec.decision.work_tolerance_bp + 9999) // 10000
    difference = abs(event_work - window_work)
    threshold_triggered = False
    reason_code = ""
    reason = ""

    if event_count <= spec.decision.event_count_threshold:
        path = "eventwise"
        reason_code = "event_count_at_or_below_threshold"
        reason = "event count is at or below the declared small-stream threshold"
        threshold_triggered = True
    elif mean_gap >= spec.decision.mean_gap_threshold_ms:
        path = "eventwise"
        reason_code = "mean_gap_at_or_above_threshold"
        reason = "mean inter-arrival gap is at or above the declared sparse-stream threshold"
        threshold_triggered = True
    elif difference <= tolerance_units:
        path = "eventwise"
        reason_code = "work_tolerance_stable_tiebreak"
        reason = "estimated work difference is within tolerance; eventwise wins the stable tie"
        threshold_triggered = True
    elif window_work < event_work:
        path = "window"
        reason_code = "lower_window_work"
        reason = "aggregated window work is lower after threshold checks"
    else:
        path = "eventwise"
        reason_code = "lower_eventwise_work"
        reason = "per-event work is lower after threshold checks"

    info = {
        "reason": reason,
        "event_count": event_count,
        "bucket_count": bucket_count,
        "window_count": window_count,
        "window_ms": spec.decision.window_ms,
        "mean_gap": fraction_text(mean_gap),
        "mean_gap_decimal": decimal_text(mean_gap, 6),
        "event_work_units": event_work,
        "window_work_units": window_work,
        "work_difference_units": difference,
        "work_tolerance_units": tolerance_units,
        "threshold_triggered": threshold_triggered,
        "stable_tiebreak": "eventwise",
        "scaling_rule": {
            "event_units_per_level": EVENT_UNITS_PER_LEVEL,
            "window_units_per_level": WINDOW_UNITS_PER_LEVEL,
            "writeback_units_per_level": WRITEBACK_UNITS_PER_LEVEL,
            "event_work": "event_count * bucket_count * event_units_per_level",
            "window_work": "window_count * bucket_count * window_units_per_level + event_count * bucket_count * writeback_units_per_level",
        },
        "boundaries": {
            "event_count_threshold": spec.decision.event_count_threshold,
            "mean_gap_threshold_ms": spec.decision.mean_gap_threshold_ms,
            "work_tolerance_bp": spec.decision.work_tolerance_bp,
            "first_event_count_above_threshold": spec.decision.event_count_threshold + 1,
            "mean_gap_below_threshold": max(0, spec.decision.mean_gap_threshold_ms - 1),
        },
    }
    return PathChoice(path=path, reason_code=reason_code, info=info)


def _window_summaries(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: dict[int, list[dict[str, Any]]] = {}
    for record in results:
        windows.setdefault(int(record["window_index"]), []).append(record)
    summaries: list[dict[str, Any]] = []
    for window_index in sorted(windows):
        records = windows[window_index]
        constrained = sum(1 for item in records if item["verdict"] != ALLOW)
        summaries.append(
            {
                "window_index": window_index,
                "first_seq": records[0]["seq"],
                "last_seq": records[-1]["seq"],
                "event_count": len(records),
                "constrained_count": constrained,
                "shaping_rate": fraction_text(Fraction(constrained, len(records))),
            }
        )
    return summaries


def _mark_window_ambiguity(
    results: list[dict[str, Any]],
    windows: Sequence[dict[str, Any]],
    total_error_lsb: int,
    exponent: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not windows:
        return checks
    total_events = sum(int(item["event_count"]) for item in windows)
    scale = 1 << exponent
    bound = Fraction(total_error_lsb, scale * max(1, total_events))
    for left, right in zip(windows, windows[1:]):
        left_rate = Fraction(*map(int, str(left["shaping_rate"]).split("/")))
        right_rate = Fraction(*map(int, str(right["shaping_rate"]).split("/")))
        delta = abs(left_rate - right_rate)
        indistinguishable = delta <= bound
        checks.append(
            {
                "left_window": left["window_index"],
                "right_window": right["window_index"],
                "left_shaping_rate": left["shaping_rate"],
                "right_shaping_rate": right["shaping_rate"],
                "rate_delta": fraction_text(delta),
                "rate_delta_decimal": decimal_text(delta, 9),
                "bound": fraction_text(bound),
                "indistinguishable": indistinguishable,
                "tiebreak": "lower_window_index" if indistinguishable else None,
            }
        )
        if indistinguishable:
            for record in results:
                if int(record["window_index"]) in {int(left["window_index"]), int(right["window_index"])}:
                    record["window_boundary_affected"] = True
                    record["error_affected"] = True
                    causes = set(record.get("causes", []))
                    causes.add("window")
                    record["causes"] = sorted(causes)
    return checks


def build_error_report(spec: Spec, events: EventStream, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    event_count = events.event_count
    scale = spec.scale_factor
    refill_by_level: list[int] = [0, 0, 0]
    if results:
        for level_index in range(3):
            refill_by_level[level_index] = int(results[-1]["levels"][level_index]["refill_count_after"])

    per_level: list[dict[str, Any]] = []
    for index, bucket in enumerate(spec.buckets):
        refill_lsb = 1
        carry_lsb = index
        window_lsb = 1
        total_lsb = refill_lsb + carry_lsb + window_lsb
        per_level.append(
            {
                "level": bucket.level,
                "name": bucket.name,
                "bucket_id": bucket.bucket_id,
                "refill_lsb": refill_lsb,
                "carry_lsb": carry_lsb,
                "window_lsb": window_lsb,
                "total_lsb": total_lsb,
                "total_tokens": fraction_text(lsb_fraction_to_tokens(total_lsb, spec.scale_exponent)),
                "cumulative_refill_calls": refill_by_level[index],
            }
        )

    refill_total = sum(item["refill_lsb"] for item in per_level)
    carry_total = sum(item["carry_lsb"] for item in per_level)
    window_total = sum(item["window_lsb"] for item in per_level)
    source_bounds = {
        "refill": refill_total,
        "carry": carry_total,
        "window": window_total,
    }
    grand_total = sum(source_bounds.values())
    source_shares: list[dict[str, Any]] = []
    for source in ("refill", "carry", "window"):
        share = Fraction(source_bounds[source], grand_total) if grand_total else Fraction(0, 1)
        source_shares.append(
            {
                "source": source,
                "bound_lsb": source_bounds[source],
                "share": fraction_text(share),
                "share_decimal": decimal_text(share, 9),
            }
        )
    if grand_total:
        largest = max(source_bounds.values())
        if largest * 10000 >= spec.decision.dominant_threshold_bp * grand_total:
            dominant = next(source for source in ("refill", "carry", "window") if source_bounds[source] == largest)
            dominant_decision = "dominant"
        else:
            dominant = "balanced"
            dominant_decision = "below_threshold"
    else:
        dominant = "none"
        dominant_decision = "no_error_budget"

    ambiguous_level_records = [
        record
        for record in results
        if any(bool(level["error_affected"]) for level in record["levels"])
    ]
    ambiguous_events = [record for record in results if record.get("error_affected")]
    source_event_counts = {"refill": 0, "carry": 0, "window": 0}
    for record in results:
        for cause in record.get("causes", []):
            if cause in source_event_counts:
                source_event_counts[cause] += 1
    aggregate_bound = Fraction(len(ambiguous_events), event_count) if event_count else Fraction(0, 1)

    window_summaries = _window_summaries(results)
    window_checks = _mark_window_ambiguity(results, window_summaries, grand_total, spec.scale_exponent)
    post_ambiguous_events = [record for record in results if record.get("error_affected")]
    if len(post_ambiguous_events) != len(ambiguous_events):
        aggregate_bound = Fraction(len(post_ambiguous_events), event_count) if event_count else Fraction(0, 1)
        source_event_counts["window"] = sum(
            1 for record in results if "window" in set(record.get("causes", []))
        )

    return {
        "fixed_point": {
            "scale_exponent": spec.scale_exponent,
            "scale_factor": scale,
            "rounding": "floor_toward_negative_infinity_nonnegative_inputs",
            "saturation": "signed_int64_max",
            "single_refill_quantization_error": {
                "upper_bound_lsb": 1,
                "upper_bound_tokens": fraction_text(Fraction(999, 1000 * scale)),
            },
        },
        "comparison_budget_note": (
            "per-level comparison budget is local: one LSB for current refill, "
            "one LSB per upstream handoff, one LSB for a window boundary; "
            "cumulative refill calls are reported for audit"
        ),
        "per_level_comparison_error": per_level,
        "end_to_end_shaping_rate_error_upper_bound": {
            "fraction": fraction_text(aggregate_bound),
            "decimal": decimal_text(aggregate_bound, 9),
            "ambiguous_event_count": len(post_ambiguous_events),
            "event_count": event_count,
        },
        "sources": source_shares,
        "dominant_threshold_bp": spec.decision.dominant_threshold_bp,
        "dominant_source": dominant,
        "dominant_decision": dominant_decision,
        "source_event_counts": source_event_counts,
        "ambiguous_level_event_count": len(ambiguous_level_records),
        "ambiguous_event_count": len(post_ambiguous_events),
        "windows": window_summaries,
        "window_candidate_checks": window_checks,
    }


def run_path(
    spec: Spec,
    events: EventStream,
    path: str,
    *,
    initial_checkpoint: dict[str, Any] | None = None,
    path_choice: PathChoice | None = None,
) -> RunOutcome:
    if path not in {"eventwise", "window"}:
        raise ValueError(f"unknown path {path!r}")
    results, _runtimes, checkpoint = _run_events(
        spec,
        events,
        path=path,
        initial_checkpoint=initial_checkpoint,
    )
    error_report = build_error_report(spec, events, results)
    digest = sha256_value({"results": results, "error_report": error_report})
    if path_choice is None:
        path_choice = choose_path(spec, events)
    chosen = PathChoice(path=path, reason_code=path_choice.reason_code, info=path_choice.info)
    return RunOutcome(
        path=path,
        path_choice=chosen,
        results=results,
        checkpoint=checkpoint,
        error_report=error_report,
        analysis_digest=digest,
    )


def first_difference(left: Any, right: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        left_keys = sorted(left)
        right_keys = sorted(right)
        if left_keys != right_keys:
            missing_left = sorted(set(right) - set(left))
            missing_right = sorted(set(left) - set(right))
            field = missing_left[0] if missing_left else missing_right[0]
            return f"{path}.{field}", left.get(field), right.get(field)
        for key in left_keys:
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            found = first_difference(left_item, right_item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if left != right:
        return path, left, right
    return None


def compare_results(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(left) != len(right):
        first_seq = min(
            [int(item["seq"]) for item in left]
            + [int(item["seq"]) for item in right]
        )
        return {
            "equal": False,
            "seq": first_seq,
            "level": None,
            "field": "event_count",
            "left": len(left),
            "right": len(right),
        }
    for left_record, right_record in zip(left, right):
        if canonical_bytes(left_record) == canonical_bytes(right_record):
            continue
        found = first_difference(left_record, right_record, "$")
        if found is None:
            field = "$"
            left_value = right_value = None
        else:
            field, left_value, right_value = found
        seq = int(left_record.get("seq", right_record.get("seq", 0)))
        level: int | None = None
        if field.startswith("$.levels["):
            try:
                level_index = int(field[len("$.levels[") :].split("]", 1)[0])
                level = level_index + 1
            except (ValueError, IndexError):
                level = None
        return {
            "equal": False,
            "seq": seq,
            "level": level,
            "field": field,
            "left": left_value,
            "right": right_value,
        }
    return {"equal": True}


def divergence_error(comparison: dict[str, Any]) -> Any:
    from .errors import DiagnosticError

    return DiagnosticError(
        "E_DIVERGE",
        "eventwise and window paths produced different deterministic results",
        entry=f"events[seq={comparison.get('seq')}]",
        field=str(comparison.get("field", "$")),
        details=comparison,
    )
