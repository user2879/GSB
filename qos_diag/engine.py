from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fixedpoint import FixedPoint, saturating_add, saturating_mul
from .model import STAGES, shaper_index


@dataclass(frozen=True)
class StageState:
    balance_q: int
    state_clock_ms: int

    def as_dict(self) -> dict[str, int]:
        return {"balance_q": self.balance_q, "state_clock_ms": self.state_clock_ms}


def initial_state(shaper: dict[str, Any], fp: FixedPoint) -> StageState:
    cap_q, _ = saturating_mul(int(shaper["burst_capacity_tokens"]), fp.q)
    return StageState(balance_q=cap_q, state_clock_ms=0)


def primary_window_start(arrival_ms: int, slide_ms: int) -> int:
    return arrival_ms - (arrival_ms % slide_ms)


def overlapping_window_starts(arrival_ms: int, window_ms: int, slide_ms: int) -> list[int]:
    first = arrival_ms - window_ms + 1
    k = -((-first) // slide_ms) if first > 0 else first // slide_ms
    # Ceiling division for arbitrary signed integers.
    k = -((-first) // slide_ms)
    start = k * slide_ms
    starts: list[int] = []
    last = arrival_ms - (arrival_ms % slide_ms)
    while start <= last:
        if start >= 0:
            starts.append(start)
        start += slide_ms
    return starts


def evaluate_stage_transition(
    shaper: dict[str, Any],
    arrival_ms: int,
    packet_length: int,
    state: StageState,
    fp: FixedPoint,
    *,
    window_start_ms: int | None = None,
) -> tuple[dict[str, Any], StageState]:
    elapsed_ms = arrival_ms - state.state_clock_ms
    if elapsed_ms < 0:
        raise ValueError("stage clock moved backwards")
    q = fp.q
    rate_q, sat_rate = saturating_mul(int(shaper["refill_rate_per_ms"]), q)
    scheduled_refill_q, sat_mul = saturating_mul(elapsed_ms, rate_q)
    raw_balance_q, sat_add = saturating_add(state.balance_q, scheduled_refill_q)
    cap_q, sat_cap = saturating_mul(int(shaper["burst_capacity_tokens"]), q)
    available_q = min(cap_q, raw_balance_q)
    actual_refill_q = available_q - state.balance_q
    required_q, sat_required = saturating_mul(packet_length, q)
    allowed = available_q >= required_q
    after_q = available_q - required_q if allowed else available_q
    saturated = any((sat_rate, sat_mul, sat_add, sat_cap, sat_required)) or raw_balance_q > cap_q
    verdict = "allow" if allowed else "shape"
    record = {
        "pipeline_id": shaper["pipeline_id"],
        "stage": shaper["stage"],
        "shaper_id": shaper["shaper_id"],
        "host": shaper["host"],
        "port": shaper["port"],
        "evaluated": True,
        "verdict": verdict,
        "decision": "allowed" if allowed else "shaped",
        "arrival_ms": arrival_ms,
        "window_start_ms": window_start_ms,
        "elapsed_ms": elapsed_ms,
        "state_clock_ms": arrival_ms,
        "rate_per_ms": shaper["refill_rate_per_ms"],
        "burst_capacity_tokens": shaper["burst_capacity_tokens"],
        "required_tokens": packet_length,
        "required_q": required_q,
        "balance_before_refill_q": state.balance_q,
        "scheduled_refill_q": scheduled_refill_q,
        "refill_q": actual_refill_q,
        "balance_before_decision_q": available_q,
        "balance_after_decision_q": after_q,
        "cap_q": cap_q,
        "overflow_saturated": bool(saturated),
        "basis": {
            "clock": "event arrival_ms acts as the logical monotonic milliseconds clock",
            "refill": "integer refill = elapsed_ms * refill_rate_per_ms * Q",
            "clamp": "balance_before_decision_q = min(burst_capacity_tokens*Q, before_refill + refill)",
            "decision": "allow iff balance_before_decision_q >= required_q",
            "truncation": fp.error_bounds()["truncation_direction"],
            "overflow": fp.error_bounds()["overflow_rule"],
        },
    }
    return record, StageState(balance_q=after_q, state_clock_ms=arrival_ms)


def skipped_stage_record(
    shaper: dict[str, Any],
    arrival_ms: int,
    packet_length: int,
    state: StageState,
    fp: FixedPoint,
    reason: str,
    *,
    window_start_ms: int | None = None,
) -> dict[str, Any]:
    required_q, _ = saturating_mul(packet_length, fp.q)
    return {
        "pipeline_id": shaper["pipeline_id"],
        "stage": shaper["stage"],
        "shaper_id": shaper["shaper_id"],
        "host": shaper["host"],
        "port": shaper["port"],
        "evaluated": False,
        "verdict": "shape",
        "decision": "skipped_upstream",
        "arrival_ms": arrival_ms,
        "window_start_ms": window_start_ms,
        "elapsed_ms": arrival_ms - state.state_clock_ms,
        "state_clock_ms": state.state_clock_ms,
        "rate_per_ms": shaper["refill_rate_per_ms"],
        "burst_capacity_tokens": shaper["burst_capacity_tokens"],
        "required_tokens": packet_length,
        "required_q": required_q,
        "balance_before_refill_q": state.balance_q,
        "scheduled_refill_q": 0,
        "refill_q": 0,
        "balance_before_decision_q": state.balance_q,
        "balance_after_decision_q": state.balance_q,
        "cap_q": saturating_mul(int(shaper["burst_capacity_tokens"]), fp.q)[0],
        "overflow_saturated": False,
        "basis": {
            "clock": "state clock is not advanced for a stage not reached by the event",
            "reason": reason,
            "decision": "chain verdict is shape because an upstream stage shaped the event",
        },
    }


class ChainEngine:
    def __init__(
        self,
        bundle: dict[str, Any],
        fp: FixedPoint | None = None,
        initial_states: dict[tuple[str, str], StageState | dict[str, int]] | None = None,
    ):
        self.bundle = bundle
        self.fp = fp or FixedPoint(int(bundle["config"].get("scale", 2)))
        self.shapers = shaper_index(bundle)
        self.states: dict[tuple[str, str], StageState] = {}
        for pipeline in bundle["pipelines"]:
            pipeline_id = pipeline["pipeline_id"]
            for stage in STAGES:
                shaper = self.shapers[(pipeline_id, stage)]
                key = (pipeline_id, stage)
                if initial_states and key in initial_states:
                    value = initial_states[key]
                    if isinstance(value, StageState):
                        self.states[key] = value
                    else:
                        self.states[key] = StageState(int(value["balance_q"]), int(value["state_clock_ms"]))
                else:
                    self.states[key] = initial_state(shaper, self.fp)

    def evaluate_event(self, event: dict[str, Any]) -> dict[str, Any]:
        pipeline_id = event["pipeline_id"]
        window_start = primary_window_start(int(event["arrival_ms"]), int(self.bundle["config"]["slide_ms"]))
        upstream_allowed = True
        records: list[dict[str, Any]] = []
        for stage in STAGES:
            key = (pipeline_id, stage)
            shaper = self.shapers[key]
            state = self.states[key]
            if not upstream_allowed:
                record = skipped_stage_record(
                    shaper,
                    int(event["arrival_ms"]),
                    int(event["packet_length"]),
                    state,
                    self.fp,
                    "upstream_stage_did_not_allow_event",
                    window_start_ms=window_start,
                )
            else:
                record, next_state = evaluate_stage_transition(
                    shaper,
                    int(event["arrival_ms"]),
                    int(event["packet_length"]),
                    state,
                    self.fp,
                    window_start_ms=window_start,
                )
                self.states[key] = next_state
                if record["verdict"] != "allow":
                    upstream_allowed = False
            records.append(record)
        chain_allowed = all(record["verdict"] == "allow" for record in records)
        return {
            "event_seq": event["event_seq"],
            "event_id": event["event_id"],
            "pipeline_id": pipeline_id,
            "arrival_ms": event["arrival_ms"],
            "packet_length": event["packet_length"],
            "window_start_ms": window_start,
            "verdict": "allow" if chain_allowed else "shape",
            "decision": "allowed" if chain_allowed else "shaped",
            "stages": records,
        }

    def run_event_path(self, events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        source = self.bundle["events"] if events is None else events
        return [self.evaluate_event(event) for event in source]


def run_event_path(
    bundle: dict[str, Any],
    fp: FixedPoint | None = None,
    initial_states: dict[tuple[str, str], StageState | dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    engine = ChainEngine(bundle, fp=fp, initial_states=initial_states)
    return engine.run_event_path()


def run_batched_event_path(bundle: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    """Run through explicit batch boundaries while carrying all stage state."""

    fp = FixedPoint(int(bundle["config"]["scale"]))
    engine = ChainEngine(bundle, fp)
    results: list[dict[str, Any]] = []
    events = bundle["events"]
    for start in range(0, len(events), max(1, batch_size)):
        batch = events[start : start + batch_size]
        results.extend(engine.evaluate_event(event) for event in batch)
    return results


def _materialize_group(
    shaper: dict[str, Any],
    group: list[dict[str, Any]],
    state: StageState,
    fp: FixedPoint,
    window_start: int,
) -> tuple[list[dict[str, Any]], StageState]:
    first = group[0]
    arrival_ms = int(first["arrival_ms"])
    length = int(first["packet_length"])
    elapsed_ms = arrival_ms - state.state_clock_ms
    rate_q, sat_rate = saturating_mul(int(shaper["refill_rate_per_ms"]), fp.q)
    scheduled_refill_q, sat_mul = saturating_mul(elapsed_ms, rate_q)
    raw_available_q, sat_add = saturating_add(state.balance_q, scheduled_refill_q)
    cap_q, sat_cap = saturating_mul(int(shaper["burst_capacity_tokens"]), fp.q)
    available_q = min(cap_q, raw_available_q)
    actual_refill_q = available_q - state.balance_q
    required_q, sat_required = saturating_mul(length, fp.q)
    allowed_count = min(len(group), available_q // required_q) if required_q > 0 else 0
    final_balance_q = available_q - allowed_count * required_q
    saturated = any((sat_rate, sat_mul, sat_add, sat_cap, sat_required)) or raw_available_q > cap_q

    records: list[dict[str, Any]] = []
    for index, event in enumerate(group):
        allowed_before = min(index, allowed_count)
        if index == 0:
            before_refill_q = state.balance_q
            refill_q = actual_refill_q
            event_elapsed_ms = elapsed_ms
        else:
            before_refill_q = available_q - allowed_before * required_q
            refill_q = 0
            event_elapsed_ms = 0
        before_decision_q = before_refill_q + refill_q
        allowed = index < allowed_count
        after_q = before_decision_q - required_q if allowed else before_decision_q
        records.append(
            {
                "pipeline_id": shaper["pipeline_id"],
                "stage": shaper["stage"],
                "shaper_id": shaper["shaper_id"],
                "host": shaper["host"],
                "port": shaper["port"],
                "evaluated": True,
                "verdict": "allow" if allowed else "shape",
                "decision": "allowed" if allowed else "shaped",
                "arrival_ms": arrival_ms,
                "window_start_ms": window_start,
                "elapsed_ms": event_elapsed_ms,
                "state_clock_ms": arrival_ms,
                "rate_per_ms": shaper["refill_rate_per_ms"],
                "burst_capacity_tokens": shaper["burst_capacity_tokens"],
                "required_tokens": length,
                "required_q": required_q,
                "balance_before_refill_q": before_refill_q,
                "scheduled_refill_q": scheduled_refill_q if index == 0 else 0,
                "refill_q": refill_q,
                "balance_before_decision_q": before_decision_q,
                "balance_after_decision_q": after_q,
                "cap_q": cap_q,
                "overflow_saturated": bool(saturated),
                "basis": {
                    "clock": "event arrival_ms acts as the logical monotonic milliseconds clock",
                    "refill": "integer refill = elapsed_ms * refill_rate_per_ms * Q",
                    "clamp": "balance_before_decision_q = min(burst_capacity_tokens*Q, before_refill + refill)",
                    "decision": "allow iff balance_before_decision_q >= required_q",
                    "truncation": fp.error_bounds()["truncation_direction"],
                    "overflow": fp.error_bounds()["overflow_rule"],
                },
            }
        )
    return records, StageState(final_balance_q, arrival_ms)


def run_window_path(
    bundle: dict[str, Any],
    fp: FixedPoint | None = None,
    initial_states: dict[tuple[str, str], StageState | dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate maximal same-arrival/same-length runs by sliding window."""

    fp = fp or FixedPoint(int(bundle["config"]["scale"]))
    engine = ChainEngine(bundle, fp, initial_states=initial_states)
    shapers = shaper_index(bundle)
    records_by_seq_stage: dict[tuple[int, str], dict[str, Any]] = {}

    for pipeline in bundle["pipelines"]:
        pipeline_id = pipeline["pipeline_id"]
        events = [event for event in bundle["events"] if event["pipeline_id"] == pipeline_id]
        upstream_allows: dict[str, list[bool]] = {}
        for stage_index, stage in enumerate(STAGES):
            shaper = shapers[(pipeline_id, stage)]
            state = engine.states[(pipeline_id, stage)]
            allows: list[bool] = []
            index = 0
            while index < len(events):
                event = events[index]
                reached = stage_index == 0 or upstream_allows[STAGES[stage_index - 1]][index]
                window_start = primary_window_start(int(event["arrival_ms"]), int(bundle["config"]["slide_ms"]))
                if not reached:
                    record = skipped_stage_record(
                        shaper,
                        int(event["arrival_ms"]),
                        int(event["packet_length"]),
                        state,
                        fp,
                        "upstream_stage_did_not_allow_event",
                        window_start_ms=window_start,
                    )
                    records_by_seq_stage[(event["event_seq"], stage)] = record
                    allows.append(False)
                    index += 1
                    continue

                end = index + 1
                while (
                    end < len(events)
                    and (stage_index == 0 or upstream_allows[STAGES[stage_index - 1]][end])
                    and events[end]["arrival_ms"] == event["arrival_ms"]
                    and events[end]["packet_length"] == event["packet_length"]
                ):
                    end += 1
                group = events[index:end]
                group_records, next_state = _materialize_group(shaper, group, state, fp, window_start)
                for group_event, record in zip(group, group_records):
                    records_by_seq_stage[(group_event["event_seq"], stage)] = record
                state = next_state
                engine.states[(pipeline_id, stage)] = state
                allows.extend([record["verdict"] == "allow" for record in group_records])
                index = end
            upstream_allows[stage] = allows

    results: list[dict[str, Any]] = []
    for event in bundle["events"]:
        stages = [records_by_seq_stage[(event["event_seq"], stage)] for stage in STAGES]
        chain_allowed = all(record["verdict"] == "allow" for record in stages)
        results.append(
            {
                "event_seq": event["event_seq"],
                "event_id": event["event_id"],
                "pipeline_id": event["pipeline_id"],
                "arrival_ms": event["arrival_ms"],
                "packet_length": event["packet_length"],
                "window_start_ms": stages[0]["window_start_ms"],
                "verdict": "allow" if chain_allowed else "shape",
                "decision": "allowed" if chain_allowed else "shaped",
                "stages": stages,
            }
        )
    return results


def states_from_results(results: list[dict[str, Any]]) -> dict[tuple[str, str], StageState]:
    states: dict[tuple[str, str], StageState] = {}
    for result in results:
        for stage_record in result["stages"]:
            key = (result["pipeline_id"], stage_record["stage"])
            states[key] = StageState(
                balance_q=int(stage_record["balance_after_decision_q"]),
                state_clock_ms=int(stage_record["state_clock_ms"]),
            )
    return states
