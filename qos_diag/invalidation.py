from __future__ import annotations

import copy
from typing import Any

from .canonical import canonical_bytes, sha256_hex
from .engine import (
    StageState,
    evaluate_stage_transition,
    overlapping_window_starts,
    primary_window_start,
    skipped_stage_record,
)
from .fixedpoint import FixedPoint
from .model import STAGES, STAGE_RANK, shaper_index


def _event_index_maps(bundle: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id = {event["event_id"]: event for event in bundle["events"]}
    by_pipeline: dict[str, list[dict[str, Any]]] = {}
    for pipeline in bundle["pipelines"]:
        pid = pipeline["pipeline_id"]
        by_pipeline[pid] = [event for event in bundle["events"] if event["pipeline_id"] == pid]
    return by_id, by_pipeline


def _window_chain_from(bundle: dict[str, Any], pid: str, first_event: dict[str, Any]) -> list[int]:
    slide = int(bundle["config"]["slide_ms"])
    first_start = primary_window_start(int(first_event["arrival_ms"]), slide)
    starts = sorted(
        {
            primary_window_start(int(event["arrival_ms"]), slide)
            for event in bundle["events"]
            if event["pipeline_id"] == pid
            and primary_window_start(int(event["arrival_ms"]), slide) >= first_start
        }
    )
    return starts


def derive_invalidation(
    old_bundle: dict[str, Any],
    new_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Derive stable chain/window invalidation closure.

    ``restart_local_index[pid][stage]`` is the first pipeline-local event for
    which that stage must be recomputed.  ``None`` means the stage prefix is
    fully reusable.
    """

    old_shapers = {item["shaper_id"]: item for item in old_bundle["shapers"]}
    new_shapers = {item["shaper_id"]: item for item in new_bundle["shapers"]}
    old_pipelines = {item["pipeline_id"]: item for item in old_bundle["pipelines"]}
    new_pipelines = {item["pipeline_id"]: item for item in new_bundle["pipelines"]}
    _, old_by_pipeline = _event_index_maps(old_bundle)
    new_by_id, new_by_pipeline = _event_index_maps(new_bundle)

    restart: dict[str, dict[str, int | None]] = {
        pid: {stage: None for stage in STAGES} for pid in new_pipelines
    }
    changes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    extra_windows: dict[str, set[int]] = {pid: set() for pid in new_pipelines}

    full_recompute = False
    full_reason: str | None = None
    old_ids = [event["event_id"] for event in old_bundle["events"]]
    new_ids = [event["event_id"] for event in new_bundle["events"]]
    if old_ids != new_ids:
        full_recompute = True
        full_reason = "event identities/order changed"
    for field in ("scale", "window_ms", "slide_ms"):
        if old_bundle["config"].get(field) != new_bundle["config"].get(field):
            full_recompute = True
            full_reason = f"diagnosis config field {field} changed"

    if full_recompute:
        for pid in restart:
            for stage in STAGES:
                restart[pid][stage] = 0
        steps.append({"step": "full_closure", "reason": full_reason, "scope": "all pipelines/stages/events"})

    for pid in sorted(set(old_pipelines) & set(new_pipelines)):
        old_map = old_pipelines[pid]["stage_shaper_ids"]
        new_map = new_pipelines[pid]["stage_shaper_ids"]
        for stage in STAGES:
            if old_map.get(stage) != new_map.get(stage):
                rank = STAGE_RANK[stage]
                change = {
                    "change_type": "pipeline_chain_relation",
                    "pipeline_id": pid,
                    "stage": stage,
                    "field": f"stage_shaper_ids.{stage}",
                    "old_value": old_map.get(stage),
                    "new_value": new_map.get(stage),
                }
                changes.append(change)
                for downstream in STAGES[rank:]:
                    restart[pid][downstream] = 0
                steps.append(
                    {
                        "step": "chain_relation_changed",
                        "pipeline_id": pid,
                        "origin_stage": stage,
                        "invalidated_stages": list(STAGES[rank:]),
                        "basis": "a replaced chain node changes that stage and every downstream stage",
                    }
                )

    for shaper_id in sorted(set(old_shapers) & set(new_shapers)):
        old_item = old_shapers[shaper_id]
        new_item = new_shapers[shaper_id]
        pid = new_item["pipeline_id"]
        if pid not in restart:
            continue
        for field in ("refill_rate_per_ms", "burst_capacity_tokens", "host", "port"):
            if old_item.get(field) == new_item.get(field):
                continue
            stage = new_item["stage"]
            rank = STAGE_RANK[stage]
            changes.append(
                {
                    "change_type": "shaper_parameter",
                    "pipeline_id": pid,
                    "stage": stage,
                    "shaper_id": shaper_id,
                    "field": field,
                    "old_value": old_item.get(field),
                    "new_value": new_item.get(field),
                }
            )
            for downstream in STAGES[rank:]:
                restart[pid][downstream] = 0
            steps.append(
                {
                    "step": "shaper_parameter_changed",
                    "pipeline_id": pid,
                    "shaper_id": shaper_id,
                    "field": field,
                    "origin_stage": stage,
                    "invalidated_stages": list(STAGES[rank:]),
                    "basis": "bucket state at the changed stage changes; chain filtering propagates to downstream stages",
                }
            )

    if not full_recompute:
        old_by_id, _ = _event_index_maps(old_bundle)
        for event_id in sorted(set(old_by_id) & set(new_by_id)):
            old_event = old_by_id[event_id]
            new_event = new_by_id[event_id]
            pid = new_event["pipeline_id"]
            if pid not in new_by_pipeline:
                continue
            local_events = new_by_pipeline[pid]
            local_index = next(index for index, item in enumerate(local_events) if item["event_id"] == event_id)
            changed_fields = [
                field
                for field in ("arrival_ms", "packet_length", "pipeline_id")
                if old_event.get(field) != new_event.get(field)
            ]
            if not changed_fields:
                continue
            if "pipeline_id" in changed_fields:
                full_recompute = True
                full_reason = "event pipeline assignment changed"
                for target_pid in restart:
                    for stage in STAGES:
                        restart[target_pid][stage] = 0
                break
            for field in changed_fields:
                changes.append(
                    {
                        "change_type": "event_stream",
                        "pipeline_id": pid,
                        "event_id": event_id,
                        "event_seq": new_event["event_seq"],
                        "arrival_ms": int(new_event["arrival_ms"]),
                        "field": field,
                        "old_value": old_event.get(field),
                        "new_value": new_event.get(field),
                    }
                )
            for stage in STAGES:
                current = restart[pid][stage]
                restart[pid][stage] = local_index if current is None else min(int(current), local_index)

            old_arrival = int(old_event["arrival_ms"])
            new_arrival = int(new_event["arrival_ms"])
            window_ms = int(new_bundle["config"]["window_ms"])
            slide_ms = int(new_bundle["config"]["slide_ms"])
            old_windows = set(overlapping_window_starts(old_arrival, window_ms, slide_ms))
            new_windows = set(overlapping_window_starts(new_arrival, window_ms, slide_ms))
            extra_windows[pid].update(old_windows | new_windows)
            steps.append(
                {
                    "step": "event_coordinate_changed",
                    "pipeline_id": pid,
                    "event_id": event_id,
                    "event_seq": new_event["event_seq"],
                    "local_index": local_index,
                    "fields": changed_fields,
                    "old_overlapping_windows": sorted(old_windows),
                    "new_overlapping_windows": sorted(new_windows),
                    "invalidated_event_suffix": [item["event_seq"] for item in local_events[local_index:]],
                    "basis": "changed elapsed time or demand changes this event and all later bucket states",
                }
            )

    affected_cells: list[dict[str, Any]] = []
    affected_event_seqs: set[int] = set()
    affected_windows: set[tuple[str, int]] = set()
    stage_windows: list[dict[str, Any]] = []

    for pid in sorted(new_pipelines):
        local_events = new_by_pipeline[pid]
        for stage in STAGES:
            cutoff = restart[pid][stage]
            if cutoff is None:
                continue
            cutoff = int(cutoff)
            window_starts = _window_chain_from(new_bundle, pid, local_events[cutoff])
            for window_start in window_starts:
                affected_windows.add((pid, window_start))
                stage_windows.append(
                    {"pipeline_id": pid, "stage": stage, "window_start_ms": window_start}
                )
            for local_index in range(cutoff, len(local_events)):
                event = local_events[local_index]
                window_start = primary_window_start(
                    int(event["arrival_ms"]), int(new_bundle["config"]["slide_ms"])
                )
                affected_cells.append(
                    {"event_seq": event["event_seq"], "pipeline_id": pid, "stage": stage}
                )
                affected_event_seqs.add(int(event["event_seq"]))
                affected_windows.add((pid, window_start))

        for window_start in extra_windows.get(pid, set()):
            affected_windows.add((pid, window_start))

    affected_events = sorted(
        (new_by_id[event_id] for event_id in new_by_id if int(new_by_id[event_id]["event_seq"]) in affected_event_seqs),
        key=lambda event: (event["pipeline_id"], int(event["arrival_ms"]), int(event["event_seq"])),
    )
    closure = {
        "schema": "qos-diag/invalidation@1",
        "old_spec_digest": old_bundle.get("spec_digest"),
        "new_spec_digest": new_bundle.get("spec_digest"),
        "full_recompute": full_recompute,
        "full_reason": full_reason,
        "changes": sorted(
            changes,
            key=lambda item: (
                str(item.get("pipeline_id") or ""),
                int(item.get("arrival_ms", -1)) if item.get("arrival_ms") is not None else -1,
                STAGE_RANK.get(str(item.get("stage")), -1),
                str(item.get("change_type")),
                str(item.get("field")),
            ),
        ),
        "restart_local_index": restart,
        "affected_pipelines": sorted({cell["pipeline_id"] for cell in affected_cells}),
        "affected_event_seqs": sorted(affected_event_seqs),
        "affected_events": [
            {
                "event_seq": event["event_seq"],
                "event_id": event["event_id"],
                "pipeline_id": event["pipeline_id"],
                "arrival_ms": event["arrival_ms"],
            }
            for event in affected_events
        ],
        "affected_windows": [
            {"pipeline_id": pid, "window_start_ms": start}
            for pid, start in sorted(affected_windows, key=lambda item: (item[0], item[1]))
        ],
        "affected_stage_windows": sorted(
            stage_windows,
            key=lambda item: (item["pipeline_id"], STAGE_RANK[item["stage"]], item["window_start_ms"]),
        ),
        "affected_cells": sorted(
            affected_cells,
            key=lambda cell: (int(cell["event_seq"]), STAGE_RANK[cell["stage"]]),
        ),
        "propagation_steps": steps,
    }
    return closure


def _state_from_record(record: dict[str, Any]) -> StageState:
    return StageState(
        balance_q=int(record["balance_after_decision_q"]),
        state_clock_ms=int(record["state_clock_ms"]),
    )


def recompute_incremental(
    old_bundle: dict[str, Any],
    new_bundle: dict[str, Any],
    old_results: list[dict[str, Any]],
    closure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if closure is None:
        closure = derive_invalidation(old_bundle, new_bundle)
    fp = FixedPoint(int(new_bundle["config"]["scale"]))
    new_shapers = shaper_index(new_bundle)
    results_by_seq = {int(result["event_seq"]): copy.deepcopy(result) for result in old_results}
    new_events_by_id = {event["event_id"]: event for event in new_bundle["events"]}
    old_events_by_id = {event["event_id"]: event for event in old_bundle["events"]}

    # Apply immutable event coordinates for events that survive.  Only affected
    # event top-level objects are rebuilt later; this keeps identity lookups new.
    for event_id, new_event in new_events_by_id.items():
        if event_id in old_events_by_id and new_event["event_seq"] in results_by_seq:
            result = results_by_seq[int(new_event["event_seq"])]
            result["arrival_ms"] = new_event["arrival_ms"]
            result["packet_length"] = new_event["packet_length"]

    restart = closure["restart_local_index"]
    new_by_pipeline: dict[str, list[dict[str, Any]]] = {}
    for pipeline in new_bundle["pipelines"]:
        pid = pipeline["pipeline_id"]
        new_by_pipeline[pid] = [event for event in new_bundle["events"] if event["pipeline_id"] == pid]

    for pid in sorted(new_by_pipeline):
        events = new_by_pipeline[pid]
        for stage_index, stage in enumerate(STAGES):
            cutoff = restart.get(pid, {}).get(stage)
            if cutoff is None:
                continue
            cutoff = int(cutoff)
            if cutoff == 0:
                shaper = new_shapers[(pid, stage)]
                cap_q = int(shaper["burst_capacity_tokens"]) * fp.q
                state = StageState(cap_q, 0)
            else:
                prior_event = events[cutoff - 1]
                prior_result = results_by_seq[int(prior_event["event_seq"])]
                prior_record = next(item for item in prior_result["stages"] if item["stage"] == stage)
                state = _state_from_record(prior_record)

            for local_index in range(cutoff, len(events)):
                event = events[local_index]
                seq = int(event["event_seq"])
                result = results_by_seq[seq]
                upstream_allows = True
                if stage_index > 0:
                    upstream_stage = STAGES[stage_index - 1]
                    upstream_record = next(item for item in result["stages"] if item["stage"] == upstream_stage)
                    upstream_allows = upstream_record["verdict"] == "allow"
                shaper = new_shapers[(pid, stage)]
                window_start = primary_window_start(int(event["arrival_ms"]), int(new_bundle["config"]["slide_ms"]))
                if upstream_allows:
                    record, state = evaluate_stage_transition(
                        shaper,
                        int(event["arrival_ms"]),
                        int(event["packet_length"]),
                        state,
                        fp,
                        window_start_ms=window_start,
                    )
                else:
                    record = skipped_stage_record(
                        shaper,
                        int(event["arrival_ms"]),
                        int(event["packet_length"]),
                        state,
                        fp,
                        "upstream_stage_did_not_allow_event",
                        window_start_ms=window_start,
                    )
                stage_position = STAGES.index(stage)
                result["stages"][stage_position] = record

        affected_seq_set = set(closure["affected_event_seqs"])
        for seq in affected_seq_set:
            result = results_by_seq[seq]
            event = new_bundle["events"][seq]
            result["event_id"] = event["event_id"]
            result["pipeline_id"] = event["pipeline_id"]
            result["arrival_ms"] = event["arrival_ms"]
            result["packet_length"] = event["packet_length"]
            result["window_start_ms"] = primary_window_start(
                int(event["arrival_ms"]), int(new_bundle["config"]["slide_ms"])
            )
            chain_allowed = all(record["verdict"] == "allow" for record in result["stages"])
            result["verdict"] = "allow" if chain_allowed else "shape"
            result["decision"] = "allowed" if chain_allowed else "shaped"

    ordered_results = [results_by_seq[seq] for seq in sorted(results_by_seq)]
    unchanged_check = verify_unaffected_bytes(old_results, ordered_results, closure)
    return {
        "results": ordered_results,
        "closure": closure,
        "unchanged_check": unchanged_check,
    }


def verify_unaffected_bytes(
    old_results: list[dict[str, Any]],
    new_results: list[dict[str, Any]],
    closure: dict[str, Any],
) -> dict[str, Any]:
    affected_cells = {
        (int(cell["event_seq"]), cell["pipeline_id"], cell["stage"])
        for cell in closure["affected_cells"]
    }
    affected_events = set(int(seq) for seq in closure["affected_event_seqs"])
    cell_checks: list[dict[str, Any]] = []
    unchanged_cells = 0
    unchanged_events = 0
    first_mismatch: dict[str, Any] | None = None
    old_by_seq = {int(item["event_seq"]): item for item in old_results}
    new_by_seq = {int(item["event_seq"]): item for item in new_results}
    for seq in sorted(old_by_seq):
        old_event = old_by_seq[seq]
        new_event = new_by_seq[seq]
        for stage_index, stage in enumerate(STAGES):
            key = (seq, old_event["pipeline_id"], stage)
            if key in affected_cells:
                continue
            old_cell = old_event["stages"][stage_index]
            new_cell = new_event["stages"][stage_index]
            same = canonical_bytes(old_cell) == canonical_bytes(new_cell)
            if same:
                unchanged_cells += 1
            elif first_mismatch is None:
                first_mismatch = {"event_seq": seq, "stage": stage, "field": "canonical_record"}
            cell_checks.append(
                {"event_seq": seq, "stage": stage, "byte_identical": same}
            )
        if seq not in affected_events:
            same_event = canonical_bytes(old_event) == canonical_bytes(new_event)
            if same_event:
                unchanged_events += 1
            elif first_mismatch is None:
                first_mismatch = {"event_seq": seq, "stage": None, "field": "canonical_event"}
    return {
        "checked_cells": len(cell_checks),
        "unchanged_cells": unchanged_cells,
        "unchanged_events": unchanged_events,
        "all_unaffected_byte_identical": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "old_sha256": sha256_hex(old_results),
        "new_sha256": sha256_hex(new_results),
    }
