"""Incremental/full fusion engine and report generation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    FusionConfig,
    config_echo_best_effort,
    load_raw_config,
    validate_config,
)
from .diagnostics import failure, sort_events, sort_failures
from .fusion import (
    assemble_fusion_record,
    build_contribution_index,
    build_observability,
    channel_contribution,
    expected_fingerprint,
    quantization_evidence,
    truncate_records,
)
from .io_utils import atomic_write_json
from .loader import (
    apply_config_injections,
    load_fragments,
    load_injections,
)
from .normalize import normalize_channel
from .reference import reference_normalized_channels
from .scheduler import allocate_budget, execute_channel_tasks
from .snapshot import SnapshotResult, SnapshotStore, SimulatedCrash


@dataclass
class RunOutcome:
    report: dict[str, Any]
    snapshot_manifest: dict[str, Any] | None = None


def _empty_invalidation() -> dict[str, Any]:
    return {
        "counts": {
            "evicted_fusions": 0,
            "invalidated_candidates": 0,
            "invalidated_fusions": 0,
            "reused_candidates": 0,
            "reused_fusions": 0,
        },
        "evicted_fusions": [],
        "invalidated_candidates": [],
        "invalidated_fusions": [],
        "reused_candidates": [],
        "reused_fusions": [],
    }


def _normalization_compatible(previous: SnapshotResult | None, config: FusionConfig) -> bool:
    if previous is None or not previous.valid:
        return False
    manifest = previous.manifest
    return (
        manifest.get("normalization") == config.normalization
        and manifest.get("quant_bits") == config.quant_bits
        and manifest.get("rounding") == config.rounding
    )


def _snapshot_report(
    config: FusionConfig | None,
    *,
    run_version: int | None,
    resumed: bool,
    stale: bool,
) -> dict[str, Any]:
    if config is None:
        return {"resumed": resumed, "stale": stale}
    return {
        "config_hash": config.config_hash(),
        "resumed": resumed,
        "run_version": run_version,
        "stale": stale,
        "struct_hash": config.structural_hash(),
    }


def run_workspace(
    workspace: Path | str,
    *,
    report_path: Path | str | None = None,
    injection_path: Path | str | None = None,
    mode: str = "auto",
    reset: bool = False,
    expected_version: int | None = None,
    simulate_crash_after: str | None = None,
) -> RunOutcome:
    """Run one load -> normalize -> fuse -> truncate transaction."""

    workspace = Path(workspace).resolve()
    config_path = workspace / "config.json"
    store = SnapshotStore(workspace)
    if reset and store.state_root.exists():
        shutil.rmtree(store.state_root)

    failures: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    injections, injection_failures = load_injections(Path(injection_path) if injection_path else None)
    failures.extend(injection_failures)
    raw_config, config_load_failures = load_raw_config(config_path)
    failures.extend(config_load_failures)
    raw_config, config_hits, config_injection_failures = apply_config_injections(raw_config, injections)
    hits.extend(config_hits)
    failures.extend(config_injection_failures)
    config_injection_ids: dict[tuple[str, ...], str] = {}
    for injection in injections:
        injection_id = str(injection.get("id"))
        if injection.get("type") == "trunc_limit_out_of_range":
            config_injection_ids[("config", "trunc_k")] = injection_id
        elif injection.get("type") == "zero_capacity" and isinstance(injection.get("channel"), str):
            config_injection_ids[("channel", injection["channel"], "capacity")] = injection_id
    config, config_failures = validate_config(raw_config, config_injection_ids)
    failures.extend(config_failures)
    config_echo = config_echo_best_effort(raw_config)

    snapshot = store.load(expected_version=expected_version)
    stale = snapshot.stale
    stale_reason = snapshot.reason
    if expected_version is not None and not snapshot.exists:
        stale = True
        stale_reason = f"expected run_version {expected_version}, but no committed snapshot exists"
    if stale:
        failures.append(
            failure(
                "snapshot",
                "E_SNAPSHOT_STALE",
                stale_reason or "committed snapshot failed integrity checks",
            )
        )

    previous: SnapshotResult | None = snapshot if snapshot.valid and not stale else None
    if mode == "full":
        previous = None
    resumed = previous is not None and mode != "full"
    version = max((snapshot.manifest_version or 0) + 1, store.next_version_hint())

    if config is None:
        report = {
            "budget": None,
            "config_echo": config_echo,
            "failures": sort_failures(failures),
            "fusion_summary": None,
            "injections_hit": sorted(hits, key=lambda item: str(item.get("id"))),
            "invalidation": None,
            "mode": "aborted",
            "normalization_events": [],
            "observability": None,
            "quantization_evidence": None,
            "run_version": None,
            "snapshot": _snapshot_report(None, run_version=None, resumed=False, stale=stale),
        }
        if report_path is not None:
            atomic_write_json(Path(report_path), report)
        return RunOutcome(report=report)

    load_state, load_failures, load_hits = load_fragments(workspace, config, injections)
    failures.extend(load_failures)
    hits.extend(load_hits)

    previous_load = previous.stages.get("load") if previous else None
    previous_normalize = previous.stages.get("normalize") if previous else None
    previous_fuse = previous.stages.get("fuse") if previous else None
    normalization_compatible = _normalization_compatible(previous, config)

    normalized_channels: dict[str, Any] = {}
    reused_candidates: list[str] = []
    invalidated_candidates: list[str] = []
    invalidated_candidate_ids: set[str] = set()
    normalization_events: list[dict[str, Any]] = []

    for channel in config.channel_names:
        current_load = load_state["channels"][channel]
        can_reuse = False
        if normalization_compatible and previous_load and previous_normalize:
            previous_channel_load = previous_load.get("channels", {}).get(channel)
            previous_channel_norm = previous_normalize.get("channels", {}).get(channel)
            can_reuse = bool(
                previous_channel_load
                and previous_channel_norm
                # Compare the complete validated load state, not only the raw
                # fragment hash. Load-stage injections can alter/remove a
                # candidate without changing the source file bytes.
                and previous_channel_load == current_load
            )
        if can_reuse:
            channel_normalized = previous_normalize["channels"][channel]
            normalized_channels[channel] = channel_normalized
            for candidate_id in sorted(current_load["candidates"]):
                if candidate_id in channel_normalized.get("candidates", {}):
                    reused_candidates.append(f"{channel}/{candidate_id}")
                else:
                    invalidated_candidates.append(f"{channel}/{candidate_id}")
                    invalidated_candidate_ids.add(candidate_id)
        else:
            channel_normalized = normalize_channel(channel, current_load, config)
            normalized_channels[channel] = channel_normalized
            for candidate_id in sorted(current_load["candidates"]):
                invalidated_candidates.append(f"{channel}/{candidate_id}")
                invalidated_candidate_ids.add(candidate_id)
        normalization_events.extend(channel_normalized.get("events", []))

    normalized_state = {
        "channels": normalized_channels,
        "parameters": config.normalization_parameters(),
    }
    contribution_index = build_contribution_index(normalized_state, config)

    previous_records: dict[str, Any] = (
        previous_fuse.get("records", {}) if previous_fuse and isinstance(previous_fuse.get("records"), dict) else {}
    )
    reused_records: list[dict[str, Any]] = []
    reused_fusions: list[str] = []
    invalidated_fusions: list[str] = []
    for candidate_id in sorted(contribution_index):
        contributions = contribution_index[candidate_id]
        fingerprint = expected_fingerprint(candidate_id, contributions, config)
        previous_record = previous_records.get(candidate_id)
        can_reuse = bool(
            previous_record
            and previous_record.get("fusion_fingerprint") == fingerprint
            and candidate_id not in invalidated_candidate_ids
        )
        if can_reuse:
            reused_records.append(previous_record)
            reused_fusions.append(candidate_id)
        else:
            invalidated_fusions.append(candidate_id)

    evicted_fusions = sorted(set(previous_records) - set(contribution_index))

    requests: dict[str, list[str]] = {}
    for candidate_id in invalidated_fusions:
        for contribution in contribution_index[candidate_id]:
            requests.setdefault(contribution["channel"], []).append(candidate_id)
    allocation = allocate_budget(requests, config)

    def channel_worker(channel: str, candidate_ids: list[str]) -> dict[str, Any]:
        return {
            candidate_id: channel_contribution(
                channel,
                candidate_id,
                normalized_channels[channel],
                config,
            )
            for candidate_id in candidate_ids
        }

    task_results, max_observed = execute_channel_tasks(
        allocation.granted,
        config.max_workers,
        channel_worker,
    )
    failures.extend(allocation.failures)

    computed_records: list[dict[str, Any]] = []
    deferred_candidates: list[str] = []
    for candidate_id in invalidated_fusions:
        ordered_contributions: list[dict[str, Any]] = []
        complete = True
        for contribution in contribution_index[candidate_id]:
            channel = contribution["channel"]
            channel_result = task_results.get(channel, {})
            if candidate_id not in channel_result:
                complete = False
                break
            ordered_contributions.append(channel_result[candidate_id])
        if complete:
            computed_records.append(assemble_fusion_record(candidate_id, ordered_contributions, config))
        else:
            deferred_candidates.append(candidate_id)

    records = sorted(reused_records + computed_records, key=lambda item: item["candidate_id"])
    kept, truncation_failures = truncate_records(records, config)
    failures.extend(truncation_failures)

    reference_channels = reference_normalized_channels(load_state, config)
    observability = build_observability(records, kept, reference_channels, config)
    evidence = quantization_evidence(normalized_state, records, kept, config)
    pending_channels = {entry.split("/", 1)[0] for entry in allocation.pending_queue}
    completed_channels = [channel for channel in config.channel_names if channel not in pending_channels]
    budget_report = allocation.report(config, max_observed=max_observed)

    invalidation = {
        "counts": {
            "evicted_fusions": len(evicted_fusions),
            "invalidated_candidates": len(invalidated_candidates),
            "invalidated_fusions": len(invalidated_fusions),
            "reused_candidates": len(reused_candidates),
            "reused_fusions": len(reused_fusions),
        },
        "evicted_fusions": evicted_fusions,
        "invalidated_candidates": sorted(invalidated_candidates),
        "invalidated_fusions": sorted(invalidated_fusions),
        "reused_candidates": sorted(reused_candidates),
        "reused_fusions": sorted(reused_fusions),
    }
    fusion_summary = {
        "deferred_candidates": sorted(deferred_candidates),
        "kept": kept,
        "kept_count": len(kept),
        "total_candidates": len(contribution_index),
        "trunc_k": config.trunc_k,
    }
    effective_mode = "incremental" if resumed else "full"
    report = {
        "budget": budget_report,
        "config_echo": config.echo(),
        "failures": sort_failures(failures),
        "fusion_summary": fusion_summary,
        "injections_hit": sorted(hits, key=lambda item: str(item.get("id"))),
        "invalidation": invalidation,
        "mode": effective_mode,
        "normalization_events": sort_events(normalization_events),
        "observability": observability,
        "quantization_evidence": evidence,
        "run_version": version,
        "snapshot": _snapshot_report(config, run_version=version, resumed=resumed, stale=stale),
    }

    fuse_stage = {
        "budget": budget_report,
        "deferred_candidates": sorted(deferred_candidates),
        "records": {record["candidate_id"]: record for record in records},
        "reused_fusions": sorted(reused_fusions),
        "invalidated_fusions": sorted(invalidated_fusions),
    }
    truncate_stage = {
        "kept": kept,
        "observability": observability,
        "quantization_evidence": evidence,
        "summary": fusion_summary,
    }
    manifest = store.commit(
        run_version=version,
        config=config,
        stage_payloads={
            "load": load_state,
            "normalize": normalized_state,
            "fuse": fuse_stage,
            "truncate": truncate_stage,
        },
        completed_channels=completed_channels,
        simulate_crash_after=simulate_crash_after,
    )
    if report_path is not None:
        atomic_write_json(Path(report_path), report)
    return RunOutcome(report=report, snapshot_manifest=manifest)
