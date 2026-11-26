from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .adjudicate import (
    adjudicate_frames,
    aggregate_interval_evidence,
    build_window_attribution,
    frame_digest,
    merge_command_conclusions,
)
from .audio import fragment_window, mix_timeline, mix_window
from .cache import (
    ReuseLedger,
    attribution_cache_key,
    conclusion_cache_key,
    feature_cache_key,
    interval_cache_key,
    posterior_cache_key,
)
from .config import DiagnosticConfig, load_config
from .endpoint import attach_windows, build_windows, detect_speech
from .errors import E_AUDIO_FORMAT, E_FIELD_TRUNCATED, E_POINT_BUDGET_PENDING, E_QUOTA_EXCEEDED, E_SCHEMA, QuotaExceededError
from .features import CommandProjectionModel, FixedWindowFeature, extract_fixed_feature
from .receipts import (
    ReceiptDocument,
    ReceiptRecord,
    parse_a1,
    to_a1,
    to_a2,
)
from .scheduler import ScheduledJob, run_scheduled_batch, stable_task_key
from .schema import Interval, LoadedSession, load_session
from .snapshot import LoadedSnapshot, SnapshotStore
from .util import (
    atomic_write_json,
    atomic_write_text,
    clamp_int,
    json_hash as util_json_hash,
    round_half_away_from_zero,
)


@dataclass(frozen=True)
class PipelineOptions:
    max_workers: int | None = None
    max_batches: int = 128
    crash_at: str | None = None
    arrival_limit: int | None = None


def _candidate_mask(endpoint_payload: Mapping[str, Any], frame_count: int) -> list[bool]:
    mask = [False] * frame_count
    for candidate in endpoint_payload.get("candidates", []):
        start = int(candidate["start_frame"])
        end = int(candidate["end_frame"])
        for index in range(max(0, start), min(frame_count, end)):
            mask[index] = True
    return mask


def _interval_char_coverage(
    interval: Interval,
    *,
    overlap_start: int,
    overlap_end: int,
) -> int:
    covered = 0
    for char in interval.chars:
        char_start = interval.start_sample + char.start_sample
        char_end = interval.start_sample + char.end_sample
        covered += max(0, min(overlap_end, char_end) - max(overlap_start, char_start))
    return covered


def _window_frame_indices(window: Mapping[str, Any], frame_count: int) -> list[int]:
    return [int(index) for index in window.get("frame_indices", []) if 0 <= int(index) < frame_count]


def _source_trust_ppm(
    *,
    speaker_trust_ppm: int,
    char_coverage_ratio: float,
    window_overlap_ratio: float,
) -> int:
    value = round(
        speaker_trust_ppm * 0.55
        + max(0.0, min(1.0, char_coverage_ratio)) * 300_000.0
        + max(0.0, min(1.0, window_overlap_ratio)) * 150_000.0
    )
    return clamp_int(value, 0, 1_000_000)


class DiagnosticPipeline:
    def __init__(
        self,
        config_path: str | Path,
        manifest_path: str | Path,
        output_dir: str | Path,
        *,
        options: PipelineOptions | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.options = options or PipelineOptions()
        self._crash_at = self.options.crash_at or os.environ.get("VOICE_DIAG_CRASH_AT")
        self._last_session: LoadedSession | None = None
        self._last_config: DiagnosticConfig | None = None

    def _maybe_crash(self, phase: str) -> None:
        if self._crash_at in (phase, "all"):
            os._exit(137)

    def run(self) -> dict[str, Any]:
        config = load_config(self.config_path)
        # Full schema and receipt validation completes before the snapshot store
        # is opened for writing.  A schema failure therefore cannot leave a
        # partial parse product behind.
        session = load_session(
            self.config_path,
            self.manifest_path,
            config,
            arrival_limit=self.options.arrival_limit,
        )
        self._last_session = session
        self._last_config = config
        max_workers = config.scheduler.max_workers
        if self.options.max_workers is not None:
            if self.options.max_workers < 1:
                raise ValueError("max_workers override must be positive")
            max_workers = min(max_workers, self.options.max_workers)
        store = SnapshotStore(self.output_dir)
        previous = store.load()
        previous_state = previous.state if previous is not None else {}
        same_family = bool(
            previous is not None
            and previous_state.get("global_version") == config.global_version
            and previous_state.get("config_hash") == config.config_hash
        )
        ledger = ReuseLedger.from_dict(previous_state.get("ledger") if previous_state else None)
        pending_ids: set[str] = set()
        if same_family:
            for item in previous_state.get("pending", []):
                if isinstance(item, dict) and isinstance(item.get("cache_id"), str):
                    pending_ids.add(item["cache_id"])
        else:
            # Keys include version/config. Retain old entries only so the audit
            # can name the actual invalidation cause when each stage is probed.
            pending_ids = set()
        total_budget_consumed = int(previous_state.get("budget_consumed_total", 0)) if same_family else 0
        state = self._base_state(config, session, previous_state if same_family else {})
        state["ledger"] = ledger.as_dict()
        state["pending"] = [item for item in previous_state.get("pending", [])] if same_family else []
        state["budget_consumed_total"] = total_budget_consumed

        # Stage 1: parse/validation (already completed in memory).
        parse_key, parse_deps = self._parse_cache_key(config, session)
        parse_probe = ledger.probe(
            cache_id="parse:global",
            stage="parse",
            key=parse_key,
            dependencies=parse_deps,
            refs={"conversation_id": session.manifest.conversation_id},
            stable_key=("stage", "parse"),
        )
        if not parse_probe.reused:
            ledger.put(
                cache_id="parse:global",
                stage="parse",
                key=parse_key,
                dependencies=parse_deps,
                refs={"conversation_id": session.manifest.conversation_id},
                stable_key=("stage", "parse"),
                payload=session.parse_summary,
            )
        state["parse_summary"] = session.parse_summary
        state["fragment_cursor"] = session.parse_summary["fragment_cursor"]
        state["arrival_cursor"] = session.parse_summary.get("arrival_cursor", state["fragment_cursor"])
        state["all_fragments_arrived"] = bool(session.parse_summary.get("all_fragments_arrived", True))
        state["stages"]["parse"] = "complete"
        state["ledger"] = ledger.as_dict()
        state = self._commit_state(store, state, session, config, final=None, status="running")
        self._maybe_crash("after_parse")

        # Stage 2: endpoint detection and window geometry.
        mixed, mixed_hash = mix_timeline(session.fragments, timeline_end=session.manifest.timeline_end)
        endpoint_key, endpoint_deps = self._endpoint_cache_key(config, session, mixed_hash)
        endpoint_probe = ledger.probe(
            cache_id="endpoint:global",
            stage="endpoint",
            key=endpoint_key,
            dependencies=endpoint_deps,
            refs={"valid_fragments": [fragment.fragment_id for fragment in session.fragments]},
            stable_key=("stage", "endpoint"),
        )
        if endpoint_probe.reused and isinstance(endpoint_probe.payload, dict):
            endpoint_payload = endpoint_probe.payload
        else:
            endpoint_result = detect_speech(mixed, endpoint=config.endpoint, content_hash=mixed_hash)
            windows = build_windows(timeline_end=session.manifest.timeline_end, analysis=config.analysis)
            endpoint_result = attach_windows(endpoint_result, windows)
            endpoint_payload = endpoint_result.as_cache_payload()
            ledger.put(
                cache_id="endpoint:global",
                stage="endpoint",
                key=endpoint_key,
                dependencies=endpoint_deps,
                refs={"valid_fragments": [fragment.fragment_id for fragment in session.fragments]},
                stable_key=("stage", "endpoint"),
                payload=endpoint_payload,
            )
        state["endpoint"] = endpoint_payload
        state["windows"] = endpoint_payload.get("windows", [])
        state["stages"]["endpoint"] = "complete"
        state["ledger"] = ledger.as_dict()
        state = self._commit_state(store, state, session, config, final=None, status="running")
        self._maybe_crash("after_endpoint")

        model = CommandProjectionModel(config, session.manifest.commands)
        contexts = self._build_contexts(session, config, endpoint_payload, model)
        state["window_references"] = [
            {
                "window": context["window"],
                "segments": [
                    {
                        "fragment_id": segment["fragment_id"],
                        "interval_id": segment["interval_id"],
                        "content_hash": segment["content_hash"],
                        "local_start": segment["local_start"],
                        "local_end": segment["local_end"],
                    }
                    for segment in context["segment_refs"]
                ],
                "intervals": [
                    {
                        "interval_id": interval["interval_id"],
                        "speaker_id": interval["speaker_id"],
                        "interval_start_sample": interval["interval_start_sample"],
                        "interval_end_sample": interval["interval_end_sample"],
                        "overlap_start_sample": interval["overlap_start_sample"],
                        "overlap_end_sample": interval["overlap_end_sample"],
                        "overlap_frames": interval["overlap_frames"],
                    }
                    for interval in context["interval_refs"]
                ],
            }
            for context in contexts
        ]
        frame_metrics = endpoint_payload.get("frame_metrics", [])
        candidate_mask = _candidate_mask(endpoint_payload, len(frame_metrics))
        state["stage_status"]["feature_total"] = len(contexts)

        batch_summaries: list[dict[str, Any]] = []
        quota_error: dict[str, Any] | None = None
        current_pending = state.get("pending", [])
        last_status = "running"
        for batch_index in range(max(1, self.options.max_batches)):
            ledger_before = copy.deepcopy(ledger)
            batch_pending_ids = {
                str(item.get("cache_id")) for item in current_pending if isinstance(item, dict)
            }
            batch_used = 0
            batch_usage: dict[str, int] = {}
            pending_descriptors: dict[tuple[str, str], dict[str, Any]] = {}
            completion_order: list[str] = []

            def run_phase(jobs: list[ScheduledJob]) -> tuple[list[ScheduledJob], dict[str, Any]]:
                nonlocal batch_used, batch_usage
                if not jobs:
                    return [], {}
                execution = run_scheduled_batch(
                    jobs,
                    pending_ids=batch_pending_ids,
                    point_budget=config.scheduler.point_budget,
                    per_interval_quota=config.scheduler.per_interval_quota,
                    max_workers=max_workers,
                    already_consumed_points=batch_used,
                    already_interval_usage=batch_usage,
                )
                batch_used = execution.consumed_points
                batch_usage = dict(execution.interval_usage)
                completion_order.extend(execution.completion_order)
                for pending_job in execution.pending:
                    descriptor = pending_job.pending_descriptor()
                    pending_descriptors[(pending_job.stage, pending_job.cache_id)] = descriptor
                return list(execution.pending), dict(execution.results)

            try:
                # Feature tasks.
                feature_jobs: list[ScheduledJob] = []
                for context in contexts:
                    if context.get("feature") is not None:
                        continue
                    if not context.get("feature_probed", False):
                        probe = ledger.probe(
                            cache_id=context["feature_cache_id"],
                            stage="features",
                            key=context["feature_key"],
                            dependencies=context["feature_dependencies"],
                            refs=context["refs"],
                            stable_key=context["stable_keys"]["features"],
                        )
                        context["feature_probed"] = True
                        if probe.reused and isinstance(probe.payload, dict):
                            context["feature"] = probe.payload
                            continue
                    feature_jobs.append(
                        ScheduledJob(
                            cache_id=context["feature_cache_id"],
                            stage="features",
                            stable_key=context["stable_keys"]["features"],
                            interval_ids=tuple(context["interval_ids"]),
                            points=config.scheduler.point_cost,
                            execute=(lambda context=context: self._compute_feature_payload(context, session, config)),
                            was_pending=context["feature_cache_id"] in batch_pending_ids,
                        )
                    )
                _, feature_results = run_phase(feature_jobs)
                for context in contexts:
                    cache_id = context["feature_cache_id"]
                    if cache_id in feature_results:
                        payload = feature_results[cache_id]
                        context["feature"] = payload
                        ledger.put(
                            cache_id=cache_id,
                            stage="features",
                            key=context["feature_key"],
                            dependencies=context["feature_dependencies"],
                            refs=context["refs"],
                            stable_key=context["stable_keys"]["features"],
                            payload=payload,
                        )

                # Posterior tasks.
                posterior_jobs: list[ScheduledJob] = []
                for context in contexts:
                    if context.get("feature") is None or context.get("posterior") is not None:
                        continue
                    if not context.get("posterior_probed", False):
                        probe = ledger.probe(
                            cache_id=context["posterior_cache_id"],
                            stage="posterior",
                            key=context["posterior_key"],
                            dependencies=context["posterior_dependencies"],
                            refs=context["refs"],
                            stable_key=context["stable_keys"]["posterior"],
                        )
                        context["posterior_probed"] = True
                        if probe.reused and isinstance(probe.payload, dict):
                            context["posterior"] = probe.payload
                            continue
                    posterior_jobs.append(
                        ScheduledJob(
                            cache_id=context["posterior_cache_id"],
                            stage="posterior",
                            stable_key=context["stable_keys"]["posterior"],
                            interval_ids=tuple(context["interval_ids"]),
                            points=config.scheduler.point_cost,
                            execute=(lambda context=context: self._compute_posterior_payload(context, model)),
                            was_pending=context["posterior_cache_id"] in batch_pending_ids,
                        )
                    )
                _, posterior_results = run_phase(posterior_jobs)
                for context in contexts:
                    cache_id = context["posterior_cache_id"]
                    if cache_id in posterior_results:
                        payload = posterior_results[cache_id]
                        context["posterior"] = payload
                        ledger.put(
                            cache_id=cache_id,
                            stage="posterior",
                            key=context["posterior_key"],
                            dependencies=context["posterior_dependencies"],
                            refs=context["refs"],
                            stable_key=context["stable_keys"]["posterior"],
                            payload=payload,
                        )

                # Attribution tasks.
                attribution_jobs: list[ScheduledJob] = []
                for context in contexts:
                    if context.get("posterior") is None or context.get("attribution") is not None:
                        continue
                    if not context.get("attribution_probed", False):
                        probe = ledger.probe(
                            cache_id=context["attribution_cache_id"],
                            stage="attribution",
                            key=context["attribution_key"],
                            dependencies=context["attribution_dependencies"],
                            refs=context["refs"],
                            stable_key=context["stable_keys"]["attribution"],
                        )
                        context["attribution_probed"] = True
                        if probe.reused and isinstance(probe.payload, dict):
                            context["attribution"] = probe.payload
                            continue
                    attribution_jobs.append(
                        ScheduledJob(
                            cache_id=context["attribution_cache_id"],
                            stage="attribution",
                            stable_key=context["stable_keys"]["attribution"],
                            interval_ids=tuple(context["interval_ids"]),
                            points=config.scheduler.point_cost,
                            execute=(lambda context=context: self._compute_attribution_payload(context, config)),
                            was_pending=context["attribution_cache_id"] in batch_pending_ids,
                        )
                    )
                _, attribution_results = run_phase(attribution_jobs)
                for context in contexts:
                    cache_id = context["attribution_cache_id"]
                    if cache_id in attribution_results:
                        payload = attribution_results[cache_id]
                        context["attribution"] = payload
                        ledger.put(
                            cache_id=cache_id,
                            stage="attribution",
                            key=context["attribution_key"],
                            dependencies=context["attribution_dependencies"],
                            refs=context["refs"],
                            stable_key=context["stable_keys"]["attribution"],
                            payload=payload,
                        )
            except QuotaExceededError as exc:
                ledger = ledger_before
                quota_error = {
                    "code": E_QUOTA_EXCEEDED,
                    "message": exc.message,
                    "details": exc.details,
                }
                state["business_code_counts"][E_QUOTA_EXCEEDED] = (
                    state["business_code_counts"].get(E_QUOTA_EXCEEDED, 0) + 1
                )
                state["last_error"] = quota_error
                state["ledger"] = ledger.as_dict()
                state["pending"] = list(current_pending)
                state = self._commit_state(
                    store,
                    state,
                    session,
                    config,
                    final=None,
                    status="quota_exceeded",
                    batch_summary={
                        "batch_index": batch_index,
                        "status": E_QUOTA_EXCEEDED,
                        "error": quota_error,
                    },
                )
                last_status = "quota_exceeded"
                break

            all_features = all(context.get("feature") is not None for context in contexts)
            all_posteriors = all(context.get("posterior") is not None for context in contexts)
            all_attributions = all(context.get("attribution") is not None for context in contexts)
            state["stages"]["features"] = "complete" if all_features else "partial"
            state["stages"]["posterior"] = "complete" if all_posteriors else "partial"
            state["stages"]["attribution"] = "complete" if all_attributions else "partial"
            state["budget_consumed_total"] += max(0, batch_used)
            pending_from_phases = list(pending_descriptors.values())
            state["pending"] = sorted(
                pending_from_phases,
                key=lambda item: (item.get("stage", ""), list(item.get("stable_key", [])), item.get("cache_id", "")),
            )

            final = None
            if all_attributions and not pending_from_phases:
                final = self._finalize(
                    ledger=ledger,
                    store=store,
                    contexts=contexts,
                    endpoint_payload=endpoint_payload,
                    session=session,
                    config=config,
                    candidate_mask=candidate_mask,
                )
                state["stages"]["adjudication"] = "complete"
                state["final"] = final
                last_status = "complete" if state.get("all_fragments_arrived", True) else "waiting_arrival"
            else:
                state["stages"]["adjudication"] = "partial"
                state["final"] = None
                last_status = "pending"
                state["business_code_counts"][E_POINT_BUDGET_PENDING] = (
                    state["business_code_counts"].get(E_POINT_BUDGET_PENDING, 0) + 1
                )
            summary = {
                "batch_index": batch_index,
                "status": last_status,
                "consumed_points": batch_used,
                "interval_usage": batch_usage,
                "completion_order": completion_order,
                "pending_count": len(state["pending"]),
            }
            batch_summaries.append(summary)
            state["last_batch"] = summary
            state["batch_summaries"] = batch_summaries[-20:]
            state["ledger"] = ledger.as_dict()
            state = self._commit_state(store, state, session, config, final=final, status=last_status)
            self._maybe_crash("after_batch")
            current_pending = state["pending"]
            if final is not None or quota_error is not None:
                break
            if batch_used == 0 and current_pending:
                # No progress is possible under the current budget.  Preserve
                # the queue instead of spinning through synthetic snapshots.
                last_status = "pending"
                break
        result = self._result_payload(state, store, session, config, status=last_status)
        self._mirror_outputs(store, state, result, session, config)
        return result

    def _base_state(
        self,
        config: DiagnosticConfig,
        session: LoadedSession,
        previous: Mapping[str, Any],
    ) -> dict[str, Any]:
        business_codes = {
            code: 0
            for code in (
                E_AUDIO_FORMAT,
                E_SCHEMA,
                E_FIELD_TRUNCATED,
                E_POINT_BUDGET_PENDING,
                E_QUOTA_EXCEEDED,
            )
        }
        business_codes[E_AUDIO_FORMAT] = len(session.issues)
        return {
            "schema": 1,
            "conversation_id": session.manifest.conversation_id,
            "global_version": config.global_version,
            "config_hash": config.config_hash,
            "manifest_hash": session.manifest.structure_hash,
            "config_path": str(self.config_path),
            "manifest_path": str(self.manifest_path),
            "config": config.as_dict(),
            "parse_summary": session.parse_summary,
            "fragment_cursor": session.parse_summary.get("fragment_cursor", 0),
            "arrival_cursor": session.parse_summary.get("arrival_cursor", 0),
            "all_fragments_arrived": bool(session.parse_summary.get("all_fragments_arrived", True)),
            "stages": {
                "parse": "pending",
                "endpoint": "pending",
                "features": "pending",
                "posterior": "pending",
                "attribution": "pending",
                "adjudication": "pending",
            },
            "stage_status": {},
            "ledger": {},
            "pending": [],
            "budget_consumed_total": int(previous.get("budget_consumed_total", 0)),
            "business_code_counts": business_codes,
            "endpoint": None,
            "windows": [],
            "final": None,
            "last_batch": None,
            "batch_summaries": [],
            "last_error": None,
        }

    def _parse_cache_key(
        self,
        config: DiagnosticConfig,
        session: LoadedSession,
    ) -> tuple[str, dict[str, Any]]:
        dependencies = {
            "global_version": config.global_version,
            "config_hash": config.config_hash,
            "manifest_hash": session.manifest.structure_hash,
            "audio": {
                fragment.fragment_id: fragment.wav.content_hash
                for fragment in sorted(session.fragments, key=lambda item: item.fragment_id)
            },
            "skipped": sorted(session.skipped_fragment_ids),
            "receipts": {
                "a1": [record.canonical_fields() for record in session.receipts_a1.records]
                if session.receipts_a1 is not None
                else None,
                "a2": [record.canonical_fields() for record in session.receipts_a2.records]
                if session.receipts_a2 is not None
                else None,
            },
        }
        return util_json_hash({"kind": "parse", **dependencies}), dependencies

    def _endpoint_cache_key(
        self,
        config: DiagnosticConfig,
        session: LoadedSession,
        mixed_hash: str,
    ) -> tuple[str, dict[str, Any]]:
        dependencies = {
            "global_version": config.global_version,
            "config_hash": config.config_hash,
            "endpoint": config.endpoint.__dict__,
            "window_length": config.analysis.window_length,
            "window_shift": config.analysis.window_shift,
            "timeline_end": session.manifest.timeline_end,
            "mixed_content_hash": mixed_hash,
        }
        return util_json_hash({"kind": "endpoint", **dependencies}), dependencies

    def _build_contexts(
        self,
        session: LoadedSession,
        config: DiagnosticConfig,
        endpoint_payload: Mapping[str, Any],
        model: CommandProjectionModel,
    ) -> list[dict[str, Any]]:
        manifest = session.manifest
        fragment_by_interval = session.fragment_by_interval
        speaker_by_id = manifest.speaker_by_id
        frame_count = len(endpoint_payload.get("frame_metrics", []))
        endpoint_candidate_mask = _candidate_mask(endpoint_payload, frame_count)
        parameters = {
            "analysis": config.analysis.as_parameters(),
            "endpoint_low_energy": config.endpoint.low_energy,
            "endpoint_high_energy": config.endpoint.high_energy,
        }
        contexts: list[dict[str, Any]] = []
        for window in endpoint_payload.get("windows", []):
            frame_indices = _window_frame_indices(window, frame_count)
            interval_refs: list[dict[str, Any]] = []
            segment_refs: list[dict[str, Any]] = []
            interval_ref_by_id: dict[str, dict[str, Any]] = {}
            for interval in sorted(manifest.intervals, key=lambda item: item.interval_id):
                fragment = fragment_by_interval.get(interval.interval_id)
                if fragment is None:
                    continue
                overlap_start = max(int(window["sample_start"]), interval.start_sample)
                overlap_end = min(int(window["sample_end"]), interval.end_sample)
                if overlap_start >= overlap_end:
                    continue
                overlap_samples = overlap_end - overlap_start
                local_start = overlap_start - interval.start_sample
                local_end = local_start + overlap_samples
                speaker = speaker_by_id[interval.speaker_id]
                char_covered = _interval_char_coverage(
                    interval,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                )
                overlap_frames = sum(
                    1
                    for index in frame_indices
                    if interval.start_sample
                    <= index * config.analysis.frame_shift + config.analysis.frame_length // 2
                    < interval.end_sample
                )
                trust_ppm = _source_trust_ppm(
                    speaker_trust_ppm=speaker.trust_ppm,
                    char_coverage_ratio=char_covered / max(1, overlap_samples),
                    window_overlap_ratio=overlap_samples
                    / max(1, int(window["sample_end"]) - int(window["sample_start"])),
                )
                interval_ref = {
                    **interval.as_dict(),
                    "interval_start_sample": interval.start_sample,
                    "interval_end_sample": interval.end_sample,
                    "overlap_start_sample": overlap_start,
                    "overlap_end_sample": overlap_end,
                    "overlap_samples": overlap_samples,
                    "overlap_frames": overlap_frames,
                    "trust_ppm": trust_ppm,
                    "speaker_trust_ppm": speaker.trust_ppm,
                    "char_coverage_ppm": round(char_covered * 1_000_000 / max(1, overlap_samples)),
                }
                interval_refs.append(interval_ref)
                interval_ref_by_id[interval.interval_id] = interval_ref
                segment_refs.append(
                    {
                        "fragment_id": fragment.fragment_id,
                        "interval_id": interval.interval_id,
                        "speaker_id": interval.speaker_id,
                        "content_hash": fragment.wav.content_hash,
                        "global_start": fragment.global_start,
                        "global_end": fragment.global_end,
                        "overlap_global_start": overlap_start,
                        "overlap_global_end": overlap_end,
                        "local_start": local_start,
                        "local_end": local_end,
                        "arrival_index": fragment.arrival_index,
                    }
                )
            window_public = {
                "window_id": str(window["window_id"]),
                "window_index": int(window["window_index"]),
                "sample_start": int(window["sample_start"]),
                "sample_end": int(window["sample_end"]),
                "frame_indices": list(frame_indices),
            }
            interval_ids = sorted(ref["interval_id"] for ref in interval_refs)
            fragment_indices = sorted(ref["arrival_index"] for ref in segment_refs)
            refs = {
                "window_id": window_public["window_id"],
                "fragment_ids": sorted(ref["fragment_id"] for ref in segment_refs),
                "interval_ids": interval_ids,
            }
            feature_key, feature_deps = feature_cache_key(
                window=window_public,
                segments=segment_refs,
                intervals=[interval_ref_by_id[interval_id] for interval_id in interval_ids],
                parameters=parameters,
                global_version=config.global_version,
                config_hash=config.config_hash,
            )
            posterior_key, posterior_deps = posterior_cache_key(
                feature_key=feature_key,
                projection_fingerprint=model.projection_fingerprint,
                command_codes=manifest.command_vocabulary,
                quant_profile=config.analysis.quant_profile,
                global_version=config.global_version,
            )
            attribution_key, attribution_deps = attribution_cache_key(
                posterior_key=posterior_key,
                window=window_public,
                interval_overlap=[
                    {
                        "interval_id": ref["interval_id"],
                        "start_sample": ref["interval_start_sample"],
                        "end_sample": ref["interval_end_sample"],
                        "overlap_frames": ref["overlap_frames"],
                        "trust_ppm": ref["trust_ppm"],
                    }
                    for ref in interval_refs
                ],
                candidate_mask_values=[endpoint_candidate_mask[index] for index in frame_indices],
                adjudication_parameters=config.adjudication.as_parameters(),
                quant_profile=config.analysis.quant_profile,
                global_version=config.global_version,
            )
            contexts.append(
                {
                    "window": window_public,
                    "frame_indices": frame_indices,
                    "frame_count": frame_count,
                    "candidate_mask": endpoint_candidate_mask,
                    "interval_refs": interval_refs,
                    "segment_refs": segment_refs,
                    "interval_ids": interval_ids,
                    "fragment_indices": fragment_indices,
                    "refs": refs,
                    "feature_cache_id": f"feature:{window_public['window_id']}",
                    "feature_key": feature_key,
                    "feature_dependencies": feature_deps,
                    "posterior_cache_id": f"posterior:{window_public['window_id']}",
                    "posterior_key": posterior_key,
                    "posterior_dependencies": posterior_deps,
                    "attribution_cache_id": f"attribution:{window_public['window_id']}",
                    "attribution_key": attribution_key,
                    "attribution_dependencies": attribution_deps,
                    "stable_keys": {
                        "features": stable_task_key(
                            fragment_indices=fragment_indices,
                            window_index=window_public["window_index"],
                            stage="features",
                        ),
                        "posterior": stable_task_key(
                            fragment_indices=fragment_indices,
                            window_index=window_public["window_index"],
                            stage="posterior",
                        ),
                        "attribution": stable_task_key(
                            fragment_indices=fragment_indices,
                            window_index=window_public["window_index"],
                            stage="attribution",
                        ),
                    },
                    "feature": None,
                    "posterior": None,
                    "attribution": None,
                }
            )
        return contexts

    def _compute_feature_payload(
        self,
        context: Mapping[str, Any],
        session: LoadedSession,
        config: DiagnosticConfig,
    ) -> dict[str, Any]:
        window = context["window"]
        start = int(window["sample_start"])
        end = int(window["sample_end"])
        frame_indices = context["frame_indices"]
        global_pcm = mix_window(session.fragments, start=start, end=end)
        global_feature = extract_fixed_feature(
            global_pcm,
            window_start=start,
            frame_indices=frame_indices,
            analysis=config.analysis,
            endpoint_low_energy=config.endpoint.low_energy,
            endpoint_high_energy=config.endpoint.high_energy,
        )
        source_features: dict[str, Any] = {}
        for interval_ref in sorted(context["interval_refs"], key=lambda item: item["interval_id"]):
            fragment = session.fragment_by_interval[interval_ref["interval_id"]]
            slice_result = fragment_window(fragment, start=start, end=end)
            if slice_result is None:
                continue
            pcm, overlap_start, overlap_end = slice_result
            feature = extract_fixed_feature(
                pcm,
                window_start=start,
                frame_indices=frame_indices,
                analysis=config.analysis,
                endpoint_low_energy=config.endpoint.low_energy,
                endpoint_high_energy=config.endpoint.high_energy,
                valid_range=(overlap_start, overlap_end),
            )
            source_features[interval_ref["interval_id"]] = feature.as_dict()
        return {
            "window_id": window["window_id"],
            "window": dict(window),
            "segment_refs": [dict(segment) for segment in context["segment_refs"]],
            "interval_refs": [
                {
                    key: interval[key]
                    for key in (
                        "interval_id",
                        "speaker_id",
                        "interval_start_sample",
                        "interval_end_sample",
                        "overlap_start_sample",
                        "overlap_end_sample",
                        "overlap_samples",
                        "overlap_frames",
                        "trust_ppm",
                    )
                }
                for interval in context["interval_refs"]
            ],
            "global": global_feature.as_dict(),
            "sources": {key: source_features[key] for key in sorted(source_features)},
            "parameters_fingerprint": util_json_hash(
                {
                    "analysis": config.analysis.as_parameters(),
                    "endpoint_low": config.endpoint.low_energy,
                    "endpoint_high": config.endpoint.high_energy,
                }
            ),
        }

    def _compute_posterior_payload(
        self,
        context: Mapping[str, Any],
        model: CommandProjectionModel,
    ) -> dict[str, Any]:
        source_payloads: dict[str, Any] = {}
        for interval_ref in sorted(context["interval_refs"], key=lambda item: item["interval_id"]):
            interval_id = interval_ref["interval_id"]
            feature_raw = context["feature"].get("sources", {}).get(interval_id)
            if feature_raw is None:
                continue
            feature = FixedWindowFeature.from_dict(feature_raw)
            posterior = model.posterior(feature)
            source_payloads[interval_id] = {
                **posterior.as_dict(),
                "speech_ppm": feature.speech_ppm,
                "feature_digest": feature.feature_digest,
                "frame_count": feature.frame_count,
                "overlap_frames": interval_ref["overlap_frames"],
                "trust_ppm": interval_ref["trust_ppm"],
                "speaker_id": interval_ref["speaker_id"],
                "interval_start_sample": interval_ref["interval_start_sample"],
                "interval_end_sample": interval_ref["interval_end_sample"],
            }
        global_sums: dict[int, int] = {}
        total_weight = 0
        for interval_id in sorted(source_payloads):
            source = source_payloads[interval_id]
            weight = max(1, int(source["overlap_frames"])) * max(1, int(source["trust_ppm"])) * max(
                1, int(source["speech_ppm"])
            )
            total_weight += weight
            for code_text, probability in source["posterior_ppm"].items():
                code = int(code_text)
                global_sums[code] = global_sums.get(code, 0) + int(probability) * weight
        if total_weight > 0 and global_sums:
            global_posterior = {
                code: round_half_away_from_zero(global_sums[code] / total_weight)
                for code in sorted(global_sums)
            }
            remainder = 1_000_000 - sum(global_posterior.values())
            best_code = max(sorted(global_posterior), key=lambda code: global_posterior[code])
            global_posterior[best_code] += remainder
        else:
            global_posterior = {}
        return {
            "window_id": context["window"]["window_id"],
            "sources": {key: source_payloads[key] for key in sorted(source_payloads)},
            "global_posterior_ppm": {str(code): value for code, value in sorted(global_posterior.items())},
            "projection_fingerprint": model.projection_fingerprint,
        }

    def _compute_attribution_payload(
        self,
        context: Mapping[str, Any],
        config: DiagnosticConfig,
    ) -> dict[str, Any]:
        source_evidence: list[dict[str, Any]] = []
        posterior_sources = context["posterior"].get("sources", {})
        for interval_ref in sorted(context["interval_refs"], key=lambda item: item["interval_id"]):
            interval_id = interval_ref["interval_id"]
            posterior = posterior_sources.get(interval_id)
            if posterior is None:
                continue
            source_evidence.append(
                {
                    "interval_id": interval_id,
                    "speaker_id": interval_ref["speaker_id"],
                    "interval_start_sample": interval_ref["interval_start_sample"],
                    "interval_end_sample": interval_ref["interval_end_sample"],
                    "overlap_frames": interval_ref["overlap_frames"],
                    "trust_ppm": interval_ref["trust_ppm"],
                    "speech_ppm": posterior["speech_ppm"],
                    "margin_ppm": posterior["margin_ppm"],
                    "best_code": posterior.get("best_code"),
                    "runner_up_code": posterior.get("runner_up_code"),
                    "posterior_ppm": posterior["posterior_ppm"],
                    "feature_digest": posterior["feature_digest"],
                }
            )
        return build_window_attribution(
            window_id=str(context["window"]["window_id"]),
            window_index=int(context["window"]["window_index"]),
            sample_start=int(context["window"]["sample_start"]),
            sample_end=int(context["window"]["sample_end"]),
            frame_indices=context["frame_indices"],
            sources=source_evidence,
            frame_count=int(context["frame_count"]),
            frame_length=config.analysis.frame_length,
            frame_shift=config.analysis.frame_shift,
            adjudication=config.adjudication,
        )

    def _finalize(
        self,
        *,
        ledger: ReuseLedger,
        store: SnapshotStore,
        contexts: list[dict[str, Any]],
        endpoint_payload: Mapping[str, Any],
        session: LoadedSession,
        config: DiagnosticConfig,
        candidate_mask: list[bool],
    ) -> dict[str, Any]:
        attribution_keys = [context["attribution_key"] for context in contexts]
        receipt_parameters = {
            "field_widths": config.receipt.field_widths,
            "quant_schemes": list(config.receipt.quant_schemes),
            "default_quant_scheme": config.receipt.default_quant_scheme,
            "command_vocabulary": {str(code): name for code, name in session.manifest.command_vocabulary.items()},
            # Truth labels are audit inputs even though they never influence
            # the acoustic decision.  Including them prevents a stale cached
            # conclusion payload from silently retaining old truth comparisons.
            "truth_command_codes": {
                interval.interval_id: interval.truth_command_code
                for interval in sorted(session.manifest.intervals, key=lambda item: item.interval_id)
            },
        }
        conclusion_key, conclusion_deps = conclusion_cache_key(
            attribution_keys=attribution_keys,
            adjudication_parameters=config.adjudication.as_parameters(),
            receipt_parameters=receipt_parameters,
            quant_profile=config.analysis.quant_profile,
            global_version=config.global_version,
        )
        probe = ledger.probe(
            cache_id="conclusion:global",
            stage="adjudication",
            key=conclusion_key,
            dependencies=conclusion_deps,
            refs={"window_ids": [context["window"]["window_id"] for context in contexts]},
            stable_key=("stage", "adjudication"),
        )
        if probe.reused and isinstance(probe.payload, dict):
            return probe.payload

        attribution_payloads = [context["attribution"] for context in contexts]
        frames, summary = adjudicate_frames(
            attribution_payloads=attribution_payloads,
            frame_metrics=endpoint_payload.get("frame_metrics", []),
            candidate_mask=candidate_mask,
            analysis=config.analysis,
            adjudication=config.adjudication,
        )
        commands = merge_command_conclusions(
            frames,
            analysis=config.analysis,
            adjudication=config.adjudication,
        )
        interval_sources: dict[str, list[dict[str, Any]]] = {}
        for context in sorted(contexts, key=lambda item: item["window"]["window_index"]):
            posterior_sources = context["posterior"].get("sources", {})
            for interval_ref in context["interval_refs"]:
                posterior = posterior_sources.get(interval_ref["interval_id"])
                if posterior is None:
                    continue
                interval_sources.setdefault(interval_ref["interval_id"], []).append(
                    {
                        "interval_id": interval_ref["interval_id"],
                        "speaker_id": interval_ref["speaker_id"],
                        "window_id": context["window"]["window_id"],
                        "window_index": context["window"]["window_index"],
                        "overlap_samples": interval_ref["overlap_samples"],
                        "trust_ppm": interval_ref["trust_ppm"],
                        "speech_ppm": posterior["speech_ppm"],
                        "posterior_ppm": posterior["posterior_ppm"],
                    }
                )
        interval_evidence: list[dict[str, Any]] = []
        for interval_id in sorted(interval_sources):
            source_rows = interval_sources[interval_id]
            source_keys = sorted(
                context["posterior_key"]
                for context in contexts
                if interval_id in context["interval_ids"]
            )
            interval_key, interval_deps = interval_cache_key(
                interval_id=interval_id,
                source_posterior_keys=source_keys,
                receipt_parameters=receipt_parameters,
                global_version=config.global_version,
            )
            interval_probe = ledger.probe(
                cache_id=f"interval:{interval_id}",
                stage="interval",
                key=interval_key,
                dependencies=interval_deps,
                refs={"interval_id": interval_id, "window_ids": sorted(row["window_id"] for row in source_rows)},
                stable_key=("stage", "interval", interval_id),
            )
            if interval_probe.reused and isinstance(interval_probe.payload, list):
                interval_evidence.extend(interval_probe.payload)
                continue
            aggregate = aggregate_interval_evidence(sources=source_rows, analysis=config.analysis)
            ledger.put(
                cache_id=f"interval:{interval_id}",
                stage="interval",
                key=interval_key,
                dependencies=interval_deps,
                refs={"interval_id": interval_id, "window_ids": sorted(row["window_id"] for row in source_rows)},
                stable_key=("stage", "interval", interval_id),
                payload=aggregate,
            )
            interval_evidence.extend(aggregate)
        interval_evidence.sort(key=lambda item: str(item["interval_id"]))
        for evidence in interval_evidence:
            truth = session.manifest.interval_by_id[str(evidence["interval_id"])].truth_command_code
            evidence["truth_command_code"] = truth
            evidence["truth_matches"] = evidence.get("command_code") == truth
        receipt_outputs = self._build_receipts(interval_evidence, session, config)
        frames_digest = frame_digest(frames, commands, interval_evidence)
        unique_digest = util_json_hash(
            {
                "frames": frames_digest,
                "commands": commands,
                "interval_evidence": interval_evidence,
                "a1": receipt_outputs["a1_text"],
                "a2": receipt_outputs["a2_text"],
            }
        )
        payload = {
            "frames": frames,
            "adjudication_summary": summary,
            "commands": commands,
            "interval_evidence": interval_evidence,
            "receipts": receipt_outputs,
            "frames_digest": frames_digest,
            "unique_replay_digest": unique_digest,
        }
        ledger.put(
            cache_id="conclusion:global",
            stage="adjudication",
            key=conclusion_key,
            dependencies=conclusion_deps,
            refs={"window_ids": [context["window"]["window_id"] for context in contexts]},
            stable_key=("stage", "adjudication"),
            payload=payload,
        )
        return payload

    def _build_receipts(
        self,
        interval_evidence: list[Mapping[str, Any]],
        session: LoadedSession,
        config: DiagnosticConfig,
    ) -> dict[str, Any]:
        command_names = session.manifest.command_vocabulary
        records: list[ReceiptRecord] = []
        for evidence in interval_evidence:
            code = evidence.get("command_code")
            if code is None:
                continue
            records.append(
                ReceiptRecord(
                    interval_id=str(evidence["interval_id"]),
                    speaker_id=str(evidence["speaker_id"]),
                    endpoint_confidence=int(evidence["endpoint_confidence_ppm"]) / 1_000_000.0,
                    command_name=command_names[int(code)],
                    command_code=int(code),
                    quant_scheme=config.receipt.default_quant_scheme,
                    source_version="runtime",
                    missing_fields=(),
                    storage={},
                )
            )
        document = ReceiptDocument(tuple(records), "runtime")
        a2_conversion = to_a2(document, vocabulary=command_names)
        a1_conversion = to_a1(a2_conversion.document, vocabulary=command_names)
        restored = parse_a1(
            a1_conversion.text,
            vocabulary=command_names,
            sidecar=a1_conversion.sidecar,
        )
        def semantic(record: ReceiptRecord) -> tuple[Any, ...]:
            return (
                record.interval_id,
                record.speaker_id,
                record.endpoint_confidence,
                record.command_name,
                record.command_code,
                record.quant_scheme,
            )

        roundtrip_ok = [semantic(record) for record in restored.records] == [
            semantic(record) for record in a2_conversion.document.records
        ]
        return {
            "a1_text": a1_conversion.text,
            "a2_text": a2_conversion.text,
            "a1_sidecar": a1_conversion.sidecar,
            "a2_sidecar": a2_conversion.sidecar,
            "a1_count": len(a1_conversion.document.records),
            "a2_count": len(a2_conversion.document.records),
            "roundtrip_ok": roundtrip_ok,
            "a1_digest": util_json_hash([record.canonical_fields() for record in a1_conversion.document.records]),
            "a2_digest": util_json_hash([record.canonical_fields() for record in a2_conversion.document.records]),
        }

    def _commit_state(
        self,
        store: SnapshotStore,
        state: dict[str, Any],
        session: LoadedSession,
        config: DiagnosticConfig,
        *,
        final: Mapping[str, Any] | None,
        status: str,
        batch_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = self._build_report(state, session, config, final=final, status=status, batch_summary=batch_summary)
        state["diagnostic_report"] = report
        report_text = self._render_report_text(report)
        committed = store.commit(state, report=report, report_text=report_text)
        state["snapshot_revision"] = committed.state.get("snapshot_revision")
        state["snapshot_dir"] = str(committed.snapshot_dir)
        return state

    def _build_report(
        self,
        state: Mapping[str, Any],
        session: LoadedSession,
        config: DiagnosticConfig,
        *,
        final: Mapping[str, Any] | None,
        status: str,
        batch_summary: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        ledger = ReuseLedger.from_dict(state.get("ledger"))
        cache_report = ledger.report()
        parse_summary = session.parse_summary
        comparison = session.receipt_comparison
        summary = final.get("adjudication_summary", {}) if final is not None else {}
        receipts = final.get("receipts", {}) if final is not None else {}
        interval_evidence = final.get("interval_evidence", []) if final is not None else []
        business_counts = dict(state.get("business_code_counts", {}))
        if status == "pending" or state.get("pending"):
            business_counts[E_POINT_BUDGET_PENDING] = max(1, int(business_counts.get(E_POINT_BUDGET_PENDING, 0)))
        return {
            "schema": 1,
            "conversation_id": session.manifest.conversation_id,
            "global_version": config.global_version,
            "config_hash": config.config_hash,
            "manifest_hash": session.manifest.structure_hash,
            "status": status,
            "complete": status == "complete",
            "stages": dict(state.get("stages", {})),
            "fragment_cursor": state.get("fragment_cursor", 0),
            "arrival_cursor": state.get("arrival_cursor", 0),
            "all_fragments_arrived": bool(state.get("all_fragments_arrived", True)),
            "budget_consumed_total": int(state.get("budget_consumed_total", 0)),
            "pending_count": len(state.get("pending", [])),
            "business_code_counts": {key: business_counts[key] for key in sorted(business_counts)},
            "last_error": state.get("last_error"),
            "last_batch": dict(batch_summary) if batch_summary is not None else state.get("last_batch"),
            "receipts": {
                "input_a1_count": parse_summary.get("receipt_a1_count", 0),
                "input_a2_count": parse_summary.get("receipt_a2_count", 0),
                "input_normalized_equal": comparison.get("ok") if comparison is not None else None,
                "input_normalized_digest": comparison.get("normalized_digest") if comparison is not None else None,
                "generated_a1_count": receipts.get("a1_count", 0),
                "generated_a2_count": receipts.get("a2_count", 0),
                "generated_roundtrip_ok": receipts.get("roundtrip_ok"),
                "generated_a1_digest": receipts.get("a1_digest"),
                "generated_a2_digest": receipts.get("a2_digest"),
            },
            "overlap_frame_count": summary.get("overlap_frame_count", 0),
            "boundary_frame_count": len(summary.get("boundary_frame_indices", [])),
            "boundary_frames": summary.get("boundary_frame_indices", []),
            "contested_frame_count": summary.get("contested_frame_count", 0),
            "frame_count": summary.get("frame_count", 0),
            "active_frame_count": summary.get("active_frame_count", 0),
            "candidate_count": len(state.get("endpoint", {}).get("candidates", [])) if state.get("endpoint") else 0,
            "multi_source_window_count": sum(
                1 for item in state.get("window_references", []) if len(item.get("intervals", [])) > 1
            ),
            "interval_truth_match_count": sum(1 for item in interval_evidence if item.get("truth_matches")),
            "interval_evidence_count": len(interval_evidence),
            "cache": cache_report,
            "final_digest": final.get("unique_replay_digest") if final is not None else None,
            "frames_digest": final.get("frames_digest") if final is not None else None,
            "command_conclusions": final.get("commands", {}) if final is not None else None,
            "interval_evidence": interval_evidence if final is not None else None,
            "audio_issues": parse_summary.get("audio_issues", []),
        }

    def _render_report_text(self, report: Mapping[str, Any]) -> str:
        lines = [
            f"Voice diagnostic: {report.get('conversation_id')}",
            f"status: {report.get('status')}",
            f"global_version: {report.get('global_version')}",
            f"final_digest: {report.get('final_digest')}",
            f"receipts a1/a2: {report.get('receipts', {}).get('generated_a1_count', 0)}/"
            f"{report.get('receipts', {}).get('generated_a2_count', 0)}",
            f"overlap frames: {report.get('overlap_frame_count', 0)}",
            f"boundary frames: {report.get('boundary_frame_count', 0)}",
            f"cache reuse/invalidation: {report.get('cache', {}).get('reuse_count', 0)}/"
            f"{report.get('cache', {}).get('invalidation_count', 0)}",
            f"pending: {report.get('pending_count', 0)}",
            "business codes: "
            + ", ".join(
                f"{code}={count}" for code, count in sorted(report.get("business_code_counts", {}).items())
            ),
        ]
        commands = report.get("command_conclusions") or {}
        for segment in commands.get("accepted", []):
            lines.append(
                f"command {segment.get('command_code')} @ {segment.get('start_sample')}..{segment.get('end_sample')} "
                f"frames={segment.get('frame_count')} contested={segment.get('contested_frame_count')}"
            )
        return "\n".join(lines) + "\n"

    def _result_payload(
        self,
        state: Mapping[str, Any],
        store: SnapshotStore,
        session: LoadedSession,
        config: DiagnosticConfig,
        *,
        status: str,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "complete": status == "complete",
            "report": state.get("diagnostic_report"),
            "snapshot_revision": state.get("snapshot_revision"),
            "snapshot_dir": state.get("snapshot_dir"),
            "fragment_cursor": state.get("fragment_cursor", 0),
            "all_fragments_arrived": bool(state.get("all_fragments_arrived", True)),
            "final_digest": (state.get("final") or {}).get("unique_replay_digest"),
            "pending": state.get("pending", []),
            "business_code_counts": state.get("business_code_counts", {}),
            "output_dir": str(self.output_dir),
        }

    def _mirror_outputs(
        self,
        store: SnapshotStore,
        state: Mapping[str, Any],
        result: Mapping[str, Any],
        session: LoadedSession,
        config: DiagnosticConfig,
    ) -> None:
        report = state.get("diagnostic_report")
        if isinstance(report, dict):
            atomic_write_json(self.output_dir / "diagnostic-report.json", report, indent=2)
            atomic_write_text(self.output_dir / "diagnostic-report.md", self._render_report_text(report))
        final = state.get("final")
        if isinstance(final, dict):
            receipt_dir = self.output_dir / "receipts"
            receipts = final.get("receipts", {})
            atomic_write_text(receipt_dir / "evidence.a1", str(receipts.get("a1_text", "")))
            atomic_write_text(receipt_dir / "evidence.a2.json", str(receipts.get("a2_text", "")))
            atomic_write_json(receipt_dir / "a1-sidecar.json", receipts.get("a1_sidecar", {}), indent=2)
            atomic_write_json(receipt_dir / "a2-sidecar.json", receipts.get("a2_sidecar", {}), indent=2)
            atomic_write_json(
                self.output_dir / "final.json",
                {
                    "unique_replay_digest": final.get("unique_replay_digest"),
                    "adjudication_summary": final.get("adjudication_summary"),
                    "commands": final.get("commands"),
                    "interval_evidence": final.get("interval_evidence"),
                },
                indent=2,
            )
        else:
            # Never leave a previous complete final/receipt beside a newly
            # committed pending or quota-aborted snapshot.  Snapshot history
            # remains immutable; only the convenience mirror is cleared.
            for stale in (
                self.output_dir / "final.json",
                self.output_dir / "receipts" / "evidence.a1",
                self.output_dir / "receipts" / "evidence.a2.json",
                self.output_dir / "receipts" / "a1-sidecar.json",
                self.output_dir / "receipts" / "a2-sidecar.json",
            ):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
