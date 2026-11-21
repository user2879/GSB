"""Voice-conversion session engine with deterministic incremental state evolution."""

from __future__ import annotations

import copy
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import AudioCatalog, WavAudio
from .config import WorkbenchConfig
from .dsp import analyze_frame, frame_count, frame_ranges
from .errors import (
    E_AUDIO_FORMAT,
    E_COMMAND_INVALID,
    E_FRAME_BUDGET_EXHAUSTED,
    E_SEGMENT_FAILURE,
    E_SEGMENT_OVERFLOW,
    E_SNAPSHOT_STALE,
    Failure,
    StageError,
    sorted_failures,
)
from .mapping import (
    build_mapping,
    mapping_identity,
    mapping_key,
    public_mapping,
)
from .plan import (
    CommandError,
    SegmentSpec,
    SessionPlan,
    apply_command,
    command_sequence,
    effective_params,
    initial_params,
    params_public,
)
from .state import (
    SnapshotError,
    SnapshotState,
    commit_snapshot,
    load_current,
    recovery_point,
)
from .stats import aggregate_stats, speaker_stats_from_analyses
from .util import canonical_json, q_from_float, sha256_bytes, sha256_value


@dataclass(slots=True)
class EngineOptions:
    fresh: bool = False
    stop_after_command: int | None = None
    crash_after_command: int | None = None
    crash_on_checkpoint: int | None = None
    output_path: Path | None = None
    report_path: Path | None = None


@dataclass(slots=True)
class AnalysisRequest:
    category: str
    key: str
    sample_count: int
    samples: np.ndarray
    audio_path: str
    content_fingerprint: str
    analysis_key: str
    order: tuple[int, int, str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _analysis_key(content_fingerprint: str, cfg: WorkbenchConfig, sample_count: int) -> str:
    return sha256_value(
        {
            "stage": "analysis",
            "content_fingerprint": content_fingerprint,
            "analysis_config": cfg.analysis_fingerprint(),
            "sample_count": sample_count,
        }
    )


def _slice_fingerprint(audio: WavAudio, start: int, end: int | None) -> str:
    stop = audio.frame_count if end is None else end
    return sha256_value(
        {
            "source_content": audio.content_fingerprint,
            "start_sample": start,
            "end_sample": stop,
            "sample_count": max(0, stop - start),
        }
    )


def _frame_worker(
    samples: np.ndarray,
    cfg: WorkbenchConfig,
    frame_index: int,
    start: int,
    valid: int,
    padded: bool,
) -> dict[str, Any]:
    length = cfg.frame_length_samples
    frame = np.zeros(length, dtype=np.float64)
    if valid > 0:
        raw = samples[start : start + valid].astype(np.float64) / 32768.0
        frame[:valid] = raw
    return analyze_frame(
        frame,
        cfg,
        frame_index=frame_index,
        start_sample=start,
        valid_samples=valid,
        padded=padded,
    )


def _mapping_worker(
    *,
    plan: SessionPlan,
    segment: SegmentSpec,
    segment_index: int,
    analysis: dict[str, Any],
    local_stats: dict[str, Any],
    target_stats: dict[str, Any],
    effective: dict[str, Any],
    cfg: WorkbenchConfig,
) -> dict[str, Any]:
    if plan.fault_for(segment.segment_id, "mapping"):
        detail = plan.fault_for(segment.segment_id, "mapping") or {}
        raise StageError(
            "mapping",
            E_SEGMENT_FAILURE,
            str(detail.get("exception_type", "InjectedFault")),
            str(detail.get("message", f"injected mapping fault for {segment.segment_id}")),
        )
    if plan.fault_for(segment.segment_id, "synthesis"):
        detail = plan.fault_for(segment.segment_id, "synthesis") or {}
        raise StageError(
            "synthesis",
            E_SEGMENT_FAILURE,
            str(detail.get("exception_type", "InjectedFault")),
            str(detail.get("message", f"injected synthesis fault for {segment.segment_id}")),
        )
    return build_mapping(
        segment_id=segment.segment_id,
        segment_index=segment_index,
        analysis=analysis,
        local_stats=local_stats,
        target_stats=target_stats,
        effective=effective,
        cfg=cfg,
    )


class WorkbenchEngine:
    def __init__(
        self,
        *,
        samples_dir: Path,
        config: WorkbenchConfig,
        plan: SessionPlan,
        state_root: Path,
        options: EngineOptions | None = None,
    ) -> None:
        self.samples_dir = samples_dir.resolve()
        self.cfg = config
        self.plan = plan
        self.state_root = state_root.resolve()
        self.options = options or EngineOptions()
        self.catalog = AudioCatalog.scan(self.samples_dir)
        self.startup_events: list[dict[str, Any]] = []
        self.stop_requested = False
        self.graceful_signal = False
        self.recovered = False
        self.checkpoint_count = 0
        self.latest_state: SnapshotState | None = None
        self._catalog_failure_payloads = self.catalog.failure_payloads()
        self.audio_fingerprint = ""
        self._load_or_initialise()
        self.previous_assembly_key = str(self.assembly.get("assembly_key", ""))

    # ------------------------------------------------------------------
    # Startup and snapshot validation
    # ------------------------------------------------------------------
    def _state_incompatibility(self, state: SnapshotState) -> str | None:
        manifest = state.manifest
        checks = [
            ("schema_version", 1, manifest.get("schema_version")),
            ("format_version", self.cfg.format_version, manifest.get("format_version")),
            ("engine_version", self.cfg.engine_version, manifest.get("engine_version")),
            ("config_hash", self.cfg.config_hash(), manifest.get("config_hash")),
            (
                "audio_fingerprint",
                self.catalog.fingerprint_for_segments(manifest.get("segments", [])),
                manifest.get("audio_fingerprint"),
            ),
        ]
        for name, expected, actual in checks:
            if actual != expected:
                return f"{name} changed: expected {expected!r}, snapshot has {actual!r}"
        cursor = int(manifest.get("command_cursor", -1))
        version = int(manifest.get("global_version", -1))
        if cursor < 0 or cursor > len(self.plan.commands):
            return f"invalid command_cursor {cursor}"
        if version != cursor:
            return f"global_version {version} does not equal command_cursor {cursor}"
        active = manifest.get("active_command_seq")
        if active is not None:
            if not isinstance(active, int) or active != cursor + 1 or active > len(self.plan.commands):
                return f"invalid active_command_seq {active!r} at cursor {cursor}"
            if manifest.get("command_complete") is not False:
                return "active snapshot must be marked incomplete"
        elif manifest.get("command_complete") is not True:
            return "completed snapshot must be marked complete"
        prefix_length = cursor if active is None else active
        if prefix_length > len(self.plan.commands):
            return f"command stream is shorter than snapshot prefix {prefix_length}"
        if manifest.get("command_prefix_hash") != self.plan.command_prefix_hash(prefix_length):
            return "command stream prefix changed"
        params = manifest.get("params")
        if not isinstance(params, dict) or int(params.get("scale", -1)) != self.cfg.scale:
            return "snapshot fixed-point scale is incompatible"
        return None

    def _load_or_initialise(self) -> None:
        state: SnapshotState | None = None
        if not self.options.fresh:
            try:
                state = load_current(self.state_root)
            except SnapshotError as exc:
                self.startup_events.append(
                    {
                        "code": E_SNAPSHOT_STALE,
                        "stage": "snapshot",
                        "detail": exc.detail,
                        "exception_type": "SnapshotError",
                    }
                )
                state = None
            if state is not None:
                incompatibility = self._state_incompatibility(state)
                if incompatibility is not None:
                    self.startup_events.append(
                        {
                            "code": E_SNAPSHOT_STALE,
                            "stage": "snapshot",
                            "detail": incompatibility,
                            "exception_type": "SnapshotCompatibilityError",
                        }
                    )
                    state = None
                else:
                    self.recovered = True
        if state is None:
            self._initialise_empty()
            self._commit(active_seq=None, command_complete=True, global_version=0)
            return
        self.latest_state = state
        manifest = state.manifest
        self.params = copy.deepcopy(manifest["params"])
        self.segments = [
            SegmentSpec(
                segment_id=str(raw["segment_id"]),
                path=str(raw["path"]),
                start_sample=int(raw.get("start_sample", 0)),
                end_sample=None if raw.get("end_sample") is None else int(raw["end_sample"]),
            )
            for raw in manifest.get("segments", [])
        ]
        self.command_cursor = int(manifest.get("command_cursor", 0))
        self.global_version = int(manifest.get("global_version", 0))
        self.ref_analyses = copy.deepcopy(state.analysis.get("reference_analyses", {}))
        self.segment_analyses = copy.deepcopy(state.analysis.get("segment_analyses", {}))
        self.speaker_stats = copy.deepcopy(state.stats.get("speakers", {}))
        self.local_stats = copy.deepcopy(state.stats.get("local", {}))
        self.mappings = copy.deepcopy(state.mappings.get("segments", {}))
        self.pcm_by_segment = dict(state.pcm_by_segment)
        self.assembly = copy.deepcopy(state.assembly)
        self.assembly_pcm = state.assembly_pcm
        self.ledger = copy.deepcopy(state.ledger)
        self.charges = copy.deepcopy(state.ledger.get("charges", []))
        self.current_failures = copy.deepcopy(state.ledger.get("failures", []))
        self.decision = copy.deepcopy(state.ledger.get("decision", {"reuse": [], "recompute": []}))
        self.todo = list(state.ledger.get("todo", []))
        self.resolved = set(state.ledger.get("resolved", []))
        self.denied = set(state.ledger.get("denied", []))
        self.command_failure_history = copy.deepcopy(state.ledger.get("command_failure_history", []))
        self.active_seq = manifest.get("active_command_seq")
        self.command_complete = bool(manifest.get("command_complete", False))
        self.audio_fingerprint = self.catalog.fingerprint_for_segments(self.segments)

    def _initialise_empty(self) -> None:
        self.params = initial_params(self.plan, self.cfg)
        self.segments = list(self.plan.initial_segments)
        self.command_cursor = 0
        self.global_version = 0
        self.ref_analyses: dict[str, Any] = {}
        self.segment_analyses: dict[str, Any] = {}
        self.speaker_stats: dict[str, Any] = {}
        self.local_stats: dict[str, Any] = {}
        self.mappings: dict[str, Any] = {}
        self.pcm_by_segment: dict[str, bytes] = {}
        self.assembly: dict[str, Any] = {}
        self.assembly_pcm = b""
        self.ledger: dict[str, Any] = {}
        self.charges: list[dict[str, Any]] = []
        self.current_failures: list[dict[str, Any]] = []
        self.decision: dict[str, list[dict[str, Any]]] = {"reuse": [], "recompute": []}
        self.todo: list[str] = []
        self.resolved: set[str] = set()
        self.denied: set[str] = set()
        self.command_failure_history: list[dict[str, Any]] = []
        self.active_seq: int | None = None
        self.command_complete = True
        self.audio_fingerprint = self.catalog.fingerprint_for_segments(self.segments)

    # ------------------------------------------------------------------
    # Analysis and statistics stages
    # ------------------------------------------------------------------
    def _segment_path_failure(self, segment: SegmentSpec, seq: int, detail: str) -> dict[str, Any]:
        return Failure(
            code=E_AUDIO_FORMAT,
            stage="analysis",
            detail=detail,
            segment_id=segment.segment_id,
            segment_index=self.segment_index(segment.segment_id),
            path=segment.path,
            command_seq=seq,
            exception_type="AudioFormatError",
        ).as_dict()

    def _catalog_failures_for_command(self, seq: int) -> list[dict[str, Any]]:
        segment_by_path: dict[str, SegmentSpec] = {}
        for segment in self.segments:
            segment_by_path.setdefault(segment.path, segment)
        result: list[dict[str, Any]] = []
        for payload in self._catalog_failure_payloads:
            item = dict(payload)
            item["command_seq"] = seq
            segment = segment_by_path.get(str(item.get("path", "")))
            if segment is not None:
                item["segment_id"] = segment.segment_id
                item["segment_index"] = self.segment_index(segment.segment_id)
            result.append(item)
        return result

    def segment_index(self, segment_id: str) -> int | None:
        for index, segment in enumerate(self.segments):
            if segment.segment_id == segment_id:
                return index
        return None

    def _analysis_requests(self, seq: int) -> tuple[list[AnalysisRequest], dict[str, Any]]:
        requests: list[AnalysisRequest] = []
        segment_markers: dict[str, Any] = {}
        for rel, audio in sorted(self.catalog.references.items()):
            content = f"reference:{rel}:{audio.content_fingerprint}"
            key = _analysis_key(content, self.cfg, audio.frame_count)
            old = self.ref_analyses.get(rel)
            if old is not None and old.get("analysis_key") == key and old.get("analysis_status") == "ok":
                continue
            requests.append(
                AnalysisRequest(
                    category="reference",
                    key=rel,
                    sample_count=audio.frame_count,
                    samples=audio.samples,
                    audio_path=rel,
                    content_fingerprint=content,
                    analysis_key=key,
                    order=(0, 0, rel),
                )
            )
        for index, segment in enumerate(self.segments):
            audio = self.catalog.sources.get(segment.path)
            if audio is None:
                known_invalid = any(failure.relative_path == segment.path for failure in self.catalog.failures)
                marker = {
                    "analysis_status": "failed",
                    "segment_id": segment.segment_id,
                    "segment_index": index,
                    "path": segment.path,
                    "analysis_key": None,
                    "failure": None
                    if known_invalid
                    else self._segment_path_failure(segment, seq, "source WAV is missing"),
                }
                segment_markers[segment.segment_id] = marker
                continue
            stop = audio.frame_count if segment.end_sample is None else segment.end_sample
            if segment.start_sample < 0 or stop <= segment.start_sample or stop > audio.frame_count:
                marker = {
                    "analysis_status": "failed",
                    "segment_id": segment.segment_id,
                    "segment_index": index,
                    "path": segment.path,
                    "analysis_key": None,
                    "failure": self._segment_path_failure(
                        segment,
                        seq,
                        f"sample range [{segment.start_sample}, {stop}) is outside source length {audio.frame_count}",
                    ),
                }
                segment_markers[segment.segment_id] = marker
                continue
            count = frame_count(stop - segment.start_sample, self.cfg.frame_length_samples, self.cfg.frame_shift_samples)
            if count > self.cfg.max_frames_per_segment:
                marker = {
                    "analysis_status": "overflow",
                    "segment_id": segment.segment_id,
                    "segment_index": index,
                    "path": segment.path,
                    "analysis_key": None,
                    "frame_count": count,
                    "failure": Failure(
                        code=E_SEGMENT_OVERFLOW,
                        stage="analysis",
                        detail=f"{segment.segment_id}: {count} frames exceed limit {self.cfg.max_frames_per_segment}",
                        segment_id=segment.segment_id,
                        segment_index=index,
                        path=segment.path,
                        command_seq=seq,
                        exception_type="SegmentOverflow",
                    ).as_dict(),
                }
                segment_markers[segment.segment_id] = marker
                continue
            if self.plan.fault_for(segment.segment_id, "analysis"):
                detail = self.plan.fault_for(segment.segment_id, "analysis") or {}
                marker = {
                    "analysis_status": "failed",
                    "segment_id": segment.segment_id,
                    "segment_index": index,
                    "path": segment.path,
                    "analysis_key": None,
                    "failure": Failure(
                        code=E_SEGMENT_FAILURE,
                        stage="analysis",
                        detail=str(detail.get("message", f"injected analysis fault for {segment.segment_id}")),
                        segment_id=segment.segment_id,
                        segment_index=index,
                        path=segment.path,
                        command_seq=seq,
                        exception_type=str(detail.get("exception_type", "InjectedFault")),
                    ).as_dict(),
                }
                segment_markers[segment.segment_id] = marker
                continue
            content = _slice_fingerprint(audio, segment.start_sample, stop)
            key = _analysis_key(content, self.cfg, stop - segment.start_sample)
            old = self.segment_analyses.get(segment.segment_id)
            if (
                old is not None
                and old.get("analysis_key") == key
                and old.get("analysis_status") == "ok"
                and old.get("source_content_fingerprint") == content
            ):
                continue
            requests.append(
                AnalysisRequest(
                    category="segment",
                    key=segment.segment_id,
                    sample_count=stop - segment.start_sample,
                    samples=audio.samples[segment.start_sample:stop],
                    audio_path=segment.path,
                    content_fingerprint=content,
                    analysis_key=key,
                    order=(1, index, segment.segment_id),
                    metadata={
                        "segment_id": segment.segment_id,
                        "segment_index": index,
                        "start_sample": segment.start_sample,
                        "end_sample": stop,
                    },
                )
            )
        return requests, segment_markers

    def _run_analyses(self, requests: list[AnalysisRequest], seq: int) -> None:
        if not requests:
            return
        request_by_id = {request.key: request for request in requests}
        frame_failures: dict[str, Exception] = {}
        frame_results: dict[tuple[str, int], dict[str, Any]] = {}
        workers = min(self.cfg.thread_pool_workers, max(1, sum(max(1, frame_count(r.sample_count, self.cfg.frame_length_samples, self.cfg.frame_shift_samples)) for r in requests)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vc-analysis") as executor:
            futures: dict[Future[dict[str, Any]], tuple[str, int, tuple[int, int, str]]] = {}
            for request in sorted(requests, key=lambda item: item.order):
                ranges = frame_ranges(request.sample_count, self.cfg)
                for frame_index, (start, valid, padded) in enumerate(ranges):
                    stable_key = (request.order[0], request.order[1], request.order[2])
                    future = executor.submit(
                        _frame_worker,
                        request.samples,
                        self.cfg,
                        frame_index,
                        start,
                        valid,
                        padded,
                    )
                    futures[future] = (request.key, frame_index, stable_key)
            for future in as_completed(futures):
                request_key, frame_index, _ = futures[future]
                try:
                    frame_results[(request_key, frame_index)] = future.result()
                except Exception as exc:  # isolation is per request, not per worker
                    frame_failures.setdefault(request_key, exc)
        for request in sorted(requests, key=lambda item: item.order):
            failure = frame_failures.get(request.key)
            if failure is not None:
                marker = {
                    "analysis_status": "failed",
                    "analysis_key": request.analysis_key,
                    "content_fingerprint": request.content_fingerprint,
                    "path": request.audio_path,
                    "failure": Failure(
                        code=E_SEGMENT_FAILURE,
                        stage="analysis",
                        detail=f"frame analysis failed: {failure}",
                        segment_id=request.metadata.get("segment_id"),
                        segment_index=request.metadata.get("segment_index"),
                        path=request.audio_path,
                        command_seq=seq,
                        exception_type=type(failure).__name__,
                    ).as_dict(),
                }
                if request.category == "reference":
                    self.ref_analyses[request.key] = marker
                else:
                    self.segment_analyses[request.key] = marker
                continue
            count = frame_count(request.sample_count, self.cfg.frame_length_samples, self.cfg.frame_shift_samples)
            frames = [frame_results[(request.key, index)] for index in range(count)]
            analysis = {
                "analysis_status": "ok",
                "analysis_key": request.analysis_key,
                "cache_key": request.analysis_key,
                "content_fingerprint": request.content_fingerprint,
                "source_content_fingerprint": request.content_fingerprint,
                "config_fingerprint": self.cfg.analysis_fingerprint(),
                "path": request.audio_path,
                "sample_count": request.sample_count,
                "frame_length_samples": self.cfg.frame_length_samples,
                "frame_shift_samples": self.cfg.frame_shift_samples,
                "frame_count": count,
                "valid_frame_count": sum(1 for frame in frames if int(frame["valid_samples"]) > 0),
                "last_frame_padded": bool(frames[-1]["padded"]) if frames else False,
                "frames": frames,
            }
            analysis.update(request.metadata)
            if request.category == "reference":
                self.ref_analyses[request.key] = analysis
            else:
                self.segment_analyses[request.key] = analysis

    def _refresh_analysis_stage(self, seq: int) -> None:
        requests, markers = self._analysis_requests(seq)
        self._run_analyses(requests, seq)
        for segment_id, marker in markers.items():
            self.segment_analyses[segment_id] = marker
        for request in requests:
            if request.category == "segment" and request.key not in self.segment_analyses:
                self.segment_analyses[request.key] = {
                    "analysis_status": "failed",
                    "analysis_key": request.analysis_key,
                    "path": request.audio_path,
                    "failure": Failure(
                        code=E_SEGMENT_FAILURE,
                        stage="analysis",
                        detail="analysis result was not produced",
                        segment_id=request.metadata.get("segment_id"),
                        segment_index=request.metadata.get("segment_index"),
                        command_seq=seq,
                        exception_type="MissingAnalysisResult",
                    ).as_dict(),
                }
        # Drop impossible stale segment analyses for removed-marker duplicates.
        current_ids = {segment.segment_id for segment in self.segments}
        for segment_id in list(self.segment_analyses):
            if segment_id not in current_ids:
                self.segment_analyses.pop(segment_id, None)

    def _refresh_stats_stage(self) -> None:
        speakers = self.catalog.reference_speakers()
        new_speaker_stats: dict[str, Any] = {}
        for speaker, paths in sorted(speakers.items()):
            analyses = [
                self.ref_analyses[path]
                for path in paths
                if path in self.ref_analyses and self.ref_analyses[path].get("analysis_status") == "ok"
            ]
            new_speaker_stats[speaker] = speaker_stats_from_analyses(analyses, self.cfg)
        empty = aggregate_stats([], self.cfg)
        for speaker in {str(self.params["target_speaker"])}:
            if speaker not in new_speaker_stats:
                new_speaker_stats[speaker] = empty
        self.speaker_stats = new_speaker_stats
        new_local: dict[str, Any] = {}
        for segment in self.segments:
            analysis = self.segment_analyses.get(segment.segment_id)
            if analysis is None or analysis.get("analysis_status") != "ok":
                continue
            new_local[segment.segment_id] = aggregate_stats(
                analysis.get("frames", []),
                self.cfg,
                cache_key=sha256_value(
                    {
                        "stage": "segment-stats",
                        "analysis_key": analysis["analysis_key"],
                        "stats_config": self.cfg.stats_fingerprint(),
                    }
                ),
                source_keys=[str(analysis["analysis_key"])],
            )
        self.local_stats = new_local

    # ------------------------------------------------------------------
    # Mapping invalidation and budget-aware parallel execution
    # ------------------------------------------------------------------
    def _desired_mapping_inputs(self, segment: SegmentSpec, index: int) -> dict[str, Any] | None:
        analysis = self.segment_analyses.get(segment.segment_id)
        local = self.local_stats.get(segment.segment_id)
        if analysis is None or analysis.get("analysis_status") != "ok" or local is None:
            return None
        effective = effective_params(self.params, segment.segment_id)
        target = self.speaker_stats.get(str(effective["target_speaker"]), aggregate_stats([], self.cfg))
        key = mapping_key(
            analysis_key=str(analysis["analysis_key"]),
            local_stats_fingerprint=str(local.get("fingerprint", "")),
            target_stats_fingerprint=str(target.get("fingerprint", "empty")),
            effective=effective,
            config_fingerprint=self.cfg.mapping_fingerprint(),
        )
        return {
            "segment": segment,
            "segment_index": index,
            "analysis": analysis,
            "local_stats": local,
            "target_stats": target,
            "effective": effective,
            "mapping_key": key,
        }

    def _failure_for_analysis_or_overflow(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for segment in self.segments:
            analysis = self.segment_analyses.get(segment.segment_id)
            if analysis is not None and analysis.get("analysis_status") in {"failed", "overflow"}:
                failure = analysis.get("failure")
                if failure:
                    result.append(dict(failure))
        return result

    def _reuse_reason(self, old: dict[str, Any] | None, desired: dict[str, Any]) -> tuple[bool, str]:
        if old is None or old.get("mapping_status") != "mapped":
            return False, "mapping-missing"
        if old.get("mapping_key") != desired["mapping_key"]:
            if old.get("analysis_key") != desired["analysis"]["analysis_key"]:
                return False, "source-changed"
            if old.get("local_stats_fingerprint") != desired["local_stats"].get("fingerprint"):
                return False, "stats-changed"
            if old.get("params") != dict(sorted(desired["effective"].items())):
                return False, "parameters-changed"
            return False, "mapping-key-changed"
        if desired["segment"].segment_id not in self.pcm_by_segment:
            return False, "mapping-piece-missing"
        if old.get("pcm_sha256") != sha256_bytes(self.pcm_by_segment[desired["segment"].segment_id]):
            return False, "mapping-piece-corrupt"
        return True, "same-content-and-parameters"

    def _prepare_decisions(self, seq: int, *, resume: bool) -> None:
        if resume:
            return
        reuse: list[dict[str, Any]] = []
        recompute: list[dict[str, Any]] = []
        for index, segment in enumerate(self.segments):
            desired = self._desired_mapping_inputs(segment, index)
            analysis = self.segment_analyses.get(segment.segment_id)
            if desired is None:
                if analysis is not None and analysis.get("analysis_status") in {"failed", "overflow"}:
                    failure = analysis.get("failure") or {}
                    self.mappings[segment.segment_id] = {
                        "segment_id": segment.segment_id,
                        "segment_index": index,
                        "mapping_status": "failed",
                        "failure": failure,
                    }
                self.pcm_by_segment.pop(segment.segment_id, None)
                continue
            old = self.mappings.get(segment.segment_id)
            is_reuse, reason = self._reuse_reason(old, desired)
            item = {
                "segment_id": segment.segment_id,
                "segment_index": index,
                "mapping_key": desired["mapping_key"],
                "reason": reason,
            }
            if is_reuse:
                reuse.append(item)
            else:
                recompute.append(item)
                self.mappings[segment.segment_id] = {
                    "segment_id": segment.segment_id,
                    "segment_index": index,
                    "mapping_status": "pending",
                    "mapping_key": desired["mapping_key"],
                    "pending_reason": reason,
                }
                self.pcm_by_segment.pop(segment.segment_id, None)
        reuse.sort(key=lambda item: int(item["segment_index"]))
        recompute.sort(key=lambda item: int(item["segment_index"]))
        self.decision = {"reuse": reuse, "recompute": recompute}
        self.resolved = set()
        self.denied = set()
        self.todo = []

    def _budget_remaining(self) -> int:
        return self.cfg.speech_frame_budget - sum(int(charge["frame_count"]) for charge in self.charges)

    def _add_budget_failure(self, entry: dict[str, Any], seq: int, cost: int, pending: bool = False) -> None:
        segment_id = str(entry["segment_id"])
        if pending:
            detail = f"{segment_id}: pending after global frame budget exhaustion"
        else:
            detail = f"{segment_id}: remaining budget is insufficient for {cost} speech frames"
        self.current_failures.append(
            Failure(
                code=E_FRAME_BUDGET_EXHAUSTED,
                stage="scheduler",
                detail=detail,
                segment_id=segment_id,
                segment_index=int(entry["segment_index"]),
                command_seq=seq,
                exception_type="BudgetExhausted",
            ).as_dict()
        )
        self.denied.add(segment_id)
        self.resolved.add(segment_id)
        old = self.mappings.get(segment_id, {})
        self.mappings[segment_id] = {
            "segment_id": segment_id,
            "segment_index": int(entry["segment_index"]),
            "mapping_status": "pending",
            "mapping_key": entry["mapping_key"],
            "pending_reason": "E_FRAME_BUDGET_EXHAUSTED",
            "budget_detail": detail,
            "old_mapping_key": old.get("mapping_key"),
        }
        self.pcm_by_segment.pop(segment_id, None)

    def _mapping_entry_by_segment(self) -> dict[str, dict[str, Any]]:
        return {str(entry["segment_id"]): entry for entry in self.decision.get("recompute", [])}

    def _run_mapping_tasks(self, seq: int, *, resume: bool) -> None:
        recompute_entries = sorted(self.decision.get("recompute", []), key=lambda item: int(item["segment_index"]))
        entry_by_segment = {str(entry["segment_id"]): entry for entry in recompute_entries}
        pending_ids = [segment_id for segment_id in entry_by_segment if segment_id not in self.resolved]
        denied_remaining = bool(self.denied)
        while pending_ids:
            batch_ids: list[str] = []
            if not denied_remaining:
                for segment_id in pending_ids:
                    if len(batch_ids) >= self.cfg.thread_pool_workers:
                        break
                    batch_ids.append(segment_id)
            if not batch_ids:
                break
            desired_batch: list[tuple[dict[str, Any], dict[str, Any], int]] = []
            reserved = 0
            denial_index: int | None = None
            for position, segment_id in enumerate(batch_ids):
                entry = entry_by_segment[segment_id]
                segment = next(s for s in self.segments if s.segment_id == segment_id)
                desired = self._desired_mapping_inputs(segment, int(entry["segment_index"]))
                if desired is None:
                    self.resolved.add(segment_id)
                    continue
                cost = int(desired["local_stats"].get("speech_frame_count", 0))
                remaining = self._budget_remaining() - reserved
                if cost > remaining:
                    denial_index = position
                    break
                reserved += cost
                desired_batch.append((entry, desired, cost))
            if not desired_batch and denial_index is None:
                for segment_id in batch_ids:
                    self.resolved.add(segment_id)
                pending_ids = [segment_id for segment_id in pending_ids if segment_id not in self.resolved]
                continue

            results: dict[str, dict[str, Any] | Exception] = {}
            with ThreadPoolExecutor(max_workers=max(1, min(self.cfg.thread_pool_workers, len(desired_batch))), thread_name_prefix="vc-mapping") as executor:
                futures: dict[Future[dict[str, Any]], str] = {}
                for entry, desired, _ in desired_batch:
                    future = executor.submit(
                        _mapping_worker,
                        plan=self.plan,
                        segment=desired["segment"],
                        segment_index=int(entry["segment_index"]),
                        analysis=desired["analysis"],
                        local_stats=desired["local_stats"],
                        target_stats=desired["target_stats"],
                        effective=desired["effective"],
                        cfg=self.cfg,
                    )
                    futures[future] = str(entry["segment_id"])
                for future in as_completed(futures):
                    segment_id = futures[future]
                    try:
                        results[segment_id] = future.result()
                    except Exception as exc:
                        results[segment_id] = exc
            for entry, desired, cost in desired_batch:
                segment_id = str(entry["segment_id"])
                if segment_id in self.resolved:
                    continue
                result = results.get(segment_id)
                self.charges.append(
                    {
                        "command_seq": int(seq),
                        "frame_count": int(cost),
                        "ordinal": len(self.charges) + 1,
                        "segment_id": segment_id,
                        "segment_identity": str(entry["mapping_key"]),
                    }
                )
                if isinstance(result, dict):
                    result["piece_file"] = f"pieces/{segment_id}.pcm.z"
                    result["mapping_status"] = "mapped"
                    self.mappings[segment_id] = result
                    self.pcm_by_segment[segment_id] = bytes(result["pcm"])
                else:
                    exc = result if isinstance(result, BaseException) else RuntimeError("mapping task produced no result")
                    stage = exc.stage if isinstance(exc, StageError) else "mapping"
                    code = exc.code if isinstance(exc, StageError) else E_SEGMENT_FAILURE
                    exception_type = exc.exception_type if isinstance(exc, StageError) else type(exc).__name__
                    failure = Failure(
                        code=code,
                        stage=stage,
                        detail=str(exc),
                        segment_id=segment_id,
                        segment_index=int(entry["segment_index"]),
                        command_seq=int(seq),
                        exception_type=exception_type,
                    ).as_dict()
                    self.current_failures.append(failure)
                    self.mappings[segment_id] = {
                        "segment_id": segment_id,
                        "segment_index": int(entry["segment_index"]),
                        "mapping_status": "failed",
                        "mapping_key": str(entry["mapping_key"]),
                        "failure": failure,
                    }
                    self.pcm_by_segment.pop(segment_id, None)
                self.resolved.add(segment_id)
                self.checkpoint_count += 1
                if self.checkpoint_count % self.cfg.snapshot_interval_segments == 0:
                    self._commit(active_seq=seq, command_complete=False, global_version=self.command_cursor)
                    self._maybe_crash_checkpoint()
            if denial_index is not None:
                for segment_id in batch_ids[denial_index:]:
                    entry = entry_by_segment[segment_id]
                    segment = next(s for s in self.segments if s.segment_id == segment_id)
                    desired = self._desired_mapping_inputs(segment, int(entry["segment_index"]))
                    cost = int(desired["local_stats"].get("speech_frame_count", 0)) if desired else 0
                    self._add_budget_failure(entry, seq, cost, pending=segment_id != batch_ids[denial_index])
                for segment_id in pending_ids:
                    if segment_id not in batch_ids:
                        entry = entry_by_segment[segment_id]
                        segment = next(s for s in self.segments if s.segment_id == segment_id)
                        desired = self._desired_mapping_inputs(segment, int(entry["segment_index"]))
                        cost = int(desired["local_stats"].get("speech_frame_count", 0)) if desired else 0
                        self._add_budget_failure(entry, seq, cost, pending=True)
                denied_remaining = True
                pending_ids = []
                break
            pending_ids = [segment_id for segment_id in pending_ids if segment_id not in self.resolved]
        self.todo = sorted(
            [
                segment_id
                for segment_id, mapping in self.mappings.items()
                if mapping.get("mapping_status") == "pending" and segment_id in self.denied
            ],
            key=lambda segment_id: int(self.mappings[segment_id]["segment_index"]),
        )

    # ------------------------------------------------------------------
    # Assembly and snapshots
    # ------------------------------------------------------------------
    def _assemble(self) -> None:
        included: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        ranges: list[dict[str, Any]] = []
        chunks: list[bytes] = []
        offset = 0
        for index, segment in enumerate(self.segments):
            mapping = self.mappings.get(segment.segment_id)
            pcm = self.pcm_by_segment.get(segment.segment_id)
            if mapping is None or mapping.get("mapping_status") != "mapped" or pcm is None:
                status = (mapping or {}).get("mapping_status", "missing")
                skipped.append(
                    {
                        "segment_id": segment.segment_id,
                        "segment_index": index,
                        "reason": status,
                    }
                )
                continue
            sample_count = len(pcm) // 2
            item = {
                "segment_id": segment.segment_id,
                "segment_index": index,
                "mapping_key": mapping["mapping_key"],
                "segment_fingerprint": mapping["segment_fingerprint"],
                "pcm_sha256": mapping["pcm_sha256"],
                "sample_count": sample_count,
            }
            included.append(item)
            ranges.append(
                {
                    "segment_id": segment.segment_id,
                    "segment_index": index,
                    "sample_offset": offset,
                    "sample_count": sample_count,
                    "segment_fingerprint": mapping["segment_fingerprint"],
                }
            )
            chunks.append(pcm)
            offset += sample_count
        assembly_pcm = b"".join(chunks)
        assembly_key = sha256_value(
            {
                "stage": "assembly",
                "config": self.cfg.assembly_fingerprint(),
                "included": [
                    {"segment_index": item["segment_index"], "mapping_key": item["mapping_key"]}
                    for item in included
                ],
            }
        )
        assembly = {
            "status": "complete" if len(included) == len(self.segments) else "partial",
            "assembly_key": assembly_key,
            "assembly_fingerprint": sha256_value(
                {
                    "assembly_key": assembly_key,
                    "segment_fingerprints": [item["segment_fingerprint"] for item in included],
                }
            ),
            "fingerprint": sha256_bytes(assembly_pcm),
            "pcm_sha256": sha256_bytes(assembly_pcm),
            "sample_count": len(assembly_pcm) // 2,
            "included": included,
            "skipped": skipped,
            "segment_ranges": ranges,
        }
        self.assembly = assembly
        self.assembly_pcm = assembly_pcm

    def _mapping_artifact(self) -> dict[str, Any]:
        segments: dict[str, Any] = {}
        for segment_id, mapping in sorted(self.mappings.items(), key=lambda item: int(item[1].get("segment_index", 0))):
            stored = {key: value for key, value in mapping.items() if key != "pcm"}
            if stored.get("mapping_status") == "mapped":
                stored["piece_file"] = f"pieces/{segment_id}.pcm.z"
            segments[segment_id] = stored
        return {
            "segments": segments,
            "decision": copy.deepcopy(self.decision),
            "params": copy.deepcopy(self.params),
            "segment_order": [segment.segment_id for segment in self.segments],
        }

    def _manifest_core(self, *, active_seq: int | None, command_complete: bool, global_version: int) -> dict[str, Any]:
        return {
            "format_version": self.cfg.format_version,
            "engine_version": self.cfg.engine_version,
            "schema_version": 1,
            "config_hash": self.cfg.config_hash(),
            "audio_fingerprint": self.audio_fingerprint,
            "command_stream_hash": self.plan.command_stream_hash(),
            "command_prefix_hash": self.plan.command_prefix_hash(self.command_cursor if active_seq is None else active_seq),
            "command_count": len(self.plan.commands),
            "command_cursor": int(self.command_cursor),
            "global_version": int(global_version),
            "command_complete": bool(command_complete),
            "active_command_seq": active_seq,
            "params": copy.deepcopy(self.params),
            "segments": [segment.as_dict() for segment in self.segments],
            "consumed_frame_budget": sum(int(charge["frame_count"]) for charge in self.charges),
            "charges": copy.deepcopy(self.charges),
            "todo": list(self.todo),
            "todo_cursor": min(
                (int(self.mappings[segment_id]["segment_index"]) for segment_id in self.todo if segment_id in self.mappings),
                default=None,
            ),
        }

    def _commit(self, *, active_seq: int | None, command_complete: bool, global_version: int) -> None:
        self._assemble()
        analysis_artifact = {
            "reference_analyses": self.ref_analyses,
            "segment_analyses": self.segment_analyses,
            "analysis_config_fingerprint": self.cfg.analysis_fingerprint(),
        }
        stats_artifact = {
            "speakers": self.speaker_stats,
            "local": self.local_stats,
            "stats_config_fingerprint": self.cfg.stats_fingerprint(),
            "target_speaker": self.params["target_speaker"],
        }
        mappings_artifact = self._mapping_artifact()
        ledger = {
            "charges": copy.deepcopy(self.charges),
            "decision": copy.deepcopy(self.decision),
            "failures": sorted_failures(self.current_failures),
            "todo": list(self.todo),
            "todo_cursor": min(
                (int(self.mappings[segment_id]["segment_index"]) for segment_id in self.todo if segment_id in self.mappings),
                default=None,
            ),
            "resolved": sorted(self.resolved),
            "denied": sorted(self.denied),
            "command_failure_history": copy.deepcopy(self.command_failure_history),
            "command_seq": active_seq if active_seq is not None else (self.command_cursor if self.command_cursor else None),
            "command_complete": bool(command_complete),
        }
        manifest_core = self._manifest_core(
            active_seq=active_seq,
            command_complete=command_complete,
            global_version=global_version,
        )
        state = commit_snapshot(
            self.state_root,
            manifest_core=manifest_core,
            analysis=analysis_artifact,
            stats=stats_artifact,
            mappings=mappings_artifact,
            assembly=self.assembly,
            ledger=ledger,
            pcm_by_segment=self.pcm_by_segment,
            assembly_pcm=self.assembly_pcm,
        )
        self.latest_state = state

    def _maybe_crash_checkpoint(self) -> None:
        target = self.options.crash_on_checkpoint
        if target is not None and self.checkpoint_count == target:
            os._exit(137)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def _refresh_failures_for_command(self, seq: int) -> None:
        self.current_failures = self._catalog_failures_for_command(seq)
        self.current_failures.extend(self._failure_for_analysis_or_overflow())

    def _apply_new_command(self, seq: int, command: dict[str, Any]) -> None:
        command_error: str | None = None
        command_seq = command_sequence(command)
        if not isinstance(command_seq, int) or isinstance(command_seq, bool) or command_seq != seq:
            command_error = f"expected command seq {seq}, got {command_seq!r}"
        else:
            try:
                new_params, new_segments = apply_command(self.params, self.segments, command, self.cfg)
                self.params = new_params
                self.segments = new_segments
            except CommandError as exc:
                command_error = exc.detail
            except Exception as exc:
                command_error = f"command application failed: {exc}"
        if command_error is not None:
            failure = Failure(
                code=E_COMMAND_INVALID,
                stage="commands",
                detail=command_error,
                command_seq=seq,
                exception_type="CommandError",
            ).as_dict()
            self.current_failures.append(failure)
            self.command_failure_history.append(dict(failure))
        self.active_seq = seq

    def _process_command(self, seq: int, *, resume: bool) -> None:
        prior_assembly_key = str(self.assembly.get("assembly_key", ""))
        if not resume:
            command = self.plan.commands[seq - 1]
            self.current_failures = self._catalog_failures_for_command(seq)
            self._apply_new_command(seq, command)
            self.audio_fingerprint = self.catalog.fingerprint_for_segments(self.segments)
            self.decision = {"reuse": [], "recompute": []}
            self.resolved = set()
            self.denied = set()
            self.todo = []
        else:
            self.active_seq = seq
            # Retain resolved-stage failures from the interrupted command and rebuild
            # the deterministic, recomputable stage failures.
            retained = [
                dict(failure)
                for failure in self.current_failures
                if failure.get("stage") in {"commands", "scheduler", "mapping", "synthesis"}
            ]
            self.current_failures = self._catalog_failures_for_command(seq)
            self.current_failures.extend(retained)

        self._refresh_analysis_stage(seq)
        preserved_command_failures = [
            dict(failure) for failure in self.current_failures if failure.get("stage") == "commands"
        ]
        self._refresh_failures_for_command(seq)
        self.current_failures.extend(preserved_command_failures)
        if resume:
            retained = [
                dict(failure)
                for failure in (self.ledger.get("failures", []) if self.ledger else [])
                if failure.get("stage") in {"commands", "scheduler", "mapping", "synthesis"}
            ]
            self.current_failures.extend(retained)
        self._refresh_stats_stage()
        self._prepare_decisions(seq, resume=resume)
        self._run_mapping_tasks(seq, resume=resume)

        # The final assembly belongs to exactly this completed command.
        old_key = self.previous_assembly_key
        self._assemble()
        self.command_cursor = seq
        self.global_version = seq
        self.active_seq = None
        self.todo = sorted(
            [
                segment_id
                for segment_id, mapping in self.mappings.items()
                if mapping.get("mapping_status") == "pending"
            ],
            key=lambda segment_id: int(self.mappings[segment_id].get("segment_index", 0)),
        )
        self._commit(active_seq=None, command_complete=True, global_version=seq)
        self._last_assembly_previous_key = prior_assembly_key
        self.previous_assembly_key = str(self.assembly.get("assembly_key", ""))
        if self.options.crash_after_command is not None and seq == self.options.crash_after_command:
            os._exit(137)

    # ------------------------------------------------------------------
    # Report and public run loop
    # ------------------------------------------------------------------
    def request_stop(self, *_args: Any) -> None:
        self.stop_requested = True
        self.graceful_signal = True

    def _assembly_decision(self) -> dict[str, Any]:
        key = str(self.assembly.get("assembly_key", ""))
        previous = getattr(self, "_last_assembly_previous_key", self.previous_assembly_key)
        return {
            "assembly_key": key,
            "decision": "reused" if key and key == previous else "recomputed",
        }

    def _budget_report(self) -> dict[str, Any]:
        budget = self.cfg.speech_frame_budget
        consumed = sum(int(charge["frame_count"]) for charge in self.charges)
        replay_consumed = 0
        for charge in sorted(self.charges, key=lambda item: int(item.get("ordinal", 0))):
            replay_consumed += int(charge["frame_count"])
        expected_todo = sorted(
            [
                segment_id
                for segment_id, mapping in self.mappings.items()
                if mapping.get("mapping_status") == "pending"
            ],
            key=lambda segment_id: int(self.mappings[segment_id].get("segment_index", 0)),
        )
        todo = list(self.todo)
        return {
            "budget": budget,
            "charges": copy.deepcopy(self.charges),
            "consumed": consumed,
            "remaining": budget - consumed,
            "replay_consumed": replay_consumed,
            "replay_remaining": budget - replay_consumed,
            "replay_fingerprint": sha256_value(self.charges),
            "replay_ok": replay_consumed == consumed,
            "todo": todo,
            "todo_cursor": min(
                (int(self.mappings[segment_id]["segment_index"]) for segment_id in todo if segment_id in self.mappings),
                default=None,
            ),
            "todo_replay_ok": expected_todo == todo,
        }

    def _segment_failure_map(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for failure in sorted_failures(self.current_failures):
            segment_id = failure.get("segment_id")
            if segment_id and segment_id not in result:
                result[str(segment_id)] = failure
        return result

    def _segments_report(self) -> list[dict[str, Any]]:
        decision_by_segment: dict[str, dict[str, Any]] = {}
        for kind in ("reuse", "recompute"):
            for entry in self.decision.get(kind, []):
                item = dict(entry)
                item["decision"] = kind
                decision_by_segment[str(entry["segment_id"])] = item
        failure_by_segment = self._segment_failure_map()
        report: list[dict[str, Any]] = []
        for index, segment in enumerate(self.segments):
            analysis = self.segment_analyses.get(segment.segment_id, {})
            mapping = self.mappings.get(segment.segment_id)
            local = self.local_stats.get(segment.segment_id, {})
            item: dict[str, Any] = {
                "segment_id": segment.segment_id,
                "index": index,
                "path": segment.path,
                "start_sample": segment.start_sample,
                "end_sample": segment.end_sample,
                "analysis_status": analysis.get("analysis_status", "missing"),
                "analysis_key": analysis.get("analysis_key"),
                "sample_count": analysis.get("sample_count"),
                "source_frame_count": analysis.get("frame_count", 0),
                "source_frame_set": [
                    f"{segment.segment_id}:{frame_index:08d}" for frame_index in range(int(analysis.get("frame_count", 0)))
                ],
                "speech_frame_count": int(local.get("speech_frame_count", 0)),
                "local_stats_key": local.get("cache_key"),
                "stats_fingerprint": local.get("fingerprint"),
                "mapping_decision": decision_by_segment.get(segment.segment_id),
                "failure": failure_by_segment.get(segment.segment_id),
            }
            if mapping is not None:
                item["mapping_status"] = mapping.get("mapping_status")
                if mapping.get("mapping_status") == "mapped":
                    public = public_mapping(mapping)
                    item.update(public)
                    item["mapping_status"] = "mapped"
                else:
                    item["mapping_key"] = mapping.get("mapping_key")
            report.append(item)
        return report

    def _reference_report(self) -> list[dict[str, Any]]:
        speakers = self.catalog.reference_speakers()
        speaker_by_path = {path: speaker for speaker, paths in speakers.items() for path in paths}
        result: list[dict[str, Any]] = []
        for rel in sorted(self.catalog.references):
            analysis = self.ref_analyses.get(rel, {})
            result.append(
                {
                    "path": rel,
                    "speaker": speaker_by_path.get(rel),
                    "status": analysis.get("analysis_status", "missing"),
                    "analysis_key": analysis.get("analysis_key"),
                    "content_fingerprint": self.catalog.references[rel].content_fingerprint,
                    "failure": analysis.get("failure"),
                }
            )
        return result

    def _report(self, *, recovered: bool, recovered_point: dict[str, Any], run_status: str) -> dict[str, Any]:
        failures = sorted_failures(self.current_failures + self.command_failure_history)
        exit_code = 0 if (not failures or self.graceful_signal) else 4
        output_path = (self.options.output_path or (self.state_root.parent / "converted.wav")).resolve()
        report_path = (self.options.report_path or (self.state_root.parent / "run_report.json")).resolve()
        command_seq = self.command_cursor if self.command_cursor else self.active_seq
        return {
            "format_version": self.cfg.format_version,
            "run_status": run_status,
            "exit_code": exit_code,
            "command_count": len(self.plan.commands),
            "command_cursor": self.command_cursor,
            "command_complete": self.active_seq is None,
            "active_command_seq": self.active_seq,
            "command_seq": command_seq,
            "global_version": self.global_version,
            "config_hash": self.cfg.config_hash(),
            "audio_fingerprint": self.audio_fingerprint,
            "command_prefix_hash": self.plan.command_prefix_hash(self.command_cursor),
            "params": params_public(self.params, self.cfg),
            "segments": self._segments_report(),
            "reference_analysis": self._reference_report(),
            "speaker_stats": self.speaker_stats,
            "reuse": copy.deepcopy(self.decision.get("reuse", [])),
            "recompute": copy.deepcopy(self.decision.get("recompute", [])),
            "assembly": copy.deepcopy(self.assembly),
            "assembly_decision": self._assembly_decision(),
            "failures": failures,
            "budget": self._budget_report(),
            "stage_versions": dict((self.latest_state.stage_versions if self.latest_state else {})),
            "recovery": {"recovered": recovered, "recovery_point": recovered_point},
            "startup_events": copy.deepcopy(self.startup_events),
            "diagnostics": {
                "configured_thread_pool_workers": self.cfg.thread_pool_workers,
                "checkpoint_count": self.checkpoint_count,
            },
            "output": {
                "path": str(output_path),
                "sample_count": len(self.assembly_pcm) // 2,
                "sha256": sha256_bytes(self.assembly_pcm),
                "written": False,
            },
            "report_path": str(report_path),
        }

    def run(self) -> dict[str, Any]:
        recovered_point = recovery_point(self.latest_state) if self.recovered else {"present": False, "recoverable": False}
        resume_active = self.active_seq is not None
        while True:
            if self.stop_requested:
                break
            if self.options.stop_after_command is not None and self.command_cursor >= self.options.stop_after_command:
                break
            if resume_active:
                if self.active_seq is None:
                    resume_active = False
                    continue
                seq = int(self.active_seq)
                self._process_command(seq, resume=True)
                resume_active = False
                continue
            if self.command_cursor >= len(self.plan.commands):
                break
            seq = self.command_cursor + 1
            self._process_command(seq, resume=False)
        run_status = "completed" if self.command_cursor >= len(self.plan.commands) else "checkpoint"
        return self._report(recovered=self.recovered, recovered_point=recovered_point, run_status=run_status)
