"""End-to-end concurrent retrieval engine and deterministic JSON reporting."""

from __future__ import annotations

import hashlib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json, atomic_write_json_group
from .errors import BusinessError
from .fingerprint import _round_half_up, fingerprint_audio
from .index import ReferenceIndex
from .models import (
    TaskSpec,
    TrackFingerprint,
    TrackSpec,
)
from .retrieval import fragment_to_dict, retrieve_similar_fragments
from .runtime import CancelToken, FrameBudget
from .task import load_task, merge_fingerprint_override
from .wavio import read_wav


RECOVERABLE_TRACK_CODES = {"E_TRUNC", "E_SILENT", "E_RATE", "E_BAND", "E_WAV", "E_IO"}


def _track_error(track: TrackSpec, error: BusinessError) -> dict[str, Any]:
    return {
        "id": track.track_id,
        "role": track.role,
        "path": str(track.path),
        "business_code": error.code,
        "message": error.message,
        "details": error.details,
    }


def _fingerprint_summary(fingerprint: TrackFingerprint, include_frame_map: bool = True) -> dict[str, Any]:
    result = {
        "id": fingerprint.track_id,
        "role": fingerprint.role,
        "path": str(fingerprint.path),
        "business_code": "E_OK",
        "sample_rate": fingerprint.sample_rate,
        "duration_samples": fingerprint.duration_samples,
        "frame_samples": fingerprint.frame_samples,
        "hop_samples_rounded": fingerprint.hop_samples_rounded,
        "frame_count": fingerprint.frame_count,
        "peak_count": len(fingerprint.peaks),
        "fingerprint_count": len(fingerprint.items),
        "time_peak_filter_drops": fingerprint.dropped_peaks_by_band_cap,
    }
    if include_frame_map:
        result["frame_map"] = [frame.to_dict() for frame in fingerprint.frames]
    return result


def _parameters(task: TaskSpec, capacity_frames: int, spectrum_frame_bytes: int) -> dict[str, Any]:
    return {
        "allowed_sample_rates": list(task.allowed_sample_rates),
        "fingerprint": {
            "frame_length": task.fingerprint.frame_length,
            "hop_length": task.fingerprint.hop_length,
            "base_sample_rate": task.fingerprint.base_sample_rate,
            "bands": [[low, high] for low, high in task.fingerprint.bands],
            "peak_neighborhood": {
                "frequency_bins": task.fingerprint.freq_neighbor_bins,
                "time_frames": task.fingerprint.time_neighbor_frames,
            },
            "min_peak_db": task.fingerprint.min_peak_db,
            "fanout": task.fingerprint.fanout,
            "max_pair_delta_frames": task.fingerprint.max_pair_delta_frames,
            "peaks_per_band": task.fingerprint.peaks_per_band,
            "window": task.fingerprint.window,
        },
        "retrieval": {
            "bucket_capacity": task.retrieval.bucket_capacity,
            "voting_window_frames": task.retrieval.voting_window_frames,
            "min_matches": task.retrieval.min_matches,
            "max_candidates": task.retrieval.max_candidates,
            "overlap_policy": task.retrieval.overlap_policy,
            "boundary_gap_frames": task.retrieval.boundary_gap_frames,
            "boundary_padding_frames": task.retrieval.boundary_padding_frames,
        },
        "tolerance": {
            "score_scale": task.tolerance.score_scale,
            "score_tolerance": task.tolerance.score_tolerance,
            "time_offset_tolerance_samples": task.tolerance.time_offset_tolerance_samples,
        },
        "concurrency": {
            "max_workers": task.concurrency.max_workers,
            "spectrum_buffer_bytes": task.concurrency.spectrum_buffer_bytes,
            "effective_spectrum_frames": capacity_frames,
            "spectrum_frame_bytes_conservative": spectrum_frame_bytes,
        },
    }


def _state_document(
    task: TaskSpec,
    status: str,
    code: str,
    *,
    completed_references: int = 0,
    completed_queries: int = 0,
    message: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "business_code": code,
        "message": message,
        "task_path": str(task.task_path),
        "task_sha256": hashlib.sha256(task.task_path.read_bytes()).hexdigest(),
        "completed_references": completed_references,
        "completed_queries": completed_queries,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_state(task: TaskSpec, document: dict[str, Any]) -> None:
    try:
        task.state_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(task.state_dir / "status.json", document)
    except (OSError, BusinessError) as exc:
        raise BusinessError("E_WR", "unable to update job state", {"reason": str(exc)}) from exc


def _prepare_budget(task: TaskSpec) -> tuple[FrameBudget, int, int]:
    maximum_rate = max(task.allowed_sample_rates)
    frame_samples = _round_half_up(Fraction(task.fingerprint.frame_length * maximum_rate, task.fingerprint.base_sample_rate))
    fft_bins = frame_samples // 2 + 1
    spectrum_frame_bytes = fft_bins * 8
    capacity_frames = max(1, task.concurrency.spectrum_buffer_bytes // spectrum_frame_bytes)
    return FrameBudget(capacity_frames), capacity_frames, spectrum_frame_bytes


def _run_query(
    task: TaskSpec,
    track: TrackSpec,
    priority: int,
    budget: FrameBudget,
    token: CancelToken,
) -> TrackFingerprint:
    token.raise_if_cancelled()
    config = merge_fingerprint_override(task.fingerprint, track.fingerprint_override)
    audio = read_wav(
        track.path,
        track_id=track.track_id,
        role="query",
        allowed_sample_rates=task.allowed_sample_rates,
        declared_sample_rate=track.declared_sample_rate,
    )
    return fingerprint_audio(audio, config, budget=budget, priority=priority, token=token)


def _build_results_document(
    task: TaskSpec, pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    deterministic_pairs = []
    for pair in pairs:
        compact_fragments = []
        for fragment in pair["fragments"]:
            compact = {key: value for key, value in fragment.items() if key != "score_trace"}
            compact_fragments.append(compact)
        deterministic_pairs.append(
            {
                "reference_id": pair["reference_id"],
                "query_id": pair["query_id"],
                "business_code": pair["business_code"],
                "fragments": compact_fragments,
            }
        )
    return {
        "schema_version": 1,
        "scoring": {
            "ranking": "fixed_point_only",
            "score_scale": task.tolerance.score_scale,
            "score_tolerance": task.tolerance.score_tolerance,
        },
        "pairs": deterministic_pairs,
    }


def run_task(
    task_or_path: TaskSpec | str | Path,
    *,
    cancel_token: CancelToken | None = None,
    include_frame_map: bool = True,
) -> dict[str, Any]:
    token = cancel_token or CancelToken()
    task = task_or_path if isinstance(task_or_path, TaskSpec) else load_task(task_or_path)
    token.raise_if_cancelled()
    _write_state(task, _state_document(task, "running", "E_OK", message="job started"))
    budget, capacity_frames, spectrum_frame_bytes = _prepare_budget(task)
    reference_tracks: list[dict[str, Any]] = []
    query_tracks: list[dict[str, Any]] = []
    index = ReferenceIndex(task.retrieval.bucket_capacity)
    try:
        for order, track in enumerate(task.references):
            token.raise_if_cancelled()
            try:
                config = merge_fingerprint_override(task.fingerprint, track.fingerprint_override)
                audio = read_wav(
                    track.path,
                    track_id=track.track_id,
                    role="reference",
                    allowed_sample_rates=task.allowed_sample_rates,
                    declared_sample_rate=track.declared_sample_rate,
                )
                fingerprint = fingerprint_audio(
                    audio, config, budget=budget, priority=order, token=token
                )
            except BusinessError as exc:
                _write_state(
                    task,
                    _state_document(task, "error", exc.code, message=f"reference {track.track_id} failed"),
                )
                raise
            index.add_fingerprint(fingerprint, reference_order=order)
            reference_tracks.append(_fingerprint_summary(fingerprint, include_frame_map))

        query_fingerprints: dict[str, TrackFingerprint] = {}
        query_errors: dict[str, BusinessError] = {}
        fatal_error: BusinessError | None = None
        fatal_index: int | None = None

        def capture_future_error(future: Future[TrackFingerprint], index_value: int) -> None:
            nonlocal fatal_error, fatal_index
            try:
                exception = future.exception()
            except Exception as exc:  # pragma: no cover - defensive
                exception = exc
            if exception is None:
                return
            if isinstance(exception, BusinessError) and exception.code in RECOVERABLE_TRACK_CODES:
                query_errors[task.queries[index_value].track_id] = exception
                return
            if fatal_error is None or fatal_index is None or index_value < fatal_index:
                fatal_index = index_value
                fatal_error = (
                    exception
                    if isinstance(exception, BusinessError)
                    else BusinessError("E_INTERNAL", "query processing failed", {"reason": str(exception)})
                )
                token.cancel(fatal_error.message)

        with ThreadPoolExecutor(max_workers=task.concurrency.max_workers, thread_name_prefix="audiofp") as pool:
            futures: dict[int, Future[TrackFingerprint]] = {}
            for query_index, track in enumerate(task.queries):
                if token.cancelled:
                    break
                futures[query_index] = pool.submit(
                    _run_query, task, track, query_index, budget, token
                )
            for future in as_completed(futures.values()):
                index_value = next(i for i, candidate in futures.items() if candidate is future)
                capture_future_error(future, index_value)
                if fatal_error is not None:
                    for pending in futures.values():
                        pending.cancel()
                    break
            for index_value, future in futures.items():
                if future.done() and not future.cancelled():
                    capture_future_error(future, index_value)
                    if future.exception() is None:
                        result = future.result()
                        query_fingerprints[result.track_id] = result
        if fatal_error is not None:
            _write_state(
                task,
                _state_document(task, "error", fatal_error.code, message=fatal_error.message),
            )
            raise fatal_error
        if token.cancelled:
            raise BusinessError("E_INTERRUPTED", token.reason or "job interrupted")

        for query in task.queries:
            error = query_errors.get(query.track_id)
            if error is not None:
                query_tracks.append(_track_error(query, error))
            else:
                fingerprint = query_fingerprints[query.track_id]
                query_tracks.append(_fingerprint_summary(fingerprint, include_frame_map))

        pairs: list[dict[str, Any]] = []
        for query in task.queries:
            error = query_errors.get(query.track_id)
            if error is not None:
                for reference in task.references:
                    pairs.append(
                        {
                            "reference_id": reference.track_id,
                            "query_id": query.track_id,
                            "business_code": error.code,
                            "fragments": [],
                        }
                    )
                continue
            fingerprint = query_fingerprints[query.track_id]
            fragments = retrieve_similar_fragments(
                index,
                fingerprint,
                task.retrieval,
                task.tolerance,
                task.fingerprint.base_sample_rate,
                task.fingerprint.hop_length,
            )
            for reference in task.references:
                selected = []
                for fragment in fragments:
                    if fragment.reference_id == reference.track_id:
                        selected.append(fragment_to_dict(fragment))
                pairs.append(
                    {
                        "reference_id": reference.track_id,
                        "query_id": query.track_id,
                        "business_code": "E_OK",
                        "fragments": selected,
                    }
                )
        pairs.sort(key=lambda p: (p["query_id"], p["reference_id"]))

        injection_audit: list[dict[str, Any]] = []
        for injection in task.injections:
            actual = "E_OK"
            for track in query_tracks:
                if track["id"] == injection.target_id:
                    actual = track["business_code"]
                    break
            if actual == "E_OK":
                for track in reference_tracks:
                    if track["id"] == injection.target_id:
                        actual = track["business_code"]
                        break
            injection_audit.append(
                {
                    "target_id": injection.target_id,
                    "kind": injection.kind,
                    "expected_code": injection.expected_code,
                    "actual_code": actual,
                    "matched": actual == injection.expected_code,
                    "details": injection.details,
                }
            )

        has_errors = bool(query_errors) or any(not item["matched"] for item in injection_audit)
        top_code = "E_PARTIAL" if has_errors else "E_OK"
        results_document = _build_results_document(task, pairs)
        report = {
            "schema_version": 1,
            "business_code": top_code,
            "status": "complete_with_track_errors" if has_errors else "complete",
            "task_path": str(task.task_path),
            "parameters": _parameters(task, capacity_frames, spectrum_frame_bytes),
            "tracks": {"references": reference_tracks, "queries": query_tracks},
            "pairs": pairs,
            "injections": injection_audit,
            "index": index.stats(),
            "resource_control": {
                "max_workers": task.concurrency.max_workers,
                "in_flight_spectrum_capacity_frames": budget.stats.capacity_frames,
                "peak_in_flight_spectrum_frames": budget.stats.peak_in_flight_frames,
                "spectrum_budget_wait_count": budget.stats.total_waits,
                "spectrum_budget_acquisitions": budget.stats.acquisitions,
                "budget_exceeded": budget.stats.peak_in_flight_frames > budget.stats.capacity_frames,
            },
            "outputs": {
                "report": str(task.report_path),
                "results": str(task.results_path),
            },
        }
        token.raise_if_cancelled()
        atomic_write_json_group({task.report_path: report, task.results_path: results_document})
        _write_state(
            task,
            _state_document(
                task,
                "complete",
                top_code,
                completed_references=len(reference_tracks),
                completed_queries=len(task.queries),
            ),
        )
        return report
    except BusinessError as exc:
        if exc.code == "E_INTERRUPTED":
            _write_state(task, _state_document(task, "interrupted", "E_INTERRUPTED", message=exc.message))
        elif exc.code != "E_WR":
            _write_state(task, _state_document(task, "error", exc.code, message=exc.message))
        raise
    except Exception as exc:
        wrapped = BusinessError("E_INTERNAL", "unexpected engine failure", {"reason": str(exc)})
        try:
            _write_state(task, _state_document(task, "error", "E_INTERNAL", message=str(exc)))
        finally:
            raise wrapped from exc
