from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from .config import (
    ExchangeConfig,
    MatchConfig,
    DetectionParams,
    load_detection_params,
    load_exchange_config,
    load_match_config,
)
from .conversion import run_conversion
from .errors import (
    E_IMAGE_INVALID,
    E_MATCH_INVALID,
    E_SERIALIZE_INVALID,
    BusinessError,
    as_business_error,
)
from .models import FailureRecord, Track, UnifiedFrame
from .matching import (
    match_series,
    recompute_pool_metadata,
    refresh_track_metadata,
    track_reusable,
)
from .snapshot import SnapshotStore
from .state import (
    failure_from_state,
    failure_to_state,
    frame_from_state,
    frame_to_state,
    load_sidecar_fingerprints,
    track_to_state,
)
from .unify import discover_series_images, frame_cache_key, unify_series
from .util import (
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    file_sha256,
    read_json,
    sanitize_for_json,
    stable_hash,
)
from .v1 import parse_v1
from .v2 import parse_v2


T = TypeVar("T")
R = TypeVar("R")


class SimulatedTermination(RuntimeError):
    """Raised after a durable stage commit to emulate an abrupt process stop."""

    def __init__(self, stage: str, version: int):
        super().__init__(f"simulated termination after {stage} stage at version {version}")
        self.stage = stage
        self.version = version


@dataclass(frozen=True)
class PipelineOptions:
    data_dir: Path
    output_dir: Path
    snapshot_root: Path
    active_version: str | None = None
    workers: int | None = None
    snapshot_commit_interval: int | None = None
    crash_after_stage: str | None = None
    force_full: bool = False
    injections_path: Path | None = None


def _chunked(values: list[T], size: int) -> Iterable[list[T]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _run_chunks(
    values: list[T], *, workers: int, chunk_size: int, function: Callable[[T], R]
) -> list[R]:
    result: list[R] = []
    for chunk in _chunked(values, chunk_size):
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk))) as executor:
            result.extend(executor.map(function, chunk))
    return result


def _failure_sort_key(failure: FailureRecord) -> tuple[Any, ...]:
    return failure.sort_key()


def _dedupe_failures(failures: list[FailureRecord]) -> list[FailureRecord]:
    unique: dict[str, FailureRecord] = {}
    for failure in failures:
        payload = failure.to_dict()
        unique[stable_hash(payload)] = failure
    return sorted(unique.values(), key=_failure_sort_key)


def _load_injections(path: Path | None, data_dir: Path) -> dict[str, Any]:
    selected = path
    if selected is None:
        candidate = data_dir / "active_injections.json"
        selected = candidate if candidate.exists() else None
    if selected is None or not selected.exists():
        return {"schema": "candidate-exchange-injections-v1", "injections": []}
    payload = read_json(selected)
    if not isinstance(payload, dict) or not isinstance(payload.get("injections", []), list):
        raise BusinessError(
            "E_CONFIG_INVALID",
            "injection config must contain an injections list",
            "config",
        )
    return payload


def _injection_items(injections: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (item for item in injections.get("injections", []) if isinstance(item, dict)),
        key=lambda item: str(item.get("id", "")),
    )


def _is_forced_injection(injection: dict[str, Any], kind: str) -> bool:
    return injection.get("type") == kind


def _forced_failures(
    injections: list[dict[str, Any]],
    *,
    kind: str,
    stage: str,
    code: str,
) -> list[FailureRecord]:
    failures: list[FailureRecord] = []
    for injection in injections:
        if not _is_forced_injection(injection, kind):
            continue
        failures.append(
            FailureRecord(
                series_name=str(injection.get("series_name", "<unknown>")),
                frame_index=int(injection["frame_index"])
                if isinstance(injection.get("frame_index"), int)
                else None,
                stage=stage,
                error_code=code,
                exception_type="InjectedFailure",
                message=f"injected runtime failure for {kind}",
                injection_id=str(injection.get("id")) if injection.get("id") is not None else None,
                details={"injection_type": kind},
            )
        )
    return failures


def _mark_frame_failed(
    frames: dict[str, list[UnifiedFrame]], series_name: str, frame_index: int
) -> None:
    for frame in frames.get(series_name, []):
        if frame.frame_index == frame_index:
            frame.status = "failed"
            frame.candidates = []


def _frame_key_map(frames: Iterable[UnifiedFrame]) -> dict[int, str]:
    return {frame.frame_index: frame.candidate_key for frame in frames}


def _status_failure(code: str, stage: str, series_name: str, frame_index: int | None, message: str, exc: BaseException) -> FailureRecord:
    error = as_business_error(
        exc if isinstance(exc, BusinessError) else BusinessError(
            code=code,
            message=message,
            stage=stage,
            series_name=series_name,
            frame_index=frame_index,
        ),
        stage=stage,
        series_name=series_name,
        frame_index=frame_index,
    )
    return FailureRecord(
        series_name=error.series_name or series_name,
        frame_index=error.frame_index,
        stage=error.stage,
        error_code=error.code,
        exception_type=error.details.get("exception_type", exc.__class__.__name__),
        message=error.message,
        details={key: value for key, value in error.details.items() if key != "exception_type"},
    )


def _prune_failed_tracks(
    tracks: list[Track],
    failed_keys: set[tuple[str, int]],
    *,
    match: MatchConfig,
    frame_keys_by_series: dict[str, dict[int, str]],
) -> tuple[list[Track], list[str]]:
    changed: list[str] = []
    result: list[Track] = []
    for track in tracks:
        kept = [
            frame
            for frame in track.frames
            if (track.series_name, frame.frame_index) not in failed_keys
        ]
        if not kept:
            changed.append(track.track_id)
            continue
        if len(kept) != len(track.frames):
            track.frames = kept
            track.reused = False
            refresh_track_metadata(
                track,
                match=match,
                frame_keys_by_index=frame_keys_by_series.get(track.series_name, {}),
            )
            changed.append(track.track_id)
        result.append(track)
    result.sort(key=lambda item: item.sort_key())
    return result, sorted(changed)


def _assign_injections(
    failures: list[FailureRecord], injections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    available = list(failures)
    hits: list[dict[str, Any]] = []
    for injection in injections:
        injection_id = str(injection.get("id"))
        preassigned = next(
            (
                failure
                for failure in available
                if failure.injection_id == injection_id
            ),
            None,
        )
        if preassigned is not None:
            available.remove(preassigned)
            hits.append(
                {
                    "id": injection_id,
                    "type": injection.get("type"),
                    "series_name": injection.get("series_name"),
                    "frame_index": injection.get("frame_index"),
                    "configured_stage": injection.get("stage"),
                    "expected_code": injection.get("expected_code"),
                    "hit": True,
                    "observed_error_code": preassigned.error_code,
                    "observed_stage": preassigned.stage,
                }
            )
            continue
        series_name = injection.get("series_name")
        frame_index = injection.get("frame_index")
        expected_code = injection.get("expected_code")
        selected: FailureRecord | None = None
        for failure in available:
            if failure.injection_id is not None:
                continue
            if series_name is not None and failure.series_name != series_name:
                continue
            if frame_index is not None and failure.frame_index != frame_index:
                continue
            if expected_code and failure.error_code != expected_code:
                continue
            selected = failure
            break
        if selected is not None:
            selected.injection_id = injection_id
            available.remove(selected)
        hits.append(
            {
                "id": injection_id,
                "type": injection.get("type"),
                "series_name": series_name,
                "frame_index": frame_index,
                "configured_stage": injection.get("stage"),
                "expected_code": expected_code,
                "hit": selected is not None,
                "observed_error_code": selected.error_code if selected else None,
                "observed_stage": selected.stage if selected else None,
            }
        )
    return hits


def _report_core(
    *,
    frames: dict[str, list[UnifiedFrame]],
    tracks: list[Track],
    failures: list[FailureRecord],
    match: MatchConfig,
    conversion: dict[str, Any] | None,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "active_version": report["active_version"],
        "frames": [
            {
                "series_name": frame.series_name,
                "frame_index": frame.frame_index,
                "status": frame.status,
                "candidate_key": frame.candidate_key,
                "candidates": [item.to_dict(match.fixed_decimal_places) for item in frame.candidates],
            }
            for series in sorted(frames)
            for frame in sorted(frames[series], key=lambda item: item.frame_index)
        ],
        "tracks": [
            {
                key: value
                for key, value in track.to_dict().items()
                if key not in {"reused"}
            }
            for track in tracks
        ],
        "failures": [failure.to_dict() for failure in failures],
        "conversion": conversion,
    }


def run_pipeline(options: PipelineOptions) -> dict[str, Any]:
    data_dir = options.data_dir.resolve()
    output_dir = ensure_dir(options.output_dir.resolve())
    snapshot_root = ensure_dir(options.snapshot_root.resolve())
    detection = load_detection_params(data_dir)
    match = load_match_config(data_dir)
    exchange = load_exchange_config(data_dir, options.active_version)
    workers = options.workers or match.concurrency
    if workers < 1:
        raise BusinessError("E_CONFIG_INVALID", "workers must be >= 1", "config")
    interval = options.snapshot_commit_interval or match.snapshot_commit_interval
    if interval < 1:
        raise BusinessError("E_CONFIG_INVALID", "snapshot interval must be >= 1", "config")
    active_version = exchange.active_version
    injections_payload = _load_injections(options.injections_path, data_dir)
    injections = _injection_items(injections_payload)

    v1_path = data_dir / "candidates_v1.txt"
    v2_path = data_dir / "candidates_v2.json"
    v1 = parse_v1(v1_path)
    v2 = parse_v2(v2_path)
    images = discover_series_images(data_dir / "images")
    sidecar_frame_fps, sidecar_hash = load_sidecar_fingerprints(data_dir)
    format_fingerprints = {
        "v1": v1.format_fingerprint,
        "v2": v2.format_fingerprint,
        "active": v1.format_fingerprint if active_version == "v1" else v2.format_fingerprint,
        "active_version": active_version,
    }
    format_fingerprint_active = format_fingerprints["active"]
    config_hash = stable_hash(
        {
            "detection": detection.raw,
            "match": match.raw,
            "exchange": exchange.raw,
            "active_version": active_version,
            "tool_schema": "candidate-exchange-pipeline-v1",
        }
    )
    input_fingerprint = stable_hash(
        {
            "v1_file_sha256": v1.file_sha256,
            "v2_file_sha256": v2.file_sha256,
            "images": {
                series: {str(index): file_sha256(path) for index, path in sorted(frame_map.items())}
                for series, frame_map in sorted(images.items())
            },
            "config_hash": config_hash,
        }
    )

    snapshot_store = SnapshotStore(snapshot_root)
    recovered = snapshot_store.recover()
    use_previous = not options.force_full and recovered.get("recovered", False)
    previous_detection = snapshot_store.latest_stage("detection") if use_previous else None
    previous_matching = snapshot_store.latest_stage("matching") if use_previous else None
    previous_frames: dict[tuple[str, int], UnifiedFrame] = {}
    previous_failures: list[FailureRecord] = []
    if previous_detection:
        for value in previous_detection.get("frames", []):
            frame = frame_from_state(value)
            previous_frames[(frame.series_name, frame.frame_index)] = frame
        previous_failures = [
            failure_from_state(value) for value in previous_detection.get("failures", [])
        ]
    previous_tracks: list[Track] = []
    previous_match_fingerprint: str | None = None
    if previous_matching:
        previous_match_fingerprint = previous_matching.get("match_fingerprint")
        previous_tracks = [
            Track.from_dict(value) for value in previous_matching.get("tracks", [])
        ]

    all_series = sorted(
        set(images)
        | {series for series, _ in v1.frames}
        | {series for series, _ in v2.frames}
        | {series for series, _ in previous_frames}
        | {
            error.series_name
            for error in v1.failures + v2.failures
            if error.series_name not in {None, "<file>", "<unknown>"}
        }
    )
    if not all_series:
        raise BusinessError(E_IMAGE_INVALID, "no input series found", "detection")

    frame_ledger: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    frames_by_series: dict[str, list[UnifiedFrame]] = {}

    def detect_series(series_name: str) -> tuple[str, list[UnifiedFrame], list[FailureRecord]]:
        series_frames, series_failures = unify_series(
            series_name=series_name,
            images=images.get(series_name, {}),
            v1=v1,
            v2=v2,
            detection=detection,
            match=match,
            active_version=active_version,
            sidecar_frame_fingerprints=sidecar_frame_fps,
        )
        for frame in series_frames:
            frame.candidate_key = frame_cache_key(
                frame,
                detection_fingerprint=detection.fingerprint,
                format_fingerprint=format_fingerprint_active,
                decimal_places=match.fixed_decimal_places,
                sidecar_fingerprint=sidecar_frame_fps.get(
                    (frame.series_name, frame.frame_index), "none"
                ),
            )
            prior = previous_frames.get((frame.series_name, frame.frame_index))
            if prior is not None and prior.candidate_key == frame.candidate_key:
                frame.candidates = prior.candidates
                frame.content_fingerprint = prior.content_fingerprint
                frame.status = prior.status
                frame_ledger.append(
                    {
                        "kind": "frame",
                        "series_name": frame.series_name,
                        "frame_index": frame.frame_index,
                        "action": "reused",
                        "cache_key": frame.candidate_key,
                        "previous_cache_key": prior.candidate_key,
                        "reason": "frame-cache-key-match",
                    }
                )
            else:
                frame_ledger.append(
                    {
                        "kind": "frame",
                        "series_name": frame.series_name,
                        "frame_index": frame.frame_index,
                        "action": "recomputed",
                        "cache_key": frame.candidate_key,
                        "previous_cache_key": prior.candidate_key if prior else None,
                        "reason": "new-frame" if prior is None else "frame-cache-key-changed",
                    }
                )
        for failure in series_failures:
            frame_ledger.append(
                {
                    "kind": "frame",
                    "series_name": failure.series_name,
                    "frame_index": failure.frame_index,
                    "action": "failed",
                    "cache_key": None,
                    "previous_cache_key": previous_frames.get(
                        (failure.series_name, failure.frame_index or -1), UnifiedFrame(
                            series_name=failure.series_name,
                            frame_index=failure.frame_index or -1,
                            image_path="",
                            image_sha256="",
                            image_size=(0, 0),
                            candidates=[],
                        )
                    ).candidate_key or None,
                    "reason": failure.error_code,
                }
            )
        return series_name, series_frames, series_failures

    def commit_stage(stage_name: str, payload: dict[str, Any], *, partial: bool, crash: bool) -> dict[str, Any]:
        manifest = snapshot_store.commit(
            stage_name=stage_name,
            payload=payload,
            config_hash=config_hash,
            format_fingerprints=format_fingerprints,
            sidecar_metadata_hash=sidecar_hash,
            input_fingerprint=input_fingerprint,
            snapshot_commit_interval=interval,
            partial=partial,
            extra={
                "active_version": active_version,
                "workers": workers,
                "series_count": len(all_series),
            },
        )
        commits.append(
            {
                "global_version": manifest["global_version"],
                "stage": stage_name,
                "partial": partial,
                "stages": manifest["complete_stages"],
            }
        )
        if crash:
            raise SimulatedTermination(stage_name, int(manifest["global_version"]))
        return manifest

    detection_chunks = list(_chunked(all_series, interval))
    for chunk_index, chunk in enumerate(detection_chunks):
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk))) as executor:
            chunk_results = list(executor.map(detect_series, chunk))
        for series_name, series_frames, series_failures in chunk_results:
            frames_by_series[series_name] = sorted(
                series_frames, key=lambda item: item.frame_index
            )
            failures.extend(series_failures)
        is_last = chunk_index == len(detection_chunks) - 1
        detection_payload = {
            "schema": "detection-stage-v1",
            "active_version": active_version,
            "format_fingerprints": format_fingerprints,
            "detection_fingerprint": detection.fingerprint,
            "frames": [
                frame_to_state(frame)
                for series in sorted(frames_by_series)
                for frame in frames_by_series[series]
            ],
            "failures": [
                failure_to_state(item)
                for item in _dedupe_failures(failures)
            ],
        }
        if not is_last:
            commit_stage("detection", detection_payload, partial=True, crash=False)

    forced_matching = _forced_failures(
        injections, kind="matching_forced_error", stage="matching", code=E_MATCH_INVALID
    )
    if forced_matching:
        for failure in forced_matching:
            if failure.frame_index is not None:
                _mark_frame_failed(
                    frames_by_series, failure.series_name, failure.frame_index
                )
        failures.extend(forced_matching)
    final_failures = _dedupe_failures(failures)
    detection_payload = {
        "schema": "detection-stage-v1",
        "active_version": active_version,
        "format_fingerprints": format_fingerprints,
        "detection_fingerprint": detection.fingerprint,
        "frames": [
            frame_to_state(frame)
            for series in sorted(frames_by_series)
            for frame in frames_by_series[series]
        ],
        "failures": [failure_to_state(item) for item in final_failures],
    }
    commit_stage(
        "detection",
        detection_payload,
        partial=False,
        crash=options.crash_after_stage == "detection",
    )

    current_frame_keys = {
        series: _frame_key_map(series_frames)
        for series, series_frames in sorted(frames_by_series.items())
    }
    previous_track_by_id = {track.track_id: track for track in previous_tracks}
    reusable_by_series: dict[str, list[Track]] = {}
    invalidated_track_reasons: dict[str, str] = {}
    same_match_config = previous_match_fingerprint == match.fingerprint
    for track in previous_tracks:
        if not same_match_config:
            invalidated_track_reasons[track.track_id] = "match-config-changed"
        elif track_reusable(
            track,
            current_frame_keys=current_frame_keys.get(track.series_name, {}),
            match=match,
        ):
            reusable_by_series.setdefault(track.series_name, []).append(track)
        else:
            invalidated_track_reasons[track.track_id] = "dependency-frame-changed"

    all_tracks: list[Track] = []
    match_audits: list[dict[str, Any]] = []

    def match_one(series_name: str) -> tuple[str, list[Track], dict[str, Any]]:
        tracks, audit = match_series(
            series_name=series_name,
            frames=frames_by_series.get(series_name, []),
            match=match,
            detection=detection,
            reusable_tracks=reusable_by_series.get(series_name, []),
        )
        return series_name, tracks, audit.to_dict()

    matching_chunks = list(_chunked(all_series, interval))
    for chunk_index, chunk in enumerate(matching_chunks):
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk))) as executor:
            chunk_results = list(executor.map(match_one, chunk))
        for series_name, series_tracks, audit in chunk_results:
            all_tracks.extend(series_tracks)
            match_audits.append(audit)
        all_tracks.sort(key=lambda item: item.sort_key())
        is_last = chunk_index == len(matching_chunks) - 1
        matching_payload = {
            "schema": "matching-stage-v1",
            "match_fingerprint": match.fingerprint,
            "match_config": {
                "iou_threshold": match.iou_threshold,
                "max_gap_frames": match.max_gap_frames,
                "pool_eviction_rule": match.pool_eviction_rule,
                "candidate_pool_capacity": detection.pool_capacity,
            },
            "frame_keys": {key: value for key, value in sorted(current_frame_keys.items())},
            "tracks": [track_to_state(track) for track in all_tracks],
            "audits": match_audits,
            "failures": [failure_to_state(item) for item in _dedupe_failures(failures)],
        }
        if not is_last:
            commit_stage("matching", matching_payload, partial=True, crash=False)
    commit_stage(
        "matching",
        matching_payload,
        partial=False,
        crash=options.crash_after_stage == "matching",
    )

    forced_serialization = _forced_failures(
        injections,
        kind="serialization_forced_error",
        stage="serialization",
        code=E_SERIALIZE_INVALID,
    )
    failures.extend(forced_serialization)
    failed_keys = {
        (failure.series_name, failure.frame_index)
        for failure in failures
        if failure.frame_index is not None
    }
    for failure in forced_serialization:
        if failure.frame_index is not None:
            _mark_frame_failed(frames_by_series, failure.series_name, failure.frame_index)
    all_tracks, tracks_pruned_by_serialization = _prune_failed_tracks(
        all_tracks,
        failed_keys,
        match=match,
        frame_keys_by_series=current_frame_keys,
    )

    serialized_frames: list[dict[str, Any]] = []
    for series_name in sorted(frames_by_series):
        for frame in sorted(frames_by_series[series_name], key=lambda item: item.frame_index):
            if (frame.series_name, frame.frame_index) in failed_keys:
                continue
            try:
                record = {
                    "series_name": frame.series_name,
                    "frame_index": frame.frame_index,
                    "candidate_key": frame.candidate_key,
                    "candidates": [
                        {
                            "class_code": candidate.class_code,
                            "class_name": candidate.class_name,
                            "integer_box": list(
                                candidate.integer_box(match.fixed_decimal_places)
                            ),
                            "score_pm": candidate.score_pm,
                            "precision_label": candidate.precision_label,
                        }
                        for candidate in frame.candidates
                    ],
                }
                json.dumps(record, allow_nan=False)
                serialized_frames.append(record)
            except Exception as exc:
                failures.append(
                    _status_failure(
                        E_SERIALIZE_INVALID,
                        "serialization",
                        frame.series_name,
                        frame.frame_index,
                        "frame serialization failed",
                        exc,
                    )
                )
                _mark_frame_failed(frames_by_series, frame.series_name, frame.frame_index)
    failed_keys = {
        (failure.series_name, failure.frame_index)
        for failure in failures
        if failure.frame_index is not None
    }
    all_tracks, pruned_again = _prune_failed_tracks(
        all_tracks,
        failed_keys,
        match=match,
        frame_keys_by_series=current_frame_keys,
    )
    tracks_pruned_by_serialization = sorted(
        set(tracks_pruned_by_serialization) | set(pruned_again)
    )
    recompute_pool_metadata(
        all_tracks,
        match=match,
        candidate_pool_capacity=detection.pool_capacity,
    )

    serialization_dir = ensure_dir(output_dir / "serialization")
    serialization_path = serialization_dir / "frames.jsonl"
    atomic_write_text(
        serialization_path,
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in serialized_frames
        ),
    )
    serialization_hash = file_sha256(serialization_path)

    conversion: dict[str, Any] | None = None
    conversion_error: dict[str, Any] | None = None
    try:
        conversion = run_conversion(
            v1=v1,
            v2=v2,
            detection=detection,
            match=match,
            output_dir=output_dir / "conversion",
        )
    except BusinessError as exc:
        conversion_error = exc.to_dict()
        failures.append(
            FailureRecord(
                series_name=exc.series_name or "<conversion>",
                frame_index=exc.frame_index,
                stage="serialization",
                error_code=exc.code,
                exception_type="BusinessError",
                message=exc.message,
                details=exc.details,
            )
        )

    failures = _dedupe_failures(failures)
    injection_hits = _assign_injections(failures, injections)
    failures = sorted(failures, key=_failure_sort_key)
    failed_keys = {
        (failure.series_name, failure.frame_index)
        for failure in failures
        if failure.frame_index is not None
    }

    frame_ledger = sorted(
        frame_ledger,
        key=lambda item: (
            item["kind"],
            item["series_name"],
            -1 if item["frame_index"] is None else item["frame_index"],
            item["action"],
        ),
    )
    current_track_by_id = {track.track_id: track for track in all_tracks}
    track_ledger: list[dict[str, Any]] = []
    for track in all_tracks:
        previous = previous_track_by_id.get(track.track_id)
        if track.reused:
            action = "reused"
            reason = "all-track-dependency-frame-keys-match"
        elif previous is None:
            action = "recomputed"
            reason = "new-track"
        else:
            action = "recomputed"
            reason = invalidated_track_reasons.get(track.track_id, "track-state-changed")
        track_ledger.append(
            {
                "kind": "track",
                "track_id": track.track_id,
                "series_name": track.series_name,
                "class_code": track.class_code,
                "action": action,
                "cache_key": track.cache_key,
                "previous_cache_key": previous.cache_key if previous else None,
                "reason": reason,
            }
        )
    for track_id, previous in sorted(previous_track_by_id.items()):
        if track_id in current_track_by_id:
            continue
        track_ledger.append(
            {
                "kind": "track",
                "track_id": track_id,
                "series_name": previous.series_name,
                "class_code": previous.class_code,
                "action": "invalidated",
                "cache_key": None,
                "previous_cache_key": previous.cache_key,
                "reason": invalidated_track_reasons.get(track_id, "track-not-produced"),
            }
        )
    track_ledger.sort(
        key=lambda item: (
            item["kind"],
            item["series_name"],
            item["track_id"],
            item["action"],
        )
    )
    invalidation_ledger = frame_ledger + track_ledger

    for series_name, series_frames in frames_by_series.items():
        for frame in series_frames:
            if (series_name, frame.frame_index) in failed_keys:
                frame.status = "failed"
                frame.candidates = []

    report: dict[str, Any] = {
        "schema_version": "candidate-exchange-report-v1",
        "tool_version": "1.0.0",
        "status": "completed_with_failures" if failures else "completed",
        "active_version": active_version,
        "process_exit_semantics": {
            "business_errors_in_exit_code": False,
            "success_exit_code": 0,
            "configuration_exit_code": 2,
            "unexpected_exit_code": 3,
            "simulated_termination_exit_code": 75,
        },
        "config": {
            "detection_parameter_fingerprint": detection.fingerprint,
            "detection_parameters": sanitize_for_json(detection.raw),
            "match_config_fingerprint": match.fingerprint,
            "match_config": sanitize_for_json(match.raw),
            "class_map": {str(key): value for key, value in sorted(detection.class_map.items())},
            "effective_workers": workers,
            "effective_snapshot_commit_interval": interval,
            "config_hash": config_hash,
        },
        "format_fingerprints": format_fingerprints,
        "sidecar_metadata_hash": sidecar_hash,
        "input_fingerprint": input_fingerprint,
        "summary": {
            "series_count": len(frames_by_series),
            "frame_count": sum(len(value) for value in frames_by_series.values()),
            "successful_frame_count": sum(
                1
                for series_frames in frames_by_series.values()
                for frame in series_frames
                if frame.status != "failed"
            ),
            "candidate_count": sum(
                len(frame.candidates)
                for series_frames in frames_by_series.values()
                for frame in series_frames
            ),
            "track_count": len(all_tracks),
            "failure_count": len(failures),
            "injection_count": len(injections),
            "injection_hit_count": sum(1 for item in injection_hits if item["hit"]),
            "reused_track_count": sum(1 for track in all_tracks if track.reused),
            "recomputed_track_count": sum(1 for track in all_tracks if not track.reused),
            "reused_frame_count": sum(
                1 for item in frame_ledger if item["action"] == "reused"
            ),
            "recomputed_frame_count": sum(
                1 for item in frame_ledger if item["action"] == "recomputed"
            ),
        },
        "frames": [
            {
                "series_name": frame.series_name,
                "frame_index": frame.frame_index,
                "status": frame.status,
                "candidate_key": frame.candidate_key,
                "image_sha256": frame.image_sha256,
                "image_size": list(frame.image_size),
                "candidates": [
                    candidate.to_dict(match.fixed_decimal_places)
                    for candidate in frame.candidates
                ],
            }
            for series_name in sorted(frames_by_series)
            for frame in sorted(
                frames_by_series[series_name], key=lambda item: item.frame_index
            )
        ],
        "tracks": [track.to_dict() for track in all_tracks],
        "conversion": conversion,
        "conversion_error": conversion_error,
        "failures": [failure.to_dict() for failure in failures],
        "failure_frames": [
            {
                "series_name": failure.series_name,
                "frame_index": failure.frame_index,
                "stage": failure.stage,
                "error_code": failure.error_code,
                "injection_id": failure.injection_id,
            }
            for failure in failures
        ],
        "injections": {
            "configured": injections,
            "hits": injection_hits,
            "all_hit": bool(injection_hits)
            and all(item["hit"] for item in injection_hits),
        },
        "invalidation": {
            "ledger": invalidation_ledger,
            "frame_ledger": frame_ledger,
            "track_ledger": track_ledger,
            "match_audits": sorted(match_audits, key=lambda item: item["series_name"]),
        },
        "serialization": {
            "path": str(serialization_path),
            "sha256": serialization_hash,
            "frame_count": len(serialized_frames),
            "tracks_pruned_after_serialization_failure": tracks_pruned_by_serialization,
        },
        "concurrency": {
            "workers": workers,
            "requested_workers": options.workers,
            "series_execution_order": all_series,
            "ordering_contract": "sorted by series name, then track sort key",
        },
        "snapshot_recovery_at_start": recovered,
        "snapshot_commits": commits,
    }

    serialization_payload = {
        "schema": "serialization-stage-v1",
        "serialization_path": str(serialization_path),
        "serialization_sha256": serialization_hash,
        "frames": serialized_frames,
        "tracks": [track_to_state(track) for track in all_tracks],
        "conversion": conversion,
        "failures": [failure_to_state(item) for item in failures],
        "deterministic_core": _report_core(
            frames=frames_by_series,
            tracks=all_tracks,
            failures=failures,
            match=match,
            conversion=conversion,
            report=report,
        ),
    }
    final_manifest = commit_stage(
        "serialization",
        serialization_payload,
        partial=False,
        crash=options.crash_after_stage == "serialization",
    )
    report["snapshot"] = {
        "root": str(snapshot_root),
        "final_manifest": final_manifest,
        "recovery_point_version": final_manifest["global_version"],
        "recovered_from_version": recovered.get("recovery_point_version"),
    }
    report_path = output_dir / "validation_report.json"
    report["report_path"] = str(report_path)
    atomic_write_json(report_path, report)
    return report
