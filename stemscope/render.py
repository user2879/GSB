from __future__ import annotations

import hashlib
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cache import CacheStore, array_entry, array_from_entry
from .config import RemixConfig, TrackSettings
from .errors import DiagnosticError, sort_errors
from .hashing import cache_key, canonical_json_bytes, sha256_bytes
from .model import Track


@dataclass(frozen=True)
class SegmentSpec:
    index: int
    clip_id: str
    chunk_index: int
    clip_frames: int
    start_frame: int
    end_frame: int
    sample_rate: int
    output_start: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def sort_key(self) -> str:
        return f"{self.index:06d}|{self.clip_id}|{self.chunk_index:06d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.index,
            "clip_id": self.clip_id,
            "chunk_index": self.chunk_index,
            "clip_frames": self.clip_frames,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": self.frames,
            "sample_rate": self.sample_rate,
            "output_start_frame": self.output_start,
            "sort_key": self.sort_key,
        }


@dataclass
class InvalidationPlan:
    affected_segments: list[int]
    reasons: dict[int, list[str]]
    all_segments: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_segments": self.all_segments,
            "affected_segments": self.affected_segments,
            "affected_count": len(self.affected_segments),
            "reasons": {str(key): value for key, value in sorted(self.reasons.items())},
        }


@dataclass
class RenderResult:
    param_version: int
    pcm: np.ndarray
    segments: list[dict[str, Any]]
    failures: list[DiagnosticError]
    reused: list[dict[str, Any]]
    recomputed: list[dict[str, Any]]
    stage_times_ms: dict[str, float]
    cache_stats: dict[str, Any]
    invalidation: dict[str, Any]
    frame_detail_sha256: str
    frame_detail_lines: list[str] = field(default_factory=list, repr=False)
    max_observed_workers: int = 1
    checkpoint_commits: int = 0
    simulated_compute_ms: float = 0.0

    @property
    def pcm_bytes(self) -> bytes:
        return np.ascontiguousarray(self.pcm, dtype="<i2").tobytes(order="C")

    @property
    def pcm_sha256(self) -> str:
        return sha256_bytes(self.pcm_bytes)

    @property
    def aggregate_peak(self) -> int:
        if self.pcm.size == 0:
            return 0
        return int(np.max(np.abs(self.pcm.astype(np.int32))))

    @property
    def completed_segments(self) -> int:
        return len(self.segments)

    @property
    def progress(self) -> float:
        return 1.0 if self.segments else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "param_version": self.param_version,
            "completed_segments": self.completed_segments,
            "total_segments": len(self.segments),
            "progress": self.progress,
            "current_mix_peak": self.aggregate_peak,
            "affected_segments": self.invalidation.get("affected_segments", []),
            "pcm_sha256": self.pcm_sha256,
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "pcm_sha256": self.pcm_sha256,
            "aggregate_peak": self.aggregate_peak,
            "segments": self.segments,
            "frame_detail_sha256": self.frame_detail_sha256,
            "failures": [error.to_dict() for error in self.failures],
        }

    def to_report_payload(self, include_details: bool = False) -> dict[str, Any]:
        payload = {
            "param_version": self.param_version,
            "pcm_sha256": self.pcm_sha256,
            "pcm_bytes": len(self.pcm_bytes),
            "aggregate_peak": self.aggregate_peak,
            "frames": int(self.pcm.shape[0]),
            "completed_segments": self.completed_segments,
            "segments": self.segments,
            "frame_detail_sha256": self.frame_detail_sha256,
            "failures": [error.to_dict() for error in self.failures],
            "reused": self.reused,
            "recomputed": self.recomputed,
            "cache_stats": self.cache_stats,
            "invalidation": self.invalidation,
            "stage_times_ms": self.stage_times_ms,
            "max_observed_workers": self.max_observed_workers,
            "checkpoint_commits": self.checkpoint_commits,
        }
        if include_details:
            payload["frame_detail_line_count"] = len(self.frame_detail_lines)
        return payload


class TaskPool:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stemscope")
        self._lock = threading.Lock()
        self._active = 0
        self.max_observed = 0

    def submit(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        return self.executor.submit(self._run, function, *args, **kwargs)

    def _run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._active += 1
            self.max_observed = max(self.max_observed, self._active)
        try:
            return function(*args, **kwargs)
        finally:
            with self._lock:
                self._active -= 1

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)


def build_segments(clips: Mapping[str, Any], render_frames: int, sample_rate: int) -> list[SegmentSpec]:
    segments: list[SegmentSpec] = []
    output_start = 0
    index = 0
    for clip_id in sorted(clips):
        frames = int(clips[clip_id].frames)
        chunk_index = 0
        for start in range(0, frames, render_frames):
            end = min(frames, start + render_frames)
            segments.append(
                SegmentSpec(
                    index=index,
                    clip_id=clip_id,
                    chunk_index=chunk_index,
                    clip_frames=frames,
                    start_frame=start,
                    end_frame=end,
                    sample_rate=sample_rate,
                    output_start=output_start,
                )
            )
            output_start += end - start
            index += 1
            chunk_index += 1
    return segments


def _crop_segment_overlap(
    segment: SegmentSpec,
    old_crop: tuple[int, int],
    new_crop: tuple[int, int],
) -> bool:
    def overlap(crop: tuple[int, int]) -> bool:
        return max(crop[0], segment.start_frame) < min(crop[1], segment.end_frame)

    if overlap(old_crop) or overlap(new_crop):
        return True
    old_inside = old_crop[0] <= segment.start_frame and segment.end_frame <= old_crop[1]
    new_inside = new_crop[0] <= segment.start_frame and segment.end_frame <= new_crop[1]
    return old_inside or new_inside


def compute_invalidation(
    previous_state: Any | None,
    current_state: Any,
    segments: Sequence[SegmentSpec],
) -> InvalidationPlan:
    all_indexes = [segment.index for segment in segments]
    if previous_state is None:
        return InvalidationPlan(
            affected_segments=all_indexes,
            reasons={segment.index: ["initial_render"] for segment in segments},
            all_segments=True,
        )
    reasons: dict[int, set[str]] = {segment.index: set() for segment in segments}
    all_segments = False
    if previous_state.config.format_version != current_state.config.format_version:
        all_segments = True
        for segment in segments:
            reasons[segment.index].add("format_version_changed")
    if (
        previous_state.config.gain_quantization_steps
        != current_state.config.gain_quantization_steps
    ):
        all_segments = True
        for segment in segments:
            reasons[segment.index].add("gain_quantization_changed")

    previous_tracks = previous_state.clips
    current_tracks = current_state.clips
    for clip_id in sorted(set(previous_tracks) | set(current_tracks)):
        old_clip = previous_tracks.get(clip_id)
        new_clip = current_tracks.get(clip_id)
        clip_segments = [segment for segment in segments if segment.clip_id == clip_id]
        if old_clip is None or new_clip is None or int(old_clip.frames) != int(new_clip.frames):
            all_segments = True
            for segment in clip_segments:
                reasons[segment.index].add("clip_geometry_changed")
            continue
        for track_name in sorted(set(old_clip.tracks) | set(new_clip.tracks)):
            old_track = old_clip.tracks.get(track_name)
            new_track = new_clip.tracks.get(track_name)
            if old_track is None or new_track is None:
                all_segments = True
                for segment in clip_segments:
                    reasons[segment.index].add(f"{track_name}:track_added_or_removed")
                continue
            if old_track.content_fingerprint() != new_track.content_fingerprint():
                for segment in clip_segments:
                    reasons[segment.index].add(f"{track_name}:pcm_replaced")

    old_settings = previous_state.config.tracks
    new_settings = current_state.config.tracks
    for track_name in sorted(set(old_settings) | set(new_settings)):
        old_setting = old_settings.get(track_name)
        new_setting = new_settings.get(track_name)
        if old_setting is None or new_setting is None:
            all_segments = True
            for segment in segments:
                reasons[segment.index].add(f"{track_name}:track_settings_added_or_removed")
            continue
        if old_setting.gain != new_setting.gain:
            for segment in segments:
                reasons[segment.index].add(f"{track_name}:gain_changed")
        if old_setting.muted != new_setting.muted:
            for segment in segments:
                reasons[segment.index].add(f"{track_name}:mute_changed")
        if old_setting.start_ms != new_setting.start_ms or old_setting.end_ms != new_setting.end_ms:
            for segment in segments:
                clip = current_tracks[segment.clip_id]
                old_crop = old_setting.crop_frames(segment.sample_rate, int(clip.frames))
                new_crop = new_setting.crop_frames(segment.sample_rate, int(clip.frames))
                if _crop_segment_overlap(segment, old_crop, new_crop):
                    reasons[segment.index].add(f"{track_name}:crop_changed")

    if all_segments:
        affected = all_indexes
        for segment in segments:
            reasons[segment.index].add("all_segments_fallback")
    else:
        affected = [segment.index for segment in segments if reasons[segment.index]]
    return InvalidationPlan(
        affected_segments=affected,
        reasons={index: sorted(values) for index, values in reasons.items() if values},
        all_segments=all_segments,
    )


def _round_div_half_away(numerator: np.ndarray, denominator: int) -> np.ndarray:
    absolute = np.abs(numerator)
    rounded = (absolute + denominator // 2) // denominator
    return np.where(numerator < 0, -rounded, rounded)


def _saturate_int16(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -32768, 32767).astype("<i2")


def _gain_key(
    config: RemixConfig,
    segment: SegmentSpec,
    track: Track,
    settings: TrackSettings,
) -> str:
    return cache_key(
        "gain",
        config.format_version,
        config.gain_quantization_steps,
        config.track_param_fingerprint(track.name),
        segment.clip_id,
        segment.index,
        segment.start_frame,
        segment.end_frame,
        track.content_fingerprint(),
        schema="stemscope-gain-v1",
    )


def _mix_key(
    config: RemixConfig,
    segment: SegmentSpec,
    gain_keys: Mapping[str, str],
) -> str:
    return cache_key(
        "mix",
        config.format_version,
        segment.clip_id,
        segment.index,
        segment.start_frame,
        segment.end_frame,
        config.gain_quantization_steps,
        [(name, gain_keys[name]) for name in sorted(gain_keys)],
        schema="stemscope-mix-v1",
    )


def _process_gain(
    segment: SegmentSpec,
    track: Track,
    settings: TrackSettings,
    config: RemixConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames = segment.frames
    result = np.zeros(frames, dtype=np.int64)
    gain_int = config.quantized_gain_int(track.name)
    crop_start, crop_end = settings.crop_frames(config.sample_rate, segment.clip_frames)
    if not settings.muted and gain_int != 0 and crop_end > crop_start:
        source_start = max(segment.start_frame, crop_start)
        source_end = min(segment.end_frame, crop_end)
        if source_end > source_start:
            source = track.pcm[source_start:source_end].astype(np.int64)
            if source.shape[0] < source_end - source_start:
                source = np.pad(source, (0, (source_end - source_start) - source.shape[0]))
            destination_start = source_start - segment.start_frame
            destination_end = destination_start + source.shape[0]
            result[destination_start:destination_end] = source * gain_int
    metadata = {
        "gain_int": gain_int,
        "quantization_steps": config.gain_quantization_steps,
        "muted": settings.muted,
        "crop_start_frame": crop_start,
        "crop_end_frame": crop_end,
        "track_fingerprint": track.content_fingerprint(),
        "param_fingerprint": config.track_param_fingerprint(track.name),
    }
    return result, metadata


def _detail_line(
    segment: SegmentSpec,
    frame_index: int,
    terms: Mapping[str, int],
    mixed: int,
) -> bytes:
    payload = {
        "segment_index": segment.index,
        "clip_id": segment.clip_id,
        "clip_frame": segment.start_frame + frame_index,
        "output_frame": segment.output_start + frame_index,
        "terms": {name: terms[name] for name in sorted(terms)},
        "mixed": mixed,
    }
    return canonical_json_bytes(payload) + b"\n"


def _detail_digest(
    segment: SegmentSpec,
    track_names: list[str],
    terms_by_track: Mapping[str, np.ndarray],
    mixed_pcm: np.ndarray,
    *,
    retain_lines: bool,
) -> tuple[bytes, str, list[str]]:
    digest = hashlib.sha256()
    digest.update(b"stemscope-frame-detail-v1\0")
    digest.update(struct.pack("<I", len(track_names)))
    for name in track_names:
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    digest.update(struct.pack("<QQ", segment.index, segment.frames))
    for name in track_names:
        digest.update(np.ascontiguousarray(terms_by_track[name], dtype="<i8").tobytes(order="C"))
    digest.update(np.ascontiguousarray(mixed_pcm, dtype="<i2").tobytes(order="C"))
    lines: list[str] = []
    if retain_lines:
        for frame_index in range(segment.frames):
            terms = {name: int(terms_by_track[name][frame_index]) for name in track_names}
            lines.append(
                _detail_line(segment, frame_index, terms, int(mixed_pcm[frame_index])).decode("utf-8")
            )
    raw_digest = digest.digest()
    return raw_digest, digest.hexdigest(), lines


def render(
    state: Any,
    cache: CacheStore,
    *,
    previous_state: Any | None = None,
    retain_frame_details: bool = False,
    checkpoint_callback: Callable[[CacheStore, list[int]], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    simulated_compute_ms: float = 0.0,
) -> RenderResult:
    started = time.perf_counter()
    config: RemixConfig = state.config
    clips = state.clips
    segments = build_segments(clips, config.render_frames, config.sample_rate)
    invalidation = compute_invalidation(previous_state, state, segments)
    failures: list[DiagnosticError] = []
    reused: list[dict[str, Any]] = []
    recomputed: list[dict[str, Any]] = []
    pool = TaskPool(config.workers)
    gain_arrays: dict[tuple[int, str], np.ndarray] = {}
    gain_keys: dict[tuple[int, str], str] = {}
    cache_stats = {
        "parse_hits": 0,
        "convert_hits": 0,
        "gain_hits": 0,
        "gain_misses": 0,
        "mix_hits": 0,
        "mix_misses": 0,
    }
    gain_started = time.perf_counter()
    gain_futures: dict[str, Future[Any]] = {}
    gain_lookup: dict[str, tuple[SegmentSpec, str, str]] = {}
    for segment in segments:
        clip = clips[segment.clip_id]
        for track_name in sorted(clip.tracks):
            track = clip.tracks[track_name]
            if track_name not in config.tracks:
                failures.append(
                    DiagnosticError(
                        "E_TRACK_PARSE_FAILED",
                        f"track {track_name} has no remix settings",
                        stage="gain",
                        format_version=config.format_version,
                        clip_id=segment.clip_id,
                        track=track_name,
                        segment_index=segment.index,
                    )
                )
                continue
            key = _gain_key(config, segment, track, config.tracks[track_name])
            sort_key = f"gain|{segment.sort_key}|{track_name}"
            gain_lookup[key] = (segment, track_name, sort_key)
            cached = cache.get("gain", key)
            if cached is not None:
                gain_arrays[(segment.index, track_name)] = array_from_entry(cached)
                gain_keys[(segment.index, track_name)] = key
                cache_stats["gain_hits"] += 1
                reused.append(
                    {
                        "stage": "gain",
                        "sort_key": sort_key,
                        "cache_key": key,
                        "segment_index": segment.index,
                        "clip_id": segment.clip_id,
                        "track": track_name,
                    }
                )
            else:
                gain_futures[sort_key] = pool.submit(
                    _process_gain,
                    segment,
                    track,
                    config.tracks[track_name],
                    config,
                )

    for sort_key in sorted(gain_futures):
        future = gain_futures[sort_key]
        try:
            terms, metadata = future.result()
        except Exception as exc:
            parts = sort_key.split("|")
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"gain processing failed: {exc}",
                    stage="gain",
                    format_version=config.format_version,
                    clip_id=parts[2] if len(parts) > 2 else "",
                    track=parts[-1],
                    segment_index=int(parts[1]) if len(parts) > 1 else None,
                )
            )
            continue
        segment_index = int(sort_key.split("|")[1])
        track_name = sort_key.split("|")[-1]
        segment = next(item for item in segments if item.index == segment_index)
        key = _gain_key(config, segment, clips[segment.clip_id].tracks[track_name], config.tracks[track_name])
        entry = array_entry(
            "gain",
            key,
            sort_key,
            terms,
            refs=[f"{segment.clip_id}:{track_name}"],
            meta=metadata,
        )
        cache.put(entry)
        gain_arrays[(segment.index, track_name)] = terms
        gain_keys[(segment.index, track_name)] = key
        cache_stats["gain_misses"] += 1
        recomputed.append(
            {
                "stage": "gain",
                "sort_key": sort_key,
                "cache_key": key,
                "segment_index": segment.index,
                "clip_id": segment.clip_id,
                "track": track_name,
            }
        )
    gain_elapsed = (time.perf_counter() - gain_started) * 1000.0

    mix_started = time.perf_counter()
    mix_futures: dict[str, Future[Any]] = {}
    mix_info: dict[str, tuple[SegmentSpec, str, list[str]]] = {}

    def mix_segment(segment: SegmentSpec) -> tuple[np.ndarray, dict[str, Any], bytes, list[str]]:
        clip = clips[segment.clip_id]
        track_names = sorted(clip.tracks)
        terms_by_track = {name: gain_arrays[(segment.index, name)] for name in track_names}
        numerator = np.zeros(segment.frames, dtype=np.int64)
        for name in track_names:
            numerator += terms_by_track[name]
        mixed_values = _round_div_half_away(numerator, config.gain_quantization_steps)
        pcm = _saturate_int16(mixed_values)
        raw_digest, detail_hex, detail_lines = _detail_digest(
            segment,
            track_names,
            terms_by_track,
            pcm,
            retain_lines=retain_frame_details,
        )
        peak = int(np.max(np.abs(pcm.astype(np.int32)))) if pcm.size else 0
        metadata = {
            "peak": peak,
            "refs": [f"{segment.clip_id}:{name}" for name in track_names],
            "track_names": track_names,
            "detail_sha256": detail_hex,
        }
        return pcm, metadata, raw_digest, detail_lines

    for segment in segments:
        clip = clips[segment.clip_id]
        track_names = sorted(clip.tracks)
        if any((segment.index, name) not in gain_arrays for name in track_names):
            continue
        segment_gain_keys = {name: gain_keys[(segment.index, name)] for name in track_names}
        key = _mix_key(config, segment, segment_gain_keys)
        sort_key = f"mix|{segment.sort_key}"
        mix_info[sort_key] = (segment, key, track_names)
        cached = cache.get("mix", key)
        if cached is not None:
            cache_stats["mix_hits"] += 1
            reused.append(
                {
                    "stage": "mix",
                    "sort_key": sort_key,
                    "cache_key": key,
                    "segment_index": segment.index,
                    "clip_id": segment.clip_id,
                    "track": "",
                }
            )
        else:
            mix_futures[sort_key] = pool.submit(mix_segment, segment)

    mix_results: dict[int, tuple[np.ndarray, dict[str, Any], bytes, list[str], str, str]] = {}
    for sort_key in sorted(mix_futures):
        future = mix_futures[sort_key]
        segment, key, track_names = mix_info[sort_key]
        try:
            pcm, metadata, detail_digest, detail_lines = future.result()
        except Exception as exc:
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"mix rendering failed: {exc}",
                    stage="mix",
                    format_version=config.format_version,
                    clip_id=segment.clip_id,
                    segment_index=segment.index,
                )
            )
            continue
        entry = array_entry(
            "mix",
            key,
            sort_key,
            pcm,
            refs=metadata["refs"],
            meta=metadata,
        )
        cache.put(entry)
        cache_stats["mix_misses"] += 1
        mix_results[segment.index] = (pcm, metadata, detail_digest, detail_lines, sort_key, key)
        recomputed.append(
            {
                "stage": "mix",
                "sort_key": sort_key,
                "cache_key": key,
                "segment_index": segment.index,
                "clip_id": segment.clip_id,
                "track": "",
            }
        )

    if simulated_compute_ms > 0:
        time.sleep(simulated_compute_ms / 1000.0)

    completed_for_checkpoint: set[int] = set()
    checkpoint_commits = 0
    for segment in segments:
        cached = cache.get("mix", mix_info[f"mix|{segment.sort_key}"][1])
        if cached is not None:
            cached_pcm = array_from_entry(cached)
            track_names = sorted(clips[segment.clip_id].tracks)
            terms_by_track = {name: gain_arrays[(segment.index, name)] for name in track_names}
            raw_digest, detail_hex, detail_lines = _detail_digest(
                segment,
                track_names,
                terms_by_track,
                cached_pcm,
                retain_lines=retain_frame_details,
            )
            metadata = dict(cached.meta)
            metadata["detail_sha256"] = detail_hex
            mix_results[segment.index] = (
                cached_pcm,
                metadata,
                raw_digest,
                detail_lines,
                f"mix|{segment.sort_key}",
                mix_info[f"mix|{segment.sort_key}"][1],
            )
        if segment.index in mix_results:
            completed_for_checkpoint.add(segment.index)
            if progress_callback is not None:
                progress_callback(len(completed_for_checkpoint), len(segments))
            if (
                checkpoint_callback is not None
                and len(completed_for_checkpoint) % config.snapshot_interval_segments == 0
            ):
                checkpoint_callback(cache, sorted(completed_for_checkpoint))
                checkpoint_commits += 1

    mix_elapsed = (time.perf_counter() - mix_started) * 1000.0
    pool.shutdown()

    ordered_segments: list[dict[str, Any]] = []
    pcm_blocks: list[np.ndarray] = []
    detail_file_digest = hashlib.sha256()
    detail_lines_all: list[str] = []
    for segment in segments:
        if segment.index not in mix_results:
            continue
        pcm, metadata, detail_digest, detail_lines, sort_key, key = mix_results[segment.index]
        pcm_blocks.append(pcm)
        detail_file_digest.update(detail_digest)
        if retain_frame_details:
            detail_lines_all.extend(detail_lines)
        peak = int(metadata.get("peak", 0))
        ordered_segments.append(
            {
                **segment.to_dict(),
                "peak": peak,
                "pcm_sha256": sha256_bytes(np.ascontiguousarray(pcm, dtype="<i2").tobytes()),
                "detail_sha256": metadata.get("detail_sha256", detail_digest.hex()),
                "referenced_tracks": sorted(metadata.get("refs", [])),
                "mix_cache_key": key,
            }
        )
    pcm = np.concatenate(pcm_blocks) if pcm_blocks else np.zeros(0, dtype="<i2")
    total_elapsed = (time.perf_counter() - started) * 1000.0
    return RenderResult(
        param_version=int(state.param_version),
        pcm=np.ascontiguousarray(pcm, dtype="<i2"),
        segments=ordered_segments,
        failures=sort_errors(failures),
        reused=sorted(reused, key=lambda item: (item["sort_key"], item["cache_key"])),
        recomputed=sorted(recomputed, key=lambda item: (item["sort_key"], item["cache_key"])),
        stage_times_ms={
            "gain": round(gain_elapsed, 3),
            "mix": round(mix_elapsed, 3),
            "render_total": round(total_elapsed, 3),
        },
        cache_stats=cache_stats,
        invalidation=invalidation.to_dict(),
        frame_detail_sha256=detail_file_digest.hexdigest(),
        frame_detail_lines=detail_lines_all,
        max_observed_workers=pool.max_observed,
        checkpoint_commits=checkpoint_commits,
        simulated_compute_ms=simulated_compute_ms,
    )


def write_frame_details(result: RenderResult, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{threading.get_ident()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        for line in result.frame_detail_lines:
            handle.write(line)
    import os

    os.replace(temporary, output)
