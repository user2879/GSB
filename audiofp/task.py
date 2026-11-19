"""JSON task parsing and validation with stable business error codes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import BusinessError, schema_error
from .models import (
    ConcurrencyConfig,
    FingerprintConfig,
    InjectionSpec,
    RetrievalConfig,
    TaskSpec,
    ToleranceConfig,
    TrackSpec,
)


DEFAULT_RATES = (16000, 44100, 48000)
INJECTION_KINDS = {
    "truncated_file",
    "silent_track",
    "rate_declaration_mismatch",
    "band_override",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise schema_error(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise schema_error(f"{name} must be {qualifier}")
    return value


def _positive_number(value: Any, name: str) -> float:
    if not _is_number(value) or float(value) <= 0:
        raise schema_error(f"{name} must be a positive finite number")
    return float(value)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise schema_error(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise schema_error(f"{name} must be an array")
    return value


def _resolve(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise schema_error(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_bands(raw: dict[str, Any], nyquist: float) -> tuple[tuple[float, float], ...]:
    if "bands" in raw:
        values = _array(raw["bands"], "fingerprint.bands")
        bands: list[tuple[float, float]] = []
        previous_high: float | None = None
        for index, pair in enumerate(values):
            pair = _array(pair, f"fingerprint.bands[{index}]")
            if len(pair) != 2 or not all(_is_number(v) for v in pair):
                raise BusinessError("E_BAND", "each band must contain two finite numeric edges")
            low, high = float(pair[0]), float(pair[1])
            if low < 0 or high <= low:
                raise BusinessError("E_BAND", "band edges must satisfy 0 <= low < high")
            if previous_high is not None and low < previous_high:
                raise BusinessError("E_BAND", "band boundaries must be strictly increasing and non-overlapping")
            if high >= nyquist:
                raise BusinessError(
                    "E_BAND",
                    "band edge reaches or exceeds Nyquist frequency",
                    band_index=index,
                    edge_hz=high,
                    nyquist_hz=nyquist,
                )
            bands.append((low, high))
            previous_high = high
        if not bands:
            raise BusinessError("E_BAND", "at least one frequency band is required")
        return tuple(bands)
    if "band_edges" in raw:
        edges_raw = _array(raw["band_edges"], "fingerprint.band_edges")
        if len(edges_raw) < 3 or not all(_is_number(v) for v in edges_raw):
            raise BusinessError("E_BAND", "band_edges needs at least three finite boundaries")
        edges = [float(v) for v in edges_raw]
        if edges[0] < 0 or any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise BusinessError("E_BAND", "band_edges must be strictly increasing")
        if edges[-1] >= nyquist:
            raise BusinessError("E_BAND", "band edge reaches or exceeds Nyquist frequency")
        return tuple((edges[i], edges[i + 1]) for i in range(len(edges) - 1))
    raise BusinessError("E_BAND", "fingerprint.bands or fingerprint.band_edges is required")


def parse_fingerprint(raw_value: Any, allowed_rates: tuple[int, ...]) -> FingerprintConfig:
    raw = _object(raw_value, "fingerprint")
    frame_length = _positive_int(raw.get("frame_length"), "fingerprint.frame_length")
    hop_length = _positive_int(raw.get("hop_length"), "fingerprint.hop_length")
    if hop_length > frame_length:
        raise schema_error("fingerprint.hop_length must not exceed frame_length")
    base_rate = _positive_int(raw.get("base_sample_rate", 16000), "fingerprint.base_sample_rate")
    if base_rate not in allowed_rates:
        raise BusinessError("E_RATE", "fingerprint.base_sample_rate is not in allowed_sample_rates")
    neighborhood = _object(raw.get("peak_neighborhood", {}), "fingerprint.peak_neighborhood")
    freq_bins = _positive_int(
        neighborhood.get("frequency_bins", neighborhood.get("freq_bins", 3)),
        "fingerprint.peak_neighborhood.frequency_bins",
        allow_zero=True,
    )
    time_frames = _positive_int(
        neighborhood.get("time_frames", 2),
        "fingerprint.peak_neighborhood.time_frames",
        allow_zero=True,
    )
    min_peak_db = raw.get("min_peak_db", -50.0)
    if not _is_number(min_peak_db) or float(min_peak_db) > 0:
        raise schema_error("fingerprint.min_peak_db must be a finite number <= 0")
    fanout = _positive_int(raw.get("fanout", 4), "fingerprint.fanout")
    max_delta = _positive_int(raw.get("max_pair_delta_frames", 24), "fingerprint.max_pair_delta_frames")
    peaks_per_band = _positive_int(raw.get("peaks_per_band", 2), "fingerprint.peaks_per_band")
    window = raw.get("window", "hann")
    if window not in {"hann", "hamming", "rect"}:
        raise schema_error("fingerprint.window must be hann, hamming, or rect")
    nyquist = base_rate / 2.0
    bands = _parse_bands(raw, nyquist)
    return FingerprintConfig(
        frame_length=frame_length,
        hop_length=hop_length,
        base_sample_rate=base_rate,
        bands=bands,
        freq_neighbor_bins=freq_bins,
        time_neighbor_frames=time_frames,
        min_peak_db=float(min_peak_db),
        fanout=fanout,
        max_pair_delta_frames=max_delta,
        peaks_per_band=peaks_per_band,
        window=window,
    )


def merge_fingerprint_override(base: FingerprintConfig, override: Any) -> FingerprintConfig:
    if override is None:
        return base
    raw_override = _object(override, "track.fingerprint_override")
    merged = {
        "frame_length": base.frame_length,
        "hop_length": base.hop_length,
        "base_sample_rate": base.base_sample_rate,
        "bands": [[low, high] for low, high in base.bands],
        "peak_neighborhood": {
            "frequency_bins": base.freq_neighbor_bins,
            "time_frames": base.time_neighbor_frames,
        },
        "min_peak_db": base.min_peak_db,
        "fanout": base.fanout,
        "max_pair_delta_frames": base.max_pair_delta_frames,
        "peaks_per_band": base.peaks_per_band,
        "window": base.window,
    }
    merged.update(raw_override)
    return parse_fingerprint(merged, (base.base_sample_rate, 16000, 44100, 48000))


def _parse_tracks(raw_task: dict[str, Any], key: str, role: str, base: Path) -> tuple[TrackSpec, ...]:
    values = _array(raw_task.get(key), key)
    if not values:
        raise schema_error(f"{key} must not be empty")
    tracks: list[TrackSpec] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        item = _object(value, f"{key}[{index}]")
        track_id = item.get("id")
        if not isinstance(track_id, str) or not track_id.strip():
            raise schema_error(f"{key}[{index}].id must be a non-empty string")
        if track_id in seen:
            raise schema_error(f"duplicate track id: {track_id}")
        seen.add(track_id)
        declared = item.get("declared_sample_rate")
        if declared is not None and (not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0):
            raise schema_error(f"{key}[{index}].declared_sample_rate must be a positive integer")
        override = item.get("fingerprint_override")
        if override is not None:
            _object(override, f"{key}[{index}].fingerprint_override")
        tracks.append(
            TrackSpec(
                track_id=track_id,
                path=_resolve(base, item.get("path"), f"{key}[{index}].path"),
                role=role,
                declared_sample_rate=declared,
                fingerprint_override=override,
            )
        )
    return tuple(tracks)


def _parse_injections(raw_task: dict[str, Any], track_ids: set[str]) -> tuple[InjectionSpec, ...]:
    values = raw_task.get("injections", [])
    values = _array(values, "injections")
    result: list[InjectionSpec] = []
    for index, value in enumerate(values):
        item = _object(value, f"injections[{index}]")
        target = item.get("target_id")
        if not isinstance(target, str) or target not in track_ids:
            raise schema_error(f"injections[{index}].target_id must identify a declared track")
        kind = item.get("kind")
        if kind not in INJECTION_KINDS:
            raise schema_error(f"injections[{index}].kind is not supported")
        expected = item.get("expected_code")
        if expected not in {"E_TRUNC", "E_SILENT", "E_RATE", "E_BAND"}:
            raise schema_error(f"injections[{index}].expected_code is not supported")
        details = {k: v for k, v in item.items() if k not in {"target_id", "kind", "expected_code"}}
        result.append(InjectionSpec(target, kind, expected, details))
    return tuple(result)


def load_task(path: str | Path) -> TaskSpec:
    task_path = Path(path).resolve()
    try:
        text = task_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError("E_IO", "unable to read task JSON", {"path": str(task_path), "reason": str(exc)}) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BusinessError(
            "E_JSON",
            "task is not valid JSON",
            {"line": exc.lineno, "column": exc.colno, "reason": exc.msg},
        ) from exc
    if not isinstance(raw, dict):
        raise schema_error("task root must be an object")
    base = task_path.parent
    allowed_raw = raw.get("allowed_sample_rates", list(DEFAULT_RATES))
    allowed_values = _array(allowed_raw, "allowed_sample_rates")
    if not allowed_values:
        raise BusinessError("E_RATE", "allowed_sample_rates must not be empty")
    allowed: list[int] = []
    for value in allowed_values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise BusinessError("E_RATE", "allowed_sample_rates must contain positive integers")
        allowed.append(value)
    allowed_rates = tuple(sorted(set(allowed)))
    references = _parse_tracks(raw, "references", "reference", base)
    queries = _parse_tracks(raw, "queries", "query", base)
    all_ids = [t.track_id for t in references] + [t.track_id for t in queries]
    if len(all_ids) != len(set(all_ids)):
        raise schema_error("reference and query ids must be globally unique")
    fingerprint = parse_fingerprint(raw.get("fingerprint"), allowed_rates)

    retrieval_raw = _object(raw.get("retrieval"), "retrieval")
    policy = retrieval_raw.get("overlap_policy", "keep_highest")
    if policy not in {"keep_highest", "merge"}:
        raise schema_error("retrieval.overlap_policy must be keep_highest or merge")
    retrieval = RetrievalConfig(
        bucket_capacity=_positive_int(retrieval_raw.get("bucket_capacity"), "retrieval.bucket_capacity"),
        voting_window_frames=_positive_int(
            retrieval_raw.get("voting_window_frames"), "retrieval.voting_window_frames"
        ),
        min_matches=_positive_int(retrieval_raw.get("min_matches"), "retrieval.min_matches"),
        max_candidates=_positive_int(retrieval_raw.get("max_candidates"), "retrieval.max_candidates"),
        overlap_policy=policy,
        boundary_gap_frames=_positive_int(
            retrieval_raw.get("boundary_gap_frames", 4),
            "retrieval.boundary_gap_frames",
            allow_zero=True,
        ),
        boundary_padding_frames=_positive_int(
            retrieval_raw.get("boundary_padding_frames", 1),
            "retrieval.boundary_padding_frames",
            allow_zero=True,
        ),
    )
    tolerance_raw = _object(raw.get("tolerance"), "tolerance")
    tolerance = ToleranceConfig(
        score_scale=_positive_int(tolerance_raw.get("score_scale"), "tolerance.score_scale"),
        score_tolerance=_positive_number(tolerance_raw.get("score_tolerance"), "tolerance.score_tolerance"),
        time_offset_tolerance_samples=_positive_number(
            tolerance_raw.get("time_offset_tolerance_samples"),
            "tolerance.time_offset_tolerance_samples",
        ),
    )
    concurrency_raw = _object(raw.get("concurrency"), "concurrency")
    concurrency = ConcurrencyConfig(
        max_workers=_positive_int(concurrency_raw.get("max_workers"), "concurrency.max_workers"),
        spectrum_buffer_bytes=_positive_int(
            concurrency_raw.get("spectrum_buffer_bytes"), "concurrency.spectrum_buffer_bytes"
        ),
    )
    row_bytes = (fingerprint.frame_length // 2 + 1) * 8
    if concurrency.spectrum_buffer_bytes < row_bytes:
        raise schema_error(
            "concurrency.spectrum_buffer_bytes must hold at least one spectrum frame",
            minimum_bytes=row_bytes,
        )

    state_dir = _resolve(base, raw.get("state_dir", "state"), "state_dir")
    outputs = _object(raw.get("outputs"), "outputs")
    report_path = _resolve(base, outputs.get("report"), "outputs.report")
    results_path = _resolve(base, outputs.get("results"), "outputs.results")
    if report_path == results_path:
        raise schema_error("outputs.report and outputs.results must differ")
    input_paths = {task_path} | {t.path for t in references} | {t.path for t in queries}
    if report_path in input_paths or results_path in input_paths:
        raise schema_error("output paths must not overwrite task or WAV inputs")
    injections = _parse_injections(raw, set(all_ids))
    return TaskSpec(
        task_path=task_path,
        base_dir=base,
        references=references,
        queries=queries,
        fingerprint=fingerprint,
        retrieval=retrieval,
        tolerance=tolerance,
        concurrency=concurrency,
        allowed_sample_rates=allowed_rates,
        state_dir=state_dir,
        report_path=report_path,
        results_path=results_path,
        injections=injections,
        raw=raw,
    )

