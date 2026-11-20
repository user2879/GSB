from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import PipelineConfig, _is_int, _is_number, _validate_bands
from .errors import (
    E_BAND_INVALID,
    E_CONFIG_INVALID,
    E_RATIO_OUT_OF_RANGE,
    E_SEGMENT_LEN_INVALID,
    E_SEGMENT_MISMATCH,
    Failure,
)
from .util import array_fingerprint, hash_object, stable_float


@dataclass(frozen=True)
class Segment:
    device_id: str
    segment_index: int
    start_sample: int
    end_sample: int
    sample_rate_hz: float
    raw_samples: np.ndarray
    samples: np.ndarray
    missing_indices: tuple[int, ...]
    resampled_indices: tuple[int, ...]
    anomaly_hint: bool
    source: str
    waveform_fingerprint: str
    preprocess_fingerprint: str

    @property
    def segment_length(self) -> int:
        return int(self.raw_samples.size)


@dataclass(frozen=True)
class DeviceData:
    device_id: str
    sample_rate_hz: float
    segment_length: int
    band_edges_hz: tuple[float, ...]
    band_count: int
    segments: tuple[Segment, ...]
    waveform_fingerprint: str


@dataclass
class LoadedData:
    devices: dict[str, DeviceData]
    failures: list[Failure]
    waveform_fingerprint: str
    segment_count: int
    synthetic_description_hash: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return value


def merge_edits(data_spec: dict[str, Any], edits_spec: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(data_spec)
    if not edits_spec:
        return merged
    local = merged.setdefault("segment_edits", {})
    external = edits_spec.get("segment_edits", edits_spec)
    if not isinstance(local, dict) or not isinstance(external, dict):
        return merged
    for device_id, edit in external.items():
        if not isinstance(edit, dict):
            continue
        target = local.setdefault(device_id, {})
        for key, value in edit.items():
            if key in ("replace", "append") and isinstance(value, list):
                target.setdefault(key, [])
                if isinstance(target[key], list):
                    target[key].extend(copy.deepcopy(value))
            elif key == "delete" and isinstance(value, list):
                target.setdefault(key, [])
                if isinstance(target[key], list):
                    target[key].extend(value)
            else:
                target[key] = copy.deepcopy(value)
    return merged


def _float_param(raw: dict[str, Any], key: str, default: float, *, device_id: str) -> float:
    value = raw.get(key, default)
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"{device_id}.{key} must be a finite number")
    return stable_float(value)


def _int_param(raw: dict[str, Any], key: str, default: int, *, device_id: str) -> int:
    value = raw.get(key, default)
    if not _is_int(value):
        raise ValueError(f"{device_id}.{key} must be an integer")
    return int(value)


def _device_bands(raw: dict[str, Any], config: PipelineConfig) -> tuple[tuple[float, ...], int]:
    if "band_edges_hz" in raw or "band_edges" in raw:
        edges_value = raw.get("band_edges_hz", raw.get("band_edges"))
        if not isinstance(edges_value, list):
            raise ValueError("band_edges_hz override must be a list")
        try:
            edges = tuple(stable_float(float(value)) for value in edges_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("band_edges_hz override must contain numbers") from exc
        declared = raw.get("band_count", max(0, len(edges) - 1))
        if not _is_int(declared):
            raise ValueError("band_count override must be an integer")
        valid, detail = _validate_bands(edges, declared)
        if not valid:
            raise ValueError(detail)
        return edges, int(declared)
    if not config.features.valid:
        raise ValueError("global band configuration is invalid")
    return config.features.band_edges_hz, config.features.band_count


def _project_markers(
    raw: dict[str, Any],
    explicit_name: str,
    stride_name: str,
    *,
    start_sample: int,
    segment_length: int,
    device_id: str,
) -> set[int]:
    explicit = raw.get(explicit_name, [])
    if explicit is None:
        explicit = []
    if not isinstance(explicit, list) or any(not _is_int(item) or item < 0 for item in explicit):
        raise ValueError(f"{device_id}.{explicit_name} must be a list of non-negative global indices")
    projected = {int(item) - start_sample for item in explicit if start_sample <= int(item) < start_sample + segment_length}
    stride = raw.get(stride_name)
    if stride is not None:
        if not _is_int(stride) or stride <= 0:
            raise ValueError(f"{device_id}.{stride_name} must be a positive integer or null")
        start = ((start_sample // stride) + 1) * stride - 1
        if start < start_sample:
            start += stride
        projected.update(global_index - start_sample for global_index in range(start, start_sample + segment_length, stride))
    return projected


def _harmonic_description(raw: dict[str, Any], device_id: str) -> tuple[list[int], list[float], list[float]]:
    orders = raw.get("harmonic_orders", [1, 2, 3])
    amplitudes = raw.get("harmonic_amplitudes", [1.0, 0.42, 0.23])
    phases = raw.get("harmonic_phases_deg", [0.0, 27.0, 63.0])
    if not isinstance(orders, list) or not isinstance(amplitudes, list) or not isinstance(phases, list):
        raise ValueError(f"{device_id} harmonic fields must be lists")
    if not (len(orders) == len(amplitudes) == len(phases) and orders):
        raise ValueError(f"{device_id} harmonic lists must have equal non-zero lengths")
    if any(not _is_int(order) or order <= 0 for order in orders):
        raise ValueError(f"{device_id}.harmonic_orders must be positive integers")
    if any(not _is_number(value) or not math.isfinite(float(value)) for value in amplitudes + phases):
        raise ValueError(f"{device_id} harmonic amplitudes and phases must be finite numbers")
    return [int(order) for order in orders], [stable_float(value) for value in amplitudes], [stable_float(value) for value in phases]


def _generate_segment(
    raw: dict[str, Any],
    device_id: str,
    *,
    segment_index: int,
    start_sample: int,
    segment_length: int,
    sample_rate_hz: float,
    seed: int,
    anomaly: bool,
    source: str,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    base_rpm = _float_param(raw, "base_rpm", 1500.0, device_id=device_id)
    rpm_drift_amplitude = _float_param(raw, "rpm_drift_amplitude", 28.0, device_id=device_id)
    rpm_drift_frequency_hz = _float_param(raw, "rpm_drift_frequency_hz", 0.7, device_id=device_id)
    rpm_drift_phase = _float_param(raw, "rpm_drift_phase_rad", 0.31, device_id=device_id)
    rpm_linear_drift = _float_param(raw, "rpm_linear_drift_per_s", 0.05, device_id=device_id)
    noise_std = _float_param(raw, "noise_std", 0.045, device_id=device_id)
    impulse_spacing = _float_param(raw, "impulse_spacing_s", 0.19, device_id=device_id)
    impulse_amplitude = _float_param(raw, "impulse_amplitude", 0.55, device_id=device_id)
    impulse_decay = _float_param(raw, "impulse_decay_per_s", 42.0, device_id=device_id)
    anomaly_scale = _float_param(raw, "anomaly_amplitude_scale", 2.4, device_id=device_id)
    if noise_std < 0 or impulse_spacing <= 0 or impulse_decay <= 0:
        raise ValueError(f"{device_id} noise/impulse parameters are outside valid ranges")
    orders, amplitudes, phases = _harmonic_description(raw, device_id)

    local = np.arange(segment_length, dtype=np.float64)
    absolute = start_sample + local
    time_s = absolute / sample_rate_hz
    drift_omega = 2.0 * math.pi * rpm_drift_frequency_hz
    if abs(rpm_drift_frequency_hz) > 1e-15:
        drift_cycles = -rpm_drift_amplitude / (60.0 * drift_omega) * np.cos(drift_omega * time_s + rpm_drift_phase)
    else:
        drift_cycles = np.zeros_like(time_s)
    rotation_cycles = base_rpm / 60.0 * time_s + drift_cycles + rpm_linear_drift / 120.0 * time_s * time_s
    rotation_phase = 2.0 * math.pi * rotation_cycles

    signal = np.zeros(segment_length, dtype=np.float64)
    for order, amplitude, phase_deg in zip(orders, amplitudes, phases, strict=True):
        signal += amplitude * np.sin(order * rotation_phase + math.radians(phase_deg))

    impulse_start = math.floor(time_s[0] / impulse_spacing) * impulse_spacing
    impulse_positions = np.arange(impulse_start, time_s[-1] + impulse_spacing, impulse_spacing)
    for position_s in impulse_positions:
        delta = time_s - float(position_s)
        mask = delta >= 0.0
        pulse_index = int(round(float(position_s) / impulse_spacing))
        pulse_sign = 1.0 if pulse_index % 2 == 0 else -1.0
        signal[mask] += pulse_sign * impulse_amplitude * np.exp(-impulse_decay * delta[mask])

    rng = np.random.default_rng(
        (int(seed) & 0xFFFFFFFFFFFFFFFF) ^ ((segment_index + 1) * 0x9E3779B97F4A7C15) ^ (0xA24BAED4963EE407 if anomaly else 0)
    )
    signal += rng.normal(0.0, noise_std, size=segment_length)

    if anomaly:
        onset = max(1, segment_length // 7)
        pulse_time = local / sample_rate_hz
        signal += pulse_sign * anomaly_scale * np.exp(-impulse_decay * 0.7 * np.maximum(pulse_time - onset / sample_rate_hz, 0.0)) * (local >= onset)
        signal *= 1.0 + 0.16 * np.sin(2.0 * math.pi * 7.0 * pulse_time)

    missing = _project_markers(
        raw,
        "missing_indices",
        "missing_stride",
        start_sample=start_sample,
        segment_length=segment_length,
        device_id=device_id,
    )
    resampled = _project_markers(
        raw,
        "resampled_indices",
        "resampled_stride",
        start_sample=start_sample,
        segment_length=segment_length,
        device_id=device_id,
    )

    for index in sorted(resampled):
        left = signal[index - 1] if index > 0 else signal[index]
        right = signal[index + 1] if index + 1 < segment_length else signal[index]
        signal[index] = 0.85 * signal[index] + 0.15 * (left + right) / 2.0
    for index in sorted(missing):
        signal[index] = np.nan
    return signal, tuple(sorted(missing)), tuple(sorted(resampled))


def _fill_segment(raw_samples: np.ndarray, missing: tuple[int, ...], resampled: tuple[int, ...]) -> np.ndarray:
    effective = np.asarray(raw_samples, dtype=np.float64).copy()
    filled = np.zeros_like(effective)
    filled[:] = effective
    marked = np.zeros(effective.size, dtype=bool)
    if missing:
        marked[np.asarray(missing, dtype=np.int64)] = True
    if resampled:
        marked[np.asarray(resampled, dtype=np.int64)] = True
    finite = np.isfinite(filled) & ~marked
    if not np.any(finite):
        filled[:] = 0.0
        return filled
    indices = np.arange(effective.size, dtype=np.float64)
    valid_indices = indices[finite]
    valid_values = filled[finite]
    filled[~finite] = np.interp(indices[~finite], valid_indices, valid_values)
    return filled


def _make_segment(
    raw: dict[str, Any],
    device_id: str,
    *,
    segment_index: int,
    start_sample: int,
    segment_length: int,
    sample_rate_hz: float,
    seed: int,
    anomaly: bool,
    source: str,
    explicit_samples: list[Any] | None = None,
) -> Segment:
    if explicit_samples is not None:
        if len(explicit_samples) != segment_length:
            raise ValueError(f"replacement segment {segment_index} has {len(explicit_samples)} samples, expected {segment_length}")
        raw_samples = np.asarray(explicit_samples, dtype=np.float64)
        # Explicit replacement data still honors marker lists from the device description.
        missing = _project_markers(
            raw,
            "missing_indices",
            "missing_stride",
            start_sample=start_sample,
            segment_length=segment_length,
            device_id=device_id,
        )
        resampled = _project_markers(
            raw,
            "resampled_indices",
            "resampled_stride",
            start_sample=start_sample,
            segment_length=segment_length,
            device_id=device_id,
        )
        for index in sorted(resampled):
            left = raw_samples[index - 1] if index > 0 else raw_samples[index]
            right = raw_samples[index + 1] if index + 1 < segment_length else raw_samples[index]
            raw_samples[index] = 0.85 * raw_samples[index] + 0.15 * (left + right) / 2.0
        missing_tuple = tuple(sorted(missing))
        resampled_tuple = tuple(sorted(resampled))
        # Explicit NaN values are treated as missing sampling points too.
        nan_indices = tuple(int(index) for index in np.flatnonzero(~np.isfinite(raw_samples)))
        missing_tuple = tuple(sorted(set(missing_tuple) | set(nan_indices)))
    else:
        raw_samples, missing_tuple, resampled_tuple = _generate_segment(
            raw,
            device_id,
            segment_index=segment_index,
            start_sample=start_sample,
            segment_length=segment_length,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
            anomaly=anomaly,
            source=source,
        )
    processed = _fill_segment(raw_samples, missing_tuple, resampled_tuple)
    waveform_fp = hash_object(
        {
            "device_id": device_id,
            "segment_index": segment_index,
            "start_sample": start_sample,
            "sample_rate_hz": sample_rate_hz,
            "array_fingerprint": array_fingerprint(raw_samples),
            "missing_indices": list(missing_tuple),
            "resampled_indices": list(resampled_tuple),
            "source": source,
        },
        namespace="segment-waveform-v1",
    )
    preprocess_fp = hash_object(
        {
            "method": "linear_interpolation_for_missing_and_resampled_v1",
            "missing_indices": list(missing_tuple),
            "resampled_indices": list(resampled_tuple),
            "processed_fingerprint": array_fingerprint(processed),
        },
        namespace="segment-preprocess-v1",
    )
    return Segment(
        device_id=device_id,
        segment_index=segment_index,
        start_sample=start_sample,
        end_sample=start_sample + segment_length,
        sample_rate_hz=sample_rate_hz,
        raw_samples=raw_samples,
        samples=processed,
        missing_indices=missing_tuple,
        resampled_indices=resampled_tuple,
        anomaly_hint=anomaly,
        source=source,
        waveform_fingerprint=waveform_fp,
        preprocess_fingerprint=preprocess_fp,
    )


def _normalize_append_entries(value: Any, device_id: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{device_id}.append must be a list")
    entries: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{device_id}.append entries must be mappings")
        count = item.get("count", 1)
        if not _is_int(count) or count <= 0:
            raise ValueError(f"{device_id}.append.count must be a positive integer")
        for offset in range(count):
            entry = copy.deepcopy(item)
            entry.pop("count", None)
            entry["_offset"] = offset
            entries.append(entry)
    return entries


def _load_device(
    raw: dict[str, Any],
    config: PipelineConfig,
    edits: dict[str, Any],
) -> DeviceData:
    device_id = raw.get("id")
    if not isinstance(device_id, str) or not device_id:
        raise ValueError("device id must be a non-empty string")
    segment_length = raw.get("segment_length", config.segment_length)
    if not _is_int(segment_length) or segment_length <= 0:
        raise ValueError("segment_length must be a positive integer")
    segment_length = int(segment_length)
    sample_count = raw.get("sample_count")
    if not _is_int(sample_count) or sample_count <= 0 or sample_count % segment_length != 0:
        raise ValueError(f"sample_count={sample_count} must be a positive multiple of segment_length={segment_length}")
    sample_count = int(sample_count)
    base_segment_count = sample_count // segment_length
    declared = raw.get("declared_segment_count", base_segment_count)
    if not _is_int(declared) or declared != base_segment_count:
        raise ValueError(f"declared_segment_count={declared} does not match {base_segment_count} segments")
    sample_rate_hz = _float_param(raw, "sample_rate_hz", 1000.0, device_id=device_id)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    seed = raw.get("seed", 1)
    if not _is_int(seed):
        raise ValueError("seed must be an integer")
    anomaly_ratio = raw.get("anomaly_ratio", 0.0)
    if not _is_number(anomaly_ratio) or not 0.0 <= float(anomaly_ratio) <= 1.0 or not math.isfinite(float(anomaly_ratio)):
        raise ValueError("anomaly_ratio must be in [0, 1]")
    anomaly_ratio = float(anomaly_ratio)
    bands, band_count = _device_bands(raw, config)
    nyquist = sample_rate_hz / 2.0
    if bands and bands[-1] > nyquist + 1e-12:
        raise ValueError(f"band boundary {bands[-1]} Hz exceeds Nyquist {nyquist} Hz")

    anomaly_count = min(base_segment_count, max(0, int(round(anomaly_ratio * base_segment_count))))
    anomaly_rng = np.random.default_rng((int(seed) & 0xFFFFFFFFFFFFFFFF) ^ 0xD1B54A32D192ED03)
    anomaly_segments = set(int(item) for item in np.sort(anomaly_rng.permutation(base_segment_count)[:anomaly_count]).tolist())

    base_segments: dict[int, Segment] = {}
    for index in range(base_segment_count):
        base_segments[index] = _make_segment(
            raw,
            device_id,
            segment_index=index,
            start_sample=index * segment_length,
            segment_length=segment_length,
            sample_rate_hz=sample_rate_hz,
            seed=int(seed),
            anomaly=index in anomaly_segments,
            source="generated",
        )

    replacement_map: dict[int, dict[str, Any]] = {}
    for key, replacement in (edits.get("replace") or {}).items() if isinstance(edits.get("replace"), dict) else []:
        if not _is_int(key):
            try:
                key = int(key)
            except (TypeError, ValueError):
                raise ValueError("replace keys must be segment indices")
        if not isinstance(replacement, dict):
            raise ValueError("replace entries must be mappings")
        replacement_map[int(key)] = replacement
    replace_list = edits.get("replace")
    if isinstance(replace_list, list):
        for replacement in replace_list:
            if not isinstance(replacement, dict) or not _is_int(replacement.get("segment_index")):
                raise ValueError("replace list entries require integer segment_index")
            replacement_map[int(replacement["segment_index"])] = replacement
    for index, replacement in sorted(replacement_map.items()):
        if index not in base_segments:
            raise ValueError(f"replacement refers to unknown segment {index}")
        seed_delta = replacement.get("seed_delta", 0)
        if not _is_int(seed_delta):
            raise ValueError("replace.seed_delta must be an integer")
        anomaly = bool(replacement.get("anomaly", index in anomaly_segments))
        explicit = replacement.get("samples")
        if explicit is not None and not isinstance(explicit, list):
            raise ValueError("replace.samples must be a list")
        source_label = str(replacement.get("source", f"replacement:{seed_delta}"))
        base_segments[index] = _make_segment(
            raw,
            device_id,
            segment_index=index,
            start_sample=index * segment_length,
            segment_length=segment_length,
            sample_rate_hz=sample_rate_hz,
            seed=int(seed) + int(seed_delta),
            anomaly=anomaly,
            source=source_label,
            explicit_samples=explicit,
        )

    delete_value = edits.get("delete", [])
    if delete_value is None:
        delete_value = []
    if not isinstance(delete_value, list) or any(not _is_int(index) for index in delete_value):
        raise ValueError("delete must be a list of integer segment indices")
    for index in sorted(set(int(item) for item in delete_value)):
        base_segments.pop(index, None)

    segments = dict(base_segments)
    append_entries = _normalize_append_entries(edits.get("append"), device_id)
    for append_position, entry in enumerate(append_entries):
        index = base_segment_count + append_position
        seed_delta = entry.get("seed_delta", append_position + 1)
        if not _is_int(seed_delta):
            raise ValueError("append.seed_delta must be an integer")
        start_sample = index * segment_length
        explicit = entry.get("samples")
        if explicit is not None and not isinstance(explicit, list):
            raise ValueError("append.samples must be a list")
        segments[index] = _make_segment(
            raw,
            device_id,
            segment_index=index,
            start_sample=start_sample,
            segment_length=segment_length,
            sample_rate_hz=sample_rate_hz,
            seed=int(seed) + int(seed_delta),
            anomaly=bool(entry.get("anomaly", False)),
            source=str(entry.get("source", f"append:{seed_delta}")),
            explicit_samples=explicit,
        )

    ordered = tuple(segments[index] for index in sorted(segments))
    waveform_fp = hash_object(
        {
            "device_id": device_id,
            "sample_rate_hz": sample_rate_hz,
            "segment_length": segment_length,
            "band_edges_hz": list(bands),
            "segments": [
                {
                    "segment_index": segment.segment_index,
                    "start_sample": segment.start_sample,
                    "waveform_fingerprint": segment.waveform_fingerprint,
                    "preprocess_fingerprint": segment.preprocess_fingerprint,
                }
                for segment in ordered
            ],
        },
        namespace="device-waveform-v1",
    )
    return DeviceData(
        device_id=device_id,
        sample_rate_hz=sample_rate_hz,
        segment_length=segment_length,
        band_edges_hz=bands,
        band_count=band_count,
        segments=ordered,
        waveform_fingerprint=waveform_fp,
    )


def _failure_for_device(error: Exception, device_id: str | None) -> Failure:
    detail = str(error)
    if "anomaly_ratio" in detail:
        code = E_RATIO_OUT_OF_RANGE
    elif "segment_length must be a positive integer" in detail:
        code = E_SEGMENT_LEN_INVALID
    elif (
        "sample_count" in detail
        or "declared_segment_count" in detail
        or "replacement segment" in detail
        or "multiple of segment_length" in detail
    ):
        code = E_SEGMENT_MISMATCH
    elif "band" in detail.lower() or "nyquist" in detail.lower():
        code = E_BAND_INVALID
    else:
        code = E_CONFIG_INVALID
    return Failure(device_id, None, None, "load", code, detail, None)


def load_data(data_spec: dict[str, Any], config: PipelineConfig, *, edits_spec: dict[str, Any] | None = None) -> LoadedData:
    spec = merge_edits(data_spec, edits_spec)
    raw_devices = spec.get("devices", [])
    failures: list[Failure] = []
    if not isinstance(raw_devices, list):
        failures.append(Failure(None, None, None, "load", E_CONFIG_INVALID, "devices must be a list", None))
        raw_devices = []
    edits_all = spec.get("segment_edits", {})
    if not isinstance(edits_all, dict):
        failures.append(Failure(None, None, None, "load", E_CONFIG_INVALID, "segment_edits must be a mapping", None))
        edits_all = {}
    devices: dict[str, DeviceData] = {}
    seen: set[str] = set()
    for raw in raw_devices:
        if not isinstance(raw, dict):
            failures.append(Failure(None, None, None, "load", E_CONFIG_INVALID, "device entries must be mappings", None))
            continue
        device_id = raw.get("id") if isinstance(raw.get("id"), str) else None
        if config.known_devices and device_id not in config.known_devices:
            failures.append(
                Failure(device_id, None, None, "load", E_CONFIG_INVALID, f"unknown device {device_id!r}", None)
            )
            continue
        if device_id in seen:
            failures.append(Failure(device_id, None, None, "load", E_CONFIG_INVALID, f"duplicate device {device_id!r}", None))
            continue
        if device_id is not None:
            seen.add(device_id)
        device_edits = edits_all.get(device_id, {}) if device_id is not None else {}
        if not isinstance(device_edits, dict):
            failures.append(Failure(device_id, None, None, "load", E_CONFIG_INVALID, "device segment_edits must be a mapping", None))
            continue
        try:
            device = _load_device(raw, config, device_edits)
        except Exception as error:  # A bad device must not stop the remaining devices.
            failures.append(_failure_for_device(error, device_id))
            continue
        devices[device.device_id] = device

    waveform_fp = hash_object(
        {
            "devices": [
                {
                    "device_id": device.device_id,
                    "waveform_fingerprint": device.waveform_fingerprint,
                    "segments": [
                        {
                            "segment_index": segment.segment_index,
                            "waveform_fingerprint": segment.waveform_fingerprint,
                            "preprocess_fingerprint": segment.preprocess_fingerprint,
                        }
                        for segment in device.segments
                    ],
                }
                for device in (devices[name] for name in sorted(devices))
            ],
            "segment_edits": edits_all,
        },
        namespace="dataset-waveform-v1",
    )
    segment_count = sum(len(device.segments) for device in devices.values())
    return LoadedData(
        devices=devices,
        failures=sorted(failures, key=Failure.sort_key),
        waveform_fingerprint=waveform_fp,
        segment_count=segment_count,
        synthetic_description_hash=hash_object(spec, namespace="synthetic-description-v1"),
    )
