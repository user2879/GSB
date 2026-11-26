"""Fixed-point per-window RMS, kurtosis, and band-energy observations."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import AppConfig
from .fingerprints import observation_cache_key
from .windows import SampleCell, Window


@dataclass(frozen=True)
class WindowObservation:
    window_id: str
    cache_key: str
    feature_names: tuple[str, ...]
    real_values: tuple[float, ...]
    integer_values: tuple[int, ...]
    valid_sample_count: int
    filled_sample_count: int
    total_sample_count: int
    resampled_sample_count: int
    fill_records: tuple[dict[str, Any], ...]
    block_fingerprints: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "cache_key": self.cache_key,
            "feature_names": list(self.feature_names),
            "real_values": list(self.real_values),
            "integer_values": list(self.integer_values),
            "valid_sample_count": self.valid_sample_count,
            "filled_sample_count": self.filled_sample_count,
            "total_sample_count": self.total_sample_count,
            "resampled_sample_count": self.resampled_sample_count,
            "fill_records": list(self.fill_records),
            "block_fingerprints": list(self.block_fingerprints),
        }


def _nearest_valid_index(valid_indexes: list[int], missing_index: int) -> int | None:
    if not valid_indexes:
        return None
    position = bisect.bisect_left(valid_indexes, missing_index)
    candidates: list[int] = []
    if position < len(valid_indexes):
        candidates.append(valid_indexes[position])
    if position > 0:
        candidates.append(valid_indexes[position - 1])
    return min(candidates, key=lambda value: (abs(value - missing_index), value))


def _fill_channel(
    *,
    channel_id: str,
    start: int,
    end: int,
    sample_map: dict[int, SampleCell],
) -> tuple[list[float], int, int, int, list[dict[str, Any]]]:
    indexes = list(range(start, end))
    values: list[float] = []
    valid_indexes: list[int] = []
    cells_by_index: dict[int, SampleCell] = {}
    resampled_count = 0
    for index in indexes:
        cell = sample_map.get(index)
        if cell is None:
            continue
        cells_by_index[index] = cell
        valid_indexes.append(index)
        if cell.resampled:
            resampled_count += 1

    fill_records: list[dict[str, Any]] = []
    fill_run: dict[str, Any] | None = None
    for index in indexes:
        cell = cells_by_index.get(index)
        if cell is not None:
            values.append(cell.value)
            fill_run = None
            continue
        source_index = _nearest_valid_index(valid_indexes, index)
        if source_index is None:
            source = "zero_no_valid_sample"
            fill_value = 0.0
            source_block = None
        else:
            source_cell = cells_by_index[source_index]
            source = "nearest_valid_in_window"
            fill_value = source_cell.value
            source_block = source_cell.block_id
        values.append(fill_value)
        token = (source, source_index, source_block)
        if fill_run is None or fill_run["_token"] != token:
            fill_run = {
                "_token": token,
                "channel": channel_id,
                "start": index,
                "end": index + 1,
                "source": source,
                "source_sample_index": source_index,
                "source_block": source_block,
            }
            fill_records.append(fill_run)
        else:
            fill_run["end"] = index + 1

    for record in fill_records:
        record.pop("_token", None)
    return values, len(valid_indexes), len(indexes) - len(valid_indexes), resampled_count, fill_records


def _channel_features(values: list[float], sample_rate_hz: int, config: AppConfig) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    centered = array - mean
    rms = float(np.sqrt(np.mean(array * array)))
    variance = float(np.mean(centered * centered))
    if variance <= 0.0:
        kurtosis = 0.0
    else:
        fourth = float(np.mean(centered * centered * centered * centered))
        kurtosis = fourth / (variance * variance)
    spectrum = np.fft.rfft(centered)
    frequency = np.fft.rfftfreq(array.size, d=1.0 / sample_rate_hz)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag) / float(array.size * array.size)
    energies: list[float] = []
    for band in config.bands:
        mask = (frequency >= band.low_hz) & (frequency < band.high_hz)
        energies.append(float(np.sum(power[mask])) if bool(np.any(mask)) else 0.0)
    return [round(rms, 12), round(kurtosis, 12), *[round(value, 12) for value in energies]]


def _quantize(value: float, scale: int) -> int:
    if not math.isfinite(value):
        raise ValueError("feature value is not finite")
    return int(round(value * scale))


def observe_window(
    window: Window,
    config: AppConfig,
    sample_map: dict[str, dict[int, SampleCell]],
    block_fingerprints: dict[str, str],
) -> dict[str, Any]:
    feature_names: list[str] = []
    real_values: list[float] = []
    fill_records: list[dict[str, Any]] = []
    valid_total = 0
    filled_total = 0
    total_samples = 0
    resampled_total = 0
    for channel in sorted(config.channels, key=lambda item: item.channel_id):
        start = window.start_sample_by_channel[channel.channel_id]
        end = window.end_sample_by_channel[channel.channel_id]
        values, valid, filled, resampled, records = _fill_channel(
            channel_id=channel.channel_id,
            start=start,
            end=end,
            sample_map=sample_map[channel.channel_id],
        )
        metrics = _channel_features(values, channel.sample_rate_hz, config)
        metric_names = ["rms", "kurtosis", *[f"band_{index}" for index in range(len(config.bands))]]
        feature_names.extend(f"{channel.channel_id}.{name}" for name in metric_names)
        real_values.extend(metrics)
        fill_records.extend(records)
        valid_total += valid
        filled_total += filled
        total_samples += end - start
        resampled_total += resampled

    fps = [
        {"block_id": block_id, "fingerprint": block_fingerprints[block_id]}
        for block_id in sorted(window.block_refs)
    ]
    key = observation_cache_key(
        window_geometry=window.geometry(config),
        block_fingerprints=fps,
        config=config,
    )
    integer_values = [_quantize(value, config.fixed_point.scale) for value in real_values]
    return {
        "window_id": window.window_id,
        "cache_key": key,
        "feature_names": feature_names,
        "real_values": real_values,
        "integer_values": integer_values,
        "valid_sample_count": valid_total,
        "filled_sample_count": filled_total,
        "total_sample_count": total_samples,
        "resampled_sample_count": resampled_total,
        "fill_records": fill_records,
        "block_fingerprints": fps,
    }


def observation_from_dict(data: dict[str, Any]) -> WindowObservation:
    return WindowObservation(
        window_id=data["window_id"],
        cache_key=data["cache_key"],
        feature_names=tuple(data["feature_names"]),
        real_values=tuple(data["real_values"]),
        integer_values=tuple(data["integer_values"]),
        valid_sample_count=data["valid_sample_count"],
        filled_sample_count=data["filled_sample_count"],
        total_sample_count=data["total_sample_count"],
        resampled_sample_count=data["resampled_sample_count"],
        fill_records=tuple(data["fill_records"]),
        block_fingerprints=tuple(data["block_fingerprints"]),
    )
