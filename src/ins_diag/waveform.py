from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    EffectiveConfig,
    WaveformChannelSpec,
    WaveformSpec,
    override_bandwidth,
    override_family_count,
    override_fractional_bits,
    override_quantization_units,
)
from .errors import BusinessError, ConfigValidationError, FailureCollector
from .util import derive_int_seed, sha256_bytes


@dataclass
class ChannelData:
    device_id: str
    channel_id: str
    samples: np.ndarray
    present: np.ndarray
    sample_rate_hz: float

    def clone(self) -> "ChannelData":
        return ChannelData(
            self.device_id,
            self.channel_id,
            self.samples.copy(),
            self.present.copy(),
            self.sample_rate_hz,
        )


@dataclass
class Dataset:
    channels: list[ChannelData]
    seed: int

    def find(self, device_id: str, channel_id: str) -> ChannelData | None:
        for channel in self.channels:
            if channel.device_id == device_id and channel.channel_id == channel_id:
                return channel
        return None

    def content_fingerprint(self) -> str:
        parts: list[bytes] = []
        for channel in self.channels:
            values = np.ascontiguousarray(channel.samples, dtype="<f8")
            present = np.ascontiguousarray(channel.present, dtype=np.uint8)
            parts.append(channel.device_id.encode("utf-8"))
            parts.append(channel.channel_id.encode("utf-8"))
            parts.append(str(channel.sample_rate_hz).encode("ascii"))
            parts.append(len(channel.samples).to_bytes(8, "little"))
            parts.append(values.tobytes(order="C"))
            parts.append(present.tobytes(order="C"))
        return sha256_bytes(b"\x1f".join(parts))


def _channel_specs(spec: WaveformSpec) -> dict[tuple[str, str], WaveformChannelSpec]:
    result: dict[tuple[str, str], WaveformChannelSpec] = {}
    for device in spec.devices:
        for channel in device.channels:
            result[(device.device_id, channel.channel_id)] = channel
    return result


def _generate_interval(
    spec: WaveformChannelSpec,
    sample_rate_hz: float,
    start: int,
    count: int,
    seed: int,
    *,
    noise_scale: float = 1.0,
    extra_jitter: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(derive_int_seed(seed, spec.channel_id, start, count))
    indices = np.arange(start, start + count, dtype=np.float64)
    values = spec.bias + spec.drift_per_sample * indices
    for term in spec.periodic:
        values += term.amplitude * np.sin(
            2.0 * math.pi * term.frequency_hz * indices / sample_rate_hz + term.phase
        )
    if spec.noise_sigma > 0:
        values += rng.normal(0.0, spec.noise_sigma * noise_scale, count)
    if extra_jitter:
        values += extra_jitter * np.sin(2.0 * math.pi * 0.37 * indices + 0.31)
    return np.asarray(values, dtype=np.float64)


def synthesize_waveform(spec: WaveformSpec, seed: int) -> Dataset:
    channels: list[ChannelData] = []
    for device in spec.devices:
        for channel in device.channels:
            values = _generate_interval(
                channel,
                spec.sample_rate_hz,
                0,
                channel.samples,
                seed,
            )
            present = np.ones(channel.samples, dtype=np.bool_)
            for start, end in channel.missing_segments:
                present[start:end] = False
            for injection in channel.non_finite_samples:
                values[injection.index] = injection.value
            channels.append(
                ChannelData(
                    device_id=device.device_id,
                    channel_id=channel.channel_id,
                    samples=values,
                    present=present,
                    sample_rate_hz=spec.sample_rate_hz,
                )
            )
    return Dataset(channels=channels, seed=seed)


def _edit_error(code: str, path: str, message: str) -> BusinessError:
    return BusinessError(code=code, path=path, message=message, stage="edit")


def _clone_dataset(dataset: Dataset) -> Dataset:
    return Dataset([channel.clone() for channel in dataset.channels], dataset.seed)


def apply_edit_commands(
    dataset: Dataset,
    config: EffectiveConfig,
    commands: tuple[dict[str, Any], ...],
) -> tuple[Dataset, EffectiveConfig, list[BusinessError], list[dict[str, Any]]]:
    from .windows import build_windows

    current = _clone_dataset(dataset)
    current_config = config
    failures: FailureCollector = FailureCollector()
    applied: list[dict[str, Any]] = []
    specs = _channel_specs(config.waveform)

    for index, command in enumerate(commands):
        path = f"edits.commands[{index}]"
        op = command.get("op")
        try:
            if op == "noop":
                applied.append({"index": index, "op": op, "status": "applied"})
                continue
            if op == "append_batch":
                channel = current.find(
                    str(command.get("device_id")), str(command.get("channel_id"))
                )
                if channel is None:
                    failures.add(
                        _edit_error(
                            "E_EDIT_TARGET_NOT_FOUND",
                            f"{path}.device_id",
                            "device/channel combination does not exist",
                        )
                    )
                    continue
                spec = specs[(channel.device_id, channel.channel_id)]
                count = int(command["count"])
                start = int(channel.samples.size)
                values = _generate_interval(
                    spec,
                    channel.sample_rate_hz,
                    start,
                    count,
                    int(command["seed"]),
                )
                channel.samples = np.concatenate([channel.samples, values])
                channel.present = np.concatenate(
                    [channel.present, np.ones(count, dtype=np.bool_)]
                )
                applied.append(
                    {
                        "index": index,
                        "op": op,
                        "status": "applied",
                        "device_id": channel.device_id,
                        "channel_id": channel.channel_id,
                        "start": start,
                        "count": count,
                    }
                )
                continue
            if op == "replace_window":
                windows = build_windows(current, current_config)
                window_id = str(command.get("window_id"))
                target = next((window for window in windows if window.window_id == window_id), None)
                if target is None:
                    failures.add(
                        _edit_error(
                            "E_WINDOW_REF_OUT_OF_RANGE",
                            f"{path}.window_id",
                            f"window reference does not exist in the current window set: {window_id}",
                        )
                    )
                    continue
                channel = current.find(target.device_id, target.channel_id)
                assert channel is not None
                spec = specs[(target.device_id, target.channel_id)]
                count = target.end - target.start
                replacement = _generate_interval(
                    spec,
                    channel.sample_rate_hz,
                    target.start,
                    count,
                    int(command["seed"]),
                    noise_scale=4.0,
                    extra_jitter=1.25,
                )
                channel.samples[target.start : target.end] = replacement
                channel.present[target.start : target.end] = True
                applied.append(
                    {
                        "index": index,
                        "op": op,
                        "status": "applied",
                        "window_id": window_id,
                        "start": target.start,
                        "end": target.end,
                    }
                )
                continue
            if op == "delete_segment":
                channel = current.find(
                    str(command.get("device_id")), str(command.get("channel_id"))
                )
                if channel is None:
                    failures.add(
                        _edit_error(
                            "E_EDIT_TARGET_NOT_FOUND",
                            f"{path}.device_id",
                            "device/channel combination does not exist",
                        )
                    )
                    continue
                start = int(command["start"])
                length = int(command["length"])
                end = start + length
                if start < 0 or end > channel.samples.size or length <= 0:
                    failures.add(
                        _edit_error(
                            "E_EDIT_RANGE_OUT_OF_BOUNDS",
                            f"{path}.start",
                            f"segment [{start}, {end}) exceeds channel length {channel.samples.size}",
                        )
                    )
                    continue
                channel.samples = np.concatenate(
                    [channel.samples[:start], channel.samples[end:]]
                )
                channel.present = np.concatenate(
                    [channel.present[:start], channel.present[end:]]
                )
                applied.append(
                    {
                        "index": index,
                        "op": op,
                        "status": "applied",
                        "device_id": channel.device_id,
                        "channel_id": channel.channel_id,
                        "start": start,
                        "length": length,
                    }
                )
                continue
            if op == "set_family_count":
                current_config = override_family_count(
                    current_config, int(command["value"]), f"{path}.value"
                )
                applied.append({"index": index, "op": op, "status": "applied", "value": command["value"]})
                continue
            if op == "set_kernel_bandwidth":
                current_config = override_bandwidth(
                    current_config, str(command["value"]), f"{path}.value"
                )
                applied.append({"index": index, "op": op, "status": "applied", "value": command["value"]})
                continue
            if op == "set_fractional_bits":
                current_config = override_fractional_bits(
                    current_config, int(command["value"]), f"{path}.value"
                )
                applied.append({"index": index, "op": op, "status": "applied", "value": command["value"]})
                continue
            if op == "set_quantization_units":
                current_config = override_quantization_units(
                    current_config, int(command["value"]), f"{path}.value"
                )
                applied.append({"index": index, "op": op, "status": "applied", "value": command["value"]})
                continue
            failures.add(
                _edit_error(
                    "E_EDIT_OP_UNKNOWN",
                    f"{path}.op",
                    f"unsupported edit operation: {op}",
                )
            )
        except ConfigValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            failures.add(
                _edit_error(
                    "E_EDIT_INVALID",
                    path,
                    f"invalid edit command: {exc}",
                )
            )
    return current, current_config, failures.sorted(), applied

