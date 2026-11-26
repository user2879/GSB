"""Signal-block declaration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import is_int, is_number, sha256_value
from .config import AppConfig
from .errors import E_CONFIG_INVALID, ValidationError


@dataclass(frozen=True)
class SampleRange:
    channel: str
    start: int
    end: int

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end

    def to_dict(self) -> dict[str, Any]:
        return {"channel": self.channel, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class ResamplingMarker:
    channel: str
    start: int
    end: int
    source_rate_hz: int
    target_rate_hz: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "start": self.start,
            "end": self.end,
            "source_rate_hz": self.source_rate_hz,
            "target_rate_hz": self.target_rate_hz,
        }


@dataclass(frozen=True)
class SignalBlock:
    block_id: str
    arrival_seq: int
    acquisition_ms: int
    signals: dict[str, tuple[float, ...]]
    missing_segments: tuple[SampleRange, ...] = ()
    resampling_markers: tuple[ResamplingMarker, ...] = ()

    def length_for(self, channel: str) -> int:
        return len(self.signals[channel])

    @property
    def content_fingerprint(self) -> str:
        return sha256_value(
            {
                "kind": "signal-block-content",
                "acquisition_ms": self.acquisition_ms,
                "signals": {name: list(values) for name, values in self.signals.items()},
                "missing_segments": [item.to_dict() for item in self.missing_segments],
                "resampling_markers": [item.to_dict() for item in self.resampling_markers],
            }
        )

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "block_id": self.block_id,
            "arrival_seq": self.arrival_seq,
            "acquisition_ms": self.acquisition_ms,
        }
        if include_samples:
            payload["signals"] = {name: list(values) for name, values in self.signals.items()}
        payload["missing_segments"] = [item.to_dict() for item in self.missing_segments]
        payload["resampling_markers"] = [item.to_dict() for item in self.resampling_markers]
        return payload


def _expect_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(E_CONFIG_INVALID, "must be an object", path)
    return value


def _expect_int(value: Any, path: str, minimum: int | None = None) -> int:
    if not is_int(value):
        raise ValidationError(E_CONFIG_INVALID, "must be an integer", path)
    if minimum is not None and value < minimum:
        raise ValidationError(E_CONFIG_INVALID, f"must be >= {minimum}", path)
    return value


def _expect_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(E_CONFIG_INVALID, "must be a non-empty string", path)
    return value


def _parse_range(obj: dict[str, Any], path: str, channels: set[str], lengths: dict[str, int]) -> SampleRange:
    channel = _expect_id(obj.get("channel"), f"{path}.channel")
    if channel not in channels:
        raise ValidationError(E_CONFIG_INVALID, "unknown channel", f"{path}.channel")
    start = _expect_int(obj.get("start"), f"{path}.start", 0)
    end = _expect_int(obj.get("end"), f"{path}.end", 0)
    if end <= start:
        raise ValidationError(E_CONFIG_INVALID, "end must be greater than start", f"{path}.end")
    if end > lengths[channel]:
        raise ValidationError(E_CONFIG_INVALID, "range exceeds block length", f"{path}.end")
    return SampleRange(channel, start, end)


def parse_block(data: Any, config: AppConfig, path: str = "$") -> SignalBlock:
    obj = _expect_object(data, path)
    allowed = config.channel_map
    block_id = _expect_id(obj.get("block_id"), f"{path}.block_id")
    arrival_seq = _expect_int(obj.get("arrival_seq"), f"{path}.arrival_seq", 0)
    acquisition_ms = _expect_int(obj.get("acquisition_ms"), f"{path}.acquisition_ms", 0)

    signals_raw = obj.get("signals")
    if not isinstance(signals_raw, dict) or not signals_raw:
        raise ValidationError(E_CONFIG_INVALID, "signals must be a non-empty object", f"{path}.signals")
    unknown = sorted(set(signals_raw) - set(allowed))
    if unknown:
        raise ValidationError(E_CONFIG_INVALID, f"unknown channel(s): {unknown}", f"{path}.signals")
    missing_channels = sorted(set(allowed) - set(signals_raw))
    if missing_channels:
        raise ValidationError(
            E_CONFIG_INVALID,
            f"missing declared channel(s): {missing_channels}",
            f"{path}.signals",
        )

    signals: dict[str, tuple[float, ...]] = {}
    lengths: dict[str, int] = {}
    for channel in sorted(signals_raw):
        values_raw = signals_raw[channel]
        signal_path = f"{path}.signals.{channel}"
        if not isinstance(values_raw, list) or not values_raw:
            raise ValidationError(E_CONFIG_INVALID, "must be a non-empty array", signal_path)
        values: list[float] = []
        for index, value in enumerate(values_raw):
            if not is_number(value):
                raise ValidationError(E_CONFIG_INVALID, "sample must be a finite number", f"{signal_path}[{index}]")
            values.append(float(value))
        signals[channel] = tuple(values)
        lengths[channel] = len(values)

    missing_raw = obj.get("missing_segments", [])
    if not isinstance(missing_raw, list):
        raise ValidationError(E_CONFIG_INVALID, "must be an array", f"{path}.missing_segments")
    missing: list[SampleRange] = []
    for index, item in enumerate(missing_raw):
        item_path = f"{path}.missing_segments[{index}]"
        missing.append(_parse_range(_expect_object(item, item_path), item_path, set(allowed), lengths))
    for left_index, left in enumerate(missing):
        for right in missing[left_index + 1 :]:
            if left.channel == right.channel and max(left.start, right.start) < min(left.end, right.end):
                raise ValidationError(E_CONFIG_INVALID, "missing segments must not overlap", f"{path}.missing_segments")

    resampling_raw = obj.get("resampling_markers", [])
    if not isinstance(resampling_raw, list):
        raise ValidationError(E_CONFIG_INVALID, "must be an array", f"{path}.resampling_markers")
    resampling: list[ResamplingMarker] = []
    for index, item in enumerate(resampling_raw):
        item_path = f"{path}.resampling_markers[{index}]"
        range_obj = _expect_object(item, item_path)
        sample_range = _parse_range(range_obj, item_path, set(allowed), lengths)
        source_rate = _expect_int(range_obj.get("source_rate_hz"), f"{item_path}.source_rate_hz", 1)
        target_rate = _expect_int(range_obj.get("target_rate_hz"), f"{item_path}.target_rate_hz", 1)
        resampling.append(
            ResamplingMarker(
                channel=sample_range.channel,
                start=sample_range.start,
                end=sample_range.end,
                source_rate_hz=source_rate,
                target_rate_hz=target_rate,
            )
        )

    return SignalBlock(
        block_id=block_id,
        arrival_seq=arrival_seq,
        acquisition_ms=acquisition_ms,
        signals=signals,
        missing_segments=tuple(missing),
        resampling_markers=tuple(resampling),
    )


def parse_block_stream(data: Any, config: AppConfig) -> list[SignalBlock]:
    obj = _expect_object(data, "$")
    blocks_raw = obj.get("blocks")
    if not isinstance(blocks_raw, list):
        raise ValidationError(E_CONFIG_INVALID, "blocks must be an array", "blocks")
    blocks: list[SignalBlock] = []
    previous_seq: int | None = None
    seen_ids: set[str] = set()
    for index, item in enumerate(blocks_raw):
        block = parse_block(item, config, f"blocks[{index}]")
        if block.block_id in seen_ids:
            raise ValidationError(E_CONFIG_INVALID, "duplicate block_id", f"blocks[{index}].block_id")
        if previous_seq is not None and block.arrival_seq <= previous_seq:
            raise ValidationError(
                E_CONFIG_INVALID,
                "arrival_seq must be strictly increasing",
                f"blocks[{index}].arrival_seq",
            )
        seen_ids.add(block.block_id)
        previous_seq = block.arrival_seq
        blocks.append(block)
    return blocks


def block_overlap_refs(blocks: Iterable[SignalBlock], config: AppConfig) -> dict[str, int]:
    """Return the earliest arrival sequence for each block id."""

    return {block.block_id: block.arrival_seq for block in blocks}
