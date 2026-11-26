"""Application configuration model and strict loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import is_int, sha256_value
from .errors import ConfigError, E_CONFIG_INVALID, ValidationError


FRACTION_BIT_LEVELS = (8, 10, 12, 16, 20, 24)
QUANTIZATION_LEVELS = (32, 64, 128, 256, 512, 1024)
MERGE_RULES = ("nearest_l1",)


@dataclass(frozen=True)
class ChannelConfig:
    channel_id: str
    sample_rate_hz: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.channel_id, "sample_rate_hz": self.sample_rate_hz}


@dataclass(frozen=True)
class WindowConfig:
    length: int
    step: int
    retention_depth: int

    def to_dict(self) -> dict[str, int]:
        return {
            "length": self.length,
            "step": self.step,
            "retention_depth": self.retention_depth,
        }


@dataclass(frozen=True)
class BandConfig:
    name: str
    low_hz: float
    high_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "low_hz": self.low_hz, "high_hz": self.high_hz}


@dataclass(frozen=True)
class ClusterConfig:
    count: int
    min_members: int
    merge_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min_members": self.min_members,
            "merge_rule": self.merge_rule,
        }


@dataclass(frozen=True)
class FixedPointConfig:
    fraction_bits: int
    quantization_levels: int

    @property
    def scale(self) -> int:
        """Combined integer work scale.

        Both selected controls are semantically active: every feature integer is
        ``round(value * 2**fraction_bits * quantization_levels)``.  Therefore
        changing either control changes persisted accumulators, not just labels.
        """

        return (1 << self.fraction_bits) * self.quantization_levels

    def to_dict(self) -> dict[str, int]:
        return {
            "fraction_bits": self.fraction_bits,
            "quantization_levels": self.quantization_levels,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    batch_point_budget: int
    cluster_point_quota: int

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_point_budget": self.batch_point_budget,
            "cluster_point_quota": self.cluster_point_quota,
        }


@dataclass(frozen=True)
class SegmentConfig:
    min_windows: int

    def to_dict(self) -> dict[str, int]:
        return {"min_windows": self.min_windows}


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    channels: tuple[ChannelConfig, ...]
    window: WindowConfig
    bands: tuple[BandConfig, ...]
    clustering: ClusterConfig
    fixed_point: FixedPointConfig
    scheduler: SchedulerConfig
    segment: SegmentConfig

    @property
    def channel_map(self) -> dict[str, ChannelConfig]:
        return {channel.channel_id: channel for channel in self.channels}

    @property
    def window_fingerprint(self) -> str:
        return sha256_value({"kind": "window", "value": self.window.to_dict()})

    @property
    def cluster_fingerprint(self) -> str:
        payload = self.clustering.to_dict()
        payload["merge_rule"] = self.clustering.merge_rule
        return sha256_value({"kind": "cluster", "value": payload})

    @property
    def fixed_point_fingerprint(self) -> str:
        return sha256_value({"kind": "fixed-point", "value": self.fixed_point.to_dict()})

    @property
    def config_hash(self) -> str:
        return sha256_value({"kind": "app-config", "value": self.to_dict()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "channels": [channel.to_dict() for channel in self.channels],
            "window": self.window.to_dict(),
            "bands": [band.to_dict() for band in self.bands],
            "clustering": self.clustering.to_dict(),
            "fixed_point": self.fixed_point.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "segment": self.segment.to_dict(),
        }


def _path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent != "$" else key


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


def _expect_positive(value: Any, path: str) -> int:
    return _expect_int(value, path, 1)


def _expect_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(E_CONFIG_INVALID, "must be a non-empty string", path)
    return value


def _expect_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(E_CONFIG_INVALID, "must be a finite number", path)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValidationError(E_CONFIG_INVALID, "must be a finite number", path)
    return number


def _parse_config(data: Any) -> AppConfig:
    root = _expect_object(data, "$")
    issues: list[ValidationError] = []

    schema_version = 1
    try:
        schema_version = _expect_int(root.get("schema_version", 1), "schema_version", 1)
        if schema_version != 1:
            raise ValidationError(E_CONFIG_INVALID, "unsupported schema_version", "schema_version")
    except ValidationError as exc:
        issues.append(exc)

    channels_raw = root.get("channels")
    channels: list[ChannelConfig] = []
    seen_channels: set[str] = set()
    if not isinstance(channels_raw, list) or not channels_raw:
        issues.append(ValidationError(E_CONFIG_INVALID, "must be a non-empty array", "channels"))
    else:
        for index, item in enumerate(channels_raw):
            path = f"channels[{index}]"
            try:
                obj = _expect_object(item, path)
                channel_id = _expect_string(obj.get("id"), f"{path}.id")
                if channel_id in seen_channels:
                    raise ValidationError(E_CONFIG_INVALID, "duplicate channel id", f"{path}.id")
                sample_rate = _expect_positive(obj.get("sample_rate_hz"), f"{path}.sample_rate_hz")
                seen_channels.add(channel_id)
                channels.append(ChannelConfig(channel_id, sample_rate))
            except ValidationError as exc:
                issues.append(exc)

    window_raw = root.get("window")
    window = WindowConfig(1, 1, 1)
    try:
        obj = _expect_object(window_raw, "window")
        length = _expect_positive(obj.get("length"), "window.length")
        step = _expect_positive(obj.get("step"), "window.step")
        if step > length:
            raise ValidationError(E_CONFIG_INVALID, "step must not exceed length", "window.step")
        retention = _expect_positive(obj.get("retention_depth"), "window.retention_depth")
        window = WindowConfig(length, step, retention)
    except ValidationError as exc:
        issues.append(exc)

    bands_raw = root.get("bands")
    bands: list[BandConfig] = []
    seen_bands: set[str] = set()
    if not isinstance(bands_raw, list) or not bands_raw:
        issues.append(ValidationError(E_CONFIG_INVALID, "must be a non-empty array", "bands"))
    else:
        for index, item in enumerate(bands_raw):
            path = f"bands[{index}]"
            try:
                obj = _expect_object(item, path)
                name = _expect_string(obj.get("name"), f"{path}.name")
                if name in seen_bands:
                    raise ValidationError(E_CONFIG_INVALID, "duplicate band name", f"{path}.name")
                low = _expect_number(obj.get("low_hz"), f"{path}.low_hz")
                high = _expect_number(obj.get("high_hz"), f"{path}.high_hz")
                if low < 0:
                    raise ValidationError(E_CONFIG_INVALID, "low_hz must be >= 0", f"{path}.low_hz")
                if high <= low:
                    raise ValidationError(E_CONFIG_INVALID, "high_hz must exceed low_hz", f"{path}.high_hz")
                seen_bands.add(name)
                bands.append(BandConfig(name, low, high))
            except ValidationError as exc:
                issues.append(exc)

    clustering_raw = root.get("clustering")
    clustering = ClusterConfig(1, 1, "nearest_l1")
    try:
        obj = _expect_object(clustering_raw, "clustering")
        count = _expect_positive(obj.get("count"), "clustering.count")
        min_members = _expect_positive(obj.get("min_members"), "clustering.min_members")
        rule = obj.get("merge_rule", "nearest_l1")
        if rule not in MERGE_RULES:
            raise ValidationError(
                E_CONFIG_INVALID,
                f"merge_rule must be one of {list(MERGE_RULES)}",
                "clustering.merge_rule",
            )
        clustering = ClusterConfig(count, min_members, rule)
    except ValidationError as exc:
        issues.append(exc)

    fixed_raw = root.get("fixed_point")
    fixed_point = FixedPointConfig(16, 256)
    try:
        obj = _expect_object(fixed_raw, "fixed_point")
        bits = _expect_int(obj.get("fraction_bits"), "fixed_point.fraction_bits")
        levels = _expect_int(obj.get("quantization_levels"), "fixed_point.quantization_levels")
        if bits not in FRACTION_BIT_LEVELS:
            raise ValidationError(
                E_CONFIG_INVALID,
                f"fraction_bits must be one of {list(FRACTION_BIT_LEVELS)}",
                "fixed_point.fraction_bits",
            )
        if levels not in QUANTIZATION_LEVELS:
            raise ValidationError(
                E_CONFIG_INVALID,
                f"quantization_levels must be one of {list(QUANTIZATION_LEVELS)}",
                "fixed_point.quantization_levels",
            )
        fixed_point = FixedPointConfig(bits, levels)
    except ValidationError as exc:
        issues.append(exc)

    scheduler_raw = root.get("scheduler")
    scheduler = SchedulerConfig(1_000_000, 1_000_000)
    try:
        obj = _expect_object(scheduler_raw, "scheduler")
        budget = _expect_positive(obj.get("batch_point_budget"), "scheduler.batch_point_budget")
        quota = _expect_positive(obj.get("cluster_point_quota"), "scheduler.cluster_point_quota")
        scheduler = SchedulerConfig(budget, quota)
    except ValidationError as exc:
        issues.append(exc)

    segment_raw = root.get("segment")
    segment = SegmentConfig(1)
    try:
        obj = _expect_object(segment_raw, "segment")
        min_windows = _expect_positive(obj.get("min_windows"), "segment.min_windows")
        segment = SegmentConfig(min_windows)
    except ValidationError as exc:
        issues.append(exc)

    if issues:
        raise ConfigError(issues)

    return AppConfig(
        schema_version=schema_version,
        channels=tuple(channels),
        window=window,
        bands=tuple(bands),
        clustering=clustering,
        fixed_point=fixed_point,
        scheduler=scheduler,
        segment=segment,
    )


def parse_config(data: Any) -> AppConfig:
    return _parse_config(data)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _parse_config(data)
