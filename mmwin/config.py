from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import content_hash, read_json
from .errors import ConfigError


MAX_CONCURRENCY = 64
MAX_WINDOWS = 200_000


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _pick(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class SensorChannel:
    name: str
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, Any]:
        return {"min": self.minimum, "max": self.maximum}


@dataclass(frozen=True)
class WindowConfig:
    window_length_ms: int
    window_step_ms: int
    allowed_lateness_ms: int
    watermark_rule: str
    late_policy: str
    visual_feature_dim: int
    text_feature_dim: int
    visual_block_rows: int
    visual_block_cols: int
    visual_frames_per_window: int
    quantization_level: str
    fractional_bits: int
    quantization_levels: Mapping[str, int]
    fusion_temperature: float
    thread_pool_size: int
    concurrency_limit: int
    max_events_per_window: int
    checkpoint_interval_windows: int
    batch_event_budget: int
    sensor_channels: Mapping[str, SensorChannel]
    frame_width: int | None
    frame_height: int | None
    config_hash: str
    semantic_hash: str

    @property
    def quant_scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def visual_raw_dim(self) -> int:
        return self.visual_block_rows * self.visual_block_cols * 5

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "window_length_ms": self.window_length_ms,
            "window_step_ms": self.window_step_ms,
            "allowed_lateness_ms": self.allowed_lateness_ms,
            "watermark_rule": self.watermark_rule,
            "late_policy": self.late_policy,
            "visual_feature_dim": self.visual_feature_dim,
            "text_feature_dim": self.text_feature_dim,
            "visual_block_rows": self.visual_block_rows,
            "visual_block_cols": self.visual_block_cols,
            "visual_frames_per_window": self.visual_frames_per_window,
            "quantization_level": self.quantization_level,
            "fractional_bits": self.fractional_bits,
            "quantization_levels": dict(sorted(self.quantization_levels.items())),
            "fusion_temperature": self.fusion_temperature,
            "thread_pool_size": self.thread_pool_size,
            "concurrency_limit": self.concurrency_limit,
            "max_events_per_window": self.max_events_per_window,
            "checkpoint_commit_interval_windows": self.checkpoint_interval_windows,
            "batch_event_budget": self.batch_event_budget,
            "sensor_channels": {
                name: channel.as_dict()
                for name, channel in sorted(self.sensor_channels.items())
            },
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
        if include_hash:
            value["config_hash"] = self.config_hash
            value["semantic_hash"] = self.semantic_hash
        return value

    def stage_config_hash(self, stage: str) -> str:
        """Stage-scoped configuration fingerprint.

        All stages include the global config hash and the active quantization
        profile.  Keeping the helper here makes the scope explicit at call
        sites and prevents accidentally hashing Python object identity.
        """
        return content_hash(
            {
                "semantic_hash": self.semantic_hash,
                "stage": stage,
                "quantization_level": self.quantization_level,
                "fractional_bits": self.fractional_bits,
            }
        )


def _parse_channels(raw: Any) -> dict[str, SensorChannel]:
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(str(item), {}) for item in raw]
    else:
        raise ConfigError("sensor_channels must be an object or a string array")
    if not items:
        raise ConfigError("sensor_channels must not be empty")
    result: dict[str, SensorChannel] = {}
    for name, spec in items:
        if not isinstance(name, str) or not name:
            raise ConfigError("sensor channel names must be non-empty strings")
        if name in result:
            raise ConfigError(f"duplicate sensor channel {name!r}")
        if spec is None:
            spec = {}
        spec = _mapping(spec, f"sensor channel {name}")
        minimum = _require_finite_float(spec.get("min", 0.0), f"{name}.min")
        maximum = _require_finite_float(spec.get("max", 1.0), f"{name}.max")
        if maximum <= minimum:
            raise ConfigError(f"sensor channel {name!r} max must exceed min")
        result[name] = SensorChannel(name=name, minimum=minimum, maximum=maximum)
    return result


def _parse_levels(raw: Any, active: str, bits: int) -> dict[str, int]:
    if raw is None:
        return {active: bits}
    levels = _mapping(raw, "quantization_levels")
    result: dict[str, int] = {}
    for level, value in levels.items():
        if not isinstance(level, str) or not level:
            raise ConfigError("quantization profile names must be non-empty strings")
        result[level] = _require_int(value, f"quantization_levels.{level}", minimum=0)
    if active not in result:
        raise ConfigError(f"unknown quantization level {active!r}")
    if result[active] != bits:
        raise ConfigError(
            f"quantization level {active!r} declares {result[active]} bits, "
            f"but fractional_bits is {bits}"
        )
    return result


def parse_config(raw: Any) -> WindowConfig:
    raw = _mapping(raw, "config")
    window = _mapping(raw.get("window", {}), "window")
    features = _mapping(raw.get("features", {}), "features")
    quantization = _mapping(raw.get("quantization", {}), "quantization")
    fusion = _mapping(raw.get("fusion", {}), "fusion")
    runtime = _mapping(raw.get("runtime", {}), "runtime")

    def field(*names: str, default: Any = None, sections: tuple[Mapping[str, Any], ...] = ()) -> Any:
        for section in sections:
            value = _pick(section, *names)
            if value is not None:
                return value
        return _pick(raw, *names, default=default)

    length = _require_int(
        field("window_length_ms", "window_size_ms", "length_ms", sections=(window,)),
        "window_length_ms",
        minimum=1,
    )
    step = _require_int(
        field("window_step_ms", "slide_step_ms", "hop_ms", "step_ms", sections=(window,)),
        "window_step_ms",
        minimum=1,
    )
    if length % step != 0:
        raise ConfigError("window_length_ms must be an exact multiple of window_step_ms")
    allowed_lateness = _require_int(
        field("allowed_lateness_ms", "allowed_late_ms", "lateness_ms", sections=(window,)),
        "allowed_lateness_ms",
        minimum=0,
    )
    watermark_rule = field(
        "watermark_rule",
        "watermark_policy",
        default="max_seen_minus_lateness",
        sections=(window,),
    )
    if watermark_rule not in {
        "max_seen_minus_lateness",
        "max_delivered_timestamp_minus_allowed_lateness",
    }:
        raise ConfigError("unsupported watermark_rule")
    late_policy = field("late_policy", "late_event_policy", default="resettle", sections=(window,))
    if late_policy not in {"resettle", "drop"}:
        raise ConfigError("late_policy must be 'resettle' or 'drop'")

    visual_dim = _require_int(
        field("visual_feature_dim", "visual_dim", sections=(features,)),
        "visual_feature_dim",
        minimum=1,
    )
    text_dim = _require_int(
        field("text_feature_dim", "text_dim", sections=(features,)),
        "text_feature_dim",
        minimum=1,
    )
    rows = _require_int(
        field("visual_block_rows", "block_rows", sections=(features,)),
        "visual_block_rows",
        minimum=1,
    )
    cols = _require_int(
        field("visual_block_cols", "block_cols", sections=(features,)),
        "visual_block_cols",
        minimum=1,
    )
    frames_per_window = _require_int(
        field("visual_frames_per_window", "frames_per_window", sections=(features,)),
        "visual_frames_per_window",
        minimum=1,
    )
    if rows * cols > 1024:
        raise ConfigError("visual block grid is too large")
    if visual_dim > 4096 or text_dim > 4096:
        raise ConfigError("feature dimensions must be <= 4096")

    levels_raw = field(
        "quantization_levels", "levels", "quantization_profiles", default=None, sections=(quantization,)
    )
    level = field(
        "quantization_level", "quant_level", "level", default="Q8", sections=(quantization,)
    )
    if not isinstance(level, str) or not level:
        raise ConfigError("quantization_level must be a non-empty string")
    bits = _require_int(
        field(
            "fractional_bits",
            "fixed_point_bits",
            "fixed_bits",
            sections=(quantization,),
        ),
        "fractional_bits",
        minimum=0,
    )
    if bits > 24:
        raise ConfigError("fractional_bits must be <= 24")
    levels = _parse_levels(levels_raw, level, bits)

    temperature = _require_finite_float(
        field("fusion_temperature", "temperature", sections=(fusion,)),
        "fusion_temperature",
    )
    if temperature <= 0:
        raise ConfigError("fusion_temperature must be > 0")
    pool_size = _require_int(
        field("thread_pool_size", "worker_count", "concurrency", sections=(runtime,)),
        "thread_pool_size",
        minimum=1,
    )
    concurrency_limit = _require_int(
        field("concurrency_limit", "max_threads", default=16, sections=(runtime,)),
        "concurrency_limit",
        minimum=1,
    )
    if concurrency_limit > MAX_CONCURRENCY:
        raise ConfigError(f"concurrency_limit must be <= {MAX_CONCURRENCY}")
    if pool_size > concurrency_limit:
        raise ConfigError("thread_pool_size must not exceed concurrency_limit")
    max_events = _require_int(
        field("max_events_per_window", "window_event_limit", sections=(runtime,)),
        "max_events_per_window",
        minimum=1,
    )
    checkpoint_interval = _require_int(
        field(
            "checkpoint_commit_interval_windows",
            "checkpoint_interval_windows",
            "checkpoint_every_windows",
            sections=(runtime,),
        ),
        "checkpoint_commit_interval_windows",
        minimum=1,
    )
    batch_budget = _require_int(
        field("batch_event_budget", default=64, sections=(runtime,)),
        "batch_event_budget",
        minimum=1,
    )
    channel_raw = raw.get(
        "sensor_channels",
        raw.get(
            "channels",
            runtime.get(
                "sensor_channels",
                runtime.get("channels", raw.get("sensor_channel_enum")),
            ),
        ),
    )
    channels = _parse_channels(channel_raw)
    frame_width = field("frame_width", default=None, sections=(features,))
    frame_height = field("frame_height", default=None, sections=(features,))
    if frame_width is not None:
        frame_width = _require_int(frame_width, "frame_width", minimum=1)
    if frame_height is not None:
        frame_height = _require_int(frame_height, "frame_height", minimum=1)

    draft = {
        "window_length_ms": length,
        "window_step_ms": step,
        "allowed_lateness_ms": allowed_lateness,
        "watermark_rule": watermark_rule,
        "late_policy": late_policy,
        "visual_feature_dim": visual_dim,
        "text_feature_dim": text_dim,
        "visual_block_rows": rows,
        "visual_block_cols": cols,
        "visual_frames_per_window": frames_per_window,
        "quantization_level": level,
        "fractional_bits": bits,
        "quantization_levels": dict(sorted(levels.items())),
        "fusion_temperature": temperature,
        "thread_pool_size": pool_size,
        "concurrency_limit": concurrency_limit,
        "max_events_per_window": max_events,
        "checkpoint_commit_interval_windows": checkpoint_interval,
        "batch_event_budget": batch_budget,
        "sensor_channels": {
            name: channel.as_dict() for name, channel in sorted(channels.items())
        },
        "frame_width": frame_width,
        "frame_height": frame_height,
    }
    config_hash = content_hash(draft)
    semantic_draft = {
        key: value
        for key, value in draft.items()
        if key
        not in {
            "thread_pool_size",
            "concurrency_limit",
            "checkpoint_commit_interval_windows",
            "batch_event_budget",
        }
    }
    semantic_hash = content_hash(semantic_draft)
    return WindowConfig(
        window_length_ms=length,
        window_step_ms=step,
        allowed_lateness_ms=allowed_lateness,
        watermark_rule=watermark_rule,
        late_policy=late_policy,
        visual_feature_dim=visual_dim,
        text_feature_dim=text_dim,
        visual_block_rows=rows,
        visual_block_cols=cols,
        visual_frames_per_window=frames_per_window,
        quantization_level=level,
        fractional_bits=bits,
        quantization_levels=levels,
        fusion_temperature=temperature,
        thread_pool_size=pool_size,
        concurrency_limit=concurrency_limit,
        max_events_per_window=max_events,
        checkpoint_interval_windows=checkpoint_interval,
        batch_event_budget=batch_budget,
        sensor_channels=channels,
        frame_width=frame_width,
        frame_height=frame_height,
        config_hash=config_hash,
        semantic_hash=semantic_hash,
    )


def load_config(path: str | Path) -> WindowConfig:
    try:
        raw = read_json(Path(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config: {exc}") from exc
    return parse_config(raw)


def validate_max_windows(window_count: int) -> None:
    if window_count > MAX_WINDOWS:
        raise ConfigError(f"window count {window_count} exceeds safety limit {MAX_WINDOWS}")
