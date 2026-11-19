from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, E_MASK_OUT_OF_RANGE, StudioError


SCHEMA_VERSION = 1
WEIGHT_SCALE = 1 << 16


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _lookup(mapping: Mapping[str, Any], paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        cursor: Any = mapping
        found = True
        for part in path.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                found = False
                break
            cursor = cursor[part]
        if found:
            return cursor
    return default


def _require(mapping: Mapping[str, Any], paths: Sequence[str], label: str) -> Any:
    sentinel = object()
    value = _lookup(mapping, paths, sentinel)
    if value is sentinel:
        raise StudioError(E_CONFIG_INVALID, f"missing required field: {label}")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_field(
    mapping: Mapping[str, Any],
    paths: Sequence[str],
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = _require(mapping, paths, label)
    if not _is_int(value):
        raise StudioError(E_CONFIG_INVALID, f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise StudioError(E_CONFIG_INVALID, f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise StudioError(E_CONFIG_INVALID, f"{label} must be <= {maximum}")
    return value


def _bool_field(mapping: Mapping[str, Any], paths: Sequence[str], label: str) -> bool:
    value = _require(mapping, paths, label)
    if not isinstance(value, bool):
        raise StudioError(E_CONFIG_INVALID, f"{label} must be a boolean")
    return value


def decimal_to_q(value: Any, label: str = "weight") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise StudioError(E_CONFIG_INVALID, f"{label} must be numeric")
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation as exc:
        raise StudioError(E_CONFIG_INVALID, f"{label} is not a finite decimal") from exc
    if not decimal_value.is_finite():
        raise StudioError(E_CONFIG_INVALID, f"{label} is not a finite decimal")
    if decimal_value < 0 or decimal_value > 1:
        raise StudioError(E_CONFIG_INVALID, f"{label} must be in [0, 1]")
    return int((decimal_value * WEIGHT_SCALE).to_integral_value(rounding=ROUND_HALF_UP))


def q_to_decimal_string(value_q: int) -> str:
    whole, fraction = divmod(value_q, WEIGHT_SCALE)
    if fraction == 0:
        return str(whole)
    return f"{whole}.{fraction:05d}".rstrip("0")


@dataclass(frozen=True)
class MaskRect:
    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def pixel_count(self) -> int:
        return self.width * self.height

    def validate(self, image_width: int, image_height: int) -> None:
        if self.width <= 0 or self.height <= 0:
            raise StudioError(
                E_MASK_OUT_OF_RANGE,
                "mask rectangle width and height must be positive",
                self.as_dict(),
            )
        if self.x < 0 or self.y < 0:
            raise StudioError(E_MASK_OUT_OF_RANGE, "mask rectangle origin is negative", self.as_dict())
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise StudioError(
                E_MASK_OUT_OF_RANGE,
                "mask rectangle exceeds image bounds",
                {**self.as_dict(), "image_width": image_width, "image_height": image_height},
            )


def parse_mask_rect(value: Any, image_width: int, image_height: int) -> MaskRect:
    if isinstance(value, (list, tuple)):
        if len(value) != 4 or any(not _is_int(item) for item in value):
            raise StudioError(E_CONFIG_INVALID, "mask rectangle array must be [x, y, width, height]")
        rect = MaskRect(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
        rect.validate(image_width, image_height)
        return rect
    if not isinstance(value, Mapping):
        raise StudioError(E_CONFIG_INVALID, "mask rectangle must be an object or four-element array")
    aliases = {
        "x": ("x",),
        "y": ("y",),
        "width": ("width", "w"),
        "height": ("height", "h"),
    }
    numbers: dict[str, int] = {}
    for target, paths in aliases.items():
        raw_value = _require(value, paths, f"mask.{target}")
        if not _is_int(raw_value):
            raise StudioError(E_CONFIG_INVALID, f"mask.{target} must be an integer")
        numbers[target] = raw_value
    rect = MaskRect(numbers["x"], numbers["y"], numbers["width"], numbers["height"])
    rect.validate(image_width, image_height)
    return rect


def parse_channel_weights(value: Any, label: str = "channel weights") -> tuple[int, int, int]:
    raw_values: list[Any]
    if isinstance(value, Mapping):
        aliases = (
            ("r", "red", "channel_0"),
            ("g", "green", "channel_1"),
            ("b", "blue", "channel_2"),
        )
        raw_values = []
        for names in aliases:
            found = next((value[name] for name in names if name in value), None)
            if found is None:
                raise StudioError(
                    E_CONFIG_INVALID,
                    f"{label} is missing one of {names[0]}/{names[1]}/{names[2]}",
                )
            raw_values.append(found)
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raise StudioError(E_CONFIG_INVALID, f"{label} must be an array or channel object")
    if len(raw_values) != 3:
        raise StudioError(E_CONFIG_INVALID, f"{label} must contain exactly three values")
    weights_q = tuple(decimal_to_q(item, f"{label} {index}") for index, item in enumerate(raw_values))
    return tuple(int(item) for item in weights_q)


@dataclass(frozen=True)
class RenderParams:
    mask: MaskRect
    repair_radius: int
    repair_iterations: int
    quant_levels: int
    channel_weights_q: tuple[int, int, int]
    preserve_mask: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "mask": self.mask.as_dict(),
            "repair_radius": self.repair_radius,
            "repair_iterations": self.repair_iterations,
            "quant_levels": self.quant_levels,
            "channel_weights_q": list(self.channel_weights_q),
            "channel_weights": [q_to_decimal_string(value) for value in self.channel_weights_q],
            "preserve_mask": self.preserve_mask,
        }

    def fingerprint(self) -> str:
        return sha256_json(self.as_dict())

    def with_mask(self, rect: MaskRect) -> "RenderParams":
        return replace(self, mask=rect)


@dataclass(frozen=True)
class StudioConfig:
    width: int
    height: int
    params: RenderParams
    latency_budget_ms: int
    chunk_rows: int
    thread_workers: int
    snapshot_interval: int
    fixed_bits: int
    seed: int | None
    image_file: str
    style_file: str
    commands_file: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "width": self.width,
            "height": self.height,
            "params": self.params.as_dict(),
            "latency_budget_ms": self.latency_budget_ms,
            "chunk_rows": self.chunk_rows,
            "thread_workers": self.thread_workers,
            "snapshot_interval": self.snapshot_interval,
            "fixed_bits": self.fixed_bits,
            "seed": self.seed,
            "image_file": self.image_file,
            "style_file": self.style_file,
            "commands_file": self.commands_file,
        }

    def fingerprint(self) -> str:
        return sha256_json(self.as_dict())


def _parse_resolution(raw: Mapping[str, Any]) -> tuple[int, int]:
    resolution = _lookup(raw, ("resolution", "image_size", "image.size"), None)
    if resolution is not None:
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise StudioError(E_CONFIG_INVALID, "resolution must be a two-element array")
        width_raw, height_raw = resolution
        if not _is_int(width_raw) or not _is_int(height_raw):
            raise StudioError(E_CONFIG_INVALID, "resolution values must be integers")
        return width_raw, height_raw
    width = _int_field(raw, ("width", "image.width"), "width", 1, 16384)
    height = _int_field(raw, ("height", "image.height"), "height", 1, 16384)
    return width, height


def load_config(path: Path) -> StudioConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudioError(E_CONFIG_INVALID, f"config file not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudioError(E_CONFIG_INVALID, f"config file is unreadable: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise StudioError(E_CONFIG_INVALID, "config root must be an object")

    width, height = _parse_resolution(raw)
    if width < 1 or height < 1 or width > 16384 or height > 16384:
        raise StudioError(E_CONFIG_INVALID, "resolution is outside supported bounds")

    mask_raw = _require(raw, ("mask", "mask_rect"), "mask")
    mask = parse_mask_rect(mask_raw, width, height)
    radius = _int_field(raw, ("repair.radius", "repair_radius"), "repair radius", 1, 64)
    iterations = _int_field(
        raw,
        ("repair.iterations", "repair_iterations", "iterations"),
        "repair iterations",
        1,
        4096,
    )
    quant_levels = _int_field(
        raw,
        ("style.quant_levels", "quant_levels"),
        "quant levels",
        2,
        256,
    )
    if quant_levels & (quant_levels - 1):
        raise StudioError(E_CONFIG_INVALID, "quant levels must be a power of two")

    weights_raw = _require(
        raw,
        ("style.channel_weights", "style.weights", "channel_weights"),
        "channel weights",
    )
    weights_q = parse_channel_weights(weights_raw)
    preserve_mask = _bool_field(
        raw,
        ("style.preserve_mask", "preserve_mask"),
        "preserve mask",
    )
    budget_ms = _int_field(
        raw,
        ("interaction.latency_budget_ms", "latency_budget_ms", "interaction_budget_ms"),
        "latency budget",
        1,
        60000,
    )
    chunk_rows = _int_field(
        raw,
        ("interaction.chunk_rows", "render.chunk_rows", "chunk_rows", "render_chunk_rows"),
        "chunk rows",
        1,
        height,
    )
    threads = _int_field(
        raw,
        (
            "interaction.thread_workers",
            "interaction.threads",
            "interaction.worker_count",
            "thread_workers",
            "threads",
            "worker_count",
        ),
        "thread workers",
        1,
        128,
    )
    snapshot_interval = _int_field(
        raw,
        (
            "interaction.snapshot_interval",
            "snapshot_interval",
            "snapshot_every_commands",
            "snapshot_interval_commands",
        ),
        "snapshot interval",
        1,
        100000,
    )
    fixed_bits = _int_field(
        raw,
        (
            "fixed_bits",
            "fixed_point_bits",
            "fixed_decimal_bits",
            "numeric.fixed_bits",
            "interaction.fixed_bits",
        ),
        "fixed bits",
        4,
        20,
    )
    seed = _lookup(raw, ("seed",), None)
    if seed is not None and not _is_int(seed):
        raise StudioError(E_CONFIG_INVALID, "seed must be an integer")

    image_file = _lookup(raw, ("image_file", "sample.image"), "image.png")
    style_file = _lookup(raw, ("style_file", "sample.style"), "style.png")
    commands_file = _lookup(raw, ("commands_file", "sample.commands"), "commands.json")
    for label, file_value in (
        ("image_file", image_file),
        ("style_file", style_file),
        ("commands_file", commands_file),
    ):
        if not isinstance(file_value, str) or not file_value:
            raise StudioError(E_CONFIG_INVALID, f"{label} must be a non-empty string")

    return StudioConfig(
        width=width,
        height=height,
        params=RenderParams(
            mask=mask,
            repair_radius=radius,
            repair_iterations=iterations,
            quant_levels=quant_levels,
            channel_weights_q=tuple(int(item) for item in weights_q),
            preserve_mask=preserve_mask,
        ),
        latency_budget_ms=budget_ms,
        chunk_rows=chunk_rows,
        thread_workers=threads,
        snapshot_interval=snapshot_interval,
        fixed_bits=fixed_bits,
        seed=seed,
        image_file=image_file,
        style_file=style_file,
        commands_file=commands_file,
    )


def apply_command(params: RenderParams, width: int, height: int, command: Mapping[str, Any]) -> RenderParams:
    if not isinstance(command, Mapping):
        raise StudioError(E_CONFIG_INVALID, "command must be an object")
    operation = command.get("op", command.get("type"))
    if not isinstance(operation, str):
        raise StudioError(E_CONFIG_INVALID, "command op must be a string")

    if operation in {"set_mask", "set_mask_rect", "mask"}:
        raw_rect = command.get("rect", command.get("mask"))
        if raw_rect is None and all(key in command for key in ("x", "y")):
            raw_rect = {
                "x": command.get("x"),
                "y": command.get("y"),
                "width": command.get("width", command.get("w")),
                "height": command.get("height", command.get("h")),
            }
        return params.with_mask(parse_mask_rect(raw_rect, width, height))
    if operation in {"set_radius", "set_repair_radius", "radius"}:
        value = command.get("value", command.get("radius"))
        if not _is_int(value) or not 1 <= value <= 64:
            raise StudioError(E_CONFIG_INVALID, "repair radius command value must be in [1, 64]")
        return replace(params, repair_radius=value)
    if operation in {"set_iterations", "set_repair_iterations"}:
        value = command.get("value", command.get("iterations"))
        if not _is_int(value) or not 1 <= value <= 4096:
            raise StudioError(E_CONFIG_INVALID, "repair iterations command value must be in [1, 4096]")
        return replace(params, repair_iterations=value)
    if operation in {"set_style_weights", "style_weights"}:
        weights = command.get("weights", command.get("value", command.get("channel_weights")))
        return replace(params, channel_weights_q=parse_channel_weights(weights, "style weights command"))
    if operation in {"set_quant_levels", "quant_levels"}:
        value = command.get("value", command.get("levels"))
        if not _is_int(value) or not 2 <= value <= 256 or value & (value - 1):
            raise StudioError(E_CONFIG_INVALID, "quant levels command value must be a power of two in [2, 256]")
        return replace(params, quant_levels=value)
    if operation in {"set_preserve_mask", "preserve_mask"}:
        value = command.get("value")
        if not isinstance(value, bool):
            raise StudioError(E_CONFIG_INVALID, "preserve mask command value must be boolean")
        return replace(params, preserve_mask=value)
    raise StudioError(E_CONFIG_INVALID, f"unsupported command op: {operation}")


def command_affected_stages(operation: str) -> list[str]:
    mapping = {
        "set_mask": ["mask", "fill", "style", "composite"],
        "set_mask_rect": ["mask", "fill", "style", "composite"],
        "mask": ["mask", "fill", "style", "composite"],
        "set_radius": ["fill", "style", "composite"],
        "set_repair_radius": ["fill", "style", "composite"],
        "radius": ["fill", "style", "composite"],
        "set_iterations": ["fill", "style", "composite"],
        "set_repair_iterations": ["fill", "style", "composite"],
        "set_style_weights": ["style", "composite"],
        "style_weights": ["style", "composite"],
        "set_quant_levels": ["style", "composite"],
        "quant_levels": ["style", "composite"],
        "set_preserve_mask": ["composite"],
        "preserve_mask": ["composite"],
    }
    return mapping.get(operation, [])


def load_commands(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudioError(E_CONFIG_INVALID, f"command stream not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudioError(E_CONFIG_INVALID, f"command stream is unreadable: {exc}") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("commands")
    if not isinstance(raw, list):
        raise StudioError(E_CONFIG_INVALID, "command stream must be an array")
    commands: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise StudioError(E_CONFIG_INVALID, f"command {index} must be an object")
        commands.append(dict(item))
    return commands
