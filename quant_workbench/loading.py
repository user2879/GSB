"""Loaders and validation for weights, calibration, YAML config and commands."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import (
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_IO,
    E_LAYER_INVALID,
    E_SHAPE_MISMATCH,
    E_VERSION_NOT_MONOTONIC,
    E_WEIGHT_NOT_FINITE,
    Issue,
    WorkbenchError,
)
from .models import (
    DEFAULT_BITS,
    DEFAULT_FRACTIONAL_BITS,
    DEFAULT_GROUP_SIZE,
    DEFAULT_SCHEME,
    CalibrationSample,
    CalibrationSet,
    Command,
    LayerConfig,
    WeightLayer,
    WorkbenchConfig,
)
from .util import file_sha256


# ---------------------------------------------------------------------------
# Small YAML reader (the required dependency list intentionally excludes PyYAML)
# ---------------------------------------------------------------------------


def _strip_yaml_comment(text: str) -> str:
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)
        elif char == "#" and depth == 0:
            return text[:index].rstrip()
    return text.rstrip()


def _split_flow(text: str) -> list[str]:
    result: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail or result:
        result.append(tail)
    return result


def _find_mapping_colon(text: str) -> int:
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == ":" and depth == 0:
            return index
    return -1


def _unquote_key(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        try:
            return str(ast.literal_eval(text))
        except (ValueError, SyntaxError):
            return text[1:-1]
    return text


def _parse_scalar(text: str) -> Any:
    text = _strip_yaml_comment(text).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_scalar(part) for part in _split_flow(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        result: dict[str, Any] = {}
        if not inner:
            return result
        for part in _split_flow(inner):
            colon = _find_mapping_colon(part)
            if colon < 0:
                raise ValueError(f"invalid inline mapping item: {part!r}")
            key = _unquote_key(part[:colon])
            result[key] = _parse_scalar(part[colon + 1 :])
        return result
    if (text[0:1] == text[-1:] and text[0] in {"'", '"'}) or text.startswith("("):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text.strip("'\"")
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        number = float(text)
        return number
    except ValueError:
        return text


def _parse_yaml_lines(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start
    is_list = lines[start][1].startswith("-")
    if is_list:
        values: list[Any] = []
        index = start
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent or not content.startswith("-"):
                break
            if current_indent > indent:
                raise ValueError("unexpected indentation in YAML list")
            rest = content[1:].strip()
            if not rest:
                if index + 1 < len(lines) and lines[index + 1][0] > indent:
                    child, index = _parse_yaml_lines(lines, index + 1, lines[index + 1][0])
                    values.append(child)
                else:
                    values.append(None)
                    index += 1
                continue
            if _find_mapping_colon(rest) >= 0:
                colon = _find_mapping_colon(rest)
                key = _unquote_key(rest[:colon])
                tail = rest[colon + 1 :].strip()
                item: dict[str, Any] = {key: _parse_scalar(tail) if tail else None}
                index += 1
                if index < len(lines) and lines[index][0] > indent:
                    child_indent = lines[index][0]
                    # Continuation mapping keys at the list item's child indentation.
                    child, index = _parse_yaml_lines(lines, index, child_indent)
                    if isinstance(child, dict):
                        item.update(child)
                values.append(item)
            else:
                values.append(_parse_scalar(rest))
                index += 1
        return values, index

    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent or content.startswith("-"):
            if current_indent > indent:
                raise ValueError("unexpected indentation in YAML mapping")
            break
        if current_indent > indent:
            raise ValueError("unexpected indentation in YAML mapping")
        colon = _find_mapping_colon(content)
        if colon < 0:
            raise ValueError(f"invalid YAML mapping line: {content!r}")
        key = _unquote_key(content[:colon])
        tail = content[colon + 1 :].strip()
        index += 1
        if tail:
            mapping[key] = _parse_scalar(tail)
        elif index < len(lines) and lines[index][0] > indent:
            child, index = _parse_yaml_lines(lines, index, lines[index][0])
            mapping[key] = child
        else:
            mapping[key] = None
    return mapping, index


def parse_yaml(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise ValueError("tabs are not allowed for YAML indentation")
        stripped = _strip_yaml_comment(raw_line)
        if not stripped.strip() or stripped.lstrip().startswith("---"):
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.lstrip(" ")))
    if not lines:
        return {}
    if lines[0][0] != 0:
        raise ValueError("YAML document must begin at indentation zero")
    parsed, index = _parse_yaml_lines(lines, 0, 0)
    if index != len(lines):
        raise ValueError("could not consume complete YAML document")
    return parsed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def normalize_scheme(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "sym": "symmetric",
        "symmetric": "symmetric",
        "signed": "symmetric",
        "asym": "asymmetric",
        "asymmetric": "asymmetric",
        "affine": "asymmetric",
        "unsigned": "asymmetric",
    }
    return aliases.get(normalized, normalized)


def _int_field(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be an integer")
    if value < minimum:
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be >= {minimum}")
    return value


def _number_field(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be finite")
    if positive and result <= 0:
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be positive")
    if nonnegative and result < 0:
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be non-negative")
    return result


def _first(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _load_layer_config(
    raw: dict[str, Any],
    *,
    defaults: LayerConfig,
    allowed_bits: tuple[int, ...],
    name: str,
) -> LayerConfig:
    bits = _int_field(raw.get("bits", defaults.bits), f"layers.{name}.bits", minimum=1)
    group_size = _int_field(
        raw.get("group_size", defaults.group_size), f"layers.{name}.group_size", minimum=1
    )
    fractional_bits = _int_field(
        raw.get("fractional_bits", defaults.fractional_bits),
        f"layers.{name}.fractional_bits",
        minimum=1,
    )
    scheme = normalize_scheme(raw.get("scheme", defaults.scheme))
    if bits not in allowed_bits:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            f"layers.{name}.bits={bits} is not in allowed bit widths {list(allowed_bits)}",
            layer=name,
        )
    if scheme not in {"symmetric", "asymmetric"}:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            f"layers.{name}.scheme must be symmetric or asymmetric",
            layer=name,
        )
    return LayerConfig(
        bits=bits,
        group_size=group_size,
        fractional_bits=fractional_bits,
        scheme=scheme,
    )


def load_config(path: str | Path) -> WorkbenchConfig:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(E_IO, f"cannot read config {source}: {exc}") from exc
    try:
        if source.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raw = parse_yaml(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_CONFIG_INVALID, f"cannot parse config {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise WorkbenchError(E_CONFIG_INVALID, "config root must be a mapping")

    quant = raw.get("quantization") if isinstance(raw.get("quantization"), dict) else {}
    interaction = raw.get("interaction") if isinstance(raw.get("interaction"), dict) else {}
    thread_pool = (
        interaction.get("thread_pool")
        if isinstance(interaction.get("thread_pool"), dict)
        else {}
    )
    if thread_pool:
        interaction = {**thread_pool, **interaction}
    snapshot = raw.get("snapshot") if isinstance(raw.get("snapshot"), dict) else {}
    calibration_section = (
        raw.get("calibration") if isinstance(raw.get("calibration"), dict) else {}
    )
    merged_root: dict[str, Any] = {}
    for section in (quant, interaction, snapshot, calibration_section):
        merged_root.update(section)
    merged_root.update(raw)

    allowed_raw = _first(
        raw,
        (
            "allowed_bit_widths",
            "bit_widths",
            "quantization_bit_widths",
            "quantization_levels",
        ),
        _first(
            quant,
            (
                "allowed_bit_widths",
                "bit_widths",
                "quantization_bit_widths",
                "quantization_levels",
                "levels",
            ),
            [2, 3, 4, 5, 6, 8, 16],
        ),
    )
    if not isinstance(allowed_raw, (list, tuple)) or not allowed_raw:
        raise WorkbenchError(E_CONFIG_INVALID, "allowed_bit_widths must be a non-empty list")
    allowed_bits = tuple(sorted({_int_field(item, "allowed_bit_widths", minimum=1) for item in allowed_raw}))
    if len(allowed_bits) != len(set(allowed_bits)):
        allowed_bits = tuple(sorted(set(allowed_bits)))

    defaults_raw = raw.get("defaults")
    if not isinstance(defaults_raw, dict):
        defaults_raw = quant.get("defaults") if isinstance(quant.get("defaults"), dict) else {}

    def pick(*names: str, default: Any = None, section: dict[str, Any] | None = None) -> Any:
        for source_mapping in (raw, section or {}, merged_root, defaults_raw):
            for name in names:
                if name in source_mapping and source_mapping[name] is not None:
                    return source_mapping[name]
        return default

    default_bits_raw = pick(
        "default_bits",
        "default_target_bits",
        "bits",
        "target_bit_width",
        default=None,
        section=quant,
    )
    target_values = pick(
        "target_bits",
        "layer_target_bits",
        "default_target_bits",
        default=None,
        section=quant,
    )
    if default_bits_raw is None:
        if isinstance(target_values, list):
            default_bits_raw = target_values[0] if target_values else DEFAULT_BITS
        elif target_values is not None and not isinstance(target_values, dict):
            default_bits_raw = target_values
        else:
            default_bits_raw = allowed_bits[0]
    if isinstance(default_bits_raw, dict):
        default_bits_raw = DEFAULT_BITS
    default_bits = _int_field(default_bits_raw, "default_bits", minimum=1)
    default_group_raw = pick(
        "default_group_size",
        "group_size",
        "grouping_granularity",
        "group_granularity",
        default=DEFAULT_GROUP_SIZE,
        section=quant,
    )
    if isinstance(default_group_raw, (dict, list)):
        default_group_raw = DEFAULT_GROUP_SIZE
    default_group = _int_field(
        default_group_raw,
        "default_group_size",
        minimum=1,
    )
    default_fraction_raw = pick(
        "default_fractional_bits",
        "fractional_bits",
        "fixed_fractional_bits",
        "fixed_point_fractional_bits",
        default=DEFAULT_FRACTIONAL_BITS,
        section=quant,
    )
    if isinstance(default_fraction_raw, (dict, list)):
        default_fraction_raw = DEFAULT_FRACTIONAL_BITS
    default_fraction = _int_field(
        default_fraction_raw,
        "default_fractional_bits",
        minimum=1,
    )
    default_scheme_raw = pick(
        "scheme",
        "quantization_scheme",
        "quantization_mode",
        default=DEFAULT_SCHEME,
        section=quant,
    )
    if isinstance(default_scheme_raw, (dict, list)):
        default_scheme_raw = DEFAULT_SCHEME
    default_scheme = normalize_scheme(default_scheme_raw)
    if default_bits not in allowed_bits:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            f"default bits={default_bits} is not in allowed bit widths {list(allowed_bits)}",
        )
    if default_scheme not in {"symmetric", "asymmetric"}:
        raise WorkbenchError(E_CONFIG_INVALID, "default scheme must be symmetric or asymmetric")
    defaults = LayerConfig(default_bits, default_group, default_fraction, default_scheme)

    max_threads_raw = pick(
        "max_threads",
        "thread_pool_size",
        "thread_workers",
        "max_workers",
        "concurrency",
        "max_concurrency",
        default=None,
        section=interaction,
    )
    thread_limit = _int_field(
        pick(
            "thread_limit",
            "max_threads_limit",
            "concurrency_limit",
            "limit",
            default=32,
            section=interaction,
        ),
        "thread_limit",
        minimum=1,
    )
    max_threads = _int_field(
        max_threads_raw if max_threads_raw is not None else min(4, thread_limit),
        "max_threads",
        minimum=1,
    )
    if max_threads > thread_limit:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            f"max_threads={max_threads} exceeds configured upper limit {thread_limit}",
        )
    latency_budget_ms = _number_field(
        pick(
            "latency_budget_ms",
            "interaction_latency_budget_ms",
            "latency_budget",
            default=100.0,
            section=interaction,
        ),
        "latency_budget_ms",
        positive=True,
    )
    point_batch_budget = _int_field(
        pick(
            "point_batch_budget",
            "quantization_point_batch_budget",
            "point_budget_per_batch",
            "point_budget",
            default=1_000_000,
            section=quant,
        ),
        "point_batch_budget",
        minimum=1,
    )
    snapshot_commit_interval = _int_field(
        pick(
            "snapshot_commit_interval",
            "snapshot_commit_interval_commands",
            "commit_interval_commands",
            default=10,
            section=snapshot,
        ),
        "snapshot_commit_interval",
        minimum=1,
    )
    snapshot_version = _int_field(
        pick(
            "snapshot_version",
            "global_version",
            "version",
            default=1,
            section=snapshot,
        ),
        "snapshot_version",
        minimum=1,
    )
    temperature = _number_field(
        pick(
            "distillation_temperature",
            "temperature",
            "distill_temperature",
            default=1.0,
            section=quant,
        ),
        "distillation_temperature",
        positive=True,
    )
    significant_digits = _int_field(
        pick("significant_digits", "scale_significant_digits", default=8, section=quant),
        "significant_digits",
        minimum=1,
    )
    worker_delay_ms = _number_field(
        pick("worker_delay_ms", "simulated_worker_delay_ms", default=0.0, section=interaction),
        "worker_delay_ms",
        nonnegative=True,
    )

    layers_raw = raw.get("layers", quant.get("layers", {}))
    if layers_raw is None:
        layers_raw = {}
    if isinstance(layers_raw, list):
        converted: dict[str, Any] = {}
        for item in layers_raw:
            if not isinstance(item, dict):
                raise WorkbenchError(E_CONFIG_INVALID, "layers list entries must be mappings")
            name = item.get("name", item.get("layer"))
            if not isinstance(name, str) or not name:
                raise WorkbenchError(E_CONFIG_INVALID, "layer list entry requires name/layer")
            converted[name] = item
        layers_raw = converted
    if not isinstance(layers_raw, dict):
        raise WorkbenchError(E_CONFIG_INVALID, "layers must be a mapping or list")

    def layer_mapping(names: tuple[str, ...]) -> dict[str, Any]:
        value = pick(*names, default=None, section=quant)
        return value if isinstance(value, dict) else {}

    bit_map = layer_mapping(
        (
            "target_bits",
            "layer_target_bits",
            "bits",
            "bit_widths",
            "quantization_levels",
        )
    )
    group_map = layer_mapping(
        ("group_sizes", "group_size", "grouping_granularity", "group_granularity")
    )
    fraction_map = layer_mapping(
        ("fractional_bits", "fixed_fractional_bits", "fixed_point_fractional_bits")
    )
    scheme_map = layer_mapping(
        ("schemes", "quantization_schemes", "scheme", "quantization_mode")
    )
    layer_names = set(layers_raw)
    for mapping in (bit_map, group_map, fraction_map, scheme_map):
        layer_names.update(str(name) for name in mapping)
    layers: dict[str, LayerConfig] = {}
    for name in sorted(layer_names):
        layer_raw = layers_raw.get(name, {})
        if not isinstance(name, str) or not name:
            raise WorkbenchError(E_CONFIG_INVALID, "layer names must be non-empty strings")
        if not isinstance(layer_raw, dict):
            raise WorkbenchError(E_CONFIG_INVALID, f"layers.{name} must be a mapping", layer=name)
        normalized = {
            "bits": _first(
                layer_raw,
                ("bits", "target_bits", "target_bit_width", "bit_width"),
                bit_map.get(name, defaults.bits),
            ),
            "group_size": _first(
                layer_raw,
                ("group_size", "grouping_granularity", "group_granularity"),
                group_map.get(name, defaults.group_size),
            ),
            "fractional_bits": _first(
                layer_raw,
                (
                    "fractional_bits",
                    "fixed_fractional_bits",
                    "fixed_point_fractional_bits",
                ),
                fraction_map.get(name, defaults.fractional_bits),
            ),
            "scheme": _first(
                layer_raw,
                ("scheme", "quantization_scheme", "quantization_mode"),
                scheme_map.get(name, defaults.scheme),
            ),
        }
        layers[name] = _load_layer_config(
            normalized, defaults=defaults, allowed_bits=allowed_bits, name=name
        )

    try:
        digest = file_sha256(source)
    except OSError as exc:
        raise WorkbenchError(E_IO, f"cannot hash config {source}: {exc}") from exc

    return WorkbenchConfig(
        snapshot_version=snapshot_version,
        allowed_bit_widths=allowed_bits,
        max_threads=max_threads,
        thread_limit=thread_limit,
        latency_budget_ms=latency_budget_ms,
        point_batch_budget=point_batch_budget,
        snapshot_commit_interval=snapshot_commit_interval,
        distillation_temperature=temperature,
        significant_digits=significant_digits,
        defaults=defaults,
        layers=layers,
        source_path=str(source),
        raw=raw,
        file_hash=digest,
        worker_delay_ms=worker_delay_ms,
    )


# ---------------------------------------------------------------------------
# Weight pack
# ---------------------------------------------------------------------------


def _iter_json_records(path: str | Path) -> list[tuple[int, Any]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(E_IO, f"cannot read {source}: {exc}") from exc
    if source.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WorkbenchError(E_IO, f"invalid JSON in {source}: {exc}") from exc
        if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
            parsed = parsed["records"]
        if not isinstance(parsed, list):
            raise WorkbenchError(E_IO, f"{source} must contain a JSON array or records list")
        return [(1, item) for item in parsed]

    records: list[tuple[int, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append((line_number, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise WorkbenchError(
                E_IO,
                f"invalid JSONL at {source}:{line_number}: {exc}",
                line_number=line_number,
            ) from exc
    return records


def _shape_tuple(raw_shape: Any) -> tuple[int, ...] | None:
    if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
        return None
    shape: list[int] = []
    for value in raw_shape:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        shape.append(value)
    return tuple(shape)


def load_weight_pack(
    path: str | Path, *, command_index: int | None = None
) -> tuple[dict[str, WeightLayer], list[Issue]]:
    records = _iter_json_records(path)
    layers: dict[str, WeightLayer] = {}
    issues: list[Issue] = []
    if not records:
        issues.append(
            Issue(
                E_LAYER_INVALID,
                f"weight pack {path} contains no layer records",
                command_index=command_index,
                line_number=1,
            )
        )
    previous_version: int | None = None
    file_order = 0
    for record_index, (line_number, record) in enumerate(records):
        if not isinstance(record, dict):
            issues.append(
                Issue(
                    E_LAYER_INVALID,
                    "weight record must be a JSON object",
                    command_index=command_index,
                    line_number=line_number,
                    details={"record_index": record_index},
                )
            )
            continue
        layer_name = record.get("layer", record.get("name"))
        if not isinstance(layer_name, str) or not layer_name:
            issues.append(
                Issue(
                    E_LAYER_INVALID,
                    "layer name must be a non-empty string",
                    command_index=command_index,
                    line_number=line_number,
                    details={"record_index": record_index},
                )
            )
            continue
        if layer_name in layers:
            issues.append(
                Issue(
                    E_LAYER_INVALID,
                    f"duplicate layer {layer_name!r}",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                )
            )
            continue
        version = record.get("version", record_index + 1)
        if isinstance(version, bool) or not isinstance(version, int):
            issues.append(
                Issue(
                    E_LAYER_INVALID,
                    f"layer {layer_name!r} version must be an integer",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                )
            )
            continue
        if previous_version is not None and version < previous_version:
            issues.append(
                Issue(
                    E_VERSION_NOT_MONOTONIC,
                    f"layer {layer_name!r} version {version} is lower than previous {previous_version}",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                )
            )
            continue
        previous_version = version
        shape = _shape_tuple(record.get("shape"))
        if shape is None:
            issues.append(
                Issue(
                    E_SHAPE_MISMATCH,
                    f"layer {layer_name!r} shape must contain positive integers",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                    details={"shape": record.get("shape")},
                )
            )
            continue
        raw_weights = record.get(
            "weights", record.get("weight", record.get("values", record.get("data")))
        )
        if raw_weights is None:
            issues.append(
                Issue(
                    E_SHAPE_MISMATCH,
                    f"layer {layer_name!r} has no weights",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                )
            )
            continue
        try:
            array = np.asarray(raw_weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            issues.append(
                Issue(
                    E_SHAPE_MISMATCH,
                    f"layer {layer_name!r} weights are not numeric: {exc}",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                )
            )
            continue
        expected_size = int(np.prod(shape, dtype=np.int64))
        nested_shape_mismatch = array.ndim > 1 and tuple(array.shape) != shape
        if array.size != expected_size or nested_shape_mismatch:
            issues.append(
                Issue(
                    E_SHAPE_MISMATCH,
                    f"layer {layer_name!r} weights shape {list(array.shape)} does not match {list(shape)}",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                    details={"expected_shape": list(shape), "actual_shape": list(array.shape)},
                )
            )
            continue
        try:
            array = array.reshape(shape)
        except ValueError as exc:
            issues.append(
                Issue(
                    E_SHAPE_MISMATCH,
                    f"layer {layer_name!r} cannot be reshaped to {list(shape)}: {exc}",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                )
            )
            continue
        finite_mask = np.isfinite(array)
        if not bool(finite_mask.all()):
            bad_index = int(np.flatnonzero(~finite_mask.ravel())[0])
            bad_value = float(array.ravel()[bad_index])
            issues.append(
                Issue(
                    E_WEIGHT_NOT_FINITE,
                    f"layer {layer_name!r} contains non-finite weight at flat index {bad_index}",
                    command_index=command_index,
                    line_number=line_number,
                    layer=layer_name,
                    details={"flat_index": bad_index, "value": str(bad_value)},
                )
            )
            continue
        layers[layer_name] = WeightLayer(
            layer=layer_name,
            version=version,
            shape=shape,
            weights=array,
            order=file_order,
            line_number=line_number,
        )
        file_order += 1
    return layers, issues


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _parse_calibration_records(
    records: list[tuple[int, Any]], *, command_index: int | None = None
) -> tuple[CalibrationSet, list[Issue]]:
    samples_by_id: dict[str, CalibrationSample] = {}
    issues: list[Issue] = []
    for record_index, (line_number, record) in enumerate(records):
        if not isinstance(record, dict):
            issues.append(
                Issue(
                    E_CONFIG_INVALID,
                    "calibration record must be a JSON object",
                    command_index=command_index,
                    line_number=line_number,
                )
            )
            continue
        sample_id = record.get("sample_id", record.get("id", f"sample-{record_index:04d}"))
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(
                Issue(
                    E_CONFIG_INVALID,
                    "calibration sample_id must be a non-empty string",
                    command_index=command_index,
                    line_number=line_number,
                )
            )
            continue
        activations_raw = record.get("activations")
        if activations_raw is None and "layer" in record:
            layer_name = record.get("layer")
            values = record.get("input", record.get("values"))
            if isinstance(layer_name, str) and values is not None:
                activations_raw = {layer_name: values}
        if activations_raw is None:
            activations_raw = {
                str(key): value
                for key, value in record.items()
                if key not in {"sample_id", "id", "line", "source"}
                and isinstance(value, list)
            }
        if not isinstance(activations_raw, dict):
            issues.append(
                Issue(
                    E_CONFIG_INVALID,
                    f"calibration sample {sample_id!r} activations must be a mapping",
                    command_index=command_index,
                    line_number=line_number,
                    sample_id=sample_id,
                )
            )
            continue
        activations: dict[str, np.ndarray] = {}
        failed = False
        for layer_name, values in activations_raw.items():
            if not isinstance(layer_name, str) or not layer_name:
                issues.append(
                    Issue(
                        E_CONFIG_INVALID,
                        f"calibration sample {sample_id!r} has invalid layer key",
                        command_index=command_index,
                        line_number=line_number,
                        sample_id=sample_id,
                    )
                )
                failed = True
                continue
            try:
                array = np.asarray(values, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                issues.append(
                    Issue(
                        E_SHAPE_MISMATCH,
                        f"calibration sample {sample_id!r} layer {layer_name!r} is not numeric: {exc}",
                        command_index=command_index,
                        line_number=line_number,
                        layer=layer_name,
                        sample_id=sample_id,
                    )
                )
                failed = True
                continue
            if array.size == 0:
                issues.append(
                    Issue(
                        E_SHAPE_MISMATCH,
                        f"calibration sample {sample_id!r} layer {layer_name!r} is empty",
                        command_index=command_index,
                        line_number=line_number,
                        layer=layer_name,
                        sample_id=sample_id,
                    )
                )
                failed = True
                continue
            if not bool(np.isfinite(array).all()):
                bad_index = int(np.flatnonzero(~np.isfinite(array))[0])
                issues.append(
                    Issue(
                        E_WEIGHT_NOT_FINITE,
                        f"calibration sample {sample_id!r} layer {layer_name!r} has non-finite value at index {bad_index}",
                        command_index=command_index,
                        line_number=line_number,
                        layer=layer_name,
                        sample_id=sample_id,
                        details={"flat_index": bad_index},
                    )
                )
                failed = True
                continue
            activations[layer_name] = array
        if failed:
            continue
        if sample_id in samples_by_id:
            samples_by_id[sample_id].activations.update(activations)
        else:
            samples_by_id[sample_id] = CalibrationSample(
                sample_id=sample_id,
                activations=activations,
                line_number=line_number,
                source=record,
            )
    if not samples_by_id:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "calibration set contains no valid samples",
                command_index=command_index,
            )
        )
    return CalibrationSet(samples=list(samples_by_id.values())), issues


def load_calibration(
    path: str | Path, *, command_index: int | None = None
) -> tuple[CalibrationSet, list[Issue]]:
    records = _iter_json_records(path)
    if len(records) == 1 and isinstance(records[0][1], dict):
        only = records[0][1]
        if isinstance(only.get("samples"), list):
            records = [(records[0][0], item) for item in only["samples"]]
    calibration, issues = _parse_calibration_records(records, command_index=command_index)
    calibration.source_path = str(path)
    return calibration, issues


def parse_calibration_payload(
    payload: Any, *, command_index: int | None = None, line_number: int | None = None
) -> tuple[CalibrationSet, list[Issue]]:
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        payload = payload["samples"]
    if not isinstance(payload, list):
        return CalibrationSet([]), [
            Issue(
                E_CONFIG_INVALID,
                "replace_calibration requires a samples array",
                command_index=command_index,
                line_number=line_number,
            )
        ]
    return _parse_calibration_records(
        [(line_number or 1, item) for item in payload],
        command_index=command_index,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_COMMAND_ALIASES = {
    "set_bits": "set_bits",
    "set_bit_width": "set_bits",
    "set_layer_bits": "set_bits",
    "change_bits": "set_bits",
    "change_target_bits": "set_bits",
    "update_bits": "set_bits",
    "modify_bits": "set_bits",
    "target_bits": "set_bits",
    "set_group_size": "set_group_size",
    "set_grouping_granularity": "set_group_size",
    "change_group_size": "set_group_size",
    "set_group": "set_group_size",
    "set_granularity": "set_group_size",
    "change_grouping": "set_group_size",
    "set_grouping": "set_group_size",
    "set_fractional_bits": "set_fractional_bits",
    "set_fixed_fractional_bits": "set_fractional_bits",
    "set_scheme": "set_scheme",
    "set_quant_scheme": "set_scheme",
    "set_quantization_scheme": "set_scheme",
    "switch_scheme": "set_scheme",
    "switch_quantization_scheme": "set_scheme",
    "change_scheme": "set_scheme",
    "change_quantization_scheme": "set_scheme",
    "set_temperature": "set_temperature",
    "set_temp": "set_temperature",
    "set_distillation_temperature": "set_temperature",
    "replace_weight": "replace_weight",
    "replace_layer_weight": "replace_weight",
    "replace_weights": "replace_weight",
    "replace_layer": "replace_weight",
    "replace_calibration": "replace_calibration",
    "replace_calibration_samples": "replace_calibration",
    "replace_calibration_set": "replace_calibration",
}


def _normalize_command_op(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _COMMAND_ALIASES.get(normalized, normalized)


def _command_records(path: str | Path) -> tuple[list[tuple[int, int, Any]], list[Issue]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(E_IO, f"cannot read {source}: {exc}") from exc
    if source.suffix.lower() in {".json", ".yaml", ".yml"}:
        try:
            if source.suffix.lower() == ".json":
                parsed = json.loads(text)
            else:
                parsed = parse_yaml(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return [], [
                Issue(
                    E_COMMAND_INVALID,
                    f"invalid command document: {exc}",
                    line_number=1,
                )
            ]
        if isinstance(parsed, dict) and isinstance(parsed.get("commands"), list):
            parsed = parsed["commands"]
        if not isinstance(parsed, list):
            return [], [Issue(E_COMMAND_INVALID, "command document must contain an array")]
        return [((index + 1), 1, item) for index, item in enumerate(parsed)], []

    records: list[tuple[int, int, Any]] = []
    issues: list[Issue] = []
    sequence = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        sequence += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    f"invalid command JSON at line {line_number}: {exc}",
                    command_index=sequence,
                    line_number=line_number,
                )
            )
            continue
        if isinstance(item, dict) and isinstance(item.get("commands"), list):
            for nested in item["commands"]:
                sequence += 1
                records.append((sequence, line_number, nested))
        else:
            records.append((sequence, line_number, item))
    return records, issues


def load_commands(path: str | Path) -> tuple[list[Command], list[Issue]]:
    records, issues = _command_records(path)
    commands: list[Command] = []
    for command_index, line_number, record in records:
        if not isinstance(record, dict):
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    "command must be a JSON object",
                    command_index=command_index,
                    line_number=line_number,
                )
            )
            continue
        raw_op = record.get("cmd", record.get("op", record.get("type", record.get("command"))))
        if not isinstance(raw_op, str) or not raw_op:
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    "command must specify cmd/op/type",
                    command_index=command_index,
                    line_number=line_number,
                )
            )
            continue
        op = _normalize_command_op(raw_op)
        layer = record.get("layer")
        if layer is None:
            layer = record.get("layer_name")
        if layer is not None and not isinstance(layer, str):
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    "command layer must be a string",
                    command_index=command_index,
                    line_number=line_number,
                )
            )
            continue
        payload = dict(record)
        for key in ("cmd", "op", "type", "command", "line_number"):
            payload.pop(key, None)
        commands.append(
            Command(
                command_index=command_index,
                line_number=line_number,
                op=op,
                layer=layer,
                payload=payload,
                raw=record,
            )
        )
    issues.sort(
        key=lambda issue: (
            issue.command_index if issue.command_index is not None else -1,
            issue.line_number if issue.line_number is not None else -1,
            issue.code,
        )
    )
    return commands, issues


def validate_replace_weight_payload(
    command: Command, existing: WeightLayer | None
) -> tuple[WeightLayer | None, list[Issue]]:
    """Validate an inline weight replacement without touching state."""

    issues: list[Issue] = []
    layer_name = command.layer
    if not layer_name:
        return None, [
            Issue(
                E_LAYER_INVALID,
                "replace_weight requires a layer",
                command_index=command.command_index,
                line_number=command.line_number,
            )
        ]
    if existing is None:
        return None, [
            Issue(
                E_LAYER_INVALID,
                f"unknown layer {layer_name!r}",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
            )
        ]
    raw_shape = command.payload.get("shape", list(existing.shape))
    shape = _shape_tuple(raw_shape)
    if shape is None:
        return None, [
            Issue(
                E_SHAPE_MISMATCH,
                f"replacement shape for {layer_name!r} must contain positive integers",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
                details={"shape": raw_shape},
            )
        ]
    if shape != existing.shape:
        return None, [
            Issue(
                E_SHAPE_MISMATCH,
                f"replacement shape {list(shape)} for {layer_name!r} does not match {list(existing.shape)}",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
            )
        ]
    raw_weights = command.payload.get(
        "weights",
        command.payload.get(
            "weight", command.payload.get("values", command.payload.get("new_weights"))
        ),
    )
    if raw_weights is None:
        return None, [
            Issue(
                E_SHAPE_MISMATCH,
                f"replace_weight for {layer_name!r} requires weights",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
            )
        ]
    try:
        array = np.asarray(raw_weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return None, [
            Issue(
                E_SHAPE_MISMATCH,
                f"replacement weights for {layer_name!r} are not numeric: {exc}",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
            )
        ]
    nested_shape_mismatch = array.ndim > 1 and tuple(array.shape) != shape
    if array.size != int(np.prod(shape, dtype=np.int64)) or nested_shape_mismatch:
        return None, [
            Issue(
                E_SHAPE_MISMATCH,
                f"replacement weights shape {list(array.shape)} for {layer_name!r} does not match {list(shape)}",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
                details={"expected_shape": list(shape), "actual_shape": list(array.shape)},
            )
        ]
    array = array.reshape(shape)
    if not bool(np.isfinite(array).all()):
        bad_index = int(np.flatnonzero(~np.isfinite(array).ravel())[0])
        return None, [
            Issue(
                E_WEIGHT_NOT_FINITE,
                f"replacement weights for {layer_name!r} contain non-finite value at flat index {bad_index}",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
                details={"flat_index": bad_index},
            )
        ]
    version = command.payload.get("version", existing.version + 1)
    if isinstance(version, bool) or not isinstance(version, int):
        return None, [
            Issue(
                E_LAYER_INVALID,
                f"replacement version for {layer_name!r} must be an integer",
                command_index=command.command_index,
                line_number=command.line_number,
                layer=layer_name,
            )
        ]
    return (
        WeightLayer(
            layer=layer_name,
            version=version,
            shape=shape,
            weights=array,
            order=existing.order,
            line_number=command.line_number,
        ),
        issues,
    )
