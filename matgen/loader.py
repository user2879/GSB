"""Deterministic sample and configuration loading with error isolation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import (
    E_COMMANDS_JSON,
    E_CONFIG_JSON,
    E_BUDGET,
    E_CONCURRENCY,
    E_LATENCY,
    E_MATERIALS_JSON,
    E_MATERIAL_WEIGHTS,
    E_LAYER_REFERENCE,
    E_NOISE_PARAMS,
    E_QUANTIZATION,
    E_RESOLUTION_TIER,
    E_RESOLUTION_TIERS,
    E_ROUNDING_MODE,
    E_SCHEMA,
    E_TEXTURE_DECODE,
    E_TEXTURE_HEADER,
    E_TEXTURE_MODE,
    E_TEXTURE_MISSING,
    E_TEXTURE_SIZE,
    Issue,
    make_issue,
)
from .fixed import CHANNELS, ROUNDING_MODES
from .hashing import fingerprint, fingerprint_file
from .models import (
    Command,
    FixedPointConfig,
    GenerationConfig,
    LoadedInput,
    Material,
    MaterialLayer,
    NoiseLayer,
    Texture,
)


SUPPORTED_BIT_WIDTHS = (8, 12, 16, 20, 24)
ALLOWED_QUANTIZATION_TIERS = (1, 2, 4, 8, 16)
ALLOWED_MIX_MODES = ("weighted", "multiply", "screen", "overlay")
MAX_CONCURRENCY_LIMIT = 32
ALLOWED_TEXTURE_MODES = frozenset({"L", "LA", "RGB", "RGBA"})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUIRED_TOLERANCES = (
    "noise_composition",
    "quantization",
    "material_mix",
    "channel_synthesis",
)


def _read_json(path: Path, code: str, issues: list[Issue]) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        issues.append(make_issue(code, f"JSON file not found: {path.name}", source=str(path)))
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(
            make_issue(code, f"cannot parse JSON: {exc}", source=str(path))
        )
        return None
    if not isinstance(value, dict):
        issues.append(make_issue(code, "JSON root must be an object", source=str(path)))
        return None
    return value


def _positive_int(value: Any, name: str, code: str, issues: list[Issue], source: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        issues.append(make_issue(code, f"{name} must be a positive integer", source=source, item=name))
        return None
    return value


def _finite_number(value: Any, name: str, issues: list[Issue], source: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(make_issue(E_NOISE_PARAMS, f"{name} must be numeric", source=source, item=name))
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        issues.append(
            make_issue(E_NOISE_PARAMS, f"{name} must be positive and finite", source=source, item=name)
        )
        return None
    return numeric


def parse_config(raw: dict[str, Any], source: str) -> tuple[GenerationConfig | None, list[Issue]]:
    issues: list[Issue] = []
    if raw.get("schema_version", 1) != 1:
        issues.append(make_issue(E_SCHEMA, "unsupported config schema_version", source=source))

    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        issues.append(make_issue(E_SCHEMA, "seed must be a non-negative integer", source=source, item="seed"))
        seed = 0

    tiers_raw = raw.get("resolution_tiers")
    resolution_tiers: tuple[int, ...] = ()
    if not isinstance(tiers_raw, list) or not tiers_raw:
        issues.append(
            make_issue(E_RESOLUTION_TIERS, "resolution_tiers must be a non-empty list", source=source)
        )
    elif any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in tiers_raw):
        issues.append(
            make_issue(E_RESOLUTION_TIERS, "resolution tiers must be positive integers", source=source)
        )
    elif any(a >= b for a, b in zip(tiers_raw, tiers_raw[1:])):
        issues.append(
            make_issue(E_RESOLUTION_TIERS, "resolution tiers must be strictly increasing", source=source)
        )
    else:
        resolution_tiers = tuple(int(item) for item in tiers_raw)

    resolution = raw.get("resolution")
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution <= 0
        or (resolution_tiers and resolution not in resolution_tiers)
    ):
        issues.append(
            make_issue(E_RESOLUTION_TIER, "resolution must be a member of resolution_tiers", source=source)
        )
        resolution_value = resolution if isinstance(resolution, int) and resolution > 0 else 0
    else:
        resolution_value = resolution

    layers_raw = raw.get("noise_layers")
    noise_layers: list[NoiseLayer] = []
    if not isinstance(layers_raw, list) or not layers_raw:
        issues.append(make_issue(E_NOISE_PARAMS, "noise_layers must be a non-empty list", source=source))
    else:
        for index, layer_raw in enumerate(layers_raw):
            if not isinstance(layer_raw, dict):
                issues.append(
                    make_issue(E_NOISE_PARAMS, "noise layer must be an object", source=source, item=str(index))
                )
                continue
            texture = layer_raw.get("texture")
            if not isinstance(texture, str) or not texture:
                issues.append(
                    make_issue(E_NOISE_PARAMS, "noise layer texture must be a non-empty string", source=source, item=str(index))
                )
                continue
            frequency = _finite_number(layer_raw.get("frequency"), "frequency", issues, source)
            amplitude = _finite_number(layer_raw.get("amplitude"), "amplitude", issues, source)
            if frequency is None or amplitude is None:
                continue
            name = layer_raw.get("name", f"layer-{index}")
            if not isinstance(name, str) or not name:
                name = f"layer-{index}"
            extras = {k: v for k, v in layer_raw.items() if k not in {"frequency", "amplitude", "texture", "name"}}
            noise_layers.append(
                NoiseLayer(
                    index=index,
                    name=name,
                    texture=texture,
                    frequency=frequency,
                    amplitude=amplitude,
                    as_dict_extra=extras,
                )
            )

    declared_count = raw.get("noise_layer_count")
    if isinstance(layers_raw, list) and declared_count is not None and declared_count != len(layers_raw):
        issues.append(
            make_issue(E_NOISE_PARAMS, "noise_layer_count must equal len(noise_layers)", source=source)
        )

    mix_mode = raw.get("material_mix_mode", "weighted")
    if mix_mode not in ALLOWED_MIX_MODES:
        issues.append(
            make_issue(
                E_SCHEMA,
                f"material_mix_mode must be one of {ALLOWED_MIX_MODES}",
                source=source,
                item="material_mix_mode",
            )
        )

    fp_raw = raw.get("fixed_point")
    fixed_point: FixedPointConfig | None = None
    if not isinstance(fp_raw, dict):
        issues.append(make_issue(E_QUANTIZATION, "fixed_point must be an object", source=source))
    else:
        bit_width = fp_raw.get("bit_width")
        if isinstance(bit_width, bool) or bit_width not in SUPPORTED_BIT_WIDTHS:
            issues.append(
                make_issue(
                    E_QUANTIZATION,
                    f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}",
                    source=source,
                    item="bit_width",
                )
            )
            bit_width_value = 16
        else:
            bit_width_value = int(bit_width)

        fraction_bits = fp_raw.get("fraction_bits")
        if (
            isinstance(fraction_bits, bool)
            or not isinstance(fraction_bits, int)
            or fraction_bits < 1
            or fraction_bits > bit_width_value - 2
        ):
            issues.append(
                make_issue(
                    E_QUANTIZATION,
                    "fraction_bits must be an integer between 1 and bit_width-2",
                    source=source,
                    item="fraction_bits",
                )
            )
            fraction_value = max(1, bit_width_value - 4)
        else:
            fraction_value = int(fraction_bits)

        tiers = fp_raw.get("quantization_tiers")
        if (
            not isinstance(tiers, list)
            or not tiers
            or any(isinstance(item, bool) or not isinstance(item, int) or item not in ALLOWED_QUANTIZATION_TIERS for item in tiers)
        ):
            issues.append(
                make_issue(
                    E_QUANTIZATION,
                    f"quantization_tiers must be a non-empty subset of {ALLOWED_QUANTIZATION_TIERS}",
                    source=source,
                    item="quantization_tiers",
                )
            )
            tier_values = (2,)
        else:
            tier_values = tuple(sorted(set(int(item) for item in tiers)))

        quantization_level = fp_raw.get("quantization_level")
        if (
            isinstance(quantization_level, bool)
            or not isinstance(quantization_level, int)
            or quantization_level not in tier_values
        ):
            issues.append(
                make_issue(
                    E_QUANTIZATION,
                    "quantization_level must belong to quantization_tiers",
                    source=source,
                    item="quantization_level",
                )
            )
            quant_value = tier_values[0]
        else:
            quant_value = int(quantization_level)

        rounding_raw = fp_raw.get("rounding")
        rounding: dict[str, str] = {}
        if not isinstance(rounding_raw, dict):
            issues.append(make_issue(E_ROUNDING_MODE, "rounding must be an object", source=source))
            rounding = {channel: "half_even" for channel in CHANNELS}
        else:
            for channel in CHANNELS:
                mode = rounding_raw.get(channel, "half_even")
                if mode not in ROUNDING_MODES:
                    issues.append(
                        make_issue(
                            E_ROUNDING_MODE,
                            f"unsupported rounding mode for {channel}",
                            source=source,
                            item=channel,
                            details={"value": mode},
                        )
                    )
                    mode = "half_even"
                rounding[channel] = mode

        weight_fraction_bits = fp_raw.get("weight_fraction_bits", 12)
        if (
            isinstance(weight_fraction_bits, bool)
            or not isinstance(weight_fraction_bits, int)
            or not 1 <= weight_fraction_bits <= 20
        ):
            issues.append(
                make_issue(
                    E_QUANTIZATION,
                    "weight_fraction_bits must be an integer between 1 and 20",
                    source=source,
                )
            )
            weight_fraction_bits = 12

        fixed_point = FixedPointConfig(
            bit_width=bit_width_value,
            fraction_bits=fraction_value,
            quantization_level=quant_value,
            quantization_tiers=tier_values,
            rounding=rounding,
            weight_fraction_bits=int(weight_fraction_bits),
        )

    trace_raw = raw.get("error_trace", {})
    if not isinstance(trace_raw, dict):
        trace_raw = {}
        issues.append(make_issue(E_SCHEMA, "error_trace must be an object", source=source))
    ladder_raw = trace_raw.get("bit_width_ladder", [12, 16, 20])
    if (
        not isinstance(ladder_raw, list)
        or not ladder_raw
        or any(isinstance(item, bool) or item not in SUPPORTED_BIT_WIDTHS for item in ladder_raw)
        or any(a >= b for a, b in zip(ladder_raw, ladder_raw[1:]))
    ):
        issues.append(
            make_issue(E_QUANTIZATION, "error_trace.bit_width_ladder must be strictly increasing supported widths", source=source)
        )
        ladder = (12, 16, 20)
    else:
        ladder = tuple(int(item) for item in ladder_raw)
    current_width = fixed_point.bit_width if fixed_point else 16
    if current_width not in ladder or not any(width > current_width for width in ladder):
        issues.append(
            make_issue(
                E_QUANTIZATION,
                "error_trace.bit_width_ladder must contain bit_width and a higher width",
                source=source,
            )
        )

    tolerances_raw = trace_raw.get("tolerance", {})
    tolerances: dict[str, float] = {}
    if not isinstance(tolerances_raw, dict):
        issues.append(make_issue(E_SCHEMA, "error_trace.tolerance must be an object", source=source))
        tolerances_raw = {}
    for stage in REQUIRED_TOLERANCES:
        value = tolerances_raw.get(stage, 1e-5)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            issues.append(
                make_issue(E_BUDGET, f"tolerance for {stage} must be finite and non-negative", source=source)
            )
            value = 1e-5
        tolerances[stage] = float(value)
    consistency = trace_raw.get("final_consistency_tolerance", 1e-5)
    if isinstance(consistency, bool) or not isinstance(consistency, (int, float)) or not math.isfinite(float(consistency)) or float(consistency) < 0:
        issues.append(make_issue(E_BUDGET, "final_consistency_tolerance must be finite and non-negative", source=source))
        consistency = 1e-5

    concurrency = _positive_int(raw.get("concurrency", 1), "concurrency", E_CONCURRENCY, issues, source)
    max_concurrency = _positive_int(raw.get("max_concurrency", 4), "max_concurrency", E_CONCURRENCY, issues, source)
    if max_concurrency is not None and max_concurrency > MAX_CONCURRENCY_LIMIT:
        issues.append(
            make_issue(E_CONCURRENCY, f"max_concurrency cannot exceed {MAX_CONCURRENCY_LIMIT}", source=source)
        )
    if concurrency is not None and max_concurrency is not None and concurrency > max_concurrency:
        issues.append(
            make_issue(E_CONCURRENCY, "concurrency cannot exceed max_concurrency", source=source)
        )
    point_budget = _positive_int(raw.get("point_budget", 1024), "point_budget", E_BUDGET, issues, source)
    latency_budget = _positive_int(raw.get("latency_budget_ms", 25), "latency_budget_ms", E_LATENCY, issues, source)
    snapshot_interval = _positive_int(
        raw.get("snapshot_commit_interval", 1),
        "snapshot_commit_interval",
        E_BUDGET,
        issues,
        source,
    )
    max_boundary_samples = raw.get("max_boundary_samples", 10000)
    if isinstance(max_boundary_samples, bool) or not isinstance(max_boundary_samples, int) or max_boundary_samples < 0:
        issues.append(make_issue(E_SCHEMA, "max_boundary_samples must be a non-negative integer", source=source))
        max_boundary_samples = 10000

    fatal = any(
        issue.code
        in {
            E_SCHEMA,
            E_RESOLUTION_TIERS,
            E_RESOLUTION_TIER,
            E_NOISE_PARAMS,
            E_QUANTIZATION,
            E_ROUNDING_MODE,
            E_CONCURRENCY,
            E_BUDGET,
            E_LATENCY,
        }
        for issue in issues
    )
    if fatal or fixed_point is None:
        return None, issues

    config = GenerationConfig(
        schema_version=1,
        seed=seed,
        resolution_tiers=resolution_tiers,
        resolution=resolution_value,
        noise_layers=tuple(noise_layers),
        material_mix_mode=mix_mode,
        fixed_point=fixed_point,
        bit_width_ladder=ladder,
        tolerances=tolerances,
        final_consistency_tolerance=float(consistency),
        concurrency=concurrency or 1,
        max_concurrency=max_concurrency or 4,
        point_budget=point_budget or 1024,
        latency_budget_ms=latency_budget or 25,
        snapshot_commit_interval=snapshot_interval or 1,
        embed_pixel_diffs=bool(trace_raw.get("embed_pixel_diffs", True)),
        max_boundary_samples=int(max_boundary_samples),
    )
    return config, issues


def _texture_luma(mode: str, pixels: np.ndarray) -> np.ndarray:
    if mode == "L":
        return np.ascontiguousarray(pixels, dtype=np.uint8)
    if mode == "LA":
        return np.ascontiguousarray(pixels[..., 0], dtype=np.uint8)
    rgb = pixels[..., :3].astype(np.uint32)
    luma = (299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2] + 500) // 1000
    return luma.astype(np.uint8)


def scan_textures(
    input_dir: Path,
    expected_resolution: int | None,
    issues: list[Issue],
) -> dict[str, Texture]:
    textures: dict[str, Texture] = {}
    for path in sorted(input_dir.rglob("*.png"), key=lambda item: item.relative_to(input_dir).as_posix()):
        relative_name = path.relative_to(input_dir).as_posix()
        try:
            with path.open("rb") as handle:
                signature = handle.read(8)
            if signature != PNG_SIGNATURE:
                issues.append(
                    make_issue(E_TEXTURE_HEADER, "invalid PNG signature", source=str(path), item=relative_name)
                )
                continue
            with Image.open(path) as image:
                image_format = image.format
                mode = image.mode
                width, height = image.size
                if image_format != "PNG":
                    issues.append(
                        make_issue(E_TEXTURE_HEADER, "container is not PNG", source=str(path), item=relative_name)
                    )
                    continue
                if mode not in ALLOWED_TEXTURE_MODES:
                    issues.append(
                        make_issue(
                            E_TEXTURE_MODE,
                            f"unsupported PNG pixel mode: {mode}",
                            source=str(path),
                            item=relative_name,
                            details={"mode": mode, "allowed": sorted(ALLOWED_TEXTURE_MODES)},
                        )
                    )
                    continue
                if expected_resolution is not None and (width != expected_resolution or height != expected_resolution):
                    issues.append(
                        make_issue(
                            E_TEXTURE_SIZE,
                            f"texture size {width}x{height} does not match resolution {expected_resolution}",
                            source=str(path),
                            item=relative_name,
                        )
                    )
                    continue
                image.load()
                pixels = np.asarray(image, dtype=np.uint8).copy()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            issues.append(
                make_issue(E_TEXTURE_DECODE, f"cannot decode PNG: {exc}", source=str(path), item=relative_name)
            )
            continue
        content_hash = fingerprint_file(path)
        cache_key = fingerprint(
            {
                "kind": "texture",
                "content_hash": content_hash,
                "mode": mode,
                "width": width,
                "height": height,
            }
        )
        textures[relative_name] = Texture(
            name=relative_name,
            path=str(path),
            width=width,
            height=height,
            mode=mode,
            pixels=pixels,
            luma=_texture_luma(mode, pixels),
            content_hash=content_hash,
            cache_key=cache_key,
        )
    return textures


def parse_materials(
    raw: dict[str, Any],
    config: GenerationConfig | None,
    source: str,
) -> tuple[tuple[Material, ...], list[Issue]]:
    issues: list[Issue] = []
    if raw.get("schema_version", 1) != 1:
        issues.append(make_issue(E_SCHEMA, "unsupported materials schema_version", source=source))
    rows = raw.get("materials")
    if not isinstance(rows, list) or not rows:
        issues.append(make_issue(E_MATERIALS_JSON, "materials must be a non-empty list", source=source))
        return (), issues
    materials: list[Material] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            issues.append(make_issue(E_SCHEMA, "material must be an object", source=source, item=str(index)))
            continue
        material_id = item.get("id")
        if not isinstance(material_id, str) or not material_id:
            issues.append(make_issue(E_SCHEMA, "material id must be a non-empty string", source=source, item=str(index)))
            continue
        if material_id in seen_ids:
            issues.append(
                make_issue(E_SCHEMA, f"duplicate material id: {material_id}", source=source, item=material_id)
            )
            continue
        seen_ids.add(material_id)
        label = item.get("label", material_id)
        if not isinstance(label, str):
            label = material_id
        layers_raw = item.get("layers")
        if not isinstance(layers_raw, list) or not layers_raw:
            issues.append(
                make_issue(E_MATERIAL_WEIGHTS, "material layers must be a non-empty list", source=source, item=material_id)
            )
            continue
        layer_rows: list[MaterialLayer] = []
        seen_layers: set[int] = set()
        valid = True
        for layer_index, layer_raw in enumerate(layers_raw):
            if not isinstance(layer_raw, dict):
                issues.append(
                    make_issue(E_LAYER_REFERENCE, "material layer must be an object", source=source, item=material_id)
                )
                valid = False
                continue
            reference = layer_raw.get("noise_layer")
            limit = config.noise_layer_count if config is not None else 0
            if (
                isinstance(reference, bool)
                or not isinstance(reference, int)
                or reference < 0
                or (config is not None and reference >= limit)
            ):
                issues.append(
                    make_issue(
                        E_LAYER_REFERENCE,
                        f"noise_layer reference must be in [0, {limit - 1}]",
                        source=source,
                        item=f"{material_id}[{layer_index}]",
                    )
                )
                valid = False
                continue
            if reference in seen_layers:
                issues.append(
                    make_issue(
                        E_LAYER_REFERENCE,
                        f"duplicate noise_layer reference: {reference}",
                        source=source,
                        item=material_id,
                    )
                )
                valid = False
                continue
            seen_layers.add(reference)
            weights_raw = layer_raw.get("weights")
            if not isinstance(weights_raw, dict):
                issues.append(
                    make_issue(E_MATERIAL_WEIGHTS, "weights must be an object", source=source, item=material_id)
                )
                valid = False
                continue
            weights: dict[str, float] = {}
            for channel in CHANNELS:
                value = weights_raw.get(channel)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    issues.append(
                        make_issue(
                            E_MATERIAL_WEIGHTS,
                            f"weight for {channel} must be finite and non-negative",
                            source=source,
                            item=f"{material_id}[{reference}]",
                        )
                    )
                    valid = False
                    break
                weights[channel] = float(value)
            if not valid:
                continue
            layer_rows.append(MaterialLayer(layer_index=reference, weights=weights))
        if not valid:
            continue
        layer_rows.sort(key=lambda row: row.layer_index)
        for channel in CHANNELS:
            if not any(row.weights[channel] > 0 for row in layer_rows):
                issues.append(
                    make_issue(
                        E_MATERIAL_WEIGHTS,
                        f"material {material_id} has no positive {channel} weight",
                        source=source,
                        item=material_id,
                    )
                )
                valid = False
                break
        if not valid:
            continue
        materials.append(
            Material(material_id=material_id, label=label, layers=tuple(layer_rows))
        )
    if not materials:
        issues.append(make_issue(E_MATERIALS_JSON, "no valid materials remain", source=source))
    return tuple(materials), issues


COMMAND_OPS = frozenset({"set_weight", "set_layer", "set_mix_mode", "set_quantization", "set_bit_width"})


def parse_commands(raw: dict[str, Any], source: str) -> tuple[tuple[Command, ...], list[Issue]]:
    issues: list[Issue] = []
    rows = raw.get("commands")
    if not isinstance(rows, list):
        issues.append(make_issue(E_COMMANDS_JSON, "commands must be a list", source=source))
        return (), issues
    commands: list[Command] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            issues.append(make_issue(E_COMMAND_INVALID, "command must be an object", source=source, item=str(index)))
            continue
        command_id = item.get("id")
        op = item.get("op")
        if not isinstance(command_id, str) or not command_id:
            issues.append(make_issue(E_COMMAND_INVALID, "command id must be a non-empty string", source=source, item=str(index)))
            continue
        if command_id in seen_ids:
            issues.append(make_issue(E_COMMAND_INVALID, "duplicate command id", source=source, item=command_id))
            continue
        seen_ids.add(command_id)
        if op not in COMMAND_OPS:
            issues.append(
                make_issue(E_COMMAND_INVALID, f"unsupported command operation: {op}", source=source, item=command_id)
            )
            continue
        payload = {key: value for key, value in item.items() if key not in {"id", "op"}}
        commands.append(Command(command_id=command_id, op=str(op), payload=payload))
    return tuple(commands), issues


def load_input(input_dir: str | Path) -> LoadedInput:
    input_path = Path(input_dir).resolve()
    issues: list[Issue] = []
    if not input_path.exists() or not input_path.is_dir():
        issues.append(make_issue(E_SCHEMA, f"input directory not found: {input_path}", source=str(input_path)))
        return LoadedInput(
            input_dir=str(input_path),
            textures={},
            config=None,
            materials=(),
            commands=(),
            issues=issues,
        )

    config_path = input_path / "config.json"
    materials_path = input_path / "materials.json"
    commands_path = input_path / "commands.json"
    config_raw = _read_json(config_path, E_CONFIG_JSON, issues) or {}
    materials_raw = _read_json(materials_path, E_MATERIALS_JSON, issues) or {}
    commands_raw = _read_json(commands_path, E_COMMANDS_JSON, issues) or {}
    config, config_issues = parse_config(config_raw, str(config_path))
    issues.extend(config_issues)
    textures = scan_textures(
        input_path,
        config.resolution if config is not None else None,
        issues,
    )
    if config is not None:
        for layer in config.noise_layers:
            if layer.texture not in textures:
                issues.append(
                    make_issue(
                        E_TEXTURE_MISSING,
                        f"noise layer {layer.index} references a missing or invalid texture: {layer.texture}",
                        source=str(config_path),
                        item=layer.texture,
                    )
                )
    materials, material_issues = parse_materials(materials_raw, config, str(materials_path))
    issues.extend(material_issues)
    commands, command_issues = parse_commands(commands_raw, str(commands_path))
    issues.extend(command_issues)
    return LoadedInput(
        input_dir=str(input_path),
        textures=textures,
        config=config,
        materials=materials,
        commands=commands,
        issues=issues,
        config_raw=config_raw,
        materials_raw=materials_raw,
        commands_raw=commands_raw,
    )


def state_fingerprint(config: GenerationConfig, materials: tuple[Material, ...]) -> str:
    return fingerprint(
        {
            "config": config.as_dict(),
            "materials": [material.as_dict() for material in materials],
        }
    )


def input_fingerprint(loaded: LoadedInput) -> str:
    if loaded.config is None:
        return fingerprint({"config_raw": loaded.config_raw, "materials_raw": loaded.materials_raw})
    return fingerprint(
        {
            "state": state_fingerprint(loaded.config, loaded.materials),
            "commands": [{"id": cmd.command_id, "op": cmd.op, "payload": cmd.payload} for cmd in loaded.commands],
        }
    )
