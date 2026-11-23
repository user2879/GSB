"""Configuration and command-stream loading / validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import CommandError, ConfigError, WorkbenchError
from .fixed import scalar_to_fixed
from .imageio import inspect_png_directory, load_rgb
from .model import Change, CommandStream, ConfigBundle, EffectiveConfig, PyramidSpec, Rect
from .util import array_fingerprint, canonical_json, sha256_json, sha256_text


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _issue(path: str, message: str) -> dict[str, str]:
    return {"path": path, "message": message}


def load_config(config_path: str | Path, image_dir: str | Path) -> ConfigBundle:
    config_path = Path(config_path)
    image_dir = Path(image_dir).resolve()
    errors: list[dict[str, str]] = []
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise ConfigError([_issue("$", f"cannot parse config JSON: {exc}")]) from exc
    if not isinstance(raw, dict):
        raise ConfigError([_issue("$", "config root must be an object")])

    if not image_dir.is_dir():
        raise ConfigError([_issue("images", f"image directory does not exist: {image_dir}")])
    image_records = inspect_png_directory(image_dir)
    bad_records = [record for record in image_records if "error" in record]
    if bad_records:
        for record in bad_records:
            errors.append(_issue(f"images.{record['name']}", record["error"]))
    if not image_records:
        errors.append(_issue("images", "image directory contains no PNG files"))
    valid_records = [record for record in image_records if "error" not in record]
    if errors and not valid_records:
        raise ConfigError(errors)

    image_raw = raw.get("image", {})
    if not isinstance(image_raw, dict):
        errors.append(_issue("image", "image must be an object"))
        image_raw = {}
    source_name = image_raw.get("source")
    style_name = image_raw.get("style")
    if not isinstance(source_name, str) or not source_name:
        errors.append(_issue("image.source", "source PNG filename is required"))
    if not isinstance(style_name, str) or not style_name:
        errors.append(_issue("image.style", "style PNG filename is required"))
    source_path: Path | None = None
    style_path: Path | None = None
    if isinstance(source_name, str):
        candidate = (image_dir / source_name).resolve()
        if candidate.parent != image_dir or candidate.suffix.lower() != ".png":
            errors.append(_issue("image.source", "source must be a PNG filename directly inside image directory"))
        elif not candidate.is_file():
            errors.append(_issue("image.source", f"source PNG not found: {source_name}"))
        else:
            source_path = candidate
    if isinstance(style_name, str):
        candidate = (image_dir / style_name).resolve()
        if candidate.parent != image_dir or candidate.suffix.lower() != ".png":
            errors.append(_issue("image.style", "style must be a PNG filename directly inside image directory"))
        elif not candidate.is_file():
            errors.append(_issue("image.style", f"style PNG not found: {style_name}"))
        else:
            style_path = candidate
    if source_path is not None and valid_records:
        source_record = next(
            (record for record in valid_records if record["name"] == source_path.name), None
        )
        if source_record is not None:
            expected_size = (source_record["width"], source_record["height"])
            for record in valid_records:
                if record["name"] == style_name:
                    continue
                if (record["width"], record["height"]) != expected_size:
                    errors.append(
                        _issue(
                            f"images.{record['name']}",
                            f"resolution mismatch: expected {expected_size[0]}x{expected_size[1]}, got {record['width']}x{record['height']}",
                        )
                    )
    if errors and (source_path is None or style_path is None):
        raise ConfigError(errors)

    source_rgb = load_rgb(source_path)
    style_rgb = load_rgb(style_path)
    base_height, base_width = source_rgb.shape[:2]

    pyramid_raw = raw.get("pyramid", {})
    if not isinstance(pyramid_raw, dict):
        errors.append(_issue("pyramid", "pyramid must be an object"))
        pyramid_raw = {}
    levels = pyramid_raw.get("levels")
    if not _is_int(levels) or levels <= 0:
        errors.append(_issue("pyramid.levels", "levels must be a positive integer"))
        levels = 0
    factors = pyramid_raw.get("downsample_factors")
    if not isinstance(factors, list) or not factors or not all(_is_int(v) and v >= 1 for v in factors):
        errors.append(_issue("pyramid.downsample_factors", "must be a non-empty list of positive integers"))
        factors = []
    elif levels and len(factors) != levels:
        errors.append(_issue("pyramid.downsample_factors", "length must equal pyramid.levels"))
    elif factors and factors[0] != 1:
        errors.append(_issue("pyramid.downsample_factors[0]", "level 0 factor must be 1"))
    if factors and any(factors[i] > factors[i + 1] for i in range(len(factors) - 1)):
        errors.append(_issue("pyramid.downsample_factors", "factors must be monotonically non-decreasing"))
    tile_size = pyramid_raw.get("tile_size")
    if not _is_int(tile_size) or tile_size <= 0:
        errors.append(_issue("pyramid.tile_size", "tile_size must be a positive integer"))
        tile_size = 0
    tile_order = pyramid_raw.get("tile_order", "row_major")
    if tile_order not in ("row_major", "row-major"):
        errors.append(_issue("pyramid.tile_order", "only stable row_major tile order is supported"))
    tile_order = "row_major"

    fraction_raw = raw.get("fixed_point", {})
    fraction_bits = fraction_raw.get("fraction_bits") if isinstance(fraction_raw, dict) else None
    if not _is_int(fraction_bits) or not 1 <= fraction_bits <= 16:
        errors.append(_issue("fixed_point.fraction_bits", "fraction_bits must be an integer in [1, 16]"))
        fraction_bits = 0

    spec: PyramidSpec | None = None
    if not errors and levels and factors and tile_size and fraction_bits:
        try:
            spec = PyramidSpec.create(
                base_width=base_width,
                base_height=base_height,
                levels=levels,
                factors=factors,
                tile_size=tile_size,
                fraction_bits=fraction_bits,
                tile_order=tile_order,
            )
        except ValueError as exc:
            errors.append(_issue("pyramid", str(exc)))
        if spec is not None:
            for level, (width, height) in enumerate(spec.sizes):
                if width % tile_size or height % tile_size:
                    errors.append(
                        _issue(
                            f"pyramid.levels[{level}]",
                            f"level size {width}x{height} is not divisible by tile_size {tile_size}",
                        )
                    )

    if spec is None:
        raise ConfigError(errors or [_issue("pyramid", "invalid pyramid specification")])

    masks_raw = raw.get("masks", [])
    if not isinstance(masks_raw, list):
        errors.append(_issue("masks", "masks must be a list of layer mask objects"))
        masks_raw = []
    mask_rects: list[list[Rect]] = [[] for _ in range(spec.levels)]
    used_mask_layers: set[int] = set()
    for index, layer_mask in enumerate(masks_raw):
        if not isinstance(layer_mask, dict):
            errors.append(_issue(f"masks[{index}]", "must be an object"))
            continue
        layer = layer_mask.get("layer")
        if not _is_int(layer) or not 0 <= layer < spec.levels:
            errors.append(_issue(f"masks[{index}].layer", "layer must be in range"))
            continue
        if layer in used_mask_layers:
            errors.append(_issue(f"masks[{index}].layer", "duplicate layer mask entry"))
        used_mask_layers.add(layer)
        rects = layer_mask.get("rects", [])
        if not isinstance(rects, list):
            errors.append(_issue(f"masks[{index}].rects", "rects must be a list"))
            continue
        layer_rects: list[Rect] = []
        for rect_index, rect in enumerate(rects):
            parsed = _parse_rect(rect, spec, layer, f"masks[{index}].rects[{rect_index}]", errors)
            if parsed is not None:
                layer_rects.append(parsed)
        mask_rects[layer] = layer_rects

    repair_raw = raw.get("repair", {})
    if not isinstance(repair_raw, dict):
        errors.append(_issue("repair", "repair must be an object"))
        repair_raw = {}
    iterations = repair_raw.get("iterations_by_layer")
    if (
        not isinstance(iterations, list)
        or len(iterations) != spec.levels
        or not all(_is_int(v) and 1 <= v <= 128 for v in iterations)
    ):
        errors.append(_issue("repair.iterations_by_layer", "must contain one integer in [1,128] per layer"))
        iterations = [1] * spec.levels
    strengths_raw = repair_raw.get("guide_strength_by_layer")
    if (
        not isinstance(strengths_raw, list)
        or len(strengths_raw) != spec.levels
        or not all(_is_number(v) and 0.0 <= float(v) <= 1.0 for v in strengths_raw)
    ):
        errors.append(_issue("repair.guide_strength_by_layer", "must contain one number in [0,1] per layer"))
        strengths_raw = [0.0] * spec.levels
    guide_strength_q = [scalar_to_fixed(float(v), spec.fraction_bits) for v in strengths_raw]

    retention_raw = raw.get("retention", {})
    if not isinstance(retention_raw, dict):
        errors.append(_issue("retention", "retention must be an object"))
        retention_raw = {}
    mask_fill = retention_raw.get("mask_fill", "diffuse")
    style_inside = retention_raw.get("style_inside", "mapped")
    style_outside = retention_raw.get("style_outside", "mapped")
    if mask_fill not in ("diffuse", "preserve_source"):
        errors.append(_issue("retention.mask_fill", "must be diffuse or preserve_source"))
    if style_inside not in ("mapped", "repaired"):
        errors.append(_issue("retention.style_inside", "must be mapped or repaired"))
    if style_outside not in ("mapped", "repaired"):
        errors.append(_issue("retention.style_outside", "must be mapped or repaired"))

    style_raw = raw.get("style", {})
    if not isinstance(style_raw, dict):
        errors.append(_issue("style", "style must be an object"))
        style_raw = {}
    quant_levels = style_raw.get("quant_levels_by_layer")
    max_quant = 1 << spec.fraction_bits
    if (
        not isinstance(quant_levels, list)
        or len(quant_levels) != spec.levels
        or not all(_is_int(v) and 2 <= v <= 256 and v <= max_quant for v in quant_levels)
    ):
        errors.append(
            _issue(
                "style.quant_levels_by_layer",
                "must contain one integer in [2, min(256, 2^fraction_bits)] per layer",
            )
        )
        quant_levels = [2] * spec.levels
    weights_raw = style_raw.get("weights_by_layer")
    if not isinstance(weights_raw, list) or len(weights_raw) != spec.levels:
        errors.append(_issue("style.weights_by_layer", "must contain one RGB weight list per layer"))
        weights_raw = [[0.0, 0.0, 0.0] for _ in range(spec.levels)]
    style_weights_q: list[tuple[int, int, int]] = []
    for level, weights in enumerate(weights_raw):
        if (
            not isinstance(weights, list)
            or len(weights) != 3
            or not all(_is_number(value) and 0.0 <= float(value) <= 1.0 for value in weights)
        ):
            errors.append(_issue(f"style.weights_by_layer[{level}]", "must contain three numbers in [0,1]"))
            style_weights_q.append((0, 0, 0))
        else:
            style_weights_q.append(tuple(scalar_to_fixed(float(v), spec.fraction_bits) for v in weights))
    composition_raw = style_raw.get("composition_weights")
    if (
        not isinstance(composition_raw, list)
        or len(composition_raw) != spec.levels
        or not all(_is_number(v) and float(v) >= 0.0 for v in composition_raw)
        or sum(float(v) for v in composition_raw) <= 0.0
    ):
        errors.append(_issue("style.composition_weights", "must contain non-negative weights for every layer with positive sum"))
        composition_raw = [1.0] * spec.levels
    composition_weights_q = [int(round(float(v) * spec.scale)) for v in composition_raw]
    if sum(composition_weights_q) <= 0:
        errors.append(
            _issue(
                "style.composition_weights",
                "fixed-point composition weights sum to zero; increase a weight or fraction_bits",
            )
        )

    scheduler_raw = raw.get("scheduler", {})
    if not isinstance(scheduler_raw, dict):
        errors.append(_issue("scheduler", "scheduler must be an object"))
        scheduler_raw = {}
    threads = scheduler_raw.get("threads")
    if not _is_int(threads) or not 1 <= threads <= 64:
        errors.append(_issue("scheduler.threads", "threads must be an integer in [1,64]"))
        threads = 1
    budget = scheduler_raw.get("recompute_tile_budget")
    if not _is_int(budget) or budget <= 0:
        errors.append(_issue("scheduler.recompute_tile_budget", "must be a positive integer"))
        budget = 1

    snapshot_raw = raw.get("snapshot", {})
    if not isinstance(snapshot_raw, dict):
        errors.append(_issue("snapshot", "snapshot must be an object"))
        snapshot_raw = {}
    commit_every = snapshot_raw.get("commit_every_commands")
    if not _is_int(commit_every) or commit_every <= 0:
        errors.append(_issue("snapshot.commit_every_commands", "must be a positive integer"))
        commit_every = 1

    if errors:
        raise ConfigError(errors)

    mutable = EffectiveConfig(
        mask_rects=mask_rects,
        iterations_by_layer=[int(v) for v in iterations],
        guide_strength_q=guide_strength_q,
        style_weights_q=style_weights_q,
        quant_levels_by_layer=[int(v) for v in quant_levels],
        composition_weights_q=composition_weights_q,
        mask_fill=mask_fill,
        style_inside=style_inside,
        style_outside=style_outside,
    )
    source_fingerprint = array_fingerprint(source_rgb)
    style_fingerprint = array_fingerprint(style_rgb)
    base_hash = sha256_json(
        {
            "config": raw,
            "source_name": source_name,
            "style_name": style_name,
            "source_fingerprint": source_fingerprint,
            "style_fingerprint": style_fingerprint,
        }
    )
    return ConfigBundle(
        spec=spec,
        initial=mutable,
        image_dir=image_dir,
        source_name=str(source_name),
        style_name=str(style_name),
        source_path=source_path,
        style_path=style_path,
        source_rgb=source_rgb,
        style_rgb=style_rgb,
        source_fingerprint=source_fingerprint,
        style_fingerprint=style_fingerprint,
        base_config_hash=base_hash,
        threads=threads,
        recompute_tile_budget=budget,
        commit_every_commands=commit_every,
        raw_config=raw,
        image_records=image_records,
    )


def _parse_rect(
    rect: Any,
    spec: PyramidSpec,
    level: int,
    path: str,
    errors: list[dict[str, str]],
) -> Rect | None:
    if not isinstance(rect, list) or len(rect) != 4 or not all(_is_int(v) for v in rect):
        errors.append(_issue(path, "rectangle must be [x,y,w,h] with integer values"))
        return None
    x, y, w, h = (int(v) for v in rect)
    width, height = spec.sizes[level]
    if w <= 0 or h <= 0:
        errors.append(_issue(path, "rectangle width and height must be positive"))
        return None
    if x < 0 or y < 0 or x + w > width or y + h > height:
        errors.append(_issue(path, f"rectangle outside level {level} bounds {width}x{height}"))
        return None
    return (x, y, w, h)


def load_commands(path: str | Path) -> CommandStream:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise WorkbenchError("E_COMMAND_INVALID", f"cannot parse command JSON: {exc}") from exc
    if isinstance(raw, dict):
        commands = raw.get("commands")
    else:
        commands = raw
    if not isinstance(commands, list):
        raise WorkbenchError("E_COMMAND_INVALID", "command stream must be a list or an object with commands list")
    commands_hash = sha256_text(canonical_json(commands))
    return CommandStream(path=path, commands=commands, commands_hash=commands_hash)


def apply_command(
    mutable: EffectiveConfig,
    raw_command: Any,
    spec: PyramidSpec,
) -> tuple[dict[str, Any], Change]:
    if not isinstance(raw_command, dict):
        raise CommandError("command must be an object")
    command_type = raw_command.get("type")
    if not isinstance(command_type, str):
        raise CommandError("command.type must be a string")
    command_id = raw_command.get("id")
    if command_id is not None and not isinstance(command_id, str):
        raise CommandError("command.id must be a string when present")
    normalized: dict[str, Any] = {"type": command_type}
    if command_id is not None:
        normalized["id"] = command_id

    if command_type == "noop":
        return normalized, Change(changed=False, kind="noop")

    layer = raw_command.get("layer")
    if not _is_int(layer) or not 0 <= layer < spec.levels:
        raise CommandError("layer must be an integer in range", {"layer": layer, "levels": spec.levels})
    normalized["layer"] = int(layer)

    if command_type in ("set_mask_rect", "clear_mask"):
        if command_type == "set_mask_rect":
            rect = _command_rect(raw_command.get("rect"), spec, layer)
            candidate = [rect]
            normalized["rect"] = list(rect)
        else:
            candidate = []
        old_mask = mutable.mask_array(layer, spec)
        old_rects = mutable.mask_rects[layer]
        mutable.mask_rects[layer] = candidate
        new_mask = mutable.mask_array(layer, spec)
        changed = not bool((old_mask == new_mask).all())
        if not changed:
            mutable.mask_rects[layer] = old_rects
            affected_rects: tuple[Rect, ...] = ()
        else:
            affected_rects = tuple(dict.fromkeys(list(old_rects) + list(candidate)))
        return normalized, Change(
            changed=changed,
            kind="mask_region",
            layer=layer,
            rect=rect if command_type == "set_mask_rect" else None,
            rects=affected_rects,
        )

    if command_type == "set_iterations":
        value = raw_command.get("value")
        if not _is_int(value) or not 1 <= value <= 128:
            raise CommandError("iterations value must be an integer in [1,128]", {"value": value})
        changed = mutable.iterations_by_layer[layer] != int(value)
        mutable.iterations_by_layer[layer] = int(value)
        normalized["value"] = int(value)
        return normalized, Change(changed=changed, kind="layer_params", layer=layer)

    if command_type == "set_guide_strength":
        value = raw_command.get("value")
        if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            raise CommandError("guide strength must be a number in [0,1]", {"value": value})
        fixed = scalar_to_fixed(float(value), spec.fraction_bits)
        changed = mutable.guide_strength_q[layer] != fixed
        mutable.guide_strength_q[layer] = fixed
        normalized["value"] = float(value)
        normalized["value_q"] = fixed
        return normalized, Change(changed=changed, kind="layer_params", layer=layer)

    if command_type in ("set_style_weight", "set_style_weights"):
        weights = raw_command.get("weights")
        if (
            not isinstance(weights, list)
            or len(weights) != 3
            or not all(_is_number(value) and 0.0 <= float(value) <= 1.0 for value in weights)
        ):
            raise CommandError("style weights must be three numbers in [0,1]", {"weights": weights})
        fixed_tuple = tuple(scalar_to_fixed(float(value), spec.fraction_bits) for value in weights)
        changed = tuple(mutable.style_weights_q[layer]) != fixed_tuple
        mutable.style_weights_q[layer] = fixed_tuple
        normalized["weights"] = [float(value) for value in weights]
        normalized["weights_q"] = list(fixed_tuple)
        return normalized, Change(changed=changed, kind="layer_params", layer=layer)

    if command_type == "set_quant_levels":
        levels = raw_command.get("levels")
        max_quant = min(256, 1 << spec.fraction_bits)
        if not _is_int(levels) or not 2 <= levels <= max_quant:
            raise CommandError(
                f"quant levels must be an integer in [2,{max_quant}]",
                {"levels": levels},
            )
        changed = mutable.quant_levels_by_layer[layer] != int(levels)
        mutable.quant_levels_by_layer[layer] = int(levels)
        normalized["levels"] = int(levels)
        return normalized, Change(changed=changed, kind="layer_params", layer=layer)

    raise CommandError(f"unknown command type: {command_type}", {"type": command_type})


def _command_rect(rect: Any, spec: PyramidSpec, layer: int) -> Rect:
    if not isinstance(rect, list) or len(rect) != 4 or not all(_is_int(v) for v in rect):
        raise CommandError("rect must be [x,y,w,h] with integer values", {"rect": rect})
    x, y, w, h = (int(v) for v in rect)
    width, height = spec.sizes[layer]
    if w <= 0 or h <= 0:
        raise CommandError("rect width and height must be positive", {"rect": rect})
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise CommandError(
            f"rect outside layer {layer} bounds {width}x{height}",
            {"rect": rect, "layer_size": [width, height]},
        )
    return (x, y, w, h)
