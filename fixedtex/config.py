"""Loading and tolerant validation for render config, material and commands."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .errors import BizError, issue
from .fixed import (
    ALLOWED_FRAC_BITS,
    ALLOWED_QUANT_STEPS,
    BLEND_MODES,
    CHANNEL_ORDER,
    EQUAL_WEIGHT_RULES,
    TIE_RULES,
    is_int,
    is_positive_finite,
    round_ratio_scalar,
)
from .io_utils import read_json
from .models import (
    BudgetConfig,
    ChannelSpec,
    FixedConfig,
    LayerSpec,
    LayerWeight,
    MaterialConfig,
    RenderConfig,
)

WEIGHT_SCALE = 1000


def _read_json_file(path: str | Path, kind: str) -> tuple[Any | None, bytes | None, list[dict[str, Any]]]:
    target = Path(path)
    try:
        payload = target.read_bytes()
    except OSError as exc:
        return None, None, [issue("E_IO", f"cannot read {kind}: {exc}", path=str(target), stage="load")]
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, payload, [issue("E_JSON", f"invalid {kind} JSON: {exc}", path=str(target), stage="load")]
    return value, payload, []


def _number_to_milli(value: Any, code: str, path: str, stage: str, issues: list[dict[str, Any]]) -> int:
    if not is_positive_finite(value):
        issues.append(issue(code, "value must be a positive finite number", path=path, stage=stage))
        return 1000
    milli = int(round(float(value) * 1000.0))
    if milli <= 0:
        issues.append(issue(code, "value underflows fixed parameter precision", path=path, stage=stage))
        return 1
    return milli


def _weight_to_q(
    value: Any,
    path: str,
    stage: str,
    issues: list[dict[str, Any]],
    *,
    code: str = "E_WEIGHT",
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(issue(code, "weight must be a finite non-negative number", path=path, stage=stage))
        return 0
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        issues.append(issue(code, "weight must be a finite non-negative number", path=path, stage=stage))
        return 0
    scaled = number * WEIGHT_SCALE
    weight_q, _, _ = round_ratio_scalar(
        int(round(scaled * 1_000_000)), 1_000_000, "ties_to_even"
    )
    if number > 0.0 and weight_q == 0:
        issues.append(
            issue(
                code,
                "positive weight underflows weight precision; clamped to one unit",
                path=path,
                stage=stage,
            )
        )
        weight_q = 1
    return weight_q


def compile_weight_q(value: Any, *, path: str = "command.weight", stage: str = "command", code: str = "E_WEIGHT") -> tuple[int, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    weight_q = _weight_to_q(value, path, stage, issues, code=code)
    return weight_q, issues


def load_render_config(path: str | Path) -> tuple[RenderConfig | None, list[dict[str, Any]]]:
    source = Path(path).resolve()
    raw, payload, issues = _read_json_file(source, "render config")
    if raw is None:
        return None, issues
    assert payload is not None
    if not isinstance(raw, dict):
        return None, [issue("E_CONFIG", "render config root must be an object", path=str(source), stage="config")]

    name = raw.get("name") if isinstance(raw.get("name"), str) else source.stem
    texture_raw = raw.get("texture", raw.get("texture_path"))
    if not isinstance(texture_raw, str) or not texture_raw:
        issues.append(issue("E_CONFIG", "texture path must be a non-empty string", path="texture", stage="config"))
        texture_path = source.parent / "missing-texture.png"
    else:
        texture_candidate = Path(texture_raw)
        texture_path = texture_candidate if texture_candidate.is_absolute() else source.parent / texture_candidate
    texture_path = texture_path.resolve()

    tiers_raw = raw.get("resolution_tiers")
    tiers: list[int] = []
    if not isinstance(tiers_raw, list):
        issues.append(issue("E_RES_TIER", "resolution_tiers must be a list", path="resolution_tiers", stage="config"))
    else:
        previous = 0
        for index, value in enumerate(tiers_raw):
            if not is_int(value) or value <= 0:
                issues.append(issue("E_RES_TIER", "resolution tier must be a positive integer", path=f"resolution_tiers[{index}]", stage="config"))
                continue
            if value <= previous:
                issues.append(issue("E_RES_TIER", "resolution tiers must be strictly increasing", path=f"resolution_tiers[{index}]", stage="config"))
                continue
            tiers.append(int(value))
            previous = int(value)
    if not tiers:
        issues.append(issue("E_RES_TIER", "no valid resolution tier; defaulting to 64", path="resolution_tiers", stage="config"))
        tiers = [64]

    resolution_value = raw.get("resolution", tiers[0])
    if not is_int(resolution_value):
        issues.append(issue("E_RESOLUTION", "resolution must be an integer tier", path="resolution", stage="config"))
        resolution = tiers[0]
    else:
        resolution = int(resolution_value)
        if resolution not in tiers:
            issues.append(issue("E_RESOLUTION", "resolution must be one of resolution_tiers", path="resolution", stage="config"))
            resolution = tiers[0]

    seed_value = raw.get("seed", 20261126)
    if not is_int(seed_value) or seed_value < 0:
        issues.append(issue("E_CONFIG", "seed must be a non-negative integer", path="seed", stage="config"))
        seed = 20261126
    else:
        seed = int(seed_value)
    algorithm = raw.get("algorithm", "value2d")
    if algorithm != "value2d":
        issues.append(issue("E_CONFIG", "only value2d noise algorithm is supported", path="algorithm", stage="config"))
        algorithm = "value2d"

    noise = raw.get("noise", {})
    if noise is None:
        noise = {}
    if not isinstance(noise, dict):
        issues.append(issue("E_CONFIG", "noise must be an object", path="noise", stage="config"))
        noise = {}
    layers_raw = noise.get("layers", raw.get("layers", []))
    layers: list[LayerSpec] = []
    if not isinstance(layers_raw, list):
        issues.append(issue("E_CONFIG", "noise.layers must be a list", path="noise.layers", stage="config"))
        layers_raw = []
    for index, item in enumerate(layers_raw):
        if not isinstance(item, dict):
            issues.append(issue("E_CONFIG", "noise layer must be an object", path=f"noise.layers[{index}]", stage="config", layer=index))
            continue
        frequency = _number_to_milli(
            item.get("frequency"),
            "E_FREQ",
            f"noise.layers[{index}].frequency",
            "config",
            issues,
        )
        amplitude = _number_to_milli(
            item.get("amplitude"),
            "E_AMPLITUDE",
            f"noise.layers[{index}].amplitude",
            "config",
            issues,
        )
        layer_seed = item.get("seed", seed + index * 1000003)
        if not is_int(layer_seed) or layer_seed < 0:
            issues.append(issue("E_CONFIG", "layer seed must be a non-negative integer", path=f"noise.layers[{index}].seed", stage="config", layer=index))
            layer_seed = seed + index * 1000003
        valid = not any(
            error.get("layer") == index and error.get("path", "").startswith(f"noise.layers[{index}]")
            for error in issues
        )
        layers.append(
            LayerSpec(
                index=index,
                frequency_milli=frequency,
                amplitude_milli=amplitude,
                seed=int(layer_seed),
                valid=valid,
            )
        )
    declared_count = noise.get("layer_count", raw.get("layer_count"))
    if declared_count is not None and (not is_int(declared_count) or declared_count != len(layers)):
        issues.append(issue("E_LAYER_COUNT", "declared layer_count disagrees with layers list", path="noise.layer_count", stage="config"))

    fixed_raw = raw.get("fixed", raw.get("fixed_point", {}))
    if not isinstance(fixed_raw, dict):
        issues.append(issue("E_CONFIG", "fixed must be an object", path="fixed", stage="config"))
        fixed_raw = {}
    fraction_bits = fixed_raw.get("fraction_bits", raw.get("fraction_bits", 12))
    if not is_int(fraction_bits) or int(fraction_bits) not in ALLOWED_FRAC_BITS:
        issues.append(issue("E_FRAC_BITS", f"fraction_bits must be one of {list(ALLOWED_FRAC_BITS)}", path="fixed.fraction_bits", stage="config"))
        fraction_bits = 12
    quant_step = fixed_raw.get("quant_step", raw.get("quant_step", 4))
    if not is_int(quant_step) or int(quant_step) not in ALLOWED_QUANT_STEPS:
        issues.append(issue("E_QUANT_STEP", f"quant_step must be one of {list(ALLOWED_QUANT_STEPS)}", path="fixed.quant_step", stage="config"))
        quant_step = 4
    tie_rule = fixed_raw.get("tie_rule", "ties_to_even")
    if tie_rule not in TIE_RULES:
        issues.append(issue("E_TIE_RULE", f"tie_rule must be one of {list(TIE_RULES)}", path="fixed.tie_rule", stage="config"))
        tie_rule = "ties_to_even"
    guard_bits = fixed_raw.get("guard_bits", 2)
    if not is_int(guard_bits) or not 0 <= int(guard_bits) <= 8:
        issues.append(issue("E_CONFIG", "guard_bits must be an integer in [0, 8]", path="fixed.guard_bits", stage="config"))
        guard_bits = 2

    budgets_raw = raw.get("budgets", {})
    if not isinstance(budgets_raw, dict):
        issues.append(issue("E_CONFIG", "budgets must be an object", path="budgets", stage="config"))
        budgets_raw = {}
    layer_points = budgets_raw.get("layer_points", raw.get("layer_point_budget", 65536))
    if not is_int(layer_points) or int(layer_points) <= 0:
        issues.append(issue("E_LAYER_BUDGET", "layer point budget must be a positive integer", path="budgets.layer_points", stage="config"))
        layer_points = 65536
    batch_points = budgets_raw.get("batch_points", raw.get("batch_point_budget", 262144))
    if not is_int(batch_points) or int(batch_points) <= 0:
        issues.append(issue("E_BATCH_BUDGET", "batch point budget must be a positive integer", path="budgets.batch_points", stage="config"))
        batch_points = 262144

    blend_mode = raw.get("blend_mode", "weighted")
    if blend_mode not in BLEND_MODES:
        issues.append(issue("E_BLEND_MODE", f"blend_mode must be one of {list(BLEND_MODES)}", path="blend_mode", stage="config"))
        blend_mode = "weighted"
    preview_budget_ms = raw.get("preview_budget_ms", 250)
    if not is_int(preview_budget_ms) or int(preview_budget_ms) < 0:
        issues.append(issue("E_PREVIEW_BUDGET", "preview_budget_ms must be a non-negative integer", path="preview_budget_ms", stage="config"))
        preview_budget_ms = 250
    thread_pool_size = raw.get("thread_pool_size", raw.get("thread_pool_concurrency", 2))
    if not is_int(thread_pool_size) or not 1 <= int(thread_pool_size) <= 64:
        issues.append(issue("E_THREAD_POOL", "thread_pool_size must be an integer in [1, 64]", path="thread_pool_size", stage="config"))
        thread_pool_size = 2
    commit_interval = raw.get("commit_interval_commands", raw.get("commit_interval", 1))
    if not is_int(commit_interval) or int(commit_interval) < 1:
        issues.append(issue("E_COMMIT_INTERVAL", "commit_interval_commands must be a positive integer", path="commit_interval_commands", stage="config"))
        commit_interval = 1
    tolerance_steps = raw.get("error_tolerance_quant_steps", 1)
    if not is_int(tolerance_steps) or int(tolerance_steps) < 0:
        issues.append(issue("E_TOLERANCE", "error_tolerance_quant_steps must be a non-negative integer", path="error_tolerance_quant_steps", stage="config"))
        tolerance_steps = 1

    config = RenderConfig(
        name=name,
        texture_path=str(texture_path),
        resolution_tiers=tuple(tiers),
        resolution=resolution,
        seed=seed,
        algorithm=algorithm,
        layers=tuple(layers),
        fixed=FixedConfig(
            fraction_bits=int(fraction_bits),
            quant_step=int(quant_step),
            tie_rule=tie_rule,
            guard_bits=int(guard_bits),
        ),
        budgets=BudgetConfig(layer_points=int(layer_points), batch_points=int(batch_points)),
        blend_mode=blend_mode,
        preview_budget_ms=int(preview_budget_ms),
        thread_pool_size=int(thread_pool_size),
        commit_interval_commands=int(commit_interval),
        error_tolerance_quant_steps=int(tolerance_steps),
        source_path=str(source),
        source_hash=hashlib.sha256(payload).hexdigest(),
        raw=raw,
    )
    return config, issues


def _default_channel(name: str) -> ChannelSpec:
    return ChannelSpec(name=name, base_weight_q=WEIGHT_SCALE, layer_weights=(), valid=False)


def load_material(path: str | Path, layer_count: int) -> tuple[MaterialConfig | None, list[dict[str, Any]]]:
    source = Path(path).resolve()
    raw, payload, issues = _read_json_file(source, "material")
    if raw is None:
        return None, issues
    assert payload is not None
    if not isinstance(raw, dict):
        return None, [issue("E_MATERIAL", "material root must be an object", path=str(source), stage="material")]

    name = raw.get("name") if isinstance(raw.get("name"), str) else source.stem
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        issues.append(issue("E_MATERIAL", "material metadata must be an object", path="metadata", stage="material"))
        metadata = {}
    tie_rule = metadata.get("tie_rule", raw.get("tie_rule", "ties_to_even"))
    if tie_rule not in TIE_RULES:
        issues.append(issue("E_TIE_RULE", f"tie_rule must be one of {list(TIE_RULES)}", path="metadata.tie_rule", stage="material"))
        tie_rule = "ties_to_even"
    equal_weight_rule = metadata.get("equal_weight_rule", raw.get("equal_weight_rule", "lower_layer_index"))
    if equal_weight_rule not in EQUAL_WEIGHT_RULES:
        issues.append(issue("E_WEIGHT", f"equal_weight_rule must be one of {list(EQUAL_WEIGHT_RULES)}", path="metadata.equal_weight_rule", stage="material"))
        equal_weight_rule = "lower_layer_index"

    channels_raw = raw.get("channels", {})
    if not isinstance(channels_raw, dict):
        issues.append(issue("E_MATERIAL", "channels must be an object", path="channels", stage="material"))
        channels_raw = {}
    channels: dict[str, ChannelSpec] = {}
    for channel_name in CHANNEL_ORDER:
        channel_raw = channels_raw.get(channel_name)
        if not isinstance(channel_raw, dict):
            issues.append(issue("E_MATERIAL_CHANNEL", f"missing channel {channel_name}", path=f"channels.{channel_name}", stage="material", channel=channel_name))
            channels[channel_name] = _default_channel(channel_name)
            continue
        base_weight_q = _weight_to_q(
            channel_raw.get("base_weight", 1.0),
            f"channels.{channel_name}.base_weight",
            "material",
            issues,
        )
        layers_raw = channel_raw.get("layers", [])
        if not isinstance(layers_raw, list):
            issues.append(issue("E_WEIGHT", "channel layers must be a list", path=f"channels.{channel_name}.layers", stage="material", channel=channel_name))
            layers_raw = []
        weights: list[LayerWeight] = []
        seen: set[int] = set()
        for ref_index, item in enumerate(layers_raw):
            if not isinstance(item, dict):
                issues.append(issue("E_WEIGHT", "layer weight must be an object", path=f"channels.{channel_name}.layers[{ref_index}]", stage="material", channel=channel_name))
                continue
            layer_index = item.get("index", item.get("layer"))
            if not is_int(layer_index) or not 0 <= int(layer_index) < layer_count:
                issues.append(issue("E_WEIGHT", "layer index is outside configured layer range", path=f"channels.{channel_name}.layers[{ref_index}].index", stage="material", channel=channel_name, layer=layer_index if is_int(layer_index) else None))
                continue
            layer_index = int(layer_index)
            if layer_index in seen:
                issues.append(issue("E_WEIGHT", "duplicate layer reference", path=f"channels.{channel_name}.layers[{ref_index}].index", stage="material", channel=channel_name, layer=layer_index))
                continue
            raw_weight = item.get("weight", 0.0)
            weight_q = _weight_to_q(
                raw_weight,
                f"channels.{channel_name}.layers[{ref_index}].weight",
                "material",
                issues,
            )
            seen.add(layer_index)
            weights.append(LayerWeight(index=layer_index, weight_q=weight_q, raw_weight=float(raw_weight) if isinstance(raw_weight, (int, float)) and not isinstance(raw_weight, bool) else 0.0))
        weights.sort(key=lambda item: item.index)
        if base_weight_q + sum(item.weight_q for item in weights) <= 0:
            issues.append(issue("E_WEIGHT", "channel has zero total weight; base weight set to one unit", path=f"channels.{channel_name}", stage="material", channel=channel_name))
            base_weight_q = 1
        channels[channel_name] = ChannelSpec(
            name=channel_name,
            base_weight_q=base_weight_q,
            layer_weights=tuple(weights),
            valid=True,
        )
    for extra_name in sorted(set(channels_raw) - set(CHANNEL_ORDER)):
        issues.append(issue("E_MATERIAL_CHANNEL", f"unknown channel ignored: {extra_name}", path=f"channels.{extra_name}", stage="material", channel=extra_name))

    material = MaterialConfig(
        name=name,
        channels=channels,
        tie_rule=tie_rule,
        equal_weight_rule=equal_weight_rule,
        source_path=str(source),
        source_hash=hashlib.sha256(payload).hexdigest(),
        raw=raw,
    )
    return material, issues


def load_commands(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [issue("E_IO", f"cannot read command stream: {exc}", path=str(source), stage="command")]
    commands: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            command = json.loads(stripped)
        except json.JSONDecodeError as exc:
            issues.append(issue("E_COMMAND", f"invalid command JSON: {exc}", path=f"{source}:{line_number}", stage="command"))
            continue
        if not isinstance(command, dict) or not isinstance(command.get("command"), str):
            issues.append(issue("E_COMMAND", "command must be an object with a string command field", path=f"{source}:{line_number}", stage="command"))
            continue
        commands.append(command)
    return commands, issues


def load_injections(path: str | Path | None, layer_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return [], []
    source = Path(path)
    try:
        raw = read_json(source)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [issue("E_INJECTION", f"cannot load injections: {exc}", path=str(source), stage="injection")]
    items = raw.get("injections", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [], [issue("E_INJECTION", "injections must be a list", path=str(source), stage="injection")]
    allowed_stages = {
        "noise_synthesis",
        "layer_quantize",
        "material_mix",
        "channel_compose",
    }
    injections: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(issue("E_INJECTION", "injection must be an object", path=f"injections[{index}]", stage="injection"))
            continue
        stage = item.get("stage")
        if stage not in allowed_stages:
            issues.append(issue("E_INJECTION", f"stage must be one of {sorted(allowed_stages)}", path=f"injections[{index}].stage", stage="injection"))
            continue
        layer = item.get("layer")
        if layer is not None and (not is_int(layer) or not 0 <= int(layer) < layer_count):
            issues.append(issue("E_INJECTION", "injection layer is outside configured range", path=f"injections[{index}].layer", stage="injection"))
            continue
        channel = item.get("channel")
        if channel is not None and channel not in CHANNEL_ORDER:
            issues.append(issue("E_INJECTION", "injection channel is invalid", path=f"injections[{index}].channel", stage="injection"))
            continue
        delay_ms = item.get("delay_ms", 0)
        if not is_int(delay_ms) or int(delay_ms) < 0:
            issues.append(issue("E_INJECTION", "delay_ms must be a non-negative integer", path=f"injections[{index}].delay_ms", stage="injection"))
            continue
        code = item.get("code", f"E_INJECTED_{stage.upper()}")
        if not isinstance(code, str) or not code.startswith("E_"):
            issues.append(issue("E_INJECTION", "injection code must start with E_", path=f"injections[{index}].code", stage="injection"))
            continue
        injections.append(
            {
                "layer": int(layer) if layer is not None else None,
                "stage": stage,
                "channel": channel,
                "delay_ms": int(delay_ms),
                "code": code,
                "message": str(item.get("message", f"injected failure at {stage}")),
            }
        )
    return injections, issues
