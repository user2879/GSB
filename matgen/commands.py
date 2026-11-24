"""Validated interactive command application."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .errors import E_COMMAND_INVALID, E_MATERIAL_WEIGHTS, E_NOISE_PARAMS, E_QUANTIZATION, Issue, make_issue
from .fixed import CHANNELS
from .loader import ALLOWED_MIX_MODES, SUPPORTED_BIT_WIDTHS
from .models import Command, FixedPointConfig, GenerationConfig, Material, MaterialLayer, NoiseLayer


def _material_with_weight(
    material: Material,
    *,
    layer_index: int,
    channel: str,
    value: float,
) -> Material:
    found = False
    rows: list[MaterialLayer] = []
    for layer in material.layers:
        weights = dict(layer.weights)
        if layer.layer_index == layer_index:
            weights[channel] = value
            found = True
        rows.append(MaterialLayer(layer_index=layer.layer_index, weights=weights))
    if not found:
        raise KeyError(layer_index)
    return replace(material, layers=tuple(rows))


def apply_command(
    config: GenerationConfig,
    materials: tuple[Material, ...],
    command: Command,
) -> tuple[GenerationConfig, tuple[Material, ...], set[int], set[str], list[Issue]]:
    """Apply one command without mutating the input state."""

    issues: list[Issue] = []
    affected_layers: set[int] = set()
    affected_channels: set[str] = set()
    new_config = config
    new_materials = materials
    payload = command.payload
    op = command.op

    if op == "set_layer":
        layer_index = payload.get("layer")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or not 0 <= layer_index < config.noise_layer_count:
            issues.append(
                make_issue(E_COMMAND_INVALID, "layer must reference an existing noise layer", source="command", item=command.command_id)
            )
            return config, materials, set(), set(), issues
        frequency = payload.get("frequency")
        amplitude = payload.get("amplitude")
        if frequency is None and amplitude is None:
            issues.append(make_issue(E_COMMAND_INVALID, "set_layer requires frequency and/or amplitude", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        updates: dict[str, Any] = {}
        if frequency is not None:
            if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not math.isfinite(float(frequency)) or float(frequency) <= 0:
                issues.append(make_issue(E_NOISE_PARAMS, "frequency must be positive and finite", source="command", item=command.command_id))
            else:
                updates["frequency"] = float(frequency)
        if amplitude is not None:
            if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)) or not math.isfinite(float(amplitude)) or float(amplitude) <= 0:
                issues.append(make_issue(E_NOISE_PARAMS, "amplitude must be positive and finite", source="command", item=command.command_id))
            else:
                updates["amplitude"] = float(amplitude)
        if issues:
            return config, materials, set(), set(), issues
        layers = list(config.noise_layers)
        layers[layer_index] = replace(layers[layer_index], **updates)
        new_config = replace(config, noise_layers=tuple(layers))
        affected_layers.add(layer_index)
        return new_config, materials, affected_layers, affected_channels, issues

    if op == "set_weight":
        material_id = payload.get("material")
        layer_index = payload.get("layer")
        channel = payload.get("channel")
        value = payload.get("value")
        if not isinstance(material_id, str) or not any(row.material_id == material_id for row in materials):
            issues.append(make_issue(E_COMMAND_INVALID, "material does not exist", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        if channel not in CHANNELS:
            issues.append(make_issue(E_COMMAND_INVALID, "channel is invalid", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            issues.append(make_issue(E_COMMAND_INVALID, "layer must be an integer", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            issues.append(make_issue(E_MATERIAL_WEIGHTS, "weight must be finite and non-negative", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        updated_materials: list[Material] = []
        for material in materials:
            if material.material_id != material_id:
                updated_materials.append(material)
                continue
            try:
                updated = _material_with_weight(material, layer_index=layer_index, channel=channel, value=float(value))
            except KeyError:
                issues.append(
                    make_issue(E_COMMAND_INVALID, "material does not contain the referenced noise layer", source="command", item=command.command_id)
                )
                return config, materials, set(), set(), issues
            if not any(row.weights[channel] > 0 for row in updated.layers):
                issues.append(
                    make_issue(E_MATERIAL_WEIGHTS, f"material {material_id} would have no positive {channel} weight", source="command", item=command.command_id)
                )
                return config, materials, set(), set(), issues
            updated_materials.append(updated)
        new_materials = tuple(updated_materials)
        affected_channels.add(f"{material_id}:{channel}")
        return new_config, new_materials, affected_layers, affected_channels, issues

    if op == "set_mix_mode":
        value = payload.get("value")
        if value not in ALLOWED_MIX_MODES:
            issues.append(make_issue(E_COMMAND_INVALID, "mix mode is invalid", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        new_config = replace(config, material_mix_mode=str(value))
        for material in materials:
            for channel in CHANNELS:
                affected_channels.add(f"{material.material_id}:{channel}")
        return new_config, materials, affected_layers, affected_channels, issues

    if op == "set_quantization":
        level = payload.get("level", payload.get("value"))
        if isinstance(level, bool) or not isinstance(level, int) or level not in config.fixed_point.quantization_tiers:
            issues.append(make_issue(E_QUANTIZATION, "quantization level is invalid", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        fixed = replace(config.fixed_point, quantization_level=level)
        new_config = replace(config, fixed_point=fixed)
        affected_layers.update(range(config.noise_layer_count))
        for material in materials:
            for channel in CHANNELS:
                affected_channels.add(f"{material.material_id}:{channel}")
        return new_config, materials, affected_layers, affected_channels, issues

    if op == "set_bit_width":
        bit_width = payload.get("bit_width", payload.get("value"))
        if isinstance(bit_width, bool) or bit_width not in SUPPORTED_BIT_WIDTHS:
            issues.append(make_issue(E_QUANTIZATION, "bit width is invalid", source="command", item=command.command_id))
            return config, materials, set(), set(), issues
        if bit_width not in config.bit_width_ladder or not any(width > bit_width for width in config.bit_width_ladder):
            issues.append(
                make_issue(E_QUANTIZATION, "bit width must leave a higher precision trace rung", source="command", item=command.command_id)
            )
            return config, materials, set(), set(), issues
        delta = int(bit_width) - config.fixed_point.bit_width
        fraction_bits = min(
            int(bit_width) - 2,
            max(1, config.fixed_point.fraction_bits + delta),
        )
        fixed = replace(config.fixed_point, bit_width=int(bit_width), fraction_bits=fraction_bits)
        new_config = replace(config, fixed_point=fixed)
        affected_layers.update(range(config.noise_layer_count))
        for material in materials:
            for channel in CHANNELS:
                affected_channels.add(f"{material.material_id}:{channel}")
        return new_config, materials, affected_layers, affected_channels, issues

    issues.append(make_issue(E_COMMAND_INVALID, f"unsupported operation: {op}", source="command", item=command.command_id))
    return config, materials, set(), set(), issues
