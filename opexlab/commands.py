"""Editing-command validation and state transitions."""

from __future__ import annotations

import copy
from typing import Any, Dict

from .config import LabConfig
from .errors import E_CONFIG, LabError
from .util import is_finite_number, is_positive_int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabError(E_CONFIG, message, stage="command")


def _normalise_parameter_override(config: LabConfig, command: Dict[str, Any]) -> Dict[str, Any]:
    layer = command.get("layer")
    _require(isinstance(layer, str) and layer, "replace_layer_params.layer is required")
    scale = command.get("scale")
    bias = command.get("bias", 0.0)
    if isinstance(scale, list):
        _require(all(is_finite_number(item) for item in scale), "scale list must be finite")
        scale_value: Any = [float(item) for item in scale]
    else:
        _require(is_finite_number(scale), "scale must be finite")
        scale_value = float(scale)
    if isinstance(bias, list):
        _require(all(is_finite_number(item) for item in bias), "bias list must be finite")
        bias_value: Any = [float(item) for item in bias]
    else:
        _require(is_finite_number(bias), "bias must be finite")
        bias_value = float(bias)
    scheme = command.get("scheme", config.v1_upgrade_scheme)
    _require(scheme in config.allowed_schemes, "scheme must be in allowed_norm_schemes")
    precision = command.get("precision", next(iter(config.quant_codebook)))
    _require(precision in config.quant_codebook, "precision must be in quant_codebook")
    return {
        "scale": scale_value,
        "bias": bias_value,
        "scheme": scheme,
        "precision": precision,
    }


def apply_command(
    config: LabConfig,
    state: Dict[str, Any],
    command: Dict[str, Any],
) -> Dict[str, Any]:
    _require(isinstance(command, dict), "command must be an object")
    command_type = command.get("type")
    _require(isinstance(command_type, str), "command.type is required")
    new_state = copy.deepcopy(state)
    if command_type == "set_exchange_version":
        value = command.get("value")
        _require(value in ("v1", "v2"), "exchange version must be v1 or v2")
        new_state["exchange_version"] = value
    elif command_type == "set_fractional_bits":
        value = command.get("value")
        _require(is_positive_int(value) and value <= 48, "fractional bits must be in 1..48")
        new_state["fractional_bits"] = value
    elif command_type == "set_quant_level":
        value = command.get("value")
        _require(value in config.allowed_quant_levels, "quant level is not allowed")
        new_state["quant_level"] = value
    elif command_type == "set_max_workers":
        value = command.get("value")
        _require(is_positive_int(value), "max workers must be positive")
        new_state["max_workers"] = value
    elif command_type == "set_batch_element_budget":
        value = command.get("value")
        _require(is_positive_int(value), "batch element budget must be positive")
        new_state["batch_element_budget"] = value
    elif command_type == "set_per_layer_element_quota":
        value = command.get("value")
        _require(is_positive_int(value), "per-layer element quota must be positive")
        new_state["per_layer_element_quota"] = value
    elif command_type == "replace_activation_row":
        layer = command.get("layer")
        row = command.get("row")
        values = command.get("values")
        _require(isinstance(layer, str) and layer, "replace_activation_row.layer is required")
        _require(isinstance(row, int) and not isinstance(row, bool) and row >= 0, "row must be non-negative")
        _require(isinstance(values, list) and values, "values must be a non-empty list")
        _require(all(is_finite_number(item) for item in values), "activation values must be finite")
        overrides = new_state.setdefault("activation_overrides", {})
        layer_map = overrides.setdefault(layer, {})
        layer_map[str(row)] = [float(item) for item in values]
    elif command_type == "replace_layer_params":
        layer = command.get("layer")
        _require(isinstance(layer, str) and layer, "replace_layer_params.layer is required")
        new_state.setdefault("parameter_overrides", {})[layer] = _normalise_parameter_override(
            config, command
        )
    elif command_type in ("run", "noop"):
        pass
    else:
        raise LabError(E_CONFIG, f"unknown command type: {command_type}", stage="command")
    new_state["revision"] = int(state.get("revision", 0)) + 1
    history = new_state.setdefault("history", [])
    history.append(copy.deepcopy(command))
    return new_state


def load_command_stream(path) -> list[Dict[str, Any]]:
    import json
    from pathlib import Path

    commands = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise LabError(E_CONFIG, f"command line {line_number} must be an object", stage="command")
            commands.append(value)
    return commands

