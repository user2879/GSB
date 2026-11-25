"""Deterministic command-stream application over config and local samples."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .errors import E_INPUT_SCHEMA, FailureList
from .hashing import fingerprint
from .models import QuantConfig, WeightLayer


def apply_commands(config: QuantConfig, layers: list[WeightLayer],
                   commands: list[dict[str, Any]],
                   failures: FailureList) -> tuple[QuantConfig, list[WeightLayer]]:
    config = replace(
        config,
        defaults=copy.deepcopy(config.defaults),
        layers=copy.deepcopy(config.layers),
        layer_tiers=dict(config.layer_tiers),
    )
    layers = list(layers)
    layer_index = {layer.name: index for index, layer in enumerate(layers)}

    for command in commands:
        op = command["op"]
        if op == "noop":
            continue
        if op == "set_receipt_dir":
            config = replace(config, receipts_dir=str(command["dir"]))
        elif op == "set_default":
            key = str(command["key"])
            if key not in config.defaults:
                failures.add(E_INPUT_SCHEMA, "command", op,
                             "unknown default key", key)
                continue
            config.defaults[key] = command["value"]
        elif op == "set_layer":
            layer_name = str(command["layer"])
            if layer_name not in layer_index:
                failures.add(E_INPUT_SCHEMA, "command", op,
                             "unknown layer", layer_name)
                continue
            overrides = dict(config.layers.get(layer_name, {}))
            for key, value in command.get("values", {}).items():
                overrides[key] = value
            config.layers[layer_name] = overrides
        elif op == "set_rounding_saturation":
            config.defaults["rounding"] = str(command["rounding"])
            config.defaults["saturation"] = str(command["saturation"])
        elif op == "set_distill":
            distill = replace(config.distill)
            for key in ("temperature", "steps", "soft_weight",
                        "learning_rate", "tolerance"):
                if key in command:
                    object.__setattr__(distill, key, command[key])
            config = replace(config, distill=distill)
        elif op == "set_tier":
            layer_name = str(command["layer"])
            if layer_name not in layer_index:
                failures.add(E_INPUT_SCHEMA, "command", op,
                             "unknown layer", layer_name)
                continue
            config.layer_tiers[layer_name] = str(command["tier"])
        elif op == "set_budget":
            budgets = replace(
                config.budgets,
                per_layer_quotas=dict(config.budgets.per_layer_quotas))
            if "global_cal_points" in command:
                object.__setattr__(budgets, "global_cal_points",
                                   int(command["global_cal_points"]))
            if "batch_quant_points" in command:
                object.__setattr__(budgets, "batch_quant_points",
                                   int(command["batch_quant_points"]))
            if "per_layer" in command:
                for name, quota in command["per_layer"].items():
                    budgets.per_layer_quotas[str(name)] = int(quota)
            config = replace(config, budgets=budgets)
        elif op == "replace_weights":
            layer_name = str(command["layer"])
            if layer_name not in layer_index:
                failures.add(E_INPUT_SCHEMA, "command", op,
                             "unknown layer", layer_name)
                continue
            source = Path(command["source"])
            record = json.loads(source.read_text(encoding="utf-8"))
            old = layers[layer_index[layer_name]]
            weights = np.asarray(record["weights"],
                                 dtype=np.float64).reshape(old.weights.shape)
            layers[layer_index[layer_name]] = WeightLayer(
                name=old.name,
                shape=old.shape,
                weights=weights,
                content_version=int(record["content_version"]),
                group_hint=old.group_hint,
            )
        else:
            failures.add(E_INPUT_SCHEMA, "command", op, "unhandled command")
    return config, layers


def effective_command_fingerprint(commands: list[dict[str, Any]]) -> str:
    return fingerprint(commands, prefix="cmd-")
