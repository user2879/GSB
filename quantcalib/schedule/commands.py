"""Command stream loading and deterministic application.

Commands are plain JSON objects applied in file order. Every mutation creates
a new global version. The runner reloads artifacts from disk each step so the
same final state can be reproduced either incrementally or by one full rebuild.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from quantcalib.data.weightpack import load_weight_pack, write_weight_pack
from quantcalib.errors import E_LAYER_UNKNOWN, QuantCalibError

VALID_COMMANDS = {
    "switch_receipt_version",
    "set_layer_bits",
    "set_layer_granularity",
    "set_rounding_mode",
    "set_saturation_policy",
    "replace_layer_weights",
    "set_distill_params",
    "set_global_budget",
}


@dataclass
class Command:
    seq: int
    name: str
    payload: dict[str, Any]


def load_command_stream(path: str | Path) -> list[Command]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    commands: list[Command] = []
    for seq, record in enumerate(raw):
        name = record["command"]
        if name not in VALID_COMMANDS:
            raise QuantCalibError("E_CONFIG_INVALID", f"unknown command {name!r}")
        commands.append(Command(seq=seq, name=name, payload=record.get("payload", {})))
    return commands


def write_command_stream(path: str | Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def apply_command(command: Command, workdir: Path, cfg) -> None:
    payload = command.payload
    if command.name == "switch_receipt_version":
        cfg_path = workdir / "config" / "quant.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        data["active_receipt_version"] = payload["version"]
        cfg.active_receipt_version = payload["version"]  # type: ignore[attr-defined]
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")

    elif command.name == "set_layer_bits":
        cfg.per_layer_bits[payload["layer"]] = int(payload["bits"])
    elif command.name == "set_layer_granularity":
        cfg.per_layer_group_granularity[payload["layer"]] = int(
            payload["granularity"])
    elif command.name == "set_rounding_mode":
        cfg.rounding_mode = payload["mode"]
    elif command.name == "set_saturation_policy":
        cfg.saturation_policy = payload["policy"]
    elif command.name == "set_distill_params":
        if "temperature" in payload:
            cfg.distill_temperature = float(payload["temperature"])
        if "steps" in payload:
            cfg.distill_steps = int(payload["steps"])
        if "soft_label_weight" in payload:
            cfg.soft_label_weight = float(payload["soft_label_weight"])
    elif command.name == "set_global_budget":
        cfg.global_point_budget = int(payload["budget"])

    elif command.name == "replace_layer_weights":
        weights_path = workdir / "weights" / "weights.jsonl"
        layers, issues = load_weight_pack(weights_path)
        names = {layer.name: layer for layer in layers}
        if payload["layer"] not in names:
            raise QuantCalibError(
                E_LAYER_UNKNOWN, "cannot replace unknown layer", payload["layer"])
        target = names[payload["layer"]]
        rng = np.random.default_rng(int(payload["seed"]))
        new_weights = rng.normal(
            loc=float(payload.get("loc", 0.0)),
            scale=float(payload.get("scale", 1.0)),
            size=target.weights.shape,
        )
        replacement = type(target)(
            name=target.name,
            shape=target.shape,
            weights=new_weights.astype(np.float64),
            content_version=target.content_version + int(
                payload.get("version_step", 1)),
            group_granularity=target.group_granularity,
        )
        new_layers = [replacement if layer.name == replacement.name else layer
                      for layer in layers]
        write_weight_pack(weights_path, new_layers)
