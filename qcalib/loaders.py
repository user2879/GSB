"""Loading and schema validation for every local input artifact."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import yamlmini
from .errors import E_INPUT_SCHEMA, FailureList
from .models import (
    KNOWN_ROUNDING,
    KNOWN_SATURATION,
    KNOWN_SCHEME,
    LEVEL_RANGES,
    CalibrationSet,
    DistillConfig,
    BudgetConfig,
    QuantConfig,
    WeightLayer,
)

WEIGHT_FIELDS = {"layer", "shape", "weights", "content_version", "group_hint"}
CAL_FIELDS = {"inputs", "references", "sample_ids", "tier_sizes"}
KNOWN_COMMANDS = {
    "set_receipt_dir",
    "set_layer",
    "replace_weights",
    "set_distill",
    "set_tier",
    "set_default",
    "set_budget",
    "set_rounding_saturation",
    "noop",
}
ALLOWED_LAYER_KEYS = {
    "bits", "level", "group_size", "q_lo", "q_hi",
    "frac_bits", "scheme", "rounding", "saturation",
}


def load_weight_package(path: str | Path, failures: FailureList) -> list[WeightLayer]:
    path = Path(path)
    layers: list[WeightLayer] = []
    seen: set[str] = set()
    last_version: int | None = None
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        target = f"{path.name}:L{line_no}"
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "malformed json line", str(exc))
            continue
        if not isinstance(record, dict):
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "record must be an object")
            continue
        unknown = set(record) - WEIGHT_FIELDS
        if unknown:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "unknown field", ",".join(sorted(unknown)))
            continue
        missing = WEIGHT_FIELDS - set(record)
        if missing:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "missing field", ",".join(sorted(missing)))
            continue
        name = record["layer"]
        if not isinstance(name, str) or not name:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "layer name must be a non-empty string")
            continue
        if name in seen:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "duplicate layer name", name)
            continue
        shape = record["shape"]
        if (not isinstance(shape, list) or not shape
                or not all(isinstance(dim, int) and dim > 0 for dim in shape)):
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "shape dims must be positive integers", repr(shape))
            continue
        weights = record["weights"]
        flat_weights = np.asarray(weights, dtype=np.float64)
        if flat_weights.size == 0 or flat_weights.ndim != 1:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "weights must be a non-empty flat array")
            continue
        if tuple(int(dim) for dim in shape) == ():
            pass
        if int(np.prod(shape)) != flat_weights.size:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "shape does not match weight element count",
                         f"shape={shape} size={flat_weights.size}")
            continue
        if not np.all(np.isfinite(flat_weights)):
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "all weight elements must be finite", name)
            continue
        version = record["content_version"]
        if not isinstance(version, int) or version < 0:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "content_version must be a non-negative integer")
            continue
        if last_version is not None and version < last_version:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "content_version must be monotonically non-decreasing",
                         f"{version} < {last_version}")
            continue
        group_hint = record["group_hint"]
        if not isinstance(group_hint, int) or group_hint <= 0:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "group_hint must be a positive integer")
            continue
        if flat_weights.size % group_hint != 0:
            failures.add(E_INPUT_SCHEMA, "weight_package", target,
                         "group_hint must divide layer element count",
                         f"size={flat_weights.size} group={group_hint}")
            continue
        seen.add(name)
        last_version = version
        layers.append(WeightLayer(
            name=name,
            shape=tuple(int(dim) for dim in shape),
            weights=flat_weights.reshape(tuple(int(dim) for dim in shape)),
            content_version=version,
            group_hint=group_hint,
        ))
    return layers


def load_calibration(path: str | Path, layers: list[WeightLayer],
                     failures: FailureList) -> CalibrationSet:
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration file is not valid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError("calibration root must be an object")
    unknown = set(record) - CAL_FIELDS
    if unknown:
        failures.add(E_INPUT_SCHEMA, "calibration", path.name,
                     "unknown field", ",".join(sorted(unknown)))
    missing = CAL_FIELDS - set(record)
    if missing:
        raise ValueError(f"calibration missing fields: {sorted(missing)}")
    inputs = np.asarray(record["inputs"], dtype=np.float64)
    references = np.asarray(record["references"], dtype=np.float64)
    sample_ids = tuple(str(sid) for sid in record["sample_ids"])
    tier_sizes = {str(k): int(v) for k, v in record["tier_sizes"].items()}
    if inputs.ndim != 2 or references.ndim != 2:
        raise ValueError("calibration inputs/references must be 2-D arrays")
    if inputs.shape[0] != references.shape[0] or inputs.shape[0] != len(sample_ids):
        raise ValueError("calibration sample counts must align")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("calibration sample_ids must be unique")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(references)):
        raise ValueError("calibration values must all be finite")
    if not tier_sizes or any(v <= 0 for v in tier_sizes.values()):
        raise ValueError("tier_sizes must contain positive values")
    dim = inputs.shape[1]
    for layer in layers:
        if layer.input_dim != dim:
            failures.add(E_INPUT_SCHEMA, "calibration", layer.name,
                         "calibration dim does not match layer input dim",
                         f"cal={dim} layer={layer.input_dim}")
    return CalibrationSet(inputs=inputs, references=references,
                          sample_ids=sample_ids, tier_sizes=tier_sizes)


def load_config(path: str | Path) -> QuantConfig:
    path = Path(path)
    raw = yamlmini.loads(path.read_text(encoding="utf-8"))
    try:
        defaults = dict(raw["defaults"])
        layer_overrides = dict(raw.get("layers", {}))
        distill_raw = dict(raw["distill"])
        budget_raw = dict(raw["budgets"])
        workers = int(raw["workers"])
        tiers = {str(k): int(v) for k, v in dict(raw["tiers"]).items()}
        default_tier = str(raw["default_tier"])
        layer_tiers = {str(k): str(v) for k, v in dict(raw.get("layer_tiers", {})).items()}
        receipts_dir = str(raw["receipts_dir"])
    except KeyError as exc:
        raise ValueError(f"config missing section: {exc}") from exc

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if default_tier not in tiers:
        raise ValueError("default_tier must be declared in tiers")
    for layer_name, tier in layer_tiers.items():
        if tier not in tiers:
            raise ValueError(f"unknown tier {tier!r} for layer {layer_name}")

    _validate_quant_material(defaults, "defaults")
    for layer_name, override in layer_overrides.items():
        unknown = set(override) - ALLOWED_LAYER_KEYS
        if unknown:
            raise ValueError(f"layer {layer_name} has unknown keys: {sorted(unknown)}")
        merged = dict(defaults)
        merged.update(override)
        _validate_quant_material(merged, f"layer:{layer_name}")

    distill = DistillConfig(
        temperature=float(distill_raw["temperature"]),
        steps=int(distill_raw["steps"]),
        soft_weight=float(distill_raw["soft_weight"]),
        learning_rate=float(distill_raw.get("learning_rate", 0.1)),
        tolerance=float(distill_raw["tolerance"]),
    )
    if distill.temperature <= 0 or distill.steps < 0 or distill.learning_rate <= 0:
        raise ValueError("invalid distillation parameters")
    if not 0.0 <= distill.soft_weight <= 1.0 or distill.tolerance < 0:
        raise ValueError("invalid distillation soft_weight/tolerance")
    budgets = BudgetConfig(
        global_cal_points=int(budget_raw["global_cal_points"]),
        batch_quant_points=int(budget_raw["batch_quant_points"]),
        per_layer_quotas={str(k): int(v)
                          for k, v in dict(budget_raw.get("per_layer", {})).items()},
    )
    if budgets.global_cal_points <= 0 or budgets.batch_quant_points <= 0:
        raise ValueError("budgets must be positive")
    quotas = {name: quota for name, quota in budgets.per_layer_quotas.items()
              if name != "none"}
    if any(quota <= 0 for quota in quotas.values()):
        raise ValueError("per-layer quotas must be positive")
    object.__setattr__(budgets, "per_layer_quotas", quotas)
    return QuantConfig(
        defaults=defaults,
        layers=layer_overrides,
        distill=distill,
        budgets=budgets,
        workers=workers,
        tiers=tiers,
        default_tier=default_tier,
        layer_tiers=layer_tiers,
        receipts_dir=receipts_dir,
    )


def _validate_quant_material(material: dict[str, Any], where: str) -> None:
    required = ALLOWED_LAYER_KEYS
    missing = required - set(material)
    if missing:
        raise ValueError(f"{where} missing quant keys: {sorted(missing)}")
    bits = int(material["bits"])
    level = str(material["level"])
    if level not in LEVEL_RANGES:
        raise ValueError(f"{where} unknown level {level}")
    lo, hi = LEVEL_RANGES[level]
    if not (lo <= 0 <= hi):
        raise ValueError(f"{where} level range must contain zero")
    if bits not in (4, 8, 16):
        raise ValueError(f"{where} bits must be one of 4/8/16")
    if int(material["group_size"]) <= 0:
        raise ValueError(f"{where} group_size must be positive")
    q_lo, q_hi = int(material["q_lo"]), int(material["q_hi"])
    if not (0 <= q_lo <= q_hi <= 1000):
        raise ValueError(f"{where} quantile permilles must satisfy 0<=lo<=hi<=1000")
    if int(material["frac_bits"]) < 0:
        raise ValueError(f"{where} frac_bits must be >= 0")
    if str(material["scheme"]) not in KNOWN_SCHEME:
        raise ValueError(f"{where} unknown scheme")
    if str(material["rounding"]) not in KNOWN_ROUNDING:
        raise ValueError(f"{where} unknown rounding mode")
    if str(material["saturation"]) not in KNOWN_SATURATION:
        raise ValueError(f"{where} unknown saturation policy")


def load_command_stream(path: str | Path,
                        failures: FailureList | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"command stream not valid JSON: {exc}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("commands"), list):
        raise ValueError("command stream must be an object with a commands list")
    commands: list[dict[str, Any]] = []
    for index, command in enumerate(record["commands"]):
        target = f"{path.name}:cmd{index}"
        if not isinstance(command, dict) or "op" not in command:
            if failures is not None:
                failures.add(E_INPUT_SCHEMA, "command_stream", target,
                             "command must be an object containing op")
            continue
        op = command["op"]
        if op not in KNOWN_COMMANDS:
            if failures is not None:
                failures.add(E_INPUT_SCHEMA, "command_stream", target,
                             "unknown command op", str(op))
            continue
        commands.append(dict(command))
    return commands
