from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from typing import Any

from .errors import ConfigInvalid


STAGES = ("position", "rotation", "dot", "normalization")


def is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def key_lines(text: str, key: str) -> list[int]:
    pattern = re.compile(r'"' + re.escape(key) + r'"\s*:')
    return [index + 1 for index, line in enumerate(text.splitlines()) if pattern.search(line)]


def nth_key_line(text: str, key: str, occurrence: int) -> int | None:
    lines = key_lines(text, key)
    return lines[occurrence] if occurrence < len(lines) else (lines[0] if lines else None)


def _require_dict(value: Any, field: str, line: int | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigInvalid("must be a JSON object", line=line, field=field)
    return value


def _validate_scheme(raw: dict[str, Any], dimension: int, text: str | None = None, index: int = 0) -> dict[str, Any]:
    field_prefix = f"schemes[{index}]"
    line = nth_key_line(text, "name", index) if text else None
    if not isinstance(raw.get("name"), str) or not raw["name"]:
        raise ConfigInvalid("scheme name must be a non-empty string", line=line, field=f"{field_prefix}.name")

    name_line = nth_key_line(text, "name", index) if text else None
    for field_name in ("base_freq", "angle_scale"):
        value = raw.get(field_name)
        if not is_real_number(value) or float(value) <= 0.0:
            raise ConfigInvalid(
                f"{field_name} must be a finite positive number",
                line=name_line,
                field=f"{field_prefix}.{field_name}",
            )

    block_size = raw.get("block_size", 2)
    if not is_positive_int(block_size):
        raise ConfigInvalid("block_size must be a positive integer", line=name_line, field=f"{field_prefix}.block_size")
    if dimension % int(block_size) != 0:
        raise ConfigInvalid(
            f"dimension {dimension} must be divisible by block_size {block_size}",
            line=name_line,
            field=f"{field_prefix}.block_size",
        )

    correction_level = raw.get("correction_level", raw.get("freq_correction_level", 0))
    if not is_nonneg_int(correction_level) or int(correction_level) > 3:
        raise ConfigInvalid(
            "correction_level must be an integer in [0, 3]",
            line=name_line,
            field=f"{field_prefix}.correction_level",
        )

    return {
        "name": str(raw["name"]),
        "base_freq": float(raw["base_freq"]),
        "angle_scale": float(raw["angle_scale"]),
        "block_size": int(block_size),
        "correction_level": int(correction_level),
    }


def _parse_config_inner(obj: Any, source_text: str | None = None) -> dict[str, Any]:
    root = _require_dict(obj, "$", 1)

    dimension = root.get("dimension", 8)
    if not is_positive_int(dimension):
        raise ConfigInvalid("dimension must be a positive integer", field="dimension")
    dimension = int(dimension)
    if dimension % 2 != 0:
        raise ConfigInvalid("dimension must be even for adjacent rotation pairs", field="dimension")

    seed = root.get("seed", 393)
    if not is_nonneg_int(seed):
        raise ConfigInvalid("seed must be a non-negative integer", field="seed")

    train_length = root.get("train_length")
    if not is_positive_int(train_length):
        raise ConfigInvalid("train_length must be a positive integer", field="train_length")
    target_length = root.get("target_length")
    if not is_positive_int(target_length):
        raise ConfigInvalid("target_length must be a positive integer", field="target_length")
    if int(target_length) < int(train_length):
        raise ConfigInvalid(
            "target_length must be greater than or equal to train_length",
            field="target_length",
        )

    fp_raw = root.get("fixed_point", {})
    if fp_raw is None:
        fp_raw = {}
    fp = _require_dict(fp_raw, "fixed_point")
    frac_bits = fp.get("frac_bits", fp.get("fractional_bits", 20))
    if not is_positive_int(frac_bits) or not (4 <= int(frac_bits) <= 40):
        raise ConfigInvalid("frac_bits must be an integer in [4, 40]", field="fixed_point.frac_bits")
    frac_bits = int(frac_bits)

    level_shifts = fp.get("level_shifts", [0, 3, 6, 9])
    if not isinstance(level_shifts, list) or not level_shifts or not all(is_nonneg_int(x) for x in level_shifts):
        raise ConfigInvalid(
            "level_shifts must be a non-empty list of non-negative integers",
            field="fixed_point.level_shifts",
        )
    if any(int(x) >= frac_bits - 2 for x in level_shifts):
        raise ConfigInvalid(
            "every level_shift must leave at least two fractional bits",
            field="fixed_point.level_shifts",
        )
    quant_level = fp.get("quant_level", 0)
    if not is_nonneg_int(quant_level) or int(quant_level) >= len(level_shifts):
        raise ConfigInvalid(
            f"quant_level is out of range [0, {len(level_shifts) - 1}]",
            field="fixed_point.quant_level",
        )
    quant_spec = {
        "frac_bits": frac_bits,
        "quant_level": int(quant_level),
        "level_shifts": [int(x) for x in level_shifts],
        "effective_frac_bits": frac_bits - int(level_shifts[int(quant_level)]),
    }

    tol_raw = root.get("tolerances", {})
    if tol_raw is None:
        tol_raw = {}
    if not isinstance(tol_raw, dict):
        raise ConfigInvalid("tolerances must be an object", field="tolerances")
    default_tolerances = {
        "position": 2.0e-3,
        "rotation": 8.0e-3,
        "dot": 2.0e-2,
        "normalization": 5.0e-3,
    }
    tolerances = dict(default_tolerances)
    for stage in STAGES:
        if stage in tol_raw:
            value = tol_raw[stage]
            if not is_real_number(value) or float(value) <= 0.0:
                raise ConfigInvalid(f"tolerance.{stage} must be a positive number", field=f"tolerances.{stage}")
            tolerances[stage] = float(value)

    budget_raw = root.get("budget", {})
    if budget_raw is None:
        budget_raw = {}
    if not isinstance(budget_raw, dict):
        raise ConfigInvalid("budget must be an object", field="budget")
    batch_budget = budget_raw.get(
        "batch_point_budget",
        budget_raw.get("max_points_per_batch", root.get("batch_point_budget", 128)),
    )
    sample_quota = budget_raw.get(
        "sample_quota_points",
        budget_raw.get("per_sample_points", root.get("sample_quota_points", batch_budget)),
    )
    if not is_positive_int(batch_budget):
        raise ConfigInvalid("batch_point_budget must be a positive integer", field="budget.batch_point_budget")
    if not is_positive_int(sample_quota):
        raise ConfigInvalid("sample_quota_points must be a positive integer", field="budget.sample_quota_points")

    concurrency_raw = root.get("concurrency", {})
    if concurrency_raw is None:
        concurrency_raw = {}
    workers = concurrency_raw.get("workers", root.get("workers", os.cpu_count() or 1))
    if not is_positive_int(workers) or int(workers) > 32:
        raise ConfigInvalid("concurrency.workers must be an integer in [1, 32]", field="concurrency.workers")

    schemes_raw = root.get("schemes")
    if not isinstance(schemes_raw, list) or not schemes_raw:
        raise ConfigInvalid("schemes must be a non-empty array", field="schemes")
    schemes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_scheme in enumerate(schemes_raw):
        scheme_obj = _require_dict(raw_scheme, f"schemes[{index}]", nth_key_line(source_text or "", "name", index))
        scheme = _validate_scheme(scheme_obj, dimension, source_text, index)
        if scheme["name"] in seen:
            raise ConfigInvalid(
                f"duplicate scheme name {scheme['name']!r}",
                line=nth_key_line(source_text or "", "name", index),
                field=f"schemes[{index}].name",
            )
        seen.add(scheme["name"])
        schemes.append(scheme)

    active_scheme = root.get("active_scheme", schemes[0]["name"])
    if active_scheme not in seen:
        line = key_lines(source_text or "", "active_scheme")
        raise ConfigInvalid(
            f"unknown active scheme {active_scheme!r}",
            line=line[0] if line else None,
            field="active_scheme",
        )

    snapshot_dir = root.get("snapshot_dir", "artifacts/snapshots")
    if not isinstance(snapshot_dir, str) or not snapshot_dir:
        raise ConfigInvalid("snapshot_dir must be a non-empty path", field="snapshot_dir")

    return {
        "schema_version": int(root.get("schema_version", 1)),
        "dimension": dimension,
        "seed": int(seed),
        "train_length": int(train_length),
        "target_length": int(target_length),
        "fixed_point": quant_spec,
        "tolerances": tolerances,
        "budget": {
            "batch_point_budget": int(batch_budget),
            "sample_quota_points": int(sample_quota),
        },
        "concurrency": {"workers": int(workers)},
        "schemes": schemes,
        "active_scheme": str(active_scheme),
        "snapshot_dir": snapshot_dir,
    }


def load_config(path: str) -> tuple[dict[str, Any], str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise ConfigInvalid(f"cannot read config: {exc}", field=path) from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigInvalid(
            f"invalid JSON: {exc.msg}",
            line=exc.lineno,
            field="$",
        ) from exc
    return parse_config(obj, text), text


def scheme_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {scheme["name"]: scheme for scheme in config["schemes"]}


def get_scheme(config: dict[str, Any], name: str) -> dict[str, Any]:
    schemes = scheme_map(config)
    if name not in schemes:
        raise ConfigInvalid(f"unknown scheme {name!r}", field="name")
    return schemes[name]


def with_quant_level(config: dict[str, Any], frac_bits: int | None = None, quant_level: int | None = None) -> dict[str, Any]:
    candidate = deepcopy(config)
    fp = candidate["fixed_point"]
    raw_like = {
        "frac_bits": frac_bits if frac_bits is not None else fp["frac_bits"],
        "quant_level": quant_level if quant_level is not None else fp["quant_level"],
        "level_shifts": fp["level_shifts"],
    }
    rebuilt = parse_config(
        {
            "dimension": candidate["dimension"],
            "train_length": candidate["train_length"],
            "target_length": candidate["target_length"],
            "fixed_point": raw_like,
            "tolerances": candidate["tolerances"],
            "schemes": candidate["schemes"],
            "active_scheme": candidate["active_scheme"],
        }
    )
    candidate["fixed_point"] = rebuilt["fixed_point"]
    return candidate


def parse_config(obj: Any, source_text: str | None = None) -> dict[str, Any]:
    try:
        return _parse_config_inner(obj, source_text)
    except ConfigInvalid as error:
        if error.line is None and source_text and error.field:
            leaf = error.field.split(".")[-1].split("[", 1)[0]
            if leaf and leaf != "$":
                lines = key_lines(source_text, leaf)
                if lines:
                    error.line = lines[0]
        raise
