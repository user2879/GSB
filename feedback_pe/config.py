from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import codes
from .hashing import fingerprint

FRAC_BITS_CHOICES = (4, 8, 12, 16)
QUANT_TIER_CHOICES = ("Q8", "Q16", "Q32")
SCHEME_KINDS = ("sinusoidal", "linear_ramp", "blockwave")
STAGES = ("position", "score", "stats", "error")


def default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": 20261127,
        "dimension": 8,
        "base_freq": 10000.0,
        "scale": 1.0,
        "block_size": 2,
        "frac_bits": 16,
        "quant_tier": "Q32",
        "tolerances": {
            "position_rel": 0.010,
            "position_abs": 1.0e-5,
            "score_rel": 0.020,
            "score_abs": 1.0e-6,
            "weight_rel": 0.050,
            "weight_abs": 1.0e-7,
        },
        "workers": 2,
        "batch_point_budget": 144,
        "sample_point_quota": 80,
        "train_length": 16,
        "target_length": 64,
        "active_scheme": "sinusoidal",
        "schemes": [
            {
                "name": "sinusoidal", "kind": "sinusoidal",
                "base_freq": 10000.0, "scale": 1.0,
                "block_size": 2, "enabled": True,
            },
            {
                "name": "linear_ramp", "kind": "linear_ramp",
                "base_freq": 10000.0, "scale": 1.0,
                "block_size": 2, "enabled": True,
            },
            {
                "name": "blockwave", "kind": "blockwave",
                "base_freq": 10000.0, "scale": 1.0,
                "block_size": 4, "enabled": True,
            },
        ],
    }


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def err(code: str, field: str, message: str) -> dict[str, str]:
    return {"code": code, "field": field, "message": message}


def validate_config(raw: Any) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not isinstance(raw, dict):
        return [err(codes.E_CONFIG_SCHEMA, "$", "configuration must be a JSON object")]

    dim = raw.get("dimension")
    if not is_int(dim) or dim <= 0 or dim % 2 != 0:
        errors.append(err(codes.E_DIMENSION, "dimension", "must be a positive even integer"))
    if not is_int(raw.get("seed")):
        errors.append(err(codes.E_CONFIG_SCHEMA, "seed", "must be an integer"))
    if not is_finite_number(raw.get("base_freq")) or float(raw.get("base_freq", 0)) <= 0:
        errors.append(err(codes.E_BASE_FREQ, "base_freq", "must be a finite number > 0"))
    if not is_finite_number(raw.get("scale")) or float(raw.get("scale", 0)) <= 0:
        errors.append(err(codes.E_SCALE, "scale", "must be a finite number > 0"))

    block_size = raw.get("block_size")
    if not is_int(block_size) or block_size <= 0:
        errors.append(err(codes.E_BLOCK_SIZE, "block_size", "must be a positive integer"))
    elif is_int(dim) and dim > 0:
        if block_size > dim or dim % block_size != 0:
            errors.append(err(codes.E_BLOCK_SIZE, "block_size", "must divide dimension"))

    if raw.get("frac_bits") not in FRAC_BITS_CHOICES:
        errors.append(err(codes.E_ENUM_FRAC_BITS, "frac_bits",
                          f"must be one of {list(FRAC_BITS_CHOICES)}"))
    if raw.get("quant_tier") not in QUANT_TIER_CHOICES:
        errors.append(err(codes.E_ENUM_QUANT_TIER, "quant_tier",
                          f"must be one of {list(QUANT_TIER_CHOICES)}"))

    tol = raw.get("tolerances")
    tol_fields = ("position_rel", "position_abs", "score_rel", "score_abs",
                  "weight_rel", "weight_abs")
    if not isinstance(tol, dict):
        errors.append(err(codes.E_TOLERANCE, "tolerances", "must be an object"))
    else:
        for field in tol_fields:
            if not is_finite_number(tol.get(field)) or float(tol[field]) <= 0:
                errors.append(err(codes.E_TOLERANCE, f"tolerances.{field}", "must be > 0"))

    if not is_int(raw.get("workers")) or not 1 <= int(raw.get("workers", 0)) <= 32:
        errors.append(err(codes.E_WORKERS, "workers", "must be an integer in [1, 32]"))
    for field in ("batch_point_budget", "sample_point_quota"):
        if not is_int(raw.get(field)) or int(raw.get(field, 0)) <= 0:
            errors.append(err(codes.E_BUDGET, field, "must be a positive integer"))

    train_len = raw.get("train_length")
    target_len = raw.get("target_length")
    if not is_int(train_len) or train_len <= 0:
        errors.append(err(codes.E_LENGTH_VALUE, "train_length", "must be a positive integer"))
    if not is_int(target_len) or target_len <= 0:
        errors.append(err(codes.E_LENGTH_VALUE, "target_length", "must be a positive integer"))
    if is_int(train_len) and is_int(target_len) and train_len > target_len:
        errors.append(err(codes.E_LENGTH_ORDER, "train_length",
                          "train_length must not exceed target_length"))

    schemes = raw.get("schemes")
    if not isinstance(schemes, list) or not schemes:
        errors.append(err(codes.E_SCHEME, "schemes", "must be a non-empty array"))
    else:
        names: set[str] = set()
        for index, scheme in enumerate(schemes):
            prefix = f"schemes[{index}]"
            if not isinstance(scheme, dict):
                errors.append(err(codes.E_SCHEME, prefix, "scheme must be an object"))
                continue
            name = scheme.get("name")
            if not isinstance(name, str) or not name:
                errors.append(err(codes.E_SCHEME, f"{prefix}.name",
                                  "must be a non-empty string"))
            elif name in names:
                errors.append(err(codes.E_SCHEME, f"{prefix}.name",
                                  f"duplicate scheme {name!r}"))
            else:
                names.add(name)
            if scheme.get("kind") not in SCHEME_KINDS:
                errors.append(err(codes.E_SCHEME, f"{prefix}.kind",
                                  f"must be one of {list(SCHEME_KINDS)}"))
            for field, code in (("base_freq", codes.E_BASE_FREQ), ("scale", codes.E_SCALE)):
                if not is_finite_number(scheme.get(field)) or float(scheme.get(field, 0)) <= 0:
                    errors.append(err(code, f"{prefix}.{field}", "must be finite and > 0"))
            block = scheme.get("block_size")
            if not is_int(block) or block <= 0:
                errors.append(err(codes.E_BLOCK_SIZE, f"{prefix}.block_size",
                                  "must be positive integer"))
            elif is_int(dim) and dim > 0 and (block > dim or dim % block != 0):
                errors.append(err(codes.E_BLOCK_SIZE, f"{prefix}.block_size",
                                  "must divide dimension"))
        active = raw.get("active_scheme")
        if isinstance(active, str) and active and active not in names:
            errors.append(err(codes.E_SCHEME, "active_scheme",
                              "must name a configured scheme"))

    return errors


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = default_config()
    config.update(raw)
    tol = default_config()["tolerances"].copy()
    if isinstance(config.get("tolerances"), dict):
        tol.update(config["tolerances"])
    config["tolerances"] = tol
    normalized_schemes = []
    for scheme in config["schemes"]:
        normalized_schemes.append({
            "name": str(scheme["name"]),
            "kind": str(scheme["kind"]),
            "base_freq": float(scheme["base_freq"]),
            "scale": float(scheme["scale"]),
            "block_size": int(scheme["block_size"]),
            "enabled": bool(scheme.get("enabled", True)),
        })
    config["schemes"] = normalized_schemes
    config["base_freq"] = float(config["base_freq"])
    config["scale"] = float(config["scale"])
    return config


def load_config(path: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [err(codes.E_CONFIG_PARSE, str(path), f"JSON parse error: {exc.msg}")]
    except OSError as exc:
        return None, [err(codes.E_CONFIG_PARSE, str(path), str(exc))]
    errors = validate_config(raw)
    if errors:
        return None, errors
    return normalize_config(raw), []


def config_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint("config:v1", config)


def scheme_fingerprint(scheme: dict[str, Any]) -> str:
    return fingerprint("scheme:v1", scheme)


def quant_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint("quant:v1", {
        "frac_bits": config["frac_bits"],
        "quant_tier": config["quant_tier"],
        "kernel": "rne-shift-v1",
    })
