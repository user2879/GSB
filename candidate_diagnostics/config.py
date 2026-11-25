from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, Failure
from .fingerprinting import fingerprint


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "diagnostics-config-v1",
    "normalization": {
        "method": "weighted_linear",
        "raw_scale": 100.0,
        "slope": 1.0,
        "bias": 0.0,
        "output_scale": 1000.0,
        "clamp": True,
        "epsilon": 1e-12,
    },
    "quantization": {
        "fixed_decimals": 3,
        "gear_exponent": 0,
        "rounding": "half_up",
        "bit_width": 32,
        "reference_bit_width": 33,
        "reference_fixed_decimals": 4,
    },
    "top_k": 5,
    "max_candidates_per_request": 12,
    "channels": {"search": 1.2, "feed": 1.0, "recall": 0.8},
    "sampling": {
        "ratio": 0.4,
        "ratio_permille": 400,
        "quota_rounding": "floor",
        "remainder": "channel_name_asc",
        "scheme": "channel_stratified_aux_seeded_v1",
    },
    "concurrency": {"max_workers": 4},
    "budget": {"batch_point_budget": 24, "request_point_quota": 8},
    "tolerance": {
        "normalization_dequant": 0.05,
        "truncation_rank": 0,
        "sampling_quota": 0,
        "sampling_flip_count": 0,
        "shuffle_position": 1,
    },
    "v1_fields": {
        "request_id": 12,
        "seq": 4,
        "candidate_id": 10,
        "score_permille": 6,
        "select_code": 2,
        "separator": "|",
    },
    "cache_dir": "artifacts/cache",
    "snapshot_dir": "artifacts/state",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    def __init__(self, raw: dict[str, Any], source: str = "<dict>"):
        failures = validate_config(raw)
        if failures:
            raise ValueError("; ".join(f.message for f in failures))
        self.raw = deep_merge(DEFAULT_CONFIG, raw)
        sampling = self.raw["sampling"]
        if "ratio_permille" not in raw.get("sampling", {}):
            sampling["ratio_permille"] = int(round(float(sampling["ratio"]) * 1000))
        if "ratio" not in raw.get("sampling", {}):
            sampling["ratio"] = float(sampling["ratio_permille"]) / 1000.0
        self.source = source
        self.config_hash = fingerprint(self.raw, 16)
        business_payload = {
            key: value
            for key, value in self.raw.items()
            if key not in {"concurrency", "budget", "cache_dir", "snapshot_dir"}
        }
        self.business_fingerprint = fingerprint(business_payload, 16)
        self.channel_weights_fingerprint = fingerprint(self.raw["channels"], 16)
        self.sampling_fingerprint = fingerprint(self.raw["sampling"], 16)
        effective_quantization = {
            key: value
            for key, value in self.raw["quantization"].items()
            if not key.startswith("reference_")
        }
        self.quantization_fingerprint = fingerprint(effective_quantization, 16)
        self.reference_fingerprint = fingerprint(
            {
                key: value
                for key, value in self.raw["quantization"].items()
                if key.startswith("reference_")
            },
            16,
        )
        self.normalization_fingerprint = fingerprint(self.raw["normalization"], 16)
        self.v1_fingerprint = fingerprint(self.raw["v1_fields"], 12)

    @property
    def top_k(self) -> int:
        return int(self.raw["top_k"])

    @property
    def max_candidates(self) -> int:
        return int(self.raw["max_candidates_per_request"])

    @property
    def max_workers(self) -> int:
        return max(1, int(self.raw["concurrency"]["max_workers"]))

    @property
    def ratio(self) -> float:
        return float(self.raw["sampling"]["ratio"])

    @property
    def ratio_permille(self) -> int:
        return int(self.raw["sampling"]["ratio_permille"])

    @property
    def batch_budget(self) -> int:
        return int(self.raw["budget"]["batch_point_budget"])

    @property
    def request_quota(self) -> int:
        return int(self.raw["budget"]["request_point_quota"])

    def derive_version_fingerprint(self, package_version: str) -> str:
        return fingerprint(
            {
                "package": package_version,
                "schema": self.raw["schema_version"],
                "v1": self.raw["v1_fields"],
            },
            16,
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(raw: dict[str, Any]) -> list[Failure]:
    failures: list[Failure] = []

    def bad(message: str) -> None:
        failures.append(Failure(E_CONFIG_INVALID, message, source="config"))

    if not isinstance(raw, dict):
        return [Failure(E_CONFIG_INVALID, "configuration must be an object", source="config")]

    merged = deep_merge(DEFAULT_CONFIG, raw)
    if not isinstance(merged.get("top_k"), int) or merged["top_k"] <= 0:
        bad("top_k must be a positive integer")
    if (
        not isinstance(merged.get("max_candidates_per_request"), int)
        or merged["max_candidates_per_request"] <= 0
    ):
        bad("max_candidates_per_request must be a positive integer")

    channels = merged.get("channels")
    if not isinstance(channels, dict) or not channels:
        bad("channels must be a non-empty object")
    else:
        for name, weight in channels.items():
            if not isinstance(name, str) or not name:
                bad("channel names must be non-empty strings")
            if not _is_number(weight) or not math.isfinite(float(weight)) or weight < 0:
                bad(f"channel weight for {name!r} must be a finite non-negative number")

    sampling = merged.get("sampling", {})
    ratio = sampling.get("ratio")
    ratio_mille = sampling.get("ratio_permille")
    ratio_ok = _is_number(ratio) and math.isfinite(float(ratio)) and 0 < float(ratio) <= 1
    mille_ok = (
        isinstance(ratio_mille, int)
        and not isinstance(ratio_mille, bool)
        and 0 < ratio_mille <= 1000
    )
    if not ratio_ok and not mille_ok:
        failures.append(
            Failure(
                E_CONFIG_INVALID,
                "sampling ratio must be positive in (0,1] or ratio_permille in (0,1000]",
                source="config.sampling",
                business_code="E_RATIO_OUT_OF_RANGE",
            )
        )
    elif ratio_ok and mille_ok and abs(round(float(ratio) * 1000) - int(ratio_mille)) > 1:
        bad("sampling.ratio and sampling.ratio_permille must agree")

    budget = merged.get("budget", {})
    for key in ("batch_point_budget", "request_point_quota"):
        value = budget.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            bad(f"budget.{key} must be a positive integer")

    workers = merged.get("concurrency", {}).get("max_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        bad("concurrency.max_workers must be a positive integer")

    quant = merged.get("quantization", {})
    decimals = quant.get("fixed_decimals")
    gear = quant.get("gear_exponent")
    if not isinstance(decimals, int) or decimals < 0 or decimals > 9:
        bad("quantization.fixed_decimals must be an integer in [0,9]")
    if not isinstance(gear, int) or gear < 0 or (
        isinstance(decimals, int) and gear > decimals
    ):
        bad("quantization.gear_exponent must be in [0,fixed_decimals]")
    if quant.get("rounding") not in {"half_up", "half_even", "floor", "ceil"}:
        bad("quantization.rounding is unsupported")

    fields = merged.get("v1_fields", {})
    for key in ("request_id", "seq", "candidate_id", "score_permille", "select_code"):
        value = fields.get(key)
        if not isinstance(value, int) or value <= 0:
            bad(f"v1_fields.{key} must be a positive width")
    return failures


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return Config(raw, source=str(path))


def config_from_file_or_default(path: str | Path | None) -> Config:
    if path is None:
        return Config({})
    return load_config(path)
