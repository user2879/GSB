"""Operator-lab configuration and effective state."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import E_CONFIG, LabError
from .util import is_finite_number, is_positive_int, read_json, resolve_path, sha256_obj


DEFAULT_FIELD_WIDTHS = {
    "layer": 12,
    "row": 5,
    "scale_permille": 8,
    "bias_code": 8,
    "quant_bits": 3,
}

DEFAULT_CODEBOOK = {"int8": 8, "int16": 16, "int32": 32}


@dataclass
class LabConfig:
    project_root: Path
    config_path: Path
    raw: Dict[str, Any]
    paths: Dict[str, Path]
    exchange_version: str
    fractional_bits: int
    quant_level: int
    allowed_quant_levels: tuple[int, ...]
    max_workers: int
    batch_element_budget: int
    per_layer_element_quota: int
    variance_floor: float
    allowed_schemes: tuple[str, ...]
    v1_upgrade_scheme: str
    scheme_bias_policy: Dict[str, bool]
    field_widths: Dict[str, int]
    quant_codebook: Dict[str, int]
    runtime_dir: Path
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent

    @property
    def config_hash(self) -> str:
        return sha256_obj(
            {
                "raw": self.raw_config,
                "field_widths": self.field_widths,
                "quant_codebook": self.quant_codebook,
            }
        )

    @property
    def bits_to_precision(self) -> Dict[int, str]:
        return {bits: label for label, bits in self.quant_codebook.items()}

    def precision_to_bits(self, label: str) -> Optional[int]:
        return self.quant_codebook.get(label)

    def exp_polynomial_order(self, quant_level: int) -> int:
        softmax = self.raw_config.get("softmax", {})
        base = int(softmax.get("polynomial_base_order", 4))
        return max(2, min(12, base + int(quant_level)))

    def exp_approximation_name(self, quant_level: int) -> str:
        softmax = self.raw_config.get("softmax", {})
        name = str(softmax.get("exp_approximation", "range_reduced_ln2_taylor"))
        return f"{name}:order={self.exp_polynomial_order(quant_level)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.exchange_version,
            "fractional_bits": self.fractional_bits,
            "quant_level": self.quant_level,
            "max_workers": self.max_workers,
            "batch_element_budget": self.batch_element_budget,
            "per_layer_element_quota": self.per_layer_element_quota,
            "variance_floor": self.variance_floor,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabError(E_CONFIG, message, stage="load_validation")


def load_config(config_path: Path, project_root: Optional[Path] = None) -> LabConfig:
    config_path = config_path.resolve()
    try:
        raw = read_json(config_path)
    except Exception as exc:
        raise LabError(E_CONFIG, f"cannot read config: {exc}", stage="load_validation")
    _require(isinstance(raw, dict), "config root must be an object")
    paths_raw = raw.get("paths", {})
    _require(isinstance(paths_raw, dict), "paths must be an object")
    if project_root is None:
        cwd = Path.cwd().resolve()
        inferred = config_path.parent.parent

        def resolves_under(base: Path) -> bool:
            return any(
                isinstance(value, str) and (base / value).resolve().exists()
                for value in paths_raw.values()
            )

        project_root = inferred if not resolves_under(cwd) and resolves_under(inferred) else cwd
    project_root = project_root.resolve()
    required_paths = ("activations", "weights_v1", "weights_v2", "commands")
    resolved_paths: Dict[str, Path] = {}
    for key in required_paths:
        _require(isinstance(paths_raw.get(key), str), f"paths.{key} is required")
        resolved_paths[key] = resolve_path(
            paths_raw[key],
            project_root=project_root,
            config_dir=config_path.parent,
            must_exist=key != "commands",
        )
    for key in ("injections", "runtime"):
        if isinstance(paths_raw.get(key), str):
            resolved_paths[key] = resolve_path(
                paths_raw[key],
                project_root=project_root,
                config_dir=config_path.parent,
                must_exist=False,
            )
    runtime_dir = resolved_paths.get("runtime", project_root / "runtime")
    fixed = raw.get("fixed_point", {})
    _require(isinstance(fixed, dict), "fixed_point must be an object")
    frac_bits = fixed.get("fractional_bits", 12)
    _require(is_positive_int(frac_bits) and frac_bits <= 48, "fractional_bits must be in 1..48")
    allowed_levels = fixed.get("allowed_quant_levels", [0, 1, 2, 3])
    _require(
        isinstance(allowed_levels, list)
        and allowed_levels
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in allowed_levels),
        "allowed_quant_levels must be a non-empty list of non-negative integers",
    )
    quant_level = fixed.get("quant_level", 1)
    _require(quant_level in allowed_levels, "quant_level must be in allowed_quant_levels")
    exchange_version = raw.get("exchange_version", "v1")
    _require(exchange_version in ("v1", "v2"), "exchange_version must be v1 or v2")
    limits = raw.get("limits", {})
    _require(isinstance(limits, dict), "limits must be an object")
    max_workers = limits.get("max_workers", 4)
    _require(is_positive_int(max_workers), "max_workers must be positive")
    budget = limits.get("batch_element_budget", 100000)
    _require(is_positive_int(budget), "batch_element_budget must be positive")
    quota = limits.get("per_layer_element_quota", 100000)
    _require(is_positive_int(quota), "per_layer_element_quota must be positive")
    variance_floor = raw.get("layernorm", {}).get("variance_floor", 1e-8)
    _require(
        is_finite_number(variance_floor) and float(variance_floor) >= 0,
        "layernorm.variance_floor must be finite and non-negative",
    )
    schemes = raw.get("allowed_norm_schemes", ["layernorm_affine", "layernorm_scale_only"])
    _require(
        isinstance(schemes, list) and schemes and all(isinstance(item, str) for item in schemes),
        "allowed_norm_schemes must be a non-empty string list",
    )
    upgrade_scheme = raw.get("v1_upgrade_scheme", "layernorm_affine")
    _require(upgrade_scheme in schemes, "v1_upgrade_scheme must be an allowed scheme")
    scheme_bias_policy = raw.get(
        "scheme_bias_policy",
        {"layernorm_affine": True, "layernorm_scale_only": False},
    )
    _require(
        isinstance(scheme_bias_policy, dict)
        and all(
            isinstance(label, str) and isinstance(enabled, bool)
            for label, enabled in scheme_bias_policy.items()
        ),
        "scheme_bias_policy must map scheme names to booleans",
    )
    _require(
        all(scheme in scheme_bias_policy for scheme in schemes),
        "scheme_bias_policy must cover every allowed scheme",
    )
    widths = dict(DEFAULT_FIELD_WIDTHS)
    widths.update(raw.get("v1_field_widths", {}))
    _require(
        all(is_positive_int(widths.get(key)) for key in DEFAULT_FIELD_WIDTHS),
        "v1_field_widths must contain positive widths for all fields",
    )
    codebook = dict(DEFAULT_CODEBOOK)
    codebook.update(raw.get("quant_codebook", {}))
    _require(
        all(isinstance(label, str) and is_positive_int(bits) for label, bits in codebook.items()),
        "quant_codebook must map labels to positive integer widths",
    )
    _require(len(set(codebook.values())) == len(codebook), "quant_codebook widths must be unique")
    return LabConfig(
        project_root=project_root,
        config_path=config_path,
        raw=raw,
        paths=resolved_paths,
        exchange_version=exchange_version,
        fractional_bits=frac_bits,
        quant_level=quant_level,
        allowed_quant_levels=tuple(sorted(set(allowed_levels))),
        max_workers=max_workers,
        batch_element_budget=budget,
        per_layer_element_quota=quota,
        variance_floor=float(variance_floor),
        allowed_schemes=tuple(schemes),
        v1_upgrade_scheme=upgrade_scheme,
        scheme_bias_policy=dict(scheme_bias_policy),
        field_widths=widths,
        quant_codebook=codebook,
        runtime_dir=runtime_dir,
        raw_config=copy.deepcopy(raw),
    )


def default_state(config: LabConfig) -> Dict[str, Any]:
    return {
        "schema": "opexlab.state.v1",
        "revision": 0,
        "exchange_version": config.exchange_version,
        "fractional_bits": config.fractional_bits,
        "quant_level": config.quant_level,
        "max_workers": config.max_workers,
        "batch_element_budget": config.batch_element_budget,
        "per_layer_element_quota": config.per_layer_element_quota,
        "activation_overrides": {},
        "parameter_overrides": {},
        "history": [],
    }


def state_hash(state: Dict[str, Any]) -> str:
    return sha256_obj(state)


def copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))


def validate_state(config: LabConfig, state: Dict[str, Any]) -> Dict[str, Any]:
    _require(isinstance(state, dict), "state must be an object")
    _require(state.get("schema") in (None, "opexlab.state.v1"), "unsupported state schema")
    _require(state.get("exchange_version") in ("v1", "v2"), "state.exchange_version invalid")
    _require(
        is_positive_int(state.get("fractional_bits")) and state["fractional_bits"] <= 48,
        "state.fractional_bits invalid",
    )
    _require(
        state.get("quant_level") in config.allowed_quant_levels,
        "state.quant_level invalid",
    )
    _require(is_positive_int(state.get("max_workers")), "state.max_workers invalid")
    _require(
        is_positive_int(state.get("batch_element_budget")),
        "state.batch_element_budget invalid",
    )
    _require(
        is_positive_int(state.get("per_layer_element_quota")),
        "state.per_layer_element_quota invalid",
    )
    _require(isinstance(state.get("activation_overrides", {}), dict), "activation_overrides invalid")
    _require(isinstance(state.get("parameter_overrides", {}), dict), "parameter_overrides invalid")
    _require(isinstance(state.get("history", []), list), "history invalid")
    return state
