"""Configuration normalization and validation for the quantization pipeline."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import (
    E_CONFIG_INVALID,
    ConfigError,
)
from .util import finite_float, is_int, sha256_json
from .yaml_lite import load_yaml


ALLOWED_SCHEMES = ("symmetric", "asymmetric")
ALLOWED_SATURATION = ("saturate", "error")
ALLOWED_ROUNDING = ("nearest_even", "nearest_away", "floor", "ceil", "truncate")
ALLOWED_INTERPOLATION = ("linear",)
ALLOWED_TIE_BREAK = ("lower_index", "higher_index", "first_occurrence")
DEFAULT_BIT_WIDTHS = tuple(range(2, 17))
MAX_FRACTIONAL_BITS = 30


@dataclass(frozen=True)
class ConfigIssue:
    message: str
    scope: str = "global"  # global | layer | distillation
    layer: str | None = None
    stage: str = "config"
    code: str = E_CONFIG_INVALID
    field: str | None = None

    def key(self) -> tuple[str, str | None, str, str | None]:
        return (self.scope, self.layer, self.stage, self.field)


@dataclass(frozen=True)
class Profile:
    name: str
    bits: int
    fractional_bits: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bits": self.bits,
            "fractional_bits": self.fractional_bits,
        }


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float
    soft_label_weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "soft_label_weight": self.soft_label_weight,
        }


@dataclass(frozen=True)
class ToleranceConfig:
    absolute: float
    relative: float

    def as_dict(self) -> dict[str, Any]:
        return {"absolute": self.absolute, "relative": self.relative}


@dataclass(frozen=True)
class RuntimeConfig:
    max_workers: int
    point_budget: int
    layer_point_quota: int
    snapshot_interval: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "point_budget": self.point_budget,
            "layer_point_quota": self.layer_point_quota,
            "snapshot_interval": self.snapshot_interval,
        }


@dataclass(frozen=True)
class EffectiveLayerConfig:
    layer: str
    profile: str
    target_bits: int
    fractional_bits: int
    group_size: int
    min_group_size: int
    scheme: str
    saturation: str
    rounding: str
    percentile_low: float
    percentile_high: float
    interpolation: str = "linear"
    tie_break: str = "lower_index"

    @property
    def qmin(self) -> int:
        return -(1 << (self.target_bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << (self.target_bits - 1)) - 1

    @property
    def fixed_multiplier(self) -> int:
        return 1 << self.fractional_bits

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "target_bits": self.target_bits,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "fractional_bits": self.fractional_bits,
            "fixed_multiplier": self.fixed_multiplier,
            "group_size": self.group_size,
            "min_group_size": self.min_group_size,
            "scheme": self.scheme,
            "saturation": self.saturation,
            "rounding": self.rounding,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "interpolation": self.interpolation,
            "tie_break": self.tie_break,
        }

    def calibration_parameters(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "target_bits": self.target_bits,
            "group_size": self.group_size,
            "min_group_size": self.min_group_size,
            "scheme": self.scheme,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "interpolation": self.interpolation,
            "tie_break": self.tie_break,
        }

    def quantization_parameters(self) -> dict[str, Any]:
        result = self.calibration_parameters()
        result.update(
            {
                "fractional_bits": self.fractional_bits,
                "saturation": self.saturation,
                "rounding": self.rounding,
                "qmin": self.qmin,
                "qmax": self.qmax,
            }
        )
        return result


_MISSING = object()


def _get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _coalesce(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_scheme(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    aliases = {
        "sym": "symmetric",
        "symmetric": "symmetric",
        "asym": "asymmetric",
        "asymmetric": "asymmetric",
    }
    return aliases.get(lowered)


def _normalize_saturation(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    aliases = {
        "saturate": "saturate",
        "clip": "saturate",
        "clamp": "saturate",
        "error": "error",
        "raise": "error",
    }
    return aliases.get(lowered)


def _normalize_rounding(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    aliases = {
        "nearest_even": "nearest_even",
        "round_half_even": "nearest_even",
        "banker": "nearest_even",
        "nearest_away": "nearest_away",
        "round_half_away": "nearest_away",
        "floor": "floor",
        "ceil": "ceil",
        "truncate": "truncate",
        "trunc": "truncate",
    }
    return aliases.get(lowered)


@dataclass
class PipelineConfig:
    raw: dict[str, Any]
    profiles: dict[str, Profile]
    profile_order: tuple[str, ...]
    active_profile: str
    defaults: dict[str, Any]
    layer_overrides: dict[str, dict[str, Any]]
    distillation: DistillationConfig
    tolerances: ToleranceConfig
    runtime: RuntimeConfig
    issues: list[ConfigIssue] = field(default_factory=list)
    distillation_valid: bool = True
    forced_profile: str | None = None
    forced_fractional_bits: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    @property
    def global_fatal_issues(self) -> list[ConfigIssue]:
        return [
            issue
            for issue in self.issues
            if issue.scope == "global" and issue.stage == "config"
        ]

    def profile_for(self, name: str | None) -> Profile:
        if name is None:
            name = self.active_profile
        if name not in self.profiles:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"unknown quantization profile: {name!r}",
                stage="config",
                details={"available_profiles": list(self.profile_order)},
            )
        return self.profiles[name]

    def layer_issues(self, layer: str) -> list[ConfigIssue]:
        return [issue for issue in self.issues if issue.scope == "layer" and issue.layer == layer]

    def resolve_layer(
        self,
        layer: str,
        numel: int,
        marker_group_size: int | None = None,
    ) -> EffectiveLayerConfig:
        issues = self.layer_issues(layer)
        if issues:
            raise ConfigError(
                E_CONFIG_INVALID,
                "; ".join(issue.message for issue in issues),
                stage="config",
                layer=layer,
                details={"fields": [issue.field for issue in issues if issue.field]},
            )

        local = dict(self.layer_overrides.get(layer, {}))
        nested_quant = _as_mapping(_get(local, "quantization", "quant"))
        local.update(nested_quant)
        base = dict(self.defaults)
        merged = {**base, **local}

        effective_profile_name: str | None
        if self.forced_profile is not None:
            effective_profile_name = self.forced_profile
        else:
            raw_profile = _coalesce(merged, "profile", "quantization_profile", "gear")
            effective_profile_name = str(raw_profile) if raw_profile is not None else self.active_profile
        profile = self.profile_for(effective_profile_name)

        explicit_bits = _coalesce(merged, "target_bits", "bits", "bit_width")
        if self.forced_profile is not None:
            bits_value: Any = profile.bits
        elif explicit_bits is not None:
            bits_value = explicit_bits
        else:
            bits_value = profile.bits
        if not is_int(bits_value):
            raise ConfigError(
                E_CONFIG_INVALID,
                f"target_bits must be an integer for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "target_bits"},
            )
        if bits_value not in self.allowed_bit_widths:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"target_bits {bits_value} is outside the allowed enumeration for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "target_bits", "allowed": list(self.allowed_bit_widths)},
            )

        explicit_frac = _coalesce(
            merged,
            "fractional_bits",
            "frac_bits",
            "fixed_fractional_bits",
            "fixed_frac_bits",
        )
        if self.forced_fractional_bits is not None:
            frac_value: Any = self.forced_fractional_bits
        elif explicit_frac is not None:
            frac_value = explicit_frac
        else:
            frac_value = profile.fractional_bits
        if not is_int(frac_value) or frac_value <= 0:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"fractional_bits must be a positive integer for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "fractional_bits"},
            )
        if frac_value > MAX_FRACTIONAL_BITS:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"fractional_bits exceeds the supported safety limit {MAX_FRACTIONAL_BITS} for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "fractional_bits", "max": MAX_FRACTIONAL_BITS},
            )

        group_value = _coalesce(
            merged,
            "group_size",
            "group_granularity",
            "granularity",
        )
        if group_value is None:
            group_value = marker_group_size
        if not is_int(group_value) or group_value <= 0:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"group_size must be a positive integer for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "group_size"},
            )
        if numel % group_value != 0:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"group_size {group_value} does not divide {numel} elements for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "group_size", "numel": numel, "group_size": group_value},
            )

        min_group_value = _coalesce(merged, "min_group_size", "minimum_group_size")
        if min_group_value is None:
            min_group_value = self.defaults.get("min_group_size", 1)
        if not is_int(min_group_value) or min_group_value <= 0:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"min_group_size must be a positive integer for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "min_group_size"},
            )

        raw_scheme = _get(merged, "scheme", "quantization_scheme", default=_MISSING)
        scheme_value = (
            "symmetric" if raw_scheme is _MISSING else _normalize_scheme(raw_scheme)
        )
        if scheme_value not in ALLOWED_SCHEMES:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"unsupported quantization scheme for layer {layer!r}: {scheme_value!r}",
                stage="config",
                layer=layer,
                details={"field": "scheme"},
            )

        raw_saturation = _get(
            merged,
            "saturation",
            "saturation_strategy",
            "clip_strategy",
            default=_MISSING,
        )
        saturation_value = (
            "saturate" if raw_saturation is _MISSING else _normalize_saturation(raw_saturation)
        )
        if saturation_value not in ALLOWED_SATURATION:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"unsupported saturation strategy for layer {layer!r}: {saturation_value!r}",
                stage="config",
                layer=layer,
                details={"field": "saturation"},
            )

        raw_rounding = _get(merged, "rounding", "rounding_mode", default=_MISSING)
        rounding_value = (
            "nearest_even" if raw_rounding is _MISSING else _normalize_rounding(raw_rounding)
        )
        if rounding_value not in ALLOWED_ROUNDING:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"unsupported rounding mode for layer {layer!r}: {rounding_value!r}",
                stage="config",
                layer=layer,
                details={"field": "rounding"},
            )

        low = _coalesce(merged, "percentile_low", "quantile_low")
        high = _coalesce(merged, "percentile_high", "quantile_high")
        low = 0.01 if low is None else low
        high = 0.99 if high is None else high
        if not finite_float(low) or not finite_float(high):
            raise ConfigError(
                E_CONFIG_INVALID,
                f"percentiles must be finite numbers for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "percentile"},
            )
        low_f, high_f = float(low), float(high)
        if not (0.0 <= low_f <= high_f <= 1.0):
            raise ConfigError(
                E_CONFIG_INVALID,
                f"percentiles must satisfy 0 <= low <= high <= 1 for layer {layer!r}",
                stage="config",
                layer=layer,
                details={"field": "percentile", "low": low_f, "high": high_f},
            )

        interpolation = str(_get(merged, "interpolation", "quantile_interpolation", default="linear"))
        if interpolation not in ALLOWED_INTERPOLATION:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"unsupported quantile interpolation for layer {layer!r}: {interpolation!r}",
                stage="config",
                layer=layer,
                details={"field": "interpolation", "allowed": list(ALLOWED_INTERPOLATION)},
            )

        tie_break = str(_get(merged, "tie_break", "tie_break_rule", default="lower_index"))
        if tie_break not in ALLOWED_TIE_BREAK:
            raise ConfigError(
                E_CONFIG_INVALID,
                f"unsupported tie-break rule for layer {layer!r}: {tie_break!r}",
                stage="config",
                layer=layer,
                details={"field": "tie_break", "allowed": list(ALLOWED_TIE_BREAK)},
            )

        return EffectiveLayerConfig(
            layer=layer,
            profile=profile.name,
            target_bits=bits_value,
            fractional_bits=frac_value,
            group_size=group_value,
            min_group_size=min_group_value,
            scheme=scheme_value,
            saturation=saturation_value,
            rounding=rounding_value,
            percentile_low=low_f,
            percentile_high=high_f,
            interpolation=interpolation,
            tie_break=tie_break,
        )

    @property
    def allowed_bit_widths(self) -> tuple[int, ...]:
        value = self.defaults.get("allowed_bit_widths")
        if isinstance(value, (list, tuple)) and value:
            return tuple(int(item) for item in value)
        return DEFAULT_BIT_WIDTHS


def load_config(
    path: str | Path,
    *,
    profile: str | None = None,
    fractional_bits: int | None = None,
    max_workers: int | None = None,
    point_budget: int | None = None,
    layer_point_quota: int | None = None,
    snapshot_interval: int | None = None,
) -> tuple[PipelineConfig, list[ConfigIssue]]:
    raw = load_yaml(path)
    if not isinstance(raw, Mapping):
        issue = ConfigIssue("configuration root must be a mapping", field="root")
        fallback = _fallback_config({}, profile, fractional_bits)
        return fallback, [issue]
    overrides: dict[str, Any] = {}
    if profile is not None:
        overrides["profile"] = profile
    if fractional_bits is not None:
        overrides["fractional_bits"] = fractional_bits
    if max_workers is not None:
        overrides["max_workers"] = max_workers
    if point_budget is not None:
        overrides["point_budget"] = point_budget
    if layer_point_quota is not None:
        overrides["layer_point_quota"] = layer_point_quota
    if snapshot_interval is not None:
        overrides["snapshot_interval"] = snapshot_interval
    return parse_config(
        dict(raw),
        profile=profile,
        fractional_bits=fractional_bits,
        runtime_overrides={
            "max_workers": max_workers,
            "point_budget": point_budget,
            "layer_point_quota": layer_point_quota,
            "snapshot_interval": snapshot_interval,
        },
        overrides=overrides,
    )


def _fallback_config(
    raw: dict[str, Any], profile: str | None, fractional_bits: int | None
) -> PipelineConfig:
    profile_name = profile or "q8"
    profile_obj = Profile(profile_name, 8, fractional_bits or 8)
    distill = DistillationConfig(1.0, 0.0)
    tolerances = ToleranceConfig(0.0, 0.0)
    runtime = RuntimeConfig(1, 0, 0, 1)
    return PipelineConfig(
        raw=raw,
        profiles={profile_name: profile_obj},
        profile_order=(profile_name,),
        active_profile=profile_name,
        defaults={"group_size": 1},
        layer_overrides={},
        distillation=distill,
        tolerances=tolerances,
        runtime=runtime,
        forced_profile=profile,
        forced_fractional_bits=fractional_bits,
        config_hash=sha256_json(raw),
    )


def parse_config(
    raw: dict[str, Any],
    *,
    profile: str | None = None,
    fractional_bits: int | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[PipelineConfig, list[ConfigIssue]]:
    raw = copy.deepcopy(raw)
    issues: list[ConfigIssue] = []
    quant = _as_mapping(_get(raw, "quantization", "quant", default={}))
    distill_raw = _as_mapping(_get(raw, "distillation", "distill", default={}))
    runtime_raw = _as_mapping(_get(raw, "runtime", "scheduler", default={}))
    tolerance_raw = _as_mapping(_get(raw, "tolerances", "tolerance", default={}))

    # Profiles and the declared allowed bit-width enumeration.
    allowed = _coalesce(quant, "allowed_bit_widths", "bit_width_enum")
    if allowed is None:
        allowed = DEFAULT_BIT_WIDTHS
    elif not isinstance(allowed, (list, tuple)) or not allowed:
        issues.append(ConfigIssue("allowed_bit_widths must be a non-empty sequence", field="allowed_bit_widths"))
        allowed = DEFAULT_BIT_WIDTHS
    else:
        normalized_allowed: list[int] = []
        for item in allowed:
            if not is_int(item) or item <= 1 or item > 32:
                issues.append(
                    ConfigIssue(
                        f"invalid allowed bit width: {item!r}",
                        field="allowed_bit_widths",
                    )
                )
            else:
                normalized_allowed.append(item)
        allowed = tuple(normalized_allowed) if normalized_allowed else DEFAULT_BIT_WIDTHS

    profiles_raw = _coalesce(
        quant,
        "profiles",
        "quantization_profiles",
        "quantization_levels",
        "gears",
    )
    profiles: dict[str, Profile] = {}
    profile_order: list[str] = []
    if profiles_raw is None:
        bits = _coalesce(quant, "target_bits", "bits", "bit_width")
        frac = _coalesce(
            quant,
            "fractional_bits",
            "frac_bits",
            "fixed_fractional_bits",
        )
        bits = 8 if bits is None else bits
        frac = bits if frac is None else frac
        default_name = profile or str(_coalesce(quant, "active_profile", "gear", default="q8"))
        if is_int(bits) and bits in allowed and is_int(frac) and 0 < frac <= MAX_FRACTIONAL_BITS:
            profiles[default_name] = Profile(default_name, bits, frac)
            profile_order.append(default_name)
        else:
            issues.append(
                ConfigIssue(
                    "default target_bits/fractional_bits are invalid",
                    field="target_bits",
                )
            )
            profiles[default_name] = Profile(default_name, 8, 8)
            profile_order.append(default_name)
    elif isinstance(profiles_raw, Mapping):
        profile_items = []
        for name, value in profiles_raw.items():
            if isinstance(value, Mapping):
                body = dict(value)
                body.setdefault("name", str(name))
            elif is_int(value):
                body = {"name": str(name), "bits": value}
            else:
                issues.append(
                    ConfigIssue(
                        f"profile {name!r} must be a mapping or integer bit width",
                        field="profiles",
                    )
                )
                continue
            profile_items.append(body)
    elif isinstance(profiles_raw, list):
        profile_items = []
        for index, value in enumerate(profiles_raw):
            if not isinstance(value, Mapping):
                issues.append(
                    ConfigIssue(f"profile at index {index} must be a mapping", field="profiles")
                )
                continue
            profile_items.append(dict(value))
    else:
        issues.append(ConfigIssue("profiles must be a mapping or sequence", field="profiles"))
        profile_items = []

    profiles_are_mapping = isinstance(profiles_raw, Mapping)
    if profiles_raw is not None:
        previous_bits: int | None = None
        for index, body in enumerate(profile_items):
            name = _coalesce(body, "name", "id", "profile")
            if name is None:
                issues.append(ConfigIssue(f"profile at index {index} has no name", field="profiles"))
                continue
            name = str(name)
            if name in profiles:
                issues.append(ConfigIssue(f"duplicate quantization profile name: {name!r}", field="profiles"))
                continue
            bits = _coalesce(body, "bits", "target_bits", "bit_width")
            frac = _coalesce(
                body,
                "fractional_bits",
                "frac_bits",
                "fixed_fractional_bits",
            )
            frac = bits if frac is None else frac
            if not is_int(bits) or bits not in allowed:
                issues.append(
                    ConfigIssue(
                        f"profile {name!r} target_bits is outside the allowed enumeration",
                        field="profiles",
                    )
                )
                continue
            if not is_int(frac) or frac <= 0 or frac > MAX_FRACTIONAL_BITS:
                issues.append(
                    ConfigIssue(
                        f"profile {name!r} fractional_bits must be a positive integer <= {MAX_FRACTIONAL_BITS}",
                        field="profiles",
                    )
                )
                continue
            if not profiles_are_mapping and previous_bits is not None and bits <= previous_bits:
                issues.append(
                    ConfigIssue(
                        "quantization profiles must be declared in strictly increasing bit-width order",
                        field="profiles",
                    )
                )
            previous_bits = bits
            profiles[name] = Profile(name, bits, frac)
            profile_order.append(name)

    if profiles_are_mapping:
        profile_order.sort(key=lambda name: (profiles[name].bits, name))

    if not profiles:
        profiles["q8"] = Profile("q8", 8, 8)
        profile_order = ["q8"]

    if profile is not None:
        active_profile = profile
    else:
        candidate = _coalesce(quant, "active_profile", "profile", "gear")
        active_profile = str(candidate) if candidate is not None else profile_order[-1]
    if active_profile not in profiles:
        issues.append(
            ConfigIssue(
                f"active quantization profile {active_profile!r} is not declared",
                field="active_profile",
            )
        )
        active_profile = profile_order[0]

    # Defaults inherited by each layer.
    defaults: dict[str, Any] = {
        "group_size": _coalesce(quant, "group_size", "group_granularity", "granularity"),
        "min_group_size": _coalesce(quant, "min_group_size", "minimum_group_size", default=1),
        "scheme": _coalesce(quant, "scheme", "quantization_scheme", default="symmetric"),
        "saturation": _coalesce(
            quant,
            "saturation",
            "saturation_strategy",
            "clip_strategy",
            default="saturate",
        ),
        "rounding": _coalesce(quant, "rounding", "rounding_mode", default="nearest_even"),
        "percentile_low": _coalesce(quant, "percentile_low", "quantile_low", default=0.01),
        "percentile_high": _coalesce(quant, "percentile_high", "quantile_high", default=0.99),
        "interpolation": _coalesce(
            quant,
            "interpolation",
            "quantile_interpolation",
            default="linear",
        ),
        "tie_break": _coalesce(quant, "tie_break", "tie_break_rule", default="lower_index"),
        "allowed_bit_widths": tuple(allowed),
    }
    top_frac = _coalesce(quant, "fractional_bits", "frac_bits", "fixed_fractional_bits")
    if top_frac is not None:
        if not is_int(top_frac) or top_frac <= 0 or top_frac > MAX_FRACTIONAL_BITS:
            issues.append(
                ConfigIssue(
                    "default fractional_bits must be a positive integer within the safety limit",
                    field="fractional_bits",
                )
            )
        else:
            defaults["fractional_bits"] = top_frac
    top_bits = _coalesce(quant, "target_bits", "bits", "bit_width")
    if top_bits is not None:
        if not is_int(top_bits) or top_bits not in allowed:
            issues.append(
                ConfigIssue(
                    "default target_bits is outside the allowed enumeration",
                    field="target_bits",
                )
            )
        else:
            defaults["target_bits"] = top_bits

    group_default = defaults.get("group_size")
    if group_default is not None and (not is_int(group_default) or group_default <= 0):
        issues.append(ConfigIssue("default group_size must be a positive integer", field="group_size"))
        defaults["group_size"] = None
    min_group_default = defaults.get("min_group_size")
    if not is_int(min_group_default) or min_group_default <= 0:
        issues.append(ConfigIssue("default min_group_size must be a positive integer", field="min_group_size"))
        defaults["min_group_size"] = 1
    if defaults.get("scheme") is not None:
        normalized = _normalize_scheme(defaults["scheme"])
        if normalized is None:
            issues.append(ConfigIssue("default quantization scheme is not declared/supported", field="scheme"))
        else:
            defaults["scheme"] = normalized
    if defaults.get("saturation") is not None:
        normalized = _normalize_saturation(defaults["saturation"])
        if normalized is None:
            issues.append(ConfigIssue("default saturation strategy is not declared/supported", field="saturation"))
        else:
            defaults["saturation"] = normalized
    if defaults.get("rounding") is not None:
        normalized = _normalize_rounding(defaults["rounding"])
        if normalized is None:
            issues.append(ConfigIssue("default rounding mode is not declared/supported", field="rounding"))
        else:
            defaults["rounding"] = normalized
    percentile_low_default = defaults.get("percentile_low")
    percentile_high_default = defaults.get("percentile_high")
    if (
        not finite_float(percentile_low_default)
        or not finite_float(percentile_high_default)
        or not (0.0 <= float(percentile_low_default) <= float(percentile_high_default) <= 1.0)
    ):
        issues.append(
            ConfigIssue(
                "default percentiles must be finite and satisfy 0 <= low <= high <= 1",
                field="percentile",
            )
        )

    # Layer overrides (mapping keyed by layer or a sequence with name/id).
    layer_overrides: dict[str, dict[str, Any]] = {}
    layers_raw = raw.get("layers", {})
    if isinstance(layers_raw, Mapping):
        for name, body in layers_raw.items():
            if body is None:
                body = {}
            if not isinstance(body, Mapping):
                issues.append(
                    ConfigIssue(f"layer override {name!r} must be a mapping", scope="layer", layer=str(name))
                )
                continue
            layer_overrides[str(name)] = dict(body)
    elif isinstance(layers_raw, list):
        for index, body in enumerate(layers_raw):
            if not isinstance(body, Mapping):
                issues.append(ConfigIssue(f"layer override at index {index} must be a mapping", field="layers"))
                continue
            name = _coalesce(body, "name", "layer")
            if name is None:
                issues.append(ConfigIssue(f"layer override at index {index} has no name", field="layers"))
                continue
            if str(name) in layer_overrides:
                issues.append(
                    ConfigIssue(f"duplicate layer override: {name!r}", scope="layer", layer=str(name), field="layers")
                )
                continue
            layer_overrides[str(name)] = {k: v for k, v in body.items() if k not in ("name", "layer")}
    elif layers_raw is not None:
        issues.append(ConfigIssue("layers must be a mapping or sequence", field="layers"))

    # Early validation of explicit layer-level bit widths and grouping values.
    for layer_name, body in layer_overrides.items():
        nested = _as_mapping(_get(body, "quantization", "quant"))
        merged = {**body, **nested}
        bits = _coalesce(merged, "target_bits", "bits", "bit_width")
        if bits is not None and (not is_int(bits) or bits not in allowed):
            issues.append(
                ConfigIssue(
                    f"target_bits for layer {layer_name!r} is outside the allowed enumeration",
                    scope="layer",
                    layer=layer_name,
                    field="target_bits",
                )
            )
        frac = _coalesce(merged, "fractional_bits", "frac_bits", "fixed_fractional_bits")
        if frac is not None and (
            not is_int(frac) or frac <= 0 or frac > MAX_FRACTIONAL_BITS
        ):
            issues.append(
                ConfigIssue(
                    f"fractional_bits for layer {layer_name!r} must be a positive integer",
                    scope="layer",
                    layer=layer_name,
                    field="fractional_bits",
                )
            )
        group = _coalesce(merged, "group_size", "group_granularity", "granularity")
        if group is not None and (not is_int(group) or group <= 0):
            issues.append(
                ConfigIssue(
                    f"group_size for layer {layer_name!r} must be a positive integer",
                    scope="layer",
                    layer=layer_name,
                    field="group_size",
                )
            )
        profile_name = _coalesce(merged, "profile", "quantization_profile", "gear")
        if profile_name is not None and str(profile_name) not in profiles:
            issues.append(
                ConfigIssue(
                    f"quantization profile {profile_name!r} for layer {layer_name!r} is not declared",
                    scope="layer",
                    layer=layer_name,
                    field="profile",
                )
            )
        scheme = _coalesce(merged, "scheme", "quantization_scheme")
        if scheme is not None and _normalize_scheme(scheme) is None:
            issues.append(
                ConfigIssue(
                    f"scheme for layer {layer_name!r} is not declared/supported",
                    scope="layer",
                    layer=layer_name,
                    field="scheme",
                )
            )
        saturation = _coalesce(merged, "saturation", "saturation_strategy", "clip_strategy")
        if saturation is not None and _normalize_saturation(saturation) is None:
            issues.append(
                ConfigIssue(
                    f"saturation for layer {layer_name!r} is not declared/supported",
                    scope="layer",
                    layer=layer_name,
                    field="saturation",
                )
            )
        rounding = _coalesce(merged, "rounding", "rounding_mode")
        if rounding is not None and _normalize_rounding(rounding) is None:
            issues.append(
                ConfigIssue(
                    f"rounding for layer {layer_name!r} is not declared/supported",
                    scope="layer",
                    layer=layer_name,
                    field="rounding",
                )
            )
        low = _coalesce(merged, "percentile_low", "quantile_low")
        high = _coalesce(merged, "percentile_high", "quantile_high")
        if low is not None or high is not None:
            low = 0.01 if low is None else low
            high = 0.99 if high is None else high
            if (
                not finite_float(low)
                or not finite_float(high)
                or not (0.0 <= float(low) <= float(high) <= 1.0)
            ):
                issues.append(
                    ConfigIssue(
                        f"percentiles for layer {layer_name!r} are invalid",
                        scope="layer",
                        layer=layer_name,
                        field="percentile",
                    )
                )

    temperature_raw = _coalesce(distill_raw, "temperature", "temp", default=1.0)
    try:
        temperature = float(temperature_raw)
    except (TypeError, ValueError):
        temperature = float("nan")
    soft_weight = _coalesce(
        distill_raw,
        "soft_label_weight",
        "soft_target_weight",
        "alpha",
        default=0.0,
    )
    distillation_valid = True
    if not finite_float(temperature) or temperature <= 0:
        issues.append(
            ConfigIssue(
                "distillation temperature must be a positive finite number",
                scope="distillation",
                stage="distillation",
                field="temperature",
            )
        )
        distillation_valid = False
        temperature = 1.0
    if not finite_float(soft_weight) or not (0.0 <= float(soft_weight) <= 1.0):
        issues.append(
            ConfigIssue(
                "soft_label_weight must be between 0 and 1",
                scope="distillation",
                stage="distillation",
                field="soft_label_weight",
            )
        )
        distillation_valid = False
        soft_weight = 0.0

    tol_abs = _coalesce(
        tolerance_raw,
        "absolute",
        "abs",
        "abs_tolerance",
        "atol",
        default=0.0,
    )
    tol_rel = _coalesce(
        tolerance_raw,
        "relative",
        "rel",
        "rel_tolerance",
        "rtol",
        default=0.0,
    )
    if not finite_float(tol_abs) or float(tol_abs) < 0:
        issues.append(ConfigIssue("absolute tolerance must be a finite non-negative number", field="tolerances.absolute"))
        tol_abs = 0.0
    if not finite_float(tol_rel) or float(tol_rel) < 0:
        issues.append(ConfigIssue("relative tolerance must be a finite non-negative number", field="tolerances.relative"))
        tol_rel = 0.0

    runtime_values = dict(runtime_raw)
    for key, value in (runtime_overrides or {}).items():
        if value is not None:
            runtime_values[key] = value
    workers = _coalesce(
        runtime_values,
        "max_workers",
        "thread_pool_concurrency",
        "concurrency",
        default=1,
    )
    budget = _coalesce(
        runtime_values,
        "point_budget",
        "quantization_point_budget",
        "point_budget_per_batch",
        "batch_point_budget",
        default=1_000_000_000,
    )
    quota = _coalesce(
        runtime_values,
        "layer_point_quota",
        "per_layer_point_quota",
        "max_points_per_layer",
        default=1_000_000_000,
    )
    interval = _coalesce(
        runtime_values,
        "snapshot_interval",
        "snapshot_commit_interval",
        "commit_interval_batches",
        default=1,
    )
    runtime_valid = True
    for name, value, minimum in (
        ("max_workers", workers, 1),
        ("point_budget", budget, 0),
        ("layer_point_quota", quota, 0),
        ("snapshot_interval", interval, 1),
    ):
        if not is_int(value) or value < minimum:
            issues.append(
                ConfigIssue(
                    f"{name} must be an integer >= {minimum}",
                    field=f"runtime.{name}",
                )
            )
            runtime_valid = False
    if not runtime_valid:
        workers, budget, quota, interval = 1, 0, 0, 1

    config = PipelineConfig(
        raw=raw,
        profiles=profiles,
        profile_order=tuple(profile_order),
        active_profile=active_profile,
        defaults=defaults,
        layer_overrides=layer_overrides,
        distillation=DistillationConfig(temperature, float(soft_weight)),
        tolerances=ToleranceConfig(float(tol_abs), float(tol_rel)),
        runtime=RuntimeConfig(int(workers), int(budget), int(quota), int(interval)),
        issues=issues,
        distillation_valid=distillation_valid,
        forced_profile=profile,
        forced_fractional_bits=fractional_bits,
        overrides=dict(overrides or {}),
    )
    config.config_hash = sha256_json(
        {
            "raw_config": config.raw,
            "overrides": config.overrides,
            "resolved_profiles": {name: profile_obj.as_dict() for name, profile_obj in profiles.items()},
        }
    )
    return config, issues


def validate_config_layer_profiles(
    config: PipelineConfig,
    layer_names: list[str],
) -> list[ConfigIssue]:
    """Validate layer override names that can be checked without a weight pack."""

    issues: list[ConfigIssue] = []
    for layer in sorted(set(layer_names)):
        try:
            config.resolve_layer(layer, 1, marker_group_size=1)
        except ConfigError as exc:
            # Divisibility with the placeholder element count is not meaningful.
            if "fractional_bits" in (exc.details.get("field") or "") or "target_bits" in (
                exc.details.get("field") or ""
            ):
                issues.append(ConfigIssue(str(exc), scope="layer", layer=layer))
    return issues


def quantize_max_for_bits(bits: int) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    return (1 << (bits - 1)) - 1


def format_effective_config_summary(config: PipelineConfig) -> dict[str, Any]:
    return {
        "active_profile": config.active_profile,
        "profiles": [config.profiles[name].as_dict() for name in config.profile_order],
        "defaults": {
            key: value
            for key, value in config.defaults.items()
            if key != "allowed_bit_widths"
        },
        "allowed_bit_widths": list(config.allowed_bit_widths),
        "distillation": config.distillation.as_dict(),
        "tolerances": config.tolerances.as_dict(),
        "runtime": config.runtime.as_dict(),
        "config_hash": config.config_hash,
    }


def snapshot_config_hash(
    raw_config: Mapping[str, Any],
    *,
    profile: str | None = None,
    fractional_bits: int | None = None,
) -> str:
    """Hash model-affecting configuration only.

    Concurrency, point budget, quota, and snapshot interval control scheduling
    rather than numerical output, so changing them must not make an unfinished
    snapshot stale.  They are still part of the full audit configuration hash.
    """

    semantic = copy.deepcopy(dict(raw_config))
    semantic.pop("runtime", None)
    semantic.pop("scheduler", None)
    return sha256_json(
        {
            "semantic_config": semantic,
            "forced_profile": profile,
            "forced_fractional_bits": fractional_bits,
        }
    )
