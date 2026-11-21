"""Compression/deployment configuration loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import E_CONFIG_INVALID, Failure, failure
from .fingerprint import sha256_json, to_builtin


ALLOWED_BITS = (2, 4, 8, 16)
ALLOWED_SCHEMES = ("symmetric", "asymmetric")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_positive_int(value: Any) -> int | None:
    if not _is_int(value) or value <= 0:
        return None
    return int(value)


def _as_nonnegative_int(value: Any) -> int | None:
    if not _is_int(value) or value < 0:
        return None
    return int(value)


def _lookup(mapping: Mapping[str, Any], names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True)
class EffectiveLayerSettings:
    bits: int
    scheme: str
    group_size: int
    group_axis: int
    per_channel: bool
    scale_significant_digits: int
    full_precision: bool = False

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "scheme": self.scheme,
            "group_size": self.group_size,
            "group_axis": self.group_axis,
            "per_channel": self.per_channel,
            "scale_significant_digits": self.scale_significant_digits,
            "full_precision": self.full_precision,
        }


@dataclass
class CompressionConfig:
    schema_version: int
    default_bits: int
    default_scheme: str
    default_group_size: int
    default_group_axis: int
    default_per_channel: bool
    scale_significant_digits: int
    layer_overrides: dict[str, dict[str, Any]]
    full_precision_layers: tuple[str, ...]
    deploy_format_version: str
    artifact_name: str
    interaction_latency_budget_ms: int
    concurrency: int
    point_budget_per_batch: int
    snapshot_commit_interval: int
    stage_delay_ms: int
    distillation_temperature: float
    soft_label_weight: float
    correction_gain: float
    issues: list[Failure] = field(default_factory=list)
    invalid_layers: set[str] = field(default_factory=set)
    usable: bool = True

    def effective(self, layer_name: str) -> EffectiveLayerSettings:
        override = self.layer_overrides.get(layer_name, {})
        full_precision = layer_name in self.full_precision_layers
        return EffectiveLayerSettings(
            bits=int(override.get("bits", self.default_bits)),
            scheme=(
                "full_precision"
                if full_precision
                else str(override.get("scheme", self.default_scheme))
            ),
            group_size=int(override.get("group_size", self.default_group_size)),
            group_axis=int(override.get("group_axis", self.default_group_axis)),
            per_channel=bool(
                override.get("per_channel", self.default_per_channel)
            ),
            scale_significant_digits=self.scale_significant_digits,
            full_precision=full_precision,
        )

    def normalized(self) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        for name in sorted(self.layer_overrides):
            override = self.layer_overrides[name]
            layers[name] = {
                "bits": int(override.get("bits", self.default_bits)),
                "group_size": int(
                    override.get("group_size", self.default_group_size)
                ),
                "group_axis": int(
                    override.get("group_axis", self.default_group_axis)
                ),
                "scheme": str(override.get("scheme", self.default_scheme)),
                "per_channel": bool(
                    override.get("per_channel", self.default_per_channel)
                ),
            }
        return {
            "schema_version": self.schema_version,
            "quantization": {
                "default_bits": self.default_bits,
                "scheme": self.default_scheme,
                "group_size": self.default_group_size,
                "group_axis": self.default_group_axis,
                "per_channel_scaling": self.default_per_channel,
                "scale_significant_digits": self.scale_significant_digits,
            },
            "layers": layers,
            "full_precision_layers": list(self.full_precision_layers),
            "deployment": {
                "package_format_version": self.deploy_format_version,
                "artifact_name": self.artifact_name,
            },
            "runtime": {
                "interaction_latency_budget_ms": self.interaction_latency_budget_ms,
                "concurrency": self.concurrency,
                "point_budget_per_batch": self.point_budget_per_batch,
                "snapshot_commit_interval": self.snapshot_commit_interval,
                "stage_delay_ms": self.stage_delay_ms,
            },
            "distillation": {
                "temperature": self.distillation_temperature,
                "soft_label_weight": self.soft_label_weight,
                "correction_gain": self.correction_gain,
            },
        }

    @property
    def config_hash(self) -> str:
        return sha256_json(self.normalized())

    def deployment_config(self) -> dict[str, Any]:
        value = self.normalized()
        value.pop("runtime", None)
        return value

    @property
    def deployment_config_hash(self) -> str:
        """Hash only fields that can change package bytes or error artifacts."""

        return sha256_json(self.deployment_config())

    def referenced_config_fields(self, layer_name: str) -> list[str]:
        fields = [
            "quantization.default_bits",
            "quantization.scheme",
            "quantization.group_size",
            "quantization.group_axis",
            "quantization.per_channel_scaling",
            "quantization.scale_significant_digits",
            "full_precision_layers",
        ]
        if layer_name in self.layer_overrides:
            fields.extend(
                f"layers.{layer_name}.{key}"
                for key in sorted(self.layer_overrides[layer_name])
            )
        return fields


def _read_yaml(path: str | Path) -> tuple[dict[str, Any], Failure | None]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}, failure(
            E_CONFIG_INVALID,
            f"configuration file does not exist: {file_path}",
            detail={"path": str(file_path)},
        )
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, failure(
            E_CONFIG_INVALID,
            f"configuration file cannot be parsed: {exc}",
            detail={"path": str(file_path)},
        )
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return {}, failure(
            E_CONFIG_INVALID,
            "configuration root must be a YAML mapping",
            detail={"path": str(file_path)},
        )
    return raw, None


def _issue(message: str, *, layer: str | None = None, detail: Any = None) -> Failure:
    return failure(
        E_CONFIG_INVALID,
        message,
        layer=layer,
        detail={"value": to_builtin(detail)} if detail is not None else None,
    )


def load_config(path: str | Path) -> CompressionConfig:
    """Load YAML configuration, collecting all recoverable invalid fields.

    Nested sections are preferred, while flat aliases are accepted to make the
    component easy to embed in existing training/deployment pipelines.
    """

    raw, read_issue = _read_yaml(path)
    issues: list[Failure] = []
    if read_issue is not None:
        issues.append(read_issue)

    quant = raw.get("quantization", {})
    if quant is None:
        quant = {}
    if not isinstance(quant, dict):
        issues.append(_issue("quantization must be a mapping", detail=quant))
        quant = {}

    deployment = raw.get("deployment", {})
    if deployment is None:
        deployment = {}
    if not isinstance(deployment, dict):
        issues.append(_issue("deployment must be a mapping", detail=deployment))
        deployment = {}

    runtime = raw.get("runtime", {})
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, dict):
        issues.append(_issue("runtime must be a mapping", detail=runtime))
        runtime = {}

    distillation = raw.get("distillation", {})
    if distillation is None:
        distillation = {}
    if not isinstance(distillation, dict):
        issues.append(_issue("distillation must be a mapping", detail=distillation))
        distillation = {}

    default_bits = _lookup(
        quant,
        ("default_bits", "target_bits", "bits"),
        _lookup(raw, ("default_bits", "target_bits", "bits"), 8),
    )
    if not _is_int(default_bits) or int(default_bits) not in ALLOWED_BITS:
        issues.append(
            _issue(
                f"default target bits must be one of {list(ALLOWED_BITS)}",
                detail=default_bits,
            )
        )
        default_bits = 8

    default_scheme = str(
        _lookup(
            quant,
            ("scheme", "quant_scheme", "quantization_scheme"),
            _lookup(raw, ("scheme", "quant_scheme"), "symmetric"),
        )
    ).lower()
    if default_scheme not in ALLOWED_SCHEMES:
        issues.append(
            _issue(
                f"quantization scheme must be one of {list(ALLOWED_SCHEMES)}",
                detail=default_scheme,
            )
        )
        default_scheme = "symmetric"

    default_group_size = _lookup(
        quant,
        ("group_size", "granularity", "group_granularity"),
        _lookup(raw, ("group_size", "granularity"), 8),
    )
    parsed_group_size = _as_positive_int(default_group_size)
    if parsed_group_size is None:
        issues.append(_issue("group size must be a positive integer", detail=default_group_size))
        parsed_group_size = 1
    default_group_size = parsed_group_size

    default_group_axis = _lookup(
        quant, ("group_axis", "axis"), _lookup(raw, ("group_axis", "axis"), -1)
    )
    if not _is_int(default_group_axis):
        issues.append(_issue("group axis must be an integer", detail=default_group_axis))
        default_group_axis = -1
    default_group_axis = int(default_group_axis)

    default_per_channel = _lookup(
        quant,
        ("per_channel_scaling", "per_channel", "channel_scaling"),
        _lookup(raw, ("per_channel_scaling", "per_channel"), False),
    )
    if not isinstance(default_per_channel, bool):
        issues.append(_issue("per-channel scaling must be boolean", detail=default_per_channel))
        default_per_channel = False

    significant_digits = _lookup(
        quant,
        ("scale_significant_digits", "significant_digits"),
        _lookup(raw, ("scale_significant_digits", "significant_digits"), 8),
    )
    parsed_digits = _as_positive_int(significant_digits)
    if parsed_digits is None:
        issues.append(
            _issue("scale significant digits must be a positive integer", detail=significant_digits)
        )
        parsed_digits = 8
    significant_digits = parsed_digits

    layer_overrides: dict[str, dict[str, Any]] = {}
    invalid_layers: set[str] = set()
    layer_map = raw.get("layers", {})
    if layer_map is None:
        layer_map = {}
    if not isinstance(layer_map, dict):
        issues.append(_issue("layers must be a mapping", detail=layer_map))
        layer_map = {}
    for name, value in layer_map.items():
        if not isinstance(name, str) or not name:
            issues.append(_issue("layer override name must be a non-empty string", detail=name))
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            value = {"bits": value}
        if not isinstance(value, dict):
            issues.append(_issue("layer override must be a mapping or integer", layer=name, detail=value))
            invalid_layers.add(name)
            continue

        override: dict[str, Any] = {}
        bits = _lookup(value, ("bits", "target_bits", "bit_width"), default_bits)
        if not _is_int(bits) or int(bits) not in ALLOWED_BITS:
            issues.append(
                _issue(
                    f"layer target bits must be one of {list(ALLOWED_BITS)}",
                    layer=name,
                    detail=bits,
                )
            )
            invalid_layers.add(name)
        else:
            override["bits"] = int(bits)

        group_size = _lookup(
            value,
            ("group_size", "granularity", "group_granularity"),
            default_group_size,
        )
        parsed_layer_group = _as_positive_int(group_size)
        if parsed_layer_group is None:
            issues.append(
                _issue("layer group size must be a positive integer", layer=name, detail=group_size)
            )
            invalid_layers.add(name)
        else:
            override["group_size"] = parsed_layer_group

        group_axis = _lookup(value, ("group_axis", "axis"), default_group_axis)
        if not _is_int(group_axis):
            issues.append(_issue("layer group axis must be an integer", layer=name, detail=group_axis))
            invalid_layers.add(name)
        else:
            override["group_axis"] = int(group_axis)

        scheme = str(
            _lookup(
                value,
                ("scheme", "quant_scheme", "quantization_scheme"),
                default_scheme,
            )
        ).lower()
        if scheme not in ALLOWED_SCHEMES:
            issues.append(
                _issue(
                    f"layer quantization scheme must be one of {list(ALLOWED_SCHEMES)}",
                    layer=name,
                    detail=scheme,
                )
            )
            invalid_layers.add(name)
        else:
            override["scheme"] = scheme

        per_channel = _lookup(
            value, ("per_channel", "per_channel_scaling"), default_per_channel
        )
        if not isinstance(per_channel, bool):
            issues.append(
                _issue("layer per-channel scaling must be boolean", layer=name, detail=per_channel)
            )
            invalid_layers.add(name)
        else:
            override["per_channel"] = per_channel
        layer_overrides[name] = override

    whitelist = raw.get(
        "full_precision_layers",
        _lookup(
            raw,
            (
                "full_precision",
                "full_precision_whitelist",
                "keep_full_precision_layers",
            ),
            [],
        ),
    )
    if whitelist is None:
        whitelist = []
    if not isinstance(whitelist, list) or any(not isinstance(item, str) for item in whitelist):
        issues.append(_issue("full-precision whitelist must be a list of layer names", detail=whitelist))
        whitelist = []
    full_precision_layers = tuple(sorted(set(whitelist)))

    deploy_format_version = _lookup(
        deployment,
        ("package_format_version", "format_version"),
        _lookup(raw, ("package_format_version", "deployment_package_format_version"), "1.0.0"),
    )
    if isinstance(deploy_format_version, (int, float)):
        deploy_format_version = str(deploy_format_version)
    if not isinstance(deploy_format_version, str) or not deploy_format_version.strip():
        issues.append(_issue("deployment package format version must be a non-empty string", detail=deploy_format_version))
        deploy_format_version = "1.0.0"

    artifact_name = _lookup(deployment, ("artifact_name", "name"), raw.get("artifact_name", "model"))
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        issues.append(_issue("deployment artifact name must be a non-empty string", detail=artifact_name))
        artifact_name = "model"

    interaction_budget = _lookup(
        runtime,
        ("interaction_latency_budget_ms", "latency_budget_ms"),
        _lookup(raw, ("interaction_latency_budget_ms", "latency_budget_ms"), 100),
    )
    if not _is_int(interaction_budget) or int(interaction_budget) < 0:
        issues.append(_issue("interaction latency budget must be a non-negative integer", detail=interaction_budget))
        interaction_budget = 100
    interaction_budget = int(interaction_budget)

    concurrency = _lookup(runtime, ("concurrency", "max_concurrency"), _lookup(raw, ("concurrency",), 4))
    parsed_concurrency = _as_positive_int(concurrency)
    if parsed_concurrency is None:
        issues.append(_issue("concurrency must be a positive integer", detail=concurrency))
        parsed_concurrency = 1
    concurrency = parsed_concurrency

    point_budget = _lookup(
        runtime,
        (
            "point_budget_per_batch",
            "single_batch_point_budget",
            "single_batch_budget",
            "batch_point_budget",
        ),
        _lookup(raw, ("point_budget_per_batch", "batch_point_budget"), 65536),
    )
    parsed_budget = _as_positive_int(point_budget)
    if parsed_budget is None:
        issues.append(_issue("single-batch point budget must be a positive integer", detail=point_budget))
        parsed_budget = 65536
    point_budget = parsed_budget

    commit_interval = _lookup(
        runtime,
        ("snapshot_commit_interval", "snapshot_interval"),
        _lookup(raw, ("snapshot_commit_interval",), 1),
    )
    parsed_interval = _as_nonnegative_int(commit_interval)
    if parsed_interval is None:
        issues.append(_issue("snapshot commit interval must be a non-negative integer", detail=commit_interval))
        parsed_interval = 1
    commit_interval = parsed_interval

    stage_delay = _lookup(runtime, ("stage_delay_ms", "simulated_stage_delay_ms"), 0)
    if not _is_int(stage_delay) or int(stage_delay) < 0:
        issues.append(_issue("stage delay must be a non-negative integer", detail=stage_delay))
        stage_delay = 0
    stage_delay = int(stage_delay)

    temperature = _lookup(
        distillation,
        ("temperature", "distillation_temperature"),
        _lookup(raw, ("distillation_temperature",), 2.0),
    )
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        issues.append(_issue("distillation temperature must be a positive number", detail=temperature))
        temperature = 2.0
    temperature = float(temperature)
    if temperature <= 0 or not (temperature < float("inf")):
        issues.append(_issue("distillation temperature must be finite and positive", detail=temperature))
        temperature = 2.0

    soft_label_weight = _lookup(
        distillation,
        ("soft_label_weight", "weight", "distillation_weight"),
        _lookup(raw, ("soft_label_weight",), 0.25),
    )
    if isinstance(soft_label_weight, bool) or not isinstance(soft_label_weight, (int, float)):
        issues.append(_issue("soft-label weight must be a number in [0, 1]", detail=soft_label_weight))
        soft_label_weight = 0.25
    soft_label_weight = float(soft_label_weight)
    if not (0.0 <= soft_label_weight <= 1.0):
        issues.append(_issue("soft-label weight must be a number in [0, 1]", detail=soft_label_weight))
        soft_label_weight = 0.25

    correction_gain = _lookup(
        distillation,
        ("correction_gain", "gain"),
        _lookup(raw, ("correction_gain",), 1.0),
    )
    if isinstance(correction_gain, bool) or not isinstance(correction_gain, (int, float)):
        issues.append(_issue("correction gain must be a finite non-negative number", detail=correction_gain))
        correction_gain = 1.0
    correction_gain = float(correction_gain)
    if correction_gain < 0 or not (correction_gain < float("inf")):
        issues.append(_issue("correction gain must be finite and non-negative", detail=correction_gain))
        correction_gain = 1.0

    schema_version = raw.get("schema_version", 1)
    if not _is_int(schema_version) or schema_version < 1:
        issues.append(_issue("schema_version must be a positive integer", detail=schema_version))
        schema_version = 1

    usable = not any(
        issue.code == E_CONFIG_INVALID
        and issue.layer is None
        and "default target bits" in issue.message
        for issue in issues
    )
    return CompressionConfig(
        schema_version=int(schema_version),
        default_bits=int(default_bits),
        default_scheme=default_scheme,
        default_group_size=int(default_group_size),
        default_group_axis=int(default_group_axis),
        default_per_channel=bool(default_per_channel),
        scale_significant_digits=int(significant_digits),
        layer_overrides=layer_overrides,
        full_precision_layers=full_precision_layers,
        deploy_format_version=str(deploy_format_version),
        artifact_name=str(artifact_name),
        interaction_latency_budget_ms=interaction_budget,
        concurrency=concurrency,
        point_budget_per_batch=point_budget,
        snapshot_commit_interval=commit_interval,
        stage_delay_ms=stage_delay,
        distillation_temperature=temperature,
        soft_label_weight=soft_label_weight,
        correction_gain=correction_gain,
        issues=issues,
        invalid_layers=invalid_layers,
        usable=usable,
    )


def validate_config_against_layers(
    config: CompressionConfig, layers: Mapping[str, Any]
) -> list[Failure]:
    """Validate config references and grouping against concrete layer shapes."""

    issues: list[Failure] = list(config.issues)
    known = set(layers)
    for name in config.full_precision_layers:
        if name not in known:
            issues.append(
                _issue(
                    f"full-precision whitelist layer does not exist: {name}",
                    layer=name,
                )
            )
    for name in sorted(config.layer_overrides):
        if name not in known:
            issues.append(_issue(f"layer override does not exist in weight package: {name}", layer=name))

    for name in sorted(known):
        if name in config.invalid_layers:
            continue
        settings = config.effective(name)
        shape = tuple(layers[name].shape)
        axis = settings.group_axis
        if not shape:
            issues.append(_issue("layer shape cannot be scalar", layer=name))
            continue
        try:
            resolved_axis = axis if axis >= 0 else len(shape) + axis
        except (TypeError, ValueError):
            resolved_axis = -1
        if resolved_axis < 0 or resolved_axis >= len(shape):
            issues.append(
                _issue(
                    f"group axis {axis} is out of range for shape {list(shape)}",
                    layer=name,
                )
            )
            config.invalid_layers.add(name)
            continue
        dimension = shape[resolved_axis]
        if dimension % settings.group_size != 0:
            issues.append(
                _issue(
                    f"group size {settings.group_size} does not divide dimension "
                    f"{dimension} at axis {axis}",
                    layer=name,
                    detail={"shape": list(shape), "axis": axis},
                )
            )
            config.invalid_layers.add(name)
    return issues
