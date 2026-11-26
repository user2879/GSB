"""Typed configuration plus enum/range validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .codes import Codes
from .diagnostics import Diagnostic, DiagnosticBag


QUANT_SCHEME_DIGITS = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BinningConfig:
    method: str = "uniform"
    bins: int = 16
    allowed_bins: tuple[int, ...] = (4, 8, 16, 32)
    value_max: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuantizationConfig:
    scheme: str = "q3"
    allowed_schemes: tuple[str, ...] = ("q1", "q2", "q3", "q4")
    fixed_decimal_bits: int = 12
    allowed_fixed_decimal_bits: tuple[int, ...] = (8, 12, 16)

    @property
    def decimal_digits(self) -> int:
        return QUANT_SCHEME_DIGITS[self.scheme]

    @property
    def rate_scale(self) -> int:
        return 10**self.decimal_digits

    @property
    def internal_scale(self) -> int:
        return 1 << self.fixed_decimal_bits

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decimal_digits"] = self.decimal_digits
        value["rate_scale"] = self.rate_scale
        value["internal_scale"] = self.internal_scale
        return value


@dataclass(frozen=True)
class FusionConfig:
    mode: str = "weighted_average"
    ctr_weight: int = 3
    cvr_weight: int = 2
    threshold: float = 0.4505

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 99173
    parameter_version: int = 12
    hidden_dim: int = 3
    fusion: FusionConfig = FusionConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "parameter_version": self.parameter_version,
            "hidden_dim": self.hidden_dim,
            "fusion": self.fusion.to_dict(),
        }


@dataclass(frozen=True)
class SchedulerConfig:
    max_workers: int = 4
    point_budget: int = 24
    request_quota: int = 4
    points_per_candidate: int = 1
    worker_delay_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisibilityConfig:
    delay_cutoff_ms: int = 15_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AppConfig:
    global_version: int
    seed: int
    sample_count: int
    request_count: int
    candidates_per_request: int
    dense_dim: int
    sparse_dim: int
    binning: BinningConfig
    quantization: QuantizationConfig
    model: ModelConfig
    scheduler: SchedulerConfig
    visibility: VisibilityConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "request_count": self.request_count,
            "candidates_per_request": self.candidates_per_request,
            "dense_dim": self.dense_dim,
            "sparse_dim": self.sparse_dim,
            "binning": self.binning.to_dict(),
            "quantization": self.quantization.to_dict(),
            "model": self.model.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "visibility": self.visibility.to_dict(),
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def parameter_fingerprint(self) -> str:
        return canonical_hash(
            {
                "model": self.model.to_dict(),
                "quantization": self.quantization.to_dict(),
                "binning": self.binning.to_dict(),
            }
        )


def _parse_section(
    data: Mapping[str, Any],
    allowed: set[str],
    section: str,
    bag: DiagnosticBag,
    source: str,
) -> Mapping[str, Any]:
    for key in data:
        if key not in allowed:
            bag.add(
                Codes.FIELD_UNKNOWN,
                f"unknown configuration field {section}.{key}",
                source=source,
                field_name=f"{section}.{key}",
            )
    return data


def _require_int(
    container: Mapping[str, Any],
    key: str,
    *,
    source: str,
    bag: DiagnosticBag,
    path: str,
    minimum: int = 0,
) -> int:
    if key not in container:
        bag.add(Codes.FIELD_MISSING, f"missing field {path}", source=source, field_name=path)
        return minimum
    value = container[key]
    if not _is_int(value):
        bag.add(
            Codes.TYPE_MISMATCH,
            f"{path} must be an integer",
            source=source,
            field_name=path,
        )
        return minimum
    if value < minimum:
        bag.add(
            Codes.VALUE_RANGE,
            f"{path} must be >= {minimum}",
            source=source,
            field_name=path,
        )
    return value


def validate_config_dict(
    data: Mapping[str, Any], *, source: str = "<config>"
) -> tuple[AppConfig | None, tuple[Diagnostic, ...]]:
    bag = DiagnosticBag()
    if not isinstance(data, Mapping):
        bag.add(Codes.TYPE_MISMATCH, "configuration root must be a mapping", source=source)
        return None, bag.items

    allowed_top = {
        "global_version",
        "seed",
        "sample_count",
        "request_count",
        "candidates_per_request",
        "dense_dim",
        "sparse_dim",
        "binning",
        "quantization",
        "model",
        "scheduler",
        "visibility",
    }
    _parse_section(data, allowed_top, "root", bag, source)

    global_version = _require_int(
        data, "global_version", source=source, bag=bag, path="global_version", minimum=1
    )
    seed = _require_int(data, "seed", source=source, bag=bag, path="seed")
    sample_count = _require_int(
        data, "sample_count", source=source, bag=bag, path="sample_count", minimum=1
    )
    request_count = _require_int(
        data, "request_count", source=source, bag=bag, path="request_count", minimum=1
    )
    candidates_per_request = _require_int(
        data,
        "candidates_per_request",
        source=source,
        bag=bag,
        path="candidates_per_request",
        minimum=1,
    )
    dense_dim = _require_int(
        data, "dense_dim", source=source, bag=bag, path="dense_dim", minimum=1
    )
    sparse_dim = _require_int(
        data, "sparse_dim", source=source, bag=bag, path="sparse_dim", minimum=1
    )

    bin_data = data.get("binning", {})
    if not isinstance(bin_data, Mapping):
        bag.add(Codes.TYPE_MISMATCH, "binning must be a mapping", source=source)
        bin_data = {}
    _parse_section(
        bin_data,
        {"method", "bins", "allowed_bins", "value_max"},
        "binning",
        bag,
        source,
    )
    method = bin_data.get("method")
    if method != "uniform":
        bag.add(
            Codes.VALUE_RANGE,
            "binning.method must be uniform",
            source=source,
            field_name="binning.method",
        )
    allowed_bins_raw = bin_data.get("allowed_bins", [])
    if not isinstance(allowed_bins_raw, list) or not all(
        _is_int(item) and item > 0 for item in allowed_bins_raw
    ):
        bag.add(
            Codes.TYPE_MISMATCH,
            "binning.allowed_bins must be a list of positive integers",
            source=source,
            field_name="binning.allowed_bins",
        )
        allowed_bins_raw = []
    allowed_bins = tuple(sorted(set(allowed_bins_raw)))
    bins = _require_int(
        bin_data, "bins", source=source, bag=bag, path="binning.bins", minimum=1
    )
    if bins not in allowed_bins:
        bag.add(
            Codes.VALUE_RANGE,
            f"binning.bins {bins} is outside allowed enum {list(allowed_bins)}",
            source=source,
            field_name="binning.bins",
        )
    value_max = _require_int(
        bin_data,
        "value_max",
        source=source,
        bag=bag,
        path="binning.value_max",
        minimum=1,
    )

    quant_data = data.get("quantization", {})
    if not isinstance(quant_data, Mapping):
        bag.add(Codes.TYPE_MISMATCH, "quantization must be a mapping", source=source)
        quant_data = {}
    _parse_section(
        quant_data,
        {
            "scheme",
            "allowed_schemes",
            "fixed_decimal_bits",
            "allowed_fixed_decimal_bits",
            "decimal_digits",
            "rate_scale",
            "internal_scale",
        },
        "quantization",
        bag,
        source,
    )
    scheme = quant_data.get("scheme")
    schemes_raw = quant_data.get("allowed_schemes", [])
    if not isinstance(schemes_raw, list) or not all(isinstance(x, str) for x in schemes_raw):
        bag.add(
            Codes.TYPE_MISMATCH,
            "quantization.allowed_schemes must be a list of strings",
            source=source,
            field_name="quantization.allowed_schemes",
        )
        schemes_raw = []
    if scheme not in schemes_raw or scheme not in QUANT_SCHEME_DIGITS:
        bag.add(
            Codes.VALUE_RANGE,
            f"quantization.scheme {scheme!r} is outside allowed enum {schemes_raw}",
            source=source,
            field_name="quantization.scheme",
        )
    bits = _require_int(
        quant_data,
        "fixed_decimal_bits",
        source=source,
        bag=bag,
        path="quantization.fixed_decimal_bits",
        minimum=1,
    )
    bits_raw = quant_data.get("allowed_fixed_decimal_bits", [])
    if not isinstance(bits_raw, list) or not all(
        _is_int(item) and item > 0 for item in bits_raw
    ):
        bag.add(
            Codes.TYPE_MISMATCH,
            "quantization.allowed_fixed_decimal_bits must be a list of positive integers",
            source=source,
            field_name="quantization.allowed_fixed_decimal_bits",
        )
        bits_raw = []
    bits_allowed = tuple(sorted(set(bits_raw)))
    if bits not in bits_allowed:
        bag.add(
            Codes.VALUE_RANGE,
            f"quantization.fixed_decimal_bits {bits} is outside allowed enum "
            f"{list(bits_allowed)}",
            source=source,
            field_name="quantization.fixed_decimal_bits",
        )

    model_data = data.get("model", {})
    if not isinstance(model_data, Mapping):
        bag.add(Codes.TYPE_MISMATCH, "model must be a mapping", source=source)
        model_data = {}
    _parse_section(
        model_data,
        {"seed", "parameter_version", "hidden_dim", "fusion"},
        "model",
        bag,
        source,
    )
    model_seed = _require_int(model_data, "seed", source=source, bag=bag, path="model.seed")
    parameter_version = _require_int(
        model_data,
        "parameter_version",
        source=source,
        bag=bag,
        path="model.parameter_version",
    )
    hidden_dim = _require_int(
        model_data,
        "hidden_dim",
        source=source,
        bag=bag,
        path="model.hidden_dim",
        minimum=1,
    )
    fusion_data = model_data.get("fusion", {})
    if not isinstance(fusion_data, Mapping):
        bag.add(Codes.TYPE_MISMATCH, "model.fusion must be a mapping", source=source)
        fusion_data = {}
    _parse_section(
        fusion_data,
        {"mode", "ctr_weight", "cvr_weight", "threshold"},
        "model.fusion",
        bag,
        source,
    )
    if fusion_data.get("mode") not in {"weighted_average", "weighted_sum"}:
        bag.add(
            Codes.VALUE_RANGE,
            "model.fusion.mode must be weighted_average or weighted_sum",
            source=source,
            field_name="model.fusion.mode",
        )
    ctr_weight = _require_int(
        fusion_data,
        "ctr_weight",
        source=source,
        bag=bag,
        path="model.fusion.ctr_weight",
    )
    cvr_weight = _require_int(
        fusion_data,
        "cvr_weight",
        source=source,
        bag=bag,
        path="model.fusion.cvr_weight",
    )
    if ctr_weight + cvr_weight <= 0:
        bag.add(
            Codes.VALUE_RANGE,
            "fusion weights must sum to a positive value",
            source=source,
            field_name="model.fusion",
        )
    threshold_raw = fusion_data.get("threshold")
    if not _is_number(threshold_raw):
        bag.add(
            Codes.TYPE_MISMATCH,
            "model.fusion.threshold must be numeric",
            source=source,
            field_name="model.fusion.threshold",
        )
        threshold = 0.0
    else:
        threshold = float(threshold_raw)
        if not 0.0 <= threshold <= 1.0:
            bag.add(
                Codes.VALUE_RANGE,
                "model.fusion.threshold must be in [0, 1]",
                source=source,
                field_name="model.fusion.threshold",
            )

    sched_data = data.get("scheduler", {})
    if not isinstance(sched_data, Mapping):
        bag.add(Codes.TYPE_MISMATCH, "scheduler must be a mapping", source=source)
        sched_data = {}
    _parse_section(
        sched_data,
        {
            "max_workers",
            "point_budget",
            "request_quota",
            "points_per_candidate",
            "worker_delay_ms",
        },
        "scheduler",
        bag,
        source,
    )
    max_workers = _require_int(
        sched_data,
        "max_workers",
        source=source,
        bag=bag,
        path="scheduler.max_workers",
        minimum=1,
    )
    point_budget = _require_int(
        sched_data, "point_budget", source=source, bag=bag, path="scheduler.point_budget"
    )
    request_quota = _require_int(
        sched_data,
        "request_quota",
        source=source,
        bag=bag,
        path="scheduler.request_quota",
    )
    points_per_candidate = _require_int(
        sched_data,
        "points_per_candidate",
        source=source,
        bag=bag,
        path="scheduler.points_per_candidate",
        minimum=1,
    )
    worker_delay_ms = _require_int(
        sched_data,
        "worker_delay_ms",
        source=source,
        bag=bag,
        path="scheduler.worker_delay_ms",
    )

    vis_data = data.get("visibility", {})
    if not isinstance(vis_data, Mapping):
        bag.add(Codes.TYPE_MISMATCH, "visibility must be a mapping", source=source)
        vis_data = {}
    _parse_section(vis_data, {"delay_cutoff_ms"}, "visibility", bag, source)
    delay_cutoff_ms = _require_int(
        vis_data,
        "delay_cutoff_ms",
        source=source,
        bag=bag,
        path="visibility.delay_cutoff_ms",
    )

    if bag.items:
        return None, bag.items

    return (
        AppConfig(
            global_version=global_version,
            seed=seed,
            sample_count=sample_count,
            request_count=request_count,
            candidates_per_request=candidates_per_request,
            dense_dim=dense_dim,
            sparse_dim=sparse_dim,
            binning=BinningConfig(
                method=method,
                bins=bins,
                allowed_bins=allowed_bins,
                value_max=value_max,
            ),
            quantization=QuantizationConfig(
                scheme=scheme,
                allowed_schemes=tuple(schemes_raw),
                fixed_decimal_bits=bits,
                allowed_fixed_decimal_bits=bits_allowed,
            ),
            model=ModelConfig(
                seed=model_seed,
                parameter_version=parameter_version,
                hidden_dim=hidden_dim,
                fusion=FusionConfig(
                    mode=fusion_data.get("mode"),
                    ctr_weight=ctr_weight,
                    cvr_weight=cvr_weight,
                    threshold=threshold,
                ),
            ),
            scheduler=SchedulerConfig(
                max_workers=max_workers,
                point_budget=point_budget,
                request_quota=request_quota,
                points_per_candidate=points_per_candidate,
                worker_delay_ms=worker_delay_ms,
            ),
            visibility=VisibilityConfig(delay_cutoff_ms=delay_cutoff_ms),
        ),
        bag.items,
    )


def load_config(path: str | Path) -> AppConfig:
    source = str(path)
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration: {exc}") from exc
    config, diagnostics = validate_config_dict(data or {}, source=source)
    if config is None:
        details = "; ".join(
            f"{item.code}:{item.field_name or ''}:{item.message}" for item in diagnostics
        )
        raise ValueError(f"invalid configuration: {details}")
    return config


def replace_quantization(
    config: AppConfig, *, scheme: str | None = None, fixed_decimal_bits: int | None = None
) -> AppConfig:
    """Return a validated copy used by quantization-sensitivity tests."""

    data = config.to_dict()
    if scheme is not None:
        data["quantization"]["scheme"] = scheme
    if fixed_decimal_bits is not None:
        data["quantization"]["fixed_decimal_bits"] = fixed_decimal_bits
    updated, diagnostics = validate_config_dict(data)
    if updated is None:
        raise ValueError("; ".join(f"{d.code}: {d.message}" for d in diagnostics))
    return updated

