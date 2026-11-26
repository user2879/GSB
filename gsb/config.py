"""Loading and validation of deterministic pipeline configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from gsb.constants import (
    ALLOWED_FIXED_BITS,
    ALLOWED_QUANT_LEVELS,
    E_CONFIG_INVALID,
    QUANT_LEVELS,
    SCHEMA_VERSION,
)
from gsb.errors import GsbError, ValidationIssue
from gsb.util import sha256_obj


@dataclass(frozen=True)
class LayoutConfig:
    width: int = 320
    height: int = 256
    pixel_mode: str = "L"


@dataclass(frozen=True)
class WindowConfig:
    width: int = 32
    height: int = 32
    stride_x: int = 16
    stride_y: int = 16


@dataclass(frozen=True)
class QuantizationConfig:
    level: str = "Q8"
    fixed_bits: int = 8


@dataclass(frozen=True)
class ThresholdConfig:
    ink: int = 180
    candidate_score: int = 1
    cc_merge_gap: int = 7
    component_tone_tolerance: int = 64
    component_min_area: int = 1
    component_max_span: int = 36
    split_min_gap: int = 1
    confidence_score_weight: int = 4
    confidence_residual_weight: int = 1
    tolerance_fixed: int = 4000


@dataclass(frozen=True)
class RecognitionConfig:
    template_version: str = "bitmap-v1"
    template_width: int = 12
    template_height: int = 12
    variant_weights: tuple[int, ...] = (64, 2, 1)
    max_workers: int = 4
    reading_conflict_priority: tuple[str, ...] = (
        "CROSS_COLUMN",
        "MISORDER",
        "MISSING",
    )


@dataclass(frozen=True)
class BudgetConfig:
    max_template_points: int = 40000


@dataclass(frozen=True)
class RunConfig:
    schema_version: int = SCHEMA_VERSION
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layout": asdict(self.layout),
            "window": asdict(self.window),
            "quantization": asdict(self.quantization),
            "thresholds": asdict(self.thresholds),
            "recognition": {
                **asdict(self.recognition),
                "variant_weights": list(self.recognition.variant_weights),
                "reading_conflict_priority": list(
                    self.recognition.reading_conflict_priority
                ),
            },
            "budget": asdict(self.budget),
        }

    @property
    def config_hash(self) -> str:
        return sha256_obj(self.to_dict())

    @property
    def fixed_bits(self) -> int:
        return self.quantization.fixed_bits

    def with_quantization(self, level: str) -> "RunConfig":
        bits = QUANT_LEVELS.get(level)
        if bits is None:
            raise GsbError(E_CONFIG_INVALID, f"unsupported quantization level: {level}")
        return replace(self, quantization=QuantizationConfig(level=level, fixed_bits=bits))

    def with_window(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        stride_x: int | None = None,
        stride_y: int | None = None,
    ) -> "RunConfig":
        window = self.window
        updated = WindowConfig(
            width=window.width if width is None else width,
            height=window.height if height is None else height,
            stride_x=window.stride_x if stride_x is None else stride_x,
            stride_y=window.stride_y if stride_y is None else stride_y,
        )
        return replace(self, window=updated)

    def with_threshold(self, name: str, value: int) -> "RunConfig":
        if not hasattr(self.thresholds, name):
            raise GsbError(E_CONFIG_INVALID, f"unknown threshold: {name}")
        return replace(self, thresholds=replace(self.thresholds, **{name: value}))

    def with_workers(self, max_workers: int) -> "RunConfig":
        return replace(
            self,
            recognition=replace(self.recognition, max_workers=max_workers),
        )

    def with_budget(self, max_template_points: int) -> "RunConfig":
        return replace(
            self,
            budget=BudgetConfig(max_template_points=max_template_points),
        )

    def stage_parameter_fingerprint(self, stage: str) -> str:
        payload = self.to_dict()
        if stage == "candidate":
            selected = {
                "layout": payload["layout"],
                "window": payload["window"],
                "thresholds": {
                    "ink": payload["thresholds"]["ink"],
                    "candidate_score": payload["thresholds"]["candidate_score"],
                },
                "quantization": payload["quantization"],
                "template_version": payload["recognition"]["template_version"],
            }
        elif stage == "component":
            selected = {
                "layout": payload["layout"],
                "window": payload["window"],
                "thresholds": payload["thresholds"],
                "quantization": payload["quantization"],
                "template_version": payload["recognition"]["template_version"],
            }
        elif stage == "keypoint":
            selected = {
                "layout": payload["layout"],
                "thresholds": {
                    "ink": payload["thresholds"]["ink"],
                    "component_min_area": payload["thresholds"]["component_min_area"],
                },
                "quantization": payload["quantization"],
                "template_version": payload["recognition"]["template_version"],
            }
        elif stage == "recognition":
            selected = {
                "quantization": payload["quantization"],
                "thresholds": {
                    "ink": payload["thresholds"]["ink"],
                    "confidence_score_weight": payload["thresholds"][
                        "confidence_score_weight"
                    ],
                    "confidence_residual_weight": payload["thresholds"][
                        "confidence_residual_weight"
                    ],
                },
                "recognition": payload["recognition"],
            }
        elif stage == "reading":
            selected = {
                "quantization": payload["quantization"],
                "recognition": {
                    "template_version": payload["recognition"]["template_version"],
                    "reading_conflict_priority": payload["recognition"][
                        "reading_conflict_priority"
                    ],
                },
            }
        else:
            selected = payload
        return sha256_obj(selected)


def _require_mapping(value: Any, label: str, issues: list[ValidationIssue]) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message=f"{label} must be a mapping",
                source=label,
            )
        )
        return {}
    return value


def _require_int(
    value: Any,
    label: str,
    issues: list[ValidationIssue],
    *,
    minimum: int | None = None,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message=f"{label} must be an integer",
                source=label,
            )
        )
        return None
    if minimum is not None and value < minimum:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message=f"{label} must be >= {minimum}",
                source=label,
            )
        )
        return None
    return value


def parse_config(raw: dict[str, Any], source: str = "<memory>") -> tuple[RunConfig | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    root = _require_mapping(raw, "config", issues)
    schema_version = root.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message=f"schema_version must be {SCHEMA_VERSION}",
                source=source,
            )
        )

    layout_raw = _require_mapping(root.get("layout", {}), "layout", issues)
    width = _require_int(layout_raw.get("width"), "layout.width", issues, minimum=1)
    height = _require_int(layout_raw.get("height"), "layout.height", issues, minimum=1)
    pixel_mode = layout_raw.get("pixel_mode", "L")
    if pixel_mode != "L":
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="layout.pixel_mode must be L",
                source=source,
            )
        )

    window_raw = _require_mapping(root.get("window", {}), "window", issues)
    window_width = _require_int(window_raw.get("width"), "window.width", issues, minimum=1)
    window_height = _require_int(window_raw.get("height"), "window.height", issues, minimum=1)
    stride_x = _require_int(window_raw.get("stride_x"), "window.stride_x", issues, minimum=1)
    stride_y = _require_int(window_raw.get("stride_y"), "window.stride_y", issues, minimum=1)
    if width and window_width and width % window_width != 0:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="window.width must divide layout.width",
                source=source,
            )
        )
    if height and window_height and height % window_height != 0:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="window.height must divide layout.height",
                source=source,
            )
        )

    quant_raw = _require_mapping(root.get("quantization", {}), "quantization", issues)
    level = quant_raw.get("level")
    fixed_bits = quant_raw.get("fixed_bits")
    if level not in ALLOWED_QUANT_LEVELS:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message=f"quantization.level must be one of {list(ALLOWED_QUANT_LEVELS)}",
                source=source,
            )
        )
    if fixed_bits not in ALLOWED_FIXED_BITS:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message=f"quantization.fixed_bits must be one of {list(ALLOWED_FIXED_BITS)}",
                source=source,
            )
        )
    if level in QUANT_LEVELS and fixed_bits != QUANT_LEVELS[level]:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="quantization.fixed_bits must match the selected level",
                source=source,
            )
        )

    thresholds_raw = _require_mapping(root.get("thresholds", {}), "thresholds", issues)
    threshold_values: dict[str, int] = {}
    threshold_defaults = ThresholdConfig()
    for name in (
        "ink",
        "candidate_score",
        "cc_merge_gap",
        "component_tone_tolerance",
        "component_min_area",
        "component_max_span",
        "split_min_gap",
        "confidence_score_weight",
        "confidence_residual_weight",
        "tolerance_fixed",
    ):
        value = _require_int(
            thresholds_raw.get(name, getattr(threshold_defaults, name)),
            f"thresholds.{name}",
            issues,
            minimum=0,
        )
        if value is not None:
            threshold_values[name] = value
    if threshold_values.get("ink", 1) > 255:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="thresholds.ink must be <= 255",
                source=source,
            )
        )

    recognition_raw = _require_mapping(root.get("recognition", {}), "recognition", issues)
    template_version = recognition_raw.get("template_version")
    if not isinstance(template_version, str) or not template_version:
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="recognition.template_version must be declared",
                source=source,
            )
        )
    template_width = _require_int(
        recognition_raw.get("template_width", 12),
        "recognition.template_width",
        issues,
        minimum=1,
    )
    template_height = _require_int(
        recognition_raw.get("template_height", 12),
        "recognition.template_height",
        issues,
        minimum=1,
    )
    weights_raw = recognition_raw.get("variant_weights", [64, 2, 1])
    weights: tuple[int, ...]
    if (
        not isinstance(weights_raw, list)
        or not weights_raw
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in weights_raw)
    ):
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="recognition.variant_weights must be a non-empty positive integer list",
                source=source,
            )
        )
        weights = (64, 2, 1)
    else:
        weights = tuple(weights_raw)
    max_workers = _require_int(
        recognition_raw.get("max_workers", 4),
        "recognition.max_workers",
        issues,
        minimum=1,
    )

    priorities_raw = recognition_raw.get(
        "reading_conflict_priority",
        ["CROSS_COLUMN", "MISORDER", "MISSING"],
    )
    if (
        not isinstance(priorities_raw, list)
        or len(priorities_raw) != 3
        or set(priorities_raw) != {"CROSS_COLUMN", "MISORDER", "MISSING"}
    ):
        issues.append(
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="recognition.reading_conflict_priority must list each conflict type once",
                source=source,
            )
        )
        priorities = ("CROSS_COLUMN", "MISORDER", "MISSING")
    else:
        priorities = tuple(priorities_raw)

    budget_raw = _require_mapping(root.get("budget", {}), "budget", issues)
    max_template_points = _require_int(
        budget_raw.get("max_template_points", 40000),
        "budget.max_template_points",
        issues,
        minimum=1,
    )

    if issues:
        return None, issues

    assert width is not None
    assert height is not None
    assert window_width is not None
    assert window_height is not None
    assert stride_x is not None
    assert stride_y is not None
    assert fixed_bits is not None
    assert template_width is not None
    assert template_height is not None
    assert max_workers is not None
    assert max_template_points is not None

    config = RunConfig(
        schema_version=SCHEMA_VERSION,
        layout=LayoutConfig(width=width, height=height, pixel_mode="L"),
        window=WindowConfig(
            width=window_width,
            height=window_height,
            stride_x=stride_x,
            stride_y=stride_y,
        ),
        quantization=QuantizationConfig(level=level, fixed_bits=fixed_bits),
        thresholds=ThresholdConfig(**threshold_values),
        recognition=RecognitionConfig(
            template_version=template_version,
            template_width=template_width,
            template_height=template_height,
            variant_weights=weights,
            max_workers=max_workers,
            reading_conflict_priority=priorities,
        ),
        budget=BudgetConfig(max_template_points=max_template_points),
    )
    return config, []


def load_config(path: str | Path) -> RunConfig:
    source = str(path)
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GsbError(E_CONFIG_INVALID, f"config not found: {path}", source=source) from exc
    except yaml.YAMLError as exc:
        raise GsbError(E_CONFIG_INVALID, f"invalid YAML: {exc}", source=source) from exc
    config, issues = parse_config(raw, source)
    if issues or config is None:
        first = issues[0]
        raise GsbError(
            first.code,
            first.message,
            source=source,
            details={"issues": [item.to_dict() for item in issues]},
        )
    return config


def validate_config_file(path: str | Path) -> list[ValidationIssue]:
    try:
        load_config(path)
    except GsbError as exc:
        nested = exc.details.get("issues")
        if isinstance(nested, list):
            return [ValidationIssue(**item) for item in nested]
        return [exc.issue()]
    return []
