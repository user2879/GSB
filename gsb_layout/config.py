"""Configuration parsing and canonical fingerprints."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigError

MATCH_METRICS = ("l1", "hamming", "iou", "cosine")
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    binarization_threshold: int = 160
    row_projection_window: int = 5
    char_min_height: int = 18
    char_max_height: int = 34
    split_penalty: int = 10
    match_metric: str = "l1"
    template_weights: tuple[int, ...] = (4, 3, 2)
    fixed_bits: int = 10
    quant_levels: int = 16
    confidence_tolerance: float = 0.62
    thread_count: int = 4
    match_point_budget: int = 1_000_000
    snapshot_interval: int = 1
    expected_width: int | None = None
    expected_height: int | None = None
    border_residual_tolerance: int = 5
    confidence_low_q: int = 0
    confidence_high_q: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def fixed_scale(self) -> int:
        return 1 << self.fixed_bits

    @property
    def weights(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.template_weights)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA_VERSION,
            "binarization_threshold": self.binarization_threshold,
            "row_projection_window": self.row_projection_window,
            "char_min_height": self.char_min_height,
            "char_max_height": self.char_max_height,
            "split_penalty": self.split_penalty,
            "match_metric": self.match_metric,
            "template_weights": list(self.template_weights),
            "fixed_bits": self.fixed_bits,
            "quant_levels": self.quant_levels,
            "confidence_tolerance": self.confidence_tolerance,
            "thread_count": self.thread_count,
            "match_point_budget": self.match_point_budget,
            "snapshot_interval": self.snapshot_interval,
            "expected_width": self.expected_width,
            "expected_height": self.expected_height,
            "border_residual_tolerance": self.border_residual_tolerance,
            "confidence_low_q": self.confidence_low_q,
            "confidence_high_q": self.confidence_high_q,
            "extras": self.extras,
        }

    def fingerprint(self) -> str:
        return canonical_hash(self.canonical_dict())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_int(raw: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} 必须为整数")
    if value < minimum:
        raise ConfigError(f"{key} 必须 >= {minimum}")
    return value


def _optional_positive_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} 必须为正整数或 null")
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"配置文件不存在: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取配置 JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("配置根节点必须为对象")
    return parse_config(raw)


_ALIASES = {
    "binarization_threshold": ("threshold", "binarize_threshold"),
    "row_projection_window": ("projection_window", "row_window"),
    "char_min_height": ("min_char_height", "min_height"),
    "char_max_height": ("max_char_height", "max_height"),
    "split_penalty": ("split_penalty_weight",),
    "match_metric": ("metric",),
    "template_weights": ("template_fusion_weights", "weights"),
    "fixed_bits": ("fixed_point_bits",),
    "quant_levels": ("quantization_levels",),
    "thread_count": ("threads", "max_workers"),
    "match_point_budget": ("batch_match_point_budget", "point_budget"),
    "snapshot_interval": ("commit_interval",),
    "expected_width": ("width",),
    "expected_height": ("height",),
    "border_residual_tolerance": ("residual_tolerance",),
}


def _normalize_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    for canonical, aliases in _ALIASES.items():
        if canonical in normalized:
            continue
        for alias in aliases:
            if alias in normalized:
                normalized[canonical] = normalized[alias]
                break
    if "expected_width" not in normalized or "expected_height" not in normalized:
        size = normalized.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            normalized.setdefault("expected_width", size[0])
            normalized.setdefault("expected_height", size[1])
    return normalized


def parse_config(raw: dict[str, Any]) -> Config:
    raw = _normalize_aliases(raw)
    threshold = _require_int(raw, "binarization_threshold", 160, 0)
    if threshold > 255:
        raise ConfigError("binarization_threshold 必须 <= 255")
    window = _require_int(raw, "row_projection_window", 5, 1)
    min_h = _require_int(raw, "char_min_height", 18, 1)
    max_h = _require_int(raw, "char_max_height", 34, 1)
    if max_h < min_h:
        raise ConfigError("char_max_height 必须 >= char_min_height")
    split_penalty = _require_int(raw, "split_penalty", 10, 0)
    metric = raw.get("match_metric", "l1")
    if metric not in MATCH_METRICS:
        raise ConfigError(f"match_metric 必须属于 {MATCH_METRICS}")
    weights_raw = raw.get("template_weights", [4, 3, 2])
    if not isinstance(weights_raw, list) or not weights_raw:
        raise ConfigError("template_weights 必须为非空整数数组")
    weights: list[int] = []
    for weight in weights_raw:
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise ConfigError("template_weights 中每项必须为正整数")
        weights.append(weight)
    if len(weights) > 8:
        raise ConfigError("template_weights 最多支持 8 个内置模板变体")
    fixed_bits = _require_int(raw, "fixed_bits", 10, 4)
    if fixed_bits > 24:
        raise ConfigError("fixed_bits 必须 <= 24")
    quant_levels = _require_int(raw, "quant_levels", 16, 2)
    if quant_levels > 256:
        raise ConfigError("quant_levels 必须 <= 256")
    tolerance_raw = raw.get("confidence_tolerance", 0.62)
    if isinstance(tolerance_raw, bool) or not isinstance(tolerance_raw, (int, float)):
        raise ConfigError("confidence_tolerance 必须为数字")
    tolerance = float(tolerance_raw)
    if not 0.0 <= tolerance <= 1.0:
        raise ConfigError("confidence_tolerance 必须在 [0,1] 内")
    thread_count = _require_int(raw, "thread_count", 4, 1)
    budget = _require_int(raw, "match_point_budget", 1_000_000, 0)
    snapshot_interval = _require_int(raw, "snapshot_interval", 1, 1)
    width = _optional_positive_int(raw, "expected_width")
    height = _optional_positive_int(raw, "expected_height")
    residual_tolerance = _require_int(raw, "border_residual_tolerance", 5, 0)
    low_q = _require_int(raw, "confidence_low_q", 0, 0)
    high_q = raw.get("confidence_high_q")
    if high_q is not None:
        if isinstance(high_q, bool) or not isinstance(high_q, int) or high_q < low_q:
            raise ConfigError("confidence_high_q 必须为 >= confidence_low_q 的整数或 null")
    known = {
        "binarization_threshold", "row_projection_window", "char_min_height",
        "char_max_height", "split_penalty", "match_metric", "template_weights",
        "fixed_bits", "quant_levels", "confidence_tolerance", "thread_count",
        "match_point_budget", "snapshot_interval", "expected_width",
        "expected_height", "border_residual_tolerance", "confidence_low_q",
        "confidence_high_q", "schema", "extras",
        "threshold", "binarize_threshold", "projection_window", "row_window",
        "min_char_height", "min_height", "max_char_height", "max_height",
        "split_penalty_weight", "metric", "template_fusion_weights", "weights",
        "fixed_point_bits", "quantization_levels", "threads", "max_workers",
        "batch_match_point_budget", "point_budget", "commit_interval",
        "width", "height", "image_size", "residual_tolerance",
    }
    inherited_extras = raw.get("extras", {})
    if inherited_extras is not None and not isinstance(inherited_extras, dict):
        raise ConfigError("extras 必须为对象")
    extras = {k: v for k, v in raw.items() if k not in known}
    extras.update(inherited_extras or {})
    return Config(
        binarization_threshold=threshold,
        row_projection_window=window,
        char_min_height=min_h,
        char_max_height=max_h,
        split_penalty=split_penalty,
        match_metric=metric,
        template_weights=tuple(weights),
        fixed_bits=fixed_bits,
        quant_levels=quant_levels,
        confidence_tolerance=tolerance,
        thread_count=thread_count,
        match_point_budget=budget,
        snapshot_interval=snapshot_interval,
        expected_width=width,
        expected_height=height,
        border_residual_tolerance=residual_tolerance,
        confidence_low_q=low_q,
        confidence_high_q=high_q,
        extras=extras,
    )


def with_overrides(config: Config, **changes: Any) -> Config:
    """Return a validated config with selected fields replaced."""
    raw = config.canonical_dict()
    raw.update(changes)
    return parse_config(raw)
