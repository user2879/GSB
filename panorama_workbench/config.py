from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .yaml_loader import load_yaml_file


ALLOWED_FIXED_BITS = frozenset({4, 8, 10, 12, 16, 20, 24, 32})


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Category:
    code: int
    name: str
    weight_milli: int = 1000

    @property
    def weight(self) -> float:
        return self.weight_milli / 1000.0


@dataclass(frozen=True)
class WorkbenchConfig:
    categories: tuple[Category, ...]
    window_size: int
    window_stride: int
    merge_iou: float
    min_instance_area: int
    fixed_bits: int
    quant_levels: tuple[int, ...]
    quant_level: int
    max_workers: int
    pixel_budget: int
    max_instances_per_image: int
    cache_budget: int
    snapshot_interval: int
    min_semantic_accuracy: float = 0.0
    min_instance_quality: float = 0.0
    instance_match_iou: float = 0.5

    @property
    def category_codes(self) -> tuple[int, ...]:
        return tuple(category.code for category in self.categories)

    @property
    def category_by_code(self) -> dict[int, Category]:
        return {category.code: category for category in self.categories}

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "cache_budget": self.cache_budget,
            "categories": [
                {
                    "code": category.code,
                    "name": category.name,
                    "weight_milli": category.weight_milli,
                }
                for category in sorted(self.categories, key=lambda item: item.code)
            ],
            "fixed_bits": self.fixed_bits,
            "instance_match_iou": self.instance_match_iou,
            "max_instances_per_image": self.max_instances_per_image,
            "max_workers": self.max_workers,
            "merge_iou": self.merge_iou,
            "min_instance_area": self.min_instance_area,
            "min_instance_quality": self.min_instance_quality,
            "min_semantic_accuracy": self.min_semantic_accuracy,
            "pixel_budget": self.pixel_budget,
            "quant_level": self.quant_level,
            "quant_levels": list(self.quant_levels),
            "snapshot_interval": self.snapshot_interval,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
        }

    @property
    def hash(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def inference_hash(self) -> str:
        payload = {
            "algorithm": "deterministic-tile-v1",
            "fixed_bits": self.fixed_bits,
            "quant_level": self.quant_level,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
        }
        return _hash_payload(payload)

    @property
    def category_hash(self) -> str:
        payload = [
            {
                "code": category.code,
                "name": category.name,
                "weight_milli": category.weight_milli,
            }
            for category in sorted(self.categories, key=lambda item: item.code)
        ]
        return _hash_payload(payload)

    @property
    def merge_hash(self) -> str:
        payload = {
            "max_instances_per_image": self.max_instances_per_image,
            "merge_iou": self.merge_iou,
            "min_instance_area": self.min_instance_area,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
        }
        return _hash_payload(payload)

    def with_updates(self, **updates: Any) -> "WorkbenchConfig":
        return replace(self, **updates)

    def as_report(self) -> dict[str, Any]:
        value = self.canonical_dict()
        value["hash"] = self.hash
        value["inference_hash"] = self.inference_hash
        value["category_hash"] = self.category_hash
        value["merge_hash"] = self.merge_hash
        return value


def load_config(path: str | Path) -> WorkbenchConfig:
    source = Path(path)
    try:
        payload = load_yaml_file(source)
    except Exception as exc:
        raise ConfigError(f"cannot parse YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be a mapping")
    return parse_config(payload)


def parse_config(payload: dict[str, Any]) -> WorkbenchConfig:
    categories_raw = payload.get(
        "categories",
        payload.get("semantic_categories", payload.get("class_set")),
    )
    if isinstance(categories_raw, dict):
        normalized_categories: list[dict[str, Any]] = []
        for key, value in categories_raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", str(key))
                if "code" not in item and "category_code" not in item:
                    try:
                        item["code"] = int(key)
                    except (TypeError, ValueError):
                        raise ConfigError(
                            "mapping-style categories require numeric keys or explicit codes"
                        )
                normalized_categories.append(item)
            else:
                try:
                    code = int(key)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        "mapping-style categories require numeric keys"
                    ) from exc
                normalized_categories.append(
                    {"code": code, "name": str(key), "weight": value}
                )
        categories_raw = normalized_categories
    if not isinstance(categories_raw, list) or not categories_raw:
        raise ConfigError("categories must be a non-empty list")

    categories: list[Category] = []
    for raw in categories_raw:
        if not isinstance(raw, dict):
            raise ConfigError("each category must be a mapping")
        code = _required_int(raw, "code", "category_code")
        if code < 0:
            raise ConfigError("category codes must be non-negative")
        name = raw.get("name", str(code))
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("category name must be a non-empty string")
        weight = raw.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ConfigError("category weight must be numeric")
        weight_milli = int(round(float(weight) * 1000))
        if weight_milli <= 0:
            raise ConfigError("category weight must be positive")
        categories.append(Category(code=code, name=name.strip(), weight_milli=weight_milli))

    if len({category.code for category in categories}) != len(categories):
        raise ConfigError("category codes must be unique")
    if len({category.name for category in categories}) != len(categories):
        raise ConfigError("category names must be unique")

    window_raw = payload.get("window", payload.get("sliding_window", {}))
    if isinstance(window_raw, int):
        window_raw = {"size": window_raw, "stride": window_raw}
    if isinstance(window_raw, (list, tuple)) and len(window_raw) == 2:
        window_raw = {"size": window_raw[0], "stride": window_raw[1]}
    if window_raw is None:
        window_raw = {}
    if not isinstance(window_raw, dict):
        raise ConfigError("window must be a mapping")
    window_size = _positive_int(
        payload.get(
            "window_size",
            payload.get("tile_size", window_raw.get("size")),
        ),
        "window size",
    )
    window_stride = _positive_int(
        payload.get(
            "window_stride",
            payload.get("tile_stride", window_raw.get("stride")),
        ),
        "window stride",
    )
    if window_stride > window_size:
        raise ConfigError("window stride must not exceed window size")

    merge_raw = payload.get("merge", payload.get("merge_config", {}))
    if merge_raw is None:
        merge_raw = {}
    if not isinstance(merge_raw, dict):
        raise ConfigError("merge must be a mapping")
    merge_iou = _probability(
        payload.get(
            "merge_iou",
            payload.get(
                "merge_iou_threshold",
                payload.get(
                    "iou_threshold",
                    merge_raw.get("iou", merge_raw.get("threshold")),
                ),
            ),
        ),
        "merge IoU",
    )
    min_area = _positive_int(
        payload.get(
            "min_instance_area",
            payload.get(
                "min_area",
                merge_raw.get(
                    "min_instance_area",
                    merge_raw.get("min_area"),
                ),
            ),
        ),
        "minimum instance area",
    )

    quantization_raw = payload.get(
        "quantization", payload.get("fixed_point", {})
    )
    if quantization_raw is None:
        quantization_raw = {}
    if not isinstance(quantization_raw, dict):
        raise ConfigError("quantization must be a mapping")
    fixed_bits = _coerce_int(
        payload.get(
            "fixed_bits",
            payload.get(
                "fixed_point_bits",
                quantization_raw.get(
                    "fixed_bits",
                    quantization_raw.get(
                        "bits", quantization_raw.get("fixed_point_bits")
                    ),
                ),
            ),
        ),
        "fixed bits",
    )
    if fixed_bits not in ALLOWED_FIXED_BITS:
        raise ConfigError(
            f"fixed bits must be one of {sorted(ALLOWED_FIXED_BITS)}"
        )
    quant_levels_raw = payload.get(
        "quant_levels",
        payload.get(
            "quantization_levels",
            quantization_raw.get("levels", quantization_raw.get("steps")),
        ),
    )
    if not isinstance(quant_levels_raw, (list, tuple)) or not quant_levels_raw:
        raise ConfigError("quant levels must be a non-empty list")
    quant_levels: list[int] = []
    for value in quant_levels_raw:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError("quant levels must be positive integers")
        quant_levels.append(value)
    if len(set(quant_levels)) != len(quant_levels):
        raise ConfigError("quant levels must be unique")
    quant_level = _coerce_int(
        payload.get(
            "quant_level",
            payload.get(
                "quantization_level",
                quantization_raw.get("level", quantization_raw.get("active")),
            ),
        ),
        "quant level",
    )
    if quant_level not in quant_levels:
        raise ConfigError("active quant level must be in quant_levels")

    scheduling_raw = payload.get(
        "scheduling", payload.get("scheduler", {})
    )
    if scheduling_raw is None:
        scheduling_raw = {}
    if not isinstance(scheduling_raw, dict):
        raise ConfigError("scheduling must be a mapping")
    max_workers = _positive_int(
        payload.get(
            "max_workers",
            payload.get(
                "thread_pool_max_workers",
                scheduling_raw.get(
                    "max_workers", scheduling_raw.get("threads")
                ),
            ),
        ),
        "max workers",
    )
    pixel_budget = _positive_int(
        payload.get(
            "pixel_budget",
            payload.get(
                "batch_pixel_budget",
                scheduling_raw.get(
                    "pixel_budget", scheduling_raw.get("batch_pixel_budget")
                ),
            ),
        ),
        "pixel budget",
    )
    max_instances = _positive_int(
        payload.get(
            "max_instances_per_image",
            payload.get(
                "max_instances",
                scheduling_raw.get(
                    "max_instances_per_image", scheduling_raw.get("max_instances")
                ),
            ),
        ),
        "max instances per image",
    )
    cache_budget = _positive_int(
        payload.get(
            "cache_budget",
            payload.get(
                "cache_resident_budget",
                scheduling_raw.get("cache_budget"),
            ),
        ),
        "cache budget",
    )
    snapshot_interval = _positive_int(
        payload.get(
            "snapshot_interval",
            payload.get(
                "snapshot_commit_interval",
                scheduling_raw.get("snapshot_interval"),
            ),
        ),
        "snapshot interval",
    )

    quality_raw = payload.get("quality", {})
    if quality_raw is None:
        quality_raw = {}
    if not isinstance(quality_raw, dict):
        raise ConfigError("quality must be a mapping")
    min_semantic = _probability(
        payload.get(
            "min_semantic_accuracy",
            payload.get(
                "semantic_accuracy_min",
                quality_raw.get(
                    "min_semantic_accuracy",
                    quality_raw.get("semantic_accuracy_min", 0.0),
                ),
            ),
        ),
        "minimum semantic accuracy",
    )
    min_quality = _probability(
        payload.get(
            "min_instance_quality",
            payload.get(
                "instance_quality_min",
                quality_raw.get(
                    "min_instance_quality",
                    quality_raw.get("instance_quality_min", 0.0),
                ),
            ),
        ),
        "minimum instance quality",
    )
    match_iou = _probability(
        payload.get(
            "instance_match_iou",
            payload.get(
                "match_iou",
                quality_raw.get(
                    "instance_match_iou", quality_raw.get("match_iou", 0.5)
                ),
            ),
        ),
        "instance match IoU",
    )

    config = WorkbenchConfig(
        categories=tuple(sorted(categories, key=lambda item: item.code)),
        window_size=window_size,
        window_stride=window_stride,
        merge_iou=merge_iou,
        min_instance_area=min_area,
        fixed_bits=fixed_bits,
        quant_levels=tuple(quant_levels),
        quant_level=quant_level,
        max_workers=max_workers,
        pixel_budget=pixel_budget,
        max_instances_per_image=max_instances,
        cache_budget=cache_budget,
        snapshot_interval=snapshot_interval,
        min_semantic_accuracy=min_semantic,
        min_instance_quality=min_quality,
        instance_match_iou=match_iou,
    )
    _validate_config(config)
    return config


def _validate_config(config: WorkbenchConfig) -> None:
    if not config.categories:
        raise ConfigError("at least one category is required")
    if config.quant_level not in config.quant_levels:
        raise ConfigError("active quant level must be listed")
    if config.window_stride > config.window_size:
        raise ConfigError("window stride must not exceed window size")
    if not 0.0 <= config.merge_iou <= 1.0:
        raise ConfigError("merge IoU must be in [0, 1]")
    for value, label in (
        (config.min_semantic_accuracy, "minimum semantic accuracy"),
        (config.min_instance_quality, "minimum instance quality"),
        (config.instance_match_iou, "instance match IoU"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{label} must be in [0, 1]")


def _required_int(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in payload and payload[key] is not None:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{key} must be an integer")
            return value
    raise ConfigError(f"missing integer field {keys[0]}")


def _coerce_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be in [0, 1]")
    return numeric


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
