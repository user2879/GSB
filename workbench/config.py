"""Verification config loading and validation."""

from pathlib import Path

import yaml

from .fingerprint import sha256_json

DETECTION_PARAM_KEYS = (
    "categories",
    "colors",
    "color_tolerance",
    "min_area",
    "aspect_ratio_range",
    "merge_gap",
    "shape_fill",
)
ARBITRATION_PARAM_KEYS = ("count_tolerance", "arbitration_priority")
ALLOWED_RULES = ("detection", "description", "label_majority", "label_a")


class ConfigError(ValueError):
    pass


class Config:
    def __init__(self, raw: dict):
        if not isinstance(raw, dict):
            raise ConfigError("config root must be a mapping")
        required = (
            ["version", "image_size", "categories", "colors"]
            + [k for k in DETECTION_PARAM_KEYS if k not in ("categories", "colors")]
            + list(ARBITRATION_PARAM_KEYS)
            + ["max_concurrency"]
        )
        for key in required:
            if key not in raw:
                raise ConfigError(f"missing config key: {key}")
        if not raw["categories"]:
            raise ConfigError("categories enum must be non-empty")
        if not raw["colors"]:
            raise ConfigError("colors enum must be non-empty")
        for rule in raw["arbitration_priority"]:
            if rule not in ALLOWED_RULES:
                raise ConfigError(f"unknown arbitration rule: {rule}")
        if int(raw["max_concurrency"]) < 1:
            raise ConfigError("max_concurrency must be >= 1")
        if int(raw["count_tolerance"]) < 0:
            raise ConfigError("count_tolerance must be >= 0")
        self.raw = raw
        self.version = raw["version"]
        self.image_size = tuple(int(v) for v in raw["image_size"])
        self.categories = list(raw["categories"])
        self.colors = {k: list(v) for k, v in raw["colors"].items()}
        self.color_tolerance = float(raw["color_tolerance"])
        self.min_area = int(raw["min_area"])
        self.aspect_ratio_range = tuple(float(v) for v in raw["aspect_ratio_range"])
        self.merge_gap = int(raw["merge_gap"])
        self.shape_fill = dict(raw["shape_fill"])
        self.count_tolerance = int(raw["count_tolerance"])
        self.arbitration_priority = list(raw["arbitration_priority"])
        self.max_concurrency = int(raw["max_concurrency"])

    @classmethod
    def load(cls, path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(raw)

    def fingerprint(self) -> str:
        return sha256_json(self.raw)

    def detection_params_fingerprint(self) -> str:
        return sha256_json({k: self.raw[k] for k in DETECTION_PARAM_KEYS})

    def arbitration_params_fingerprint(self) -> str:
        return sha256_json({k: self.raw[k] for k in ARBITRATION_PARAM_KEYS})
