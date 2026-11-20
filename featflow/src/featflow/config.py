"""Pipeline configuration (YAML) loading and parameter validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import yaml

from .common import (
    CODE_DEF_PARAM,
    FixedPoint,
    Failure,
)


@dataclass
class PipelineConfig:
    thread_concurrency: int
    global_event_budget: int          # total feature point budget; -1 = unlimited
    feature_point_quota: int          # per-feature quota; -1 = unlimited
    cache_block_limit: int            # max resident blocks; 0 disables cache
    snapshot_commit_every: int        # commit every N processed points/batches
    fixed_point_scale: int
    quant_step: str
    behaviors: list[str]
    context_fields: list[str]
    business_codes: dict[str, str]
    default_empty_value: str
    fp: FixedPoint = field(init=False)

    @staticmethod
    def load(path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return PipelineConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "PipelineConfig":
        failures: list[Failure] = []

        def need_int(key: str, default: int, min_value: int) -> int:
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
                failures.append(Failure("load", key, CODE_DEF_PARAM,
                                        f"{key} must be int >= {min_value}"))
                return default
            return value

        concurrency = need_int("thread_concurrency", 1, 1)
        global_budget = need_int("global_event_budget", -1, -1)
        quota = need_int("feature_point_quota", -1, -1)
        cache_limit = need_int("cache_block_limit", 0, 0)
        commit_every = need_int("snapshot_commit_every", 50, 1)
        scale = need_int("fixed_point_scale", 4, 0)
        if scale > 18:
            failures.append(Failure("load", "fixed_point_scale", CODE_DEF_PARAM,
                                    "fixed_point_scale must be <= 18"))
            scale = min(scale, 18)

        step = str(raw.get("quant_step", "1"))
        try:
            fp = FixedPoint.build(scale, step)
        except ValueError as exc:
            failures.append(Failure("load", "quant_step", CODE_DEF_PARAM, str(exc)))
            fp = FixedPoint.build(scale, "1")
            step = "1"

        behaviors = raw.get("behaviors")
        if not isinstance(behaviors, list) or not behaviors or \
                any(not isinstance(b, str) for b in behaviors):
            failures.append(Failure("load", "behaviors", CODE_DEF_PARAM,
                                    "behaviors must be a non-empty string list"))
            behaviors = ["click", "buy", "cart"]

        context_fields = raw.get("context_fields")
        if not isinstance(context_fields, list) or \
                any(not isinstance(c, str) for c in context_fields):
            failures.append(Failure("load", "context_fields", CODE_DEF_PARAM,
                                    "context_fields must be a string list"))
            context_fields = ["price", "duration"]

        codes = raw.get("business_codes", {})
        if not isinstance(codes, dict):
            failures.append(Failure("load", "business_codes", CODE_DEF_PARAM,
                                    "business_codes must be a mapping"))
            codes = {}

        empty_default = str(raw.get("default_empty_value", "0"))
        try:
            Decimal(empty_default)
        except Exception:
            failures.append(Failure("load", "default_empty_value", CODE_DEF_PARAM,
                                    "default_empty_value must be numeric"))
            empty_default = "0"

        cfg = PipelineConfig(
            thread_concurrency=concurrency,
            global_event_budget=global_budget,
            feature_point_quota=quota,
            cache_block_limit=cache_limit,
            snapshot_commit_every=commit_every,
            fixed_point_scale=scale,
            quant_step=step,
            behaviors=list(behaviors),
            context_fields=list(context_fields),
            business_codes=dict(codes),
            default_empty_value=empty_default,
        )
        cfg.fp = fp
        cfg.load_failures = failures  # type: ignore[attr-defined]
        return cfg
