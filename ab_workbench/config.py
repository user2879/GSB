"""Experiment configuration model and loader."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .fingerprint import fingerprint_json
from .fixedpoint import FixedPoint


@dataclass
class ExperimentConfig:
    experiment_id: str
    control_group: str
    treatment_groups: List[str]
    hash_seed: int
    num_buckets: int
    group_weights: Dict[str, int]
    primary_metric: str
    guardrail_metrics: List[str]
    metric_direction: str
    min_sample_size: int
    min_lift: str
    attribution_tolerance: str
    decimal_width: int
    quantization_step: Optional[str]
    concurrency: int
    batch_size: int
    snapshot_every_batches: int
    z_quantile: float = 1.959963984540054
    confidence_level: float = 0.95

    @property
    def group_order(self) -> List[str]:
        return [self.control_group, *self.treatment_groups]

    @property
    def all_metrics(self) -> List[str]:
        metrics = [self.primary_metric, *self.guardrail_metrics]
        ordered: List[str] = []
        for name in metrics:
            if name not in ordered:
                ordered.append(name)
        return ordered

    @property
    def fixedpoint(self) -> FixedPoint:
        return FixedPoint(self.decimal_width, self.quantization_step)

    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return config_from_dict(raw)


def config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    if "treatment_groups" in raw and "groups" not in raw:
        control = raw["control_group"]
        treatment = list(raw["treatment_groups"])
        groups = [control, *treatment]
        weights_raw = raw.get("group_weights", {g: 1 for g in groups})
    else:
        groups = list(raw["groups"])
        control = raw["control_group"]
        if control not in groups:
            raise ValueError("control_group must be present in groups")
        treatment = [g for g in groups if g != control]
        weights_raw = raw.get("group_weights", {g: 1 for g in groups})
    weights = {g: int(weights_raw.get(g, 1)) for g in groups}
    direction = raw.get("metric_direction", "higher_is_better")
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError("metric_direction must be higher_is_better or lower_is_better")
    width = int(raw["decimal_width"])
    quant = raw.get("quantization_step")
    FixedPoint(width, quant)  # validate early
    min_sample = int(raw["min_sample_size"])
    if min_sample < 0:
        raise ValueError("min_sample_size must be non-negative")
    concurrency = int(raw.get("concurrency", 1))
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    batch_size = int(raw.get("batch_size", 1000))
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return ExperimentConfig(
        experiment_id=raw.get("experiment_id", "exp-0"),
        control_group=control,
        treatment_groups=treatment,
        hash_seed=int(raw["hash_seed"]),
        num_buckets=int(raw["num_buckets"]),
        group_weights=weights,
        primary_metric=raw["primary_metric"],
        guardrail_metrics=list(raw.get("guardrail_metrics", [])),
        metric_direction=direction,
        min_sample_size=min_sample,
        min_lift=str(raw["min_lift"]),
        attribution_tolerance=str(raw.get("attribution_tolerance", "0.0001")),
        decimal_width=width,
        quantization_step=quant,
        concurrency=concurrency,
        batch_size=batch_size,
        snapshot_every_batches=int(raw.get("snapshot_every_batches", 1)),
        z_quantile=float(raw.get("z_quantile", 1.959963984540054)),
        confidence_level=float(raw.get("confidence_level", 0.95)),
    )
