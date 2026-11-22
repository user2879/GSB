"""Content-fingerprint cache keys and precise invalidation auditing."""

from __future__ import annotations

from dataclasses import dataclass

from .features import feature_params_fingerprint_input
from .hashing import cache_key, fingerprint_json, fingerprint_strings, scale_tier
from .classifiers import hyper_fingerprint_input

VALIDATION = "validation"
FEATURES = "features"
SCORING = "scoring"
DECISION = "decision"
METRICS = "metrics"

STAGE_ORDER = (VALIDATION, FEATURES, SCORING, DECISION, METRICS)


def validation_artifact_key(sample_name: str, image_fingerprint: str, annotation_fingerprint: str, labels):
    return cache_key(
        "validation",
        sample_name,
        image_fingerprint,
        annotation_fingerprint,
        fingerprint_strings(labels),
    )


def feature_artifact_key(sample_name: str, image_fingerprint: str, config) -> str:
    return cache_key(
        "feature",
        sample_name,
        image_fingerprint,
        feature_params_fingerprint_input(config),
    )


def model_artifact_key(
    classifier: str,
    label: str,
    referenced_samples: list[str],
    sample_fingerprints: dict[str, str],
    config,
) -> str:
    ordered_samples = sorted(referenced_samples)
    matrix_fingerprint = fingerprint_json(
        {name: sample_fingerprints[name] for name in ordered_samples}
    )
    tier = scale_tier(len(ordered_samples))
    return cache_key(
        "model",
        classifier,
        label,
        hyper_fingerprint_input(classifier, config),
        matrix_fingerprint,
        tier,
    )


def decision_artifact_key(model_key: str, label: str, threshold: float, quant_level: str, frac_bits: int):
    return cache_key(
        "decision",
        model_key,
        label,
        {"threshold": threshold, "quant_level": quant_level, "frac_bits": frac_bits},
    )


def metric_artifact_key(decision_key: str, classifier: str, label: str):
    return cache_key("metric", decision_key, classifier, label)


@dataclass
class AuditEntry:
    stage: str
    name: str
    action: str  # "reuse" | "recompute"
    reason: str
    cache_key: str
    sort_key: str
    referenced_samples: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "name": self.name,
            "action": self.action,
            "reason": self.reason,
            "cache_key": self.cache_key,
            "sort_key": self.sort_key,
            "referenced_samples": list(self.referenced_samples),
        }


def sort_audit(entries: list[AuditEntry]) -> list[AuditEntry]:
    """Stable audit ordering shared by serial and concurrent runs."""
    stage_rank = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    return sorted(
        entries,
        key=lambda entry: (
            stage_rank.get(entry.stage, len(STAGE_ORDER)),
            str(entry.sort_key),
            str(entry.name),
            str(entry.cache_key),
        ),
    )
