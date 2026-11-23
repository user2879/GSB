"""Configuration loading and strict business validation."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Iterable

from .canonical import sha256_json
from .errors import E_CONFIG_INVALID, Failure
from .fixedpoint import quantize_units
from .normalization_rules import normalize_token
from .yaml_subset import YamlSubsetError, load as load_yaml


REQUIRED_FIELDS = (
    "window_count",
    "signature_dimension",
    "similarity_threshold",
    "bucket_bits",
    "fixed_decimal_places",
    "quantization_levels",
    "signature_batch_budget",
    "max_workers",
    "snapshot_commit_interval",
    "source_labels",
    "vocabulary",
)

ALGORITHM_FIELDS = (
    "window_count",
    "signature_dimension",
    "hash_seed",
    "schema_version",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
        and len(set(value)) == len(value)
    )


@dataclass(frozen=True)
class FingerprintConfig:
    schema_version: int
    window_count: int
    signature_dimension: int
    similarity_threshold: float
    bucket_bits: int
    fixed_decimal_places: int
    quantization_levels: int
    signature_batch_budget: int
    max_workers: int
    snapshot_commit_interval: int
    source_labels: tuple[str, ...]
    vocabulary: tuple[str, ...]
    hash_seed: str

    @cached_property
    def scale(self) -> int:
        return 10**self.fixed_decimal_places

    @cached_property
    def threshold_units(self) -> int:
        raw = int(round(self.similarity_threshold * self.scale))
        return quantize_units(raw, self.scale, self.quantization_levels)

    @cached_property
    def normalized_vocabulary(self) -> frozenset[str]:
        return frozenset(normalize_token(item) for item in self.vocabulary)

    @cached_property
    def algorithm_payload(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in ALGORITHM_FIELDS}

    @cached_property
    def algorithm_fingerprint(self) -> str:
        return sha256_json(self.algorithm_payload)

    @cached_property
    def bucket_fingerprint(self) -> str:
        return sha256_json(
            {
                "signature_algorithm_fingerprint": self.algorithm_fingerprint,
                "bucket_bits": self.bucket_bits,
                "bucket_version": 1,
            }
        )

    @cached_property
    def config_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "window_count": self.window_count,
            "signature_dimension": self.signature_dimension,
            "similarity_threshold": self.similarity_threshold,
            "bucket_bits": self.bucket_bits,
            "fixed_decimal_places": self.fixed_decimal_places,
            "quantization_levels": self.quantization_levels,
            "signature_batch_budget": self.signature_batch_budget,
            "max_workers": self.max_workers,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "source_labels": list(self.source_labels),
            "vocabulary": list(self.vocabulary),
            "hash_seed": self.hash_seed,
        }

    @cached_property
    def config_hash(self) -> str:
        return sha256_json(self.config_payload)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.config_payload)


def _fail(field: str, message: str, *, index: int | None = None) -> Failure:
    detail = {"field": field}
    if index is not None:
        detail["operation_index"] = index
    return Failure(
        E_CONFIG_INVALID,
        "config_validation",
        message,
        item_id=field,
        detail=detail,
    )


def validate_config_payload(raw: Any, *, source: str = "config.yaml") -> tuple[FingerprintConfig | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        return None, [Failure(E_CONFIG_INVALID, "config_validation", "configuration must be a mapping", source=source)]

    for field in REQUIRED_FIELDS:
        if field not in raw:
            failures.append(_fail(field, f"missing required field {field!r}"))

    schema_version = raw.get("schema_version", 1)
    if not _is_int(schema_version) or schema_version != 1:
        failures.append(_fail("schema_version", "schema_version must be integer 1"))

    window_count = raw.get("window_count")
    if not _is_int(window_count) or window_count <= 0:
        failures.append(_fail("window_count", "window_count must be a positive integer"))

    signature_dimension = raw.get("signature_dimension")
    if (
        not _is_int(signature_dimension)
        or signature_dimension <= 0
        or signature_dimension % 2 != 0
    ):
        failures.append(_fail("signature_dimension", "signature_dimension must be a positive even integer"))

    threshold = raw.get("similarity_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        failures.append(_fail("similarity_threshold", "similarity_threshold must be a finite number in [0, 1]"))
    elif not math.isfinite(float(threshold)) or not (0.0 <= float(threshold) <= 1.0):
        failures.append(_fail("similarity_threshold", "similarity_threshold must be a finite number in [0, 1]"))

    bucket_bits = raw.get("bucket_bits")
    if not _is_int(bucket_bits) or bucket_bits <= 0:
        failures.append(_fail("bucket_bits", "bucket_bits must be a positive integer"))
    elif _is_int(signature_dimension) and bucket_bits > signature_dimension:
        failures.append(_fail("bucket_bits", "bucket_bits must not exceed signature_dimension"))

    decimal_places = raw.get("fixed_decimal_places")
    if not _is_int(decimal_places) or not (1 <= decimal_places <= 12):
        failures.append(_fail("fixed_decimal_places", "fixed_decimal_places must be an integer in [1, 12]"))

    levels = raw.get("quantization_levels")
    if not _is_int(levels) or levels <= 0:
        failures.append(_fail("quantization_levels", "quantization_levels must be a positive integer"))
    elif _is_int(decimal_places) and 1 <= decimal_places <= 12 and levels > 10**decimal_places:
        failures.append(_fail("quantization_levels", "quantization_levels must not exceed 10**fixed_decimal_places"))

    budget = raw.get("signature_batch_budget")
    if not _is_int(budget) or budget <= 0:
        failures.append(_fail("signature_batch_budget", "signature_batch_budget must be a positive integer"))

    workers = raw.get("max_workers")
    if not _is_int(workers) or workers <= 0:
        failures.append(_fail("max_workers", "max_workers must be a positive integer"))

    commit_interval = raw.get("snapshot_commit_interval")
    if not _is_int(commit_interval) or commit_interval <= 0:
        failures.append(_fail("snapshot_commit_interval", "snapshot_commit_interval must be a positive integer"))

    source_labels_raw = raw.get("source_labels")
    if not _nonempty_string_list(source_labels_raw):
        failures.append(_fail("source_labels", "source_labels must be a non-empty unique string list"))

    vocabulary_raw = raw.get("vocabulary")
    if not _nonempty_string_list(vocabulary_raw):
        failures.append(_fail("vocabulary", "vocabulary must be a non-empty unique string list"))
    else:
        normalized = [normalize_token(item) for item in vocabulary_raw]
        if any(not item for item in normalized):
            failures.append(_fail("vocabulary", "vocabulary contains a token that normalizes to empty"))
        elif len(set(normalized)) != len(normalized):
            failures.append(_fail("vocabulary", "vocabulary contains duplicate normalized tokens"))

    hash_seed = raw.get("hash_seed", "near-dup-v1")
    if not isinstance(hash_seed, str) or not hash_seed:
        failures.append(_fail("hash_seed", "hash_seed must be a non-empty string"))

    if failures:
        return None, [replace(item, source=source) for item in failures]

    config = FingerprintConfig(
        schema_version=schema_version,
        window_count=window_count,
        signature_dimension=signature_dimension,
        similarity_threshold=float(threshold),
        bucket_bits=bucket_bits,
        fixed_decimal_places=decimal_places,
        quantization_levels=levels,
        signature_batch_budget=budget,
        max_workers=workers,
        snapshot_commit_interval=commit_interval,
        source_labels=tuple(source_labels_raw),
        vocabulary=tuple(vocabulary_raw),
        hash_seed=hash_seed,
    )
    return config, []


def load_config(path: str) -> tuple[FingerprintConfig | None, list[Failure]]:
    try:
        raw = load_yaml(path)
    except (OSError, YamlSubsetError) as exc:
        return None, [
            Failure(E_CONFIG_INVALID, "config_validation", f"cannot load configuration: {exc}", source=path)
        ]
    return validate_config_payload(raw, source=path)


def config_from_overrides(
    base: FingerprintConfig,
    overrides: Iterable[tuple[str, Any]],
    *,
    source: str = "edit_plan",
) -> tuple[FingerprintConfig | None, list[Failure]]:
    raw = base.to_dict()
    for field, value in overrides:
        raw[field] = value
    return validate_config_payload(raw, source=source)
