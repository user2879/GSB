"""Configuration loading, strict validation and stable configuration hashes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .codes import E_CONFIG_INVALID
from .models import Failure
from .util import sha256_json

REPRESENTATIVE_RULE_ALIASES = {
    "source_weight_then_doc_id": "source_weight_desc_doc_id_asc",
    "source_weight_desc_doc_id_asc": "source_weight_desc_doc_id_asc",
    "weight_then_doc_id": "source_weight_desc_doc_id_asc",
    "weight_desc_doc_id_asc": "source_weight_desc_doc_id_asc",
    "max_weight_min_doc_id": "source_weight_desc_doc_id_asc",
}

INJECTION_FAULTS = {
    "empty_text",
    "non_utf8",
    "unknown_source",
    "duplicate_doc_id",
}


class ConfigValidationError(ValueError):
    def __init__(self, failures: list[Failure]):
        self.failures = failures
        super().__init__("; ".join(failure.message for failure in failures))


@dataclass(frozen=True)
class BalanceConfig:
    source_weights: dict[str, float]
    target_sample_count: int
    sampling_salt: str

    def hash_value(self) -> dict[str, Any]:
        return {
            "source_weights": self.source_weights,
            "target_sample_count": self.target_sample_count,
            "sampling_salt": self.sampling_salt,
            "algorithm": "weighted-largest-remainder-splitmix64-v1",
        }

    def hash(self) -> str:
        return sha256_json(self.hash_value())


@dataclass(frozen=True)
class DedupConfig:
    shingle_length: int
    signature_dim: int
    jaccard_threshold: float
    representative_rule: str = "source_weight_desc_doc_id_asc"

    def hash_value(self) -> dict[str, Any]:
        return {
            "shingle_length": self.shingle_length,
            "signature_dim": self.signature_dim,
            "jaccard_threshold": self.jaccard_threshold,
            "representative_rule": self.representative_rule,
            "algorithm": "minhash-affine-v1",
        }

    def hash(self) -> str:
        return sha256_json(self.hash_value())


@dataclass(frozen=True)
class SchedulerConfig:
    max_workers: int
    global_shingle_budget: int
    source_quotas: dict[str, int]
    batch_size: int
    snapshot_interval_batches: int

    def hash_value(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "global_shingle_budget": self.global_shingle_budget,
            "source_quotas": self.source_quotas,
            "batch_size": self.batch_size,
            "snapshot_interval_batches": self.snapshot_interval_batches,
        }

    def hash(self) -> str:
        return sha256_json(self.hash_value())


@dataclass(frozen=True)
class InjectionItem:
    injection_id: str
    source: str
    doc_id: str
    fault: str
    business_code_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "source": self.source,
            "doc_id": self.doc_id,
            "fault": self.fault,
            "business_code_hint": self.business_code_hint,
            "metadata": self.metadata,
        }


def _failure(message: str, file: str | None = None) -> Failure:
    return Failure(
        source="",
        doc_id="",
        stage="config",
        code=E_CONFIG_INVALID,
        message=message,
        file=file,
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ConfigValidationError([_failure(f"configuration file not found: {path}", str(path))])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigValidationError([_failure(f"invalid JSON configuration {path}: {exc}", str(path))])


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_balance_config(path: str | Path) -> BalanceConfig:
    path = Path(path)
    raw = _load_json(path)
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        raise ConfigValidationError([_failure("balance config must be a JSON object", str(path))])

    weights = raw.get("source_weights", raw.get("weights"))
    target = raw.get("target_sample_count", raw.get("target"))
    salt = raw.get("sampling_salt", raw.get("salt"))
    cleaned_weights: dict[str, float] = {}

    if not isinstance(weights, dict) or not weights:
        failures.append(_failure("source_weights must be a non-empty object", str(path)))
    else:
        for source, weight in weights.items():
            if not isinstance(source, str) or not source:
                failures.append(_failure("source weight keys must be non-empty strings", str(path)))
                continue
            if not _is_number(weight) or float(weight) <= 0:
                failures.append(_failure(f"weight for source {source!r} must be a positive number", str(path)))
                continue
            cleaned_weights[source] = float(weight)

    if not _is_int(target) or target <= 0:
        failures.append(_failure("target_sample_count must be a positive integer", str(path)))
    if not isinstance(salt, str) or not salt:
        failures.append(_failure("sampling_salt must be a non-empty string", str(path)))
    if failures:
        raise ConfigValidationError(failures)
    return BalanceConfig(
        source_weights=dict(sorted(cleaned_weights.items())),
        target_sample_count=int(target),
        sampling_salt=str(salt),
    )


def load_dedup_config(path: str | Path) -> DedupConfig:
    path = Path(path)
    raw = _load_json(path)
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        raise ConfigValidationError([_failure("dedup config must be a JSON object", str(path))])

    shingle = raw.get("shingle_length")
    dimension = raw.get("signature_dim", raw.get("signature_dimension"))
    threshold = raw.get("jaccard_threshold", raw.get("threshold"))
    rule_raw = raw.get("representative_rule", "source_weight_desc_doc_id_asc")

    if not _is_int(shingle) or shingle <= 0:
        failures.append(_failure("shingle_length must be a positive integer", str(path)))
    if not _is_int(dimension) or dimension <= 0:
        failures.append(_failure("signature_dim must be a positive integer", str(path)))
    if not _is_number(threshold) or not (0.0 <= float(threshold) <= 1.0):
        failures.append(_failure("jaccard_threshold must be a number in [0, 1]", str(path)))
    if not isinstance(rule_raw, str) or rule_raw not in REPRESENTATIVE_RULE_ALIASES:
        failures.append(
            _failure(
                "representative_rule must be one of "
                + ", ".join(sorted(REPRESENTATIVE_RULE_ALIASES)),
                str(path),
            )
        )
    if failures:
        raise ConfigValidationError(failures)
    return DedupConfig(
        shingle_length=int(shingle),
        signature_dim=int(dimension),
        jaccard_threshold=float(threshold),
        representative_rule=REPRESENTATIVE_RULE_ALIASES[str(rule_raw)],
    )


def load_scheduler_config(path: str | Path) -> SchedulerConfig:
    path = Path(path)
    raw = _load_json(path)
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        raise ConfigValidationError([_failure("scheduler config must be a JSON object", str(path))])

    workers = raw.get("max_workers", raw.get("max_concurrency"))
    budget = raw.get("global_shingle_budget", raw.get("shingle_point_budget"))
    quotas = raw.get("source_quotas", {})
    batch_size = raw.get("batch_size")
    interval = raw.get("snapshot_interval_batches")

    if not _is_int(workers) or workers <= 0:
        failures.append(_failure("max_workers must be a positive integer", str(path)))
    if not _is_int(budget) or budget <= 0:
        failures.append(_failure("global_shingle_budget must be a positive integer", str(path)))
    if not isinstance(quotas, dict):
        failures.append(_failure("source_quotas must be an object of non-negative integers", str(path)))
        cleaned_quotas: dict[str, int] = {}
    else:
        cleaned_quotas = {}
        for source, quota in quotas.items():
            if not isinstance(source, str) or not source:
                failures.append(_failure("source_quotas keys must be non-empty strings", str(path)))
            elif not _is_int(quota) or quota < 0:
                failures.append(_failure(f"quota for source {source!r} must be a non-negative integer", str(path)))
            else:
                cleaned_quotas[source] = int(quota)
    if not _is_int(batch_size) or batch_size <= 0:
        failures.append(_failure("batch_size must be a positive integer", str(path)))
    if not _is_int(interval) or interval <= 0:
        failures.append(_failure("snapshot_interval_batches must be a positive integer", str(path)))
    if failures:
        raise ConfigValidationError(failures)
    return SchedulerConfig(
        max_workers=int(workers),
        global_shingle_budget=int(budget),
        source_quotas=dict(sorted(cleaned_quotas.items())),
        batch_size=int(batch_size),
        snapshot_interval_batches=int(interval),
    )


def load_injection_config(path: str | Path | None) -> list[InjectionItem]:
    if path is None:
        return []
    path = Path(path)
    raw = _load_json(path)
    failures: list[Failure] = []
    items = raw.get("injections") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ConfigValidationError([_failure("injection config must be a list or contain an injections list", str(path))])

    parsed: list[InjectionItem] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            failures.append(_failure(f"injection item {index} must be an object", str(path)))
            continue
        injection_id = item.get("injection_id")
        source = item.get("source")
        doc_id = item.get("doc_id")
        fault = item.get("fault")
        if not isinstance(injection_id, str) or not injection_id:
            failures.append(_failure(f"injection item {index} needs a non-empty injection_id", str(path)))
            continue
        if injection_id in seen_ids:
            failures.append(_failure(f"duplicate injection_id {injection_id!r}", str(path)))
            continue
        seen_ids.add(injection_id)
        item_valid = True
        if not isinstance(source, str) or not source:
            failures.append(_failure(f"injection {injection_id!r} needs a non-empty source", str(path)))
            item_valid = False
        if not isinstance(doc_id, str) or not doc_id:
            failures.append(_failure(f"injection {injection_id!r} needs a non-empty doc_id", str(path)))
            item_valid = False
        if fault not in INJECTION_FAULTS:
            failures.append(
                _failure(
                    f"injection {injection_id!r} fault must be one of {', '.join(sorted(INJECTION_FAULTS))}",
                    str(path),
                )
            )
            item_valid = False
        if item_valid:
            parsed.append(
                InjectionItem(
                    injection_id=str(injection_id),
                    source=str(source),
                    doc_id=str(doc_id),
                    fault=str(fault),
                    business_code_hint=item.get("business_code_hint"),
                    metadata=item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {},
                )
            )
    if failures:
        raise ConfigValidationError(failures)
    return parsed


def load_all_configs(
    balance_path: str | Path,
    dedup_path: str | Path,
    scheduler_path: str | Path,
    injection_path: str | Path | None = None,
) -> tuple[BalanceConfig, DedupConfig, SchedulerConfig, list[InjectionItem]]:
    failures: list[Failure] = []
    configs: list[Any] = []
    loaders: Iterable[tuple[str, Any]] = (
        ("balance", load_balance_config),
        ("dedup", load_dedup_config),
        ("scheduler", load_scheduler_config),
        ("injection", load_injection_config),
    )
    paths = (balance_path, dedup_path, scheduler_path, injection_path)
    for (name, loader), path in zip(loaders, paths):
        try:
            configs.append(loader(path))
        except ConfigValidationError as exc:
            failures.extend(exc.failures)
            configs.append(None)
    if failures:
        raise ConfigValidationError(sorted(failures, key=Failure.sort_key))
    balance, dedup, scheduler, injections = configs
    return balance, dedup, scheduler, injections
