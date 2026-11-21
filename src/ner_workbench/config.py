from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .errors import E_CONFIG_INVALID, WorkbenchError


SNAPSHOT_FORMAT_VERSION = 1


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_canonical(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InferenceConfig:
    snapshot_format_version: int
    processing_unit: str
    window_chars: int
    window_stride: int
    char_batch_size: int
    max_iterations: int
    fixed_decimals: int
    quantization_level: int
    max_workers: int
    global_char_budget: int
    per_sample_quota: int
    cache_block_budget: int
    char_block_size: int
    latency_budget_ms: int
    snapshot_interval_batches: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return sha256_canonical(self.as_dict())

    def with_workers(self, workers: int | None) -> "InferenceConfig":
        if workers is None:
            return self
        return replace(self, max_workers=int(workers))


def _as_int(raw: dict[str, Any], key: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if value is None:
        raise WorkbenchError(E_CONFIG_INVALID, "load", f"missing config field: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WorkbenchError(E_CONFIG_INVALID, "load", f"config field {key} must be an integer") from exc


def load_config(path: Path, workers_override: int | None = None) -> InferenceConfig:
    try:
        raw_obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkbenchError(E_CONFIG_INVALID, "load", f"cannot read config: {path}") from exc
    except yaml.YAMLError as exc:
        raise WorkbenchError(E_CONFIG_INVALID, "load", f"invalid YAML config: {exc}") from exc
    if not isinstance(raw_obj, dict):
        raise WorkbenchError(E_CONFIG_INVALID, "load", "config root must be a mapping")
    raw: dict[str, Any] = raw_obj

    processing_unit = str(raw.get("processing_unit", raw.get("unit", "sentence"))).strip()
    processing_unit = processing_unit.replace("-", "_")
    if processing_unit in {"whole_sentence", "full_sentence", "sentence"}:
        processing_unit = "sentence"
    elif processing_unit in {"sliding", "slide_window", "sliding_window", "window"}:
        processing_unit = "sliding_window"
    else:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "load",
            f"unsupported processing_unit: {processing_unit!r}",
        )

    cfg = InferenceConfig(
        snapshot_format_version=_as_int(raw, "snapshot_format_version", SNAPSHOT_FORMAT_VERSION),
        processing_unit=processing_unit,
        window_chars=_as_int(raw, "window_chars", 12),
        window_stride=_as_int(raw, "window_stride", 10),
        char_batch_size=_as_int(raw, "char_batch_size", 4),
        max_iterations=_as_int(raw, "max_iterations", 512),
        fixed_decimals=_as_int(raw, "fixed_decimals", 4),
        quantization_level=_as_int(raw, "quantization_level", 6),
        max_workers=_as_int(raw, "max_workers", 4),
        global_char_budget=_as_int(raw, "global_char_budget", 4096),
        per_sample_quota=_as_int(raw, "per_sample_quota", 128),
        cache_block_budget=_as_int(raw, "cache_blocks", raw.get("cache_block_budget", 8)),
        char_block_size=_as_int(raw, "char_block_size", 4),
        latency_budget_ms=_as_int(raw, "latency_budget_ms", 500),
        snapshot_interval_batches=_as_int(raw, "snapshot_interval_batches", 4),
        seed=_as_int(raw, "seed", 20261121),
    ).with_workers(workers_override)
    validate_config(cfg)
    return cfg


def validate_config(cfg: InferenceConfig) -> None:
    checks = {
        "snapshot_format_version": cfg.snapshot_format_version == SNAPSHOT_FORMAT_VERSION,
        "window_chars": cfg.window_chars >= 1,
        "window_stride": 1 <= cfg.window_stride <= cfg.window_chars,
        "char_batch_size": cfg.char_batch_size >= 1,
        "max_iterations": cfg.max_iterations >= 1,
        "fixed_decimals": 0 <= cfg.fixed_decimals <= 12,
        "quantization_level": 0 <= cfg.quantization_level <= 30,
        "max_workers": cfg.max_workers >= 1,
        "global_char_budget": cfg.global_char_budget >= 0,
        "per_sample_quota": cfg.per_sample_quota >= 0,
        "cache_block_budget": cfg.cache_block_budget >= 0,
        "char_block_size": cfg.char_block_size >= 1,
        "latency_budget_ms": cfg.latency_budget_ms >= 0,
        "snapshot_interval_batches": cfg.snapshot_interval_batches >= 1,
    }
    invalid = [name for name, ok in checks.items() if not ok]
    if invalid:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "load",
            "invalid config values: " + ", ".join(invalid),
        )
